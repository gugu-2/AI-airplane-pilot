"""
Aegis OS - AirSim Bridge
=========================
Connects your trained PPO brain to Microsoft AirSim for real 3D flight testing.

HOW IT WORKS:
    1. AirSim runs a photorealistic Unreal Engine drone simulation
    2. This script reads position / velocity / orientation from AirSim
    3. Converts the real sensor data into your model's 10-dim physics format
    4. Reads 32 distance sensors for the depth scan
    5. Runs your trained VisionPPOActor to compute the best action
    6. Sends pitch / roll / yaw / throttle commands back to AirSim at 20Hz

USAGE:
    1. Start AirSim (double-click the .exe you downloaded)
    2. Open a terminal in AI-airplane-pilot/ and run:
       python scripts/airsim_bridge.py

OPTIONAL ARGUMENTS:
    --model   path to .pth file  (default: newest in models/)
    --target  x y z              (default: fly to 50m ahead, 20m up)
    --speed   control Hz         (default: 20)

REQUIREMENTS:
    - AirSim running (Blocks or Neighborhood environment)
    - airsim_client/ folder in this project root (already set up)
"""

import sys
import os
import argparse
import time
import glob
import math
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'airsim_client'))

try:
    import airsim
    print(f"[OK] AirSim {airsim.__version__} ready.")
except ImportError:
    print("[ERROR] AirSim not installed!")
    print("        Run:  pip install airsim --no-build-isolation")
    sys.exit(1)

from cognitive.rl_models import VisionPPOActor


# ==============================================================================
# HELPERS
# ==============================================================================

def find_best_model():
    """Find the newest valid checkpoint in the models/ directory."""
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    skip = ['aegis_pilot_v1', 'aegis_vision', 'flight_rl']
    pths = glob.glob(os.path.join(model_dir, '**', '*.pth'), recursive=True)
    pths = [p for p in pths if not any(s in p for s in skip)]
    if not pths:
        raise FileNotFoundError("No checkpoint found in models/ folder!")
    pths.sort(key=os.path.getmtime)
    return pths[-1]


def quaternion_to_euler(q):
    """Convert AirSim quaternion to (pitch, roll, yaw) in radians."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return pitch, roll, yaw


# ==============================================================================
# AIRSIM BRIDGE
# ==============================================================================

class AirSimBridge:
    """
    Real-time bridge between the trained VisionPPOActor and AirSim.
    Runs at ~20Hz sending control commands based on the AI's inference.
    """

    NUM_RAYS = 32         # must match training (32 distance sensors)
    MAX_RANGE = 50.0      # metres — matches training normalisation
    CONTROL_HZ = 20       # inference frequency
    DT = 1.0 / CONTROL_HZ

    def __init__(self, model_path: str, target_xyz: list, vehicle_name: str = ""):
        self.vehicle_name = vehicle_name
        # Convert target from AirSim NED (Z down) to Model (Z up)
        self.target = np.array([target_xyz[0], target_xyz[1], -target_xyz[2]], dtype=np.float32)

        # ── Load model ─────────────────────────────────────────────────────
        print(f"\nLoading brain: {os.path.basename(model_path)}")
        ckpt      = torch.load(model_path, map_location='cpu', weights_only=False)
        phys_dim  = ckpt['actor']['physics_net.0.weight'].shape[1]
        session   = ckpt.get('session', '?')
        update    = ckpt.get('update', '?')
        steps     = ckpt.get('total_steps', 0)
        print(f"  Session {session} | Update {update} | Steps {steps:,} | Dim {phys_dim}")

        self.actor = VisionPPOActor(depth_rays=self.NUM_RAYS,
                                    physics_dim=phys_dim, action_dim=4)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.eval()
        print("  Brain loaded!")

        self.vehicle_type_val = 0.0   # 0 = drone, 1 = fixed-wing

        # ── Connect to AirSim ──────────────────────────────────────────────
        print("\nConnecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        print("  Connected to AirSim!")
        print(f"  Target position (NED): x={target_xyz[0]:.1f} y={target_xyz[1]:.1f} z={target_xyz[2]:.1f}")

    def get_terrain_height(self, x, y):
        # No hills active — just return flat ground (Z=0 in NED is sky, 0 = ground)
        return 0.0

    # ── Sensor reading ──────────────────────────────────────────────────────

    def _read_depth_scan(self):
        """Compute synthetic LiDAR from stored obstacle positions.
        
        AirSim's ray-cast sensors CANNOT detect objects spawned with physics_enabled=False.
        They return 1.0 (max range) for every ray, making the AI effectively blind.
        
        Instead, we analytically compute the 32 ray distances directly from the obstacle
        positions stored in self.solid_obstacles — exactly as the training env did.
        Each of the 32 rays sweeps 360° in 11.25° increments relative to drone heading.
        """
        depths = np.ones(self.NUM_RAYS, dtype=np.float32)  # default = max range (no obstacle)

        if not hasattr(self, 'solid_obstacles') or not self.solid_obstacles:
            return depths

        # Get current drone pose for ray directions
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kine  = state.kinematics_estimated
        drone_x = kine.position.x_val
        drone_y = kine.position.y_val

        _, _, yaw = quaternion_to_euler(kine.orientation)

        import math
        for i in range(self.NUM_RAYS):
            # ── MATCH TRAINING LIDAR ──
            # Training used: torch.linspace(-math.pi/2, math.pi/2, 32)
            # Ray 0 is -90 deg (Right in training), Ray 31 is +90 deg (Left in training).
            # To mirror to AirSim's NED (where Right is +90 deg and Left is -90 deg):
            # AirSim relative angle = - (Training Angle)
            
            theta_train = -math.pi/2.0 + i * (math.pi / 31.0)
            theta_airsim_relative = -theta_train
            
            ray_angle = yaw + theta_airsim_relative
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            min_dist = self.MAX_RANGE

            for obs in self.solid_obstacles:
                # Vector from drone to obstacle center
                cx = obs['x'] - drone_x
                cy = obs['y'] - drone_y
                r  = obs['hit_dist']   # obstacle collision radius

                # Ray-circle intersection: solve |P + t*D - C|^2 = r^2
                # where P=drone pos, D=ray dir, C=obstacle center
                # Simplified (P is origin): |t*D - C|^2 = r^2
                # => t^2 - 2t(D·C) + (|C|^2 - r^2) = 0
                dot = ray_dx * cx + ray_dy * cy
                disc = dot * dot - (cx*cx + cy*cy - r*r)

                if disc < 0:
                    continue  # ray misses this obstacle

                t = dot - math.sqrt(disc)   # nearest intersection
                if 0 < t < min_dist:
                    min_dist = t

            depths[i] = min(min_dist, self.MAX_RANGE) / self.MAX_RANGE

        return depths

    def _read_state(self, depth_arr: np.ndarray):
        """Read kinematics from AirSim and spoof altitude for horizontal/over-obstacle flight."""
        """Read kinematics and map AirSim NED coordinates to Training Cartesian coordinates."""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kine = state.kinematics_estimated
        
        # Global NED positions (for logging)
        pos = np.array([kine.position.x_val, kine.position.y_val, -kine.position.z_val], dtype=np.float32)
        vel_global = np.array([kine.linear_velocity.x_val, kine.linear_velocity.y_val, -kine.linear_velocity.z_val], dtype=np.float32)

        # ── MAP AIRSIM NED TO TRAINING CARTESIAN ──
        # Training: +X Forward, +Y Left, +Z Up
        # AirSim:   +X North,   +Y East (Right), +Z Down
        
        # 1. Target Vector (Mirror Y)
        dist_vec_train = np.zeros(3, dtype=np.float32)
        dist_vec_train[0] = self.target[0] - kine.position.x_val
        dist_vec_train[1] = -(self.target[1] - kine.position.y_val) # Negate Y!
        
        norm_dist = np.zeros(3, dtype=np.float32)
        norm_dist[0] = np.clip(dist_vec_train[0] / 50.0, -1.0, 1.0)
        norm_dist[1] = np.clip(dist_vec_train[1] / 50.0, -1.0, 1.0)
        norm_dist[2] = 0.0  # Force flat altitude

        # 2. Velocity (Mirror Y and Z)
        norm_vel = np.zeros(3, dtype=np.float32)
        norm_vel[0] = np.clip(kine.linear_velocity.x_val / 15.0, -1.0, 1.0)
        norm_vel[1] = np.clip(-kine.linear_velocity.y_val / 15.0, -1.0, 1.0) # Negate Y!
        norm_vel[2] = np.clip(-kine.linear_velocity.z_val / 15.0, -1.0, 1.0) # Negate Z!

        # 3. Attitude (Mirror Roll and Yaw)
        pitch, roll, yaw = quaternion_to_euler(kine.orientation)
        attitude = np.array([pitch, -roll, -yaw], dtype=np.float32) # Negate Roll and Yaw!

        physics = np.concatenate([
            attitude, norm_vel, norm_dist, [self.vehicle_type_val]
        ]).astype(np.float32)
        return pos, vel_global, attitude, physics

    # ── Action → AirSim command ─────────────────────────────────────────────

    def _send_action(self, action: np.ndarray):
        """Map network actions to AirSim commands."""
        # ── MAP TRAINING ACTIONS TO AIRSIM ──
        # Training Action:
        # action[0]: Pitch (positive = nose down) -> AirSim Pitch (positive = nose down)
        # action[1]: Roll (positive = lean left)  -> AirSim Roll (positive = lean right)
        # action[2]: Yaw Rate (positive = turn left) -> AirSim Yaw Rate (positive = turn right)
        
        # We must strictly cap the pitch. The AI's vision is trained up to 15 m/s. 
        # If it flies at 20 m/s, its sensors clip and it cannot brake in time.
        flu_pitch    = float(action[0]) * 0.15  # Cap speed at ~12 m/s
        flu_roll     = -float(action[1]) * 0.35 # Negate to map left to left
        flu_yaw_rate = -float(action[2]) * 1.00 # Negate to map left to left
        
        # Altitude Hold (P-Controller on Throttle, NOT pitch)
        desired_physical_z = -8.0  # Cruise at 8 meters
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        z_val = state.kinematics_estimated.position.z_val
        vz_val = state.kinematics_estimated.linear_velocity.z_val
        
        z_error = z_val - desired_physical_z  # e.g. -0.7 - (-8.0) = +7.3 (we are too low, need to go up!)
        
        # AirSim hover throttle is ~0.59
        base_throttle = 0.59
        p_term = z_error * 0.15  # Increased P-term for stronger hold
        
        # vz_val is POSITIVE when falling DOWN. 
        # If we are falling, we need MORE throttle to stop falling, so we ADD the d_term.
        d_term = vz_val * 0.10
        
        throttle = float(np.clip(base_throttle + p_term + d_term, 0.0, 1.0))

        self.client.moveByRollPitchYawrateThrottleAsync(
            flu_roll, flu_pitch, flu_yaw_rate, throttle,
            duration=0.5,                     # Prevents 'API call not received' hover safety mode
            vehicle_name=self.vehicle_name
        )

    # ── Obstacles ───────────────────────────────────────────────────────────

    def _spawn_obstacles(self, start_pos: np.ndarray):
        """Spawns a dense forest of randomly sized pillars for the AI to weave through."""
        

        print("  Cleaning up old obstacles and spawning a cylinder gauntlet...")
        try:
            for i in range(1200):
                self.client.simDestroyObject(f"AegisObstacle_{i}")
        except:
            pass

        try:
            import math
            self.hills = []
            self.solid_obstacles = []
            
            # Spawn 1000 pure cylinders for a very dense forest!
            # Safe zone of 100m gives the AI time to build speed and react before its first obstacle.
            # Density increases gradually: tight in inner ring, sparser in outer ring.
            count = 1000
            for i in range(count):
                obj_name = f"AegisObstacle_{i}"

                # Graduated density: 60% of obstacles in 100-250m range, 40% in 250-600m
                if i < int(count * 0.6):
                    radius = 100.0 + np.random.rand() * 150.0   # Inner ring: 100-250m
                else:
                    radius = 250.0 + np.random.rand() * 350.0   # Outer ring: 250-600m

                angle = np.random.rand() * 2 * math.pi
                obs_x = float(start_pos[0]) + radius * math.cos(angle)
                obs_y = float(start_pos[1]) + radius * math.sin(angle)
                obs_z = 0.0

                # Width: 5.0m to match original training distribution (OBS_RADIUS = 2.5)
                # The skinny 1m pillars were slipping perfectly between the 32 LiDAR rays!
                width = 5.0
                # Height: 30m — always taller than drone's cruise altitude
                height = 30.0

                pose  = airsim.Pose(airsim.Vector3r(obs_x, obs_y, obs_z), airsim.to_quaternion(0, 0, 0))
                scale = airsim.Vector3r(width, width, height)
                self.client.simSpawnObject(obj_name, "Cylinder", pose, scale, physics_enabled=False)

                self.solid_obstacles.append({
                    'name':     obj_name,
                    'x':        obs_x,
                    'y':        obs_y,
                    'hit_dist': (width / 2.0) + 0.7   # radius + drone bounding box
                })

            print(f"  {count} cylinder obstacles spawned (safe zone: 100m, cruise alt: 8m)")
        except Exception as e:
            print(f"  [Notice] Could not spawn objects: {e}")
        except Exception as e:
            print(f"  [Notice] Could not spawn objects: {e}")

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self, max_steps: int = 3000):
        """Take off and hand control to the AI."""
        print("\n" + "="*60)
        
        # Teleport to a completely open field (known safe location from previous logs!)
        print("  Teleporting drone to safe open field...")
        pose = airsim.Pose(airsim.Vector3r(777.2, -5119.1, -2.0), airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(pose, True)
        time.sleep(1.0)
        
        # Read initial position to make the target relative
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        start_x = state.kinematics_estimated.position.x_val
        start_y = state.kinematics_estimated.position.y_val
        start_z = -state.kinematics_estimated.position.z_val  # Convert to Z up
        
        # Target is placed inside the massive 600m radius obstacle circle!
        self.target = np.array([start_x + 500.0, start_y, start_z - 20.0], dtype=np.float32)
        print(f"  Spawned at: X={start_x:.1f}, Y={start_y:.1f}")
        print(f"  New Target: X={self.target[0]:.1f}, Y={self.target[1]:.1f}, Z={self.target[2]:.1f}")
        
        # Spawn our custom pillars in this new open area
        self._spawn_obstacles(np.array([start_x, start_y, start_z]))
        
        print("  Taking off...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(1.0)
        print("  AIRBORNE! Aegis AI has control.")
        print("  Press Ctrl+C at any time to land and exit.")
        print("="*60)
        print(f"\n  {'Step':>5} | {'X':>7} | {'Y':>7} | {'Z':>7} | "
              f"{'Speed':>7} | {'DistToTarget':>12}")
        print(f"  {'-'*60}")

        for step in range(max_steps):
            t0 = time.time()

            # 1. Read sensors
            depth_arr = self._read_depth_scan()
            pos, vel, attitude, physics = self._read_state(depth_arr)

            # Manual collision check against spawned obstacles
            drone_xy = np.array([pos[0], pos[1]])
            crashed = False
            for obs in getattr(self, 'solid_obstacles', []):
                if np.linalg.norm(drone_xy - np.array([obs['x'], obs['y']])) < obs['hit_dist']:
                    print(f"\n  [CRASH] Drone smashed head-on into {obs['name']} at {np.linalg.norm(vel):.1f} m/s!")
                    crashed = True
                    break
            
            if crashed:
                # FREEZE the drone in mid-air and immediately exit the simulation
                self.client.moveByVelocityAsync(0, 0, 0, 0.1).join()
                import sys
                sys.exit(1)

            # 2. Convert to tensors
            depth_t = torch.FloatTensor(depth_arr).unsqueeze(0).unsqueeze(0)  # (1,1,32)
            phys_t  = torch.FloatTensor(physics).unsqueeze(0)                 # (1,10)

            # 3. Run inference (deterministic — no exploration noise)
            with torch.no_grad():
                mean, _ = self.actor(depth_t, phys_t)
                action  = torch.clamp(mean, -1.0, 1.0).numpy()[0]

            # 4. Send command to AirSim
            self._send_action(action)

            # 5. Log progress every second (~20 steps)
            if step % 20 == 0:
                dist   = float(np.linalg.norm(self.target - pos))
                speed  = float(np.linalg.norm(vel))
                print(f"  {step:>5} | {pos[0]:>7.1f} | {pos[1]:>7.1f} | {pos[2]:>7.1f} | "
                      f"{speed:>7.2f} | {dist:>12.1f}m")

                if dist < 3.0:
                    print("\n  TARGET REACHED!")
                    break

            # 6. Maintain control rate
            elapsed = time.time() - t0
            sleep_t = max(0.0, self.DT - elapsed)
            time.sleep(sleep_t)

        # Land safely ON the terrain!
        print("\n  Landing on terrain...")
        try:
            # Determine how high the ground is directly below the drone
            final_terrain_z = self.get_terrain_height(pos[0], pos[1])
            # Tell the drone to descend precisely to that height! (Subtract 0.5m so it sits on top)
            self.client.moveToZAsync(final_terrain_z - 0.5, velocity=2.0, vehicle_name=self.vehicle_name).join()
        except:
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            
        self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        print("  Landed successfully! Session complete.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aegis OS AirSim Bridge - AI drone flight in Unreal Engine"
    )
    parser.add_argument("--model",  type=str, default=None,
                        help="Path to .pth brain file (default: newest in models/)")
    parser.add_argument("--target", type=float, nargs=3, default=[50.0, 0.0, -20.0],
                        metavar=("X", "Y", "Z"),
                        help="Target position in NED metres (default: 50 0 -20)")
    parser.add_argument("--steps",  type=int, default=3000,
                        help="Max inference steps (default: 3000 = 2.5 min at 20Hz)")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    print(f"\nUsing model: {model_path}")

    bridge = AirSimBridge(
        model_path=model_path,
        target_xyz=args.target
    )
    bridge.run(max_steps=args.steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted. Landing...")
