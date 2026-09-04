import torch
import torch.nn as nn
import math
import random
import os

class DeepQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for Autonomous Path Planning.
    Takes a 5-dimensional state vector and outputs Q-values for 5 discrete actions.
    """
    def __init__(self, state_dim=5, action_dim=5):
        super(DeepQNetwork, self).__init__()
        
        # 3 Hidden layers for complex non-linear spatial reasoning
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class RLInferenceEngine:
    """
    Wrapper for the PyTorch Neural Network.
    Translates physical drone GPS and Semantic Map data into tensors, runs inference,
    and translates the raw tensor output back into physical GPS maneuvers.
    """
    def __init__(self, model_path=None):
        # Forced to CPU because RTX 5050 (sm_120) is too new for PyTorch stable
        self.device = torch.device("cpu")
        print(f"[Cognitive-RL] Initializing PyTorch DQN on {self.device}...")
        
        self.model = DeepQNetwork(state_dim=5, action_dim=5).to(self.device)
        
        # Look for the default trained weights if no path is provided
        if model_path is None:
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/aegis_pilot_v1.pth'))
            if os.path.exists(default_path):
                model_path = default_path
                
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                print(f"[Cognitive-RL] SUCCESS: Loaded pre-trained weights from {model_path}")
                self.untrained_mode = False
            except Exception as e:
                print(f"[Cognitive-RL] WARNING: Could not load weights: {e}")
                self.untrained_mode = True
        else:
            print("[Cognitive-RL] WARNING: No pre-trained weights found. Using rule-based heuristic fallback.")
            self.untrained_mode = True
                
        self.model.eval()
        
        # Action space mapping
        self.actions = {
            0: "MAINTAIN_HEADING",
            1: "VEER_LEFT",
            2: "VEER_RIGHT",
            3: "CLIMB",
            4: "DESCEND"
        }

    def _normalize_state(self, current_lat, current_lon, dest_lat, dest_lon, obstacle_dist):
        """Converts raw physics data into normalized tensor inputs (-1.0 to 1.0).
        B3 FIX: Removed random noise from state vector.
        The same physical state must always produce the same Q-values (determinism).
        """
        dist_x = dest_lon - current_lon
        dist_y = dest_lat - current_lat
        norm_x = max(-1.0, min(1.0, dist_x / 0.005))
        norm_y = max(-1.0, min(1.0, dist_y / 0.005))
        heading = math.atan2(dist_y, dist_x)
        norm_heading = heading / math.pi
        # Obstacle proximity (0 = far, 1 = extremely close)
        norm_obs = max(0.0, min(1.0, 1.0 / (obstacle_dist + 0.1)))
        state_vector = [norm_x, norm_y, norm_heading, norm_obs, 0.0]  # 5th dim = reserved
        return torch.FloatTensor(state_vector).to(self.device)

    def decide_next_action(self, current_lat, current_lon, dest_lat, dest_lon, obstacle_dist=100.0):
        """
        Runs the state through the Neural Network and returns a physical waypoint delta.
        B3 FIX: When no trained model is loaded, falls back to a deterministic rule-based
                heuristic (always face the target, climb if obstacle close) instead of
                random noise-driven inference.
        """
        # Rule-based fallback for untrained model
        if self.untrained_mode:
            dist_x = dest_lon - current_lon
            dist_y = dest_lat - current_lat
            angle = math.atan2(dist_y, dist_x)
            base_step = 0.0001
            if obstacle_dist < 20.0:
                return "CLIMB", math.sin(angle) * base_step * 0.5, math.cos(angle) * base_step * 0.5, 5.0
            return "MAINTAIN_HEADING", math.sin(angle) * base_step, math.cos(angle) * base_step, 0.0

        state_tensor = self._normalize_state(current_lat, current_lon, dest_lat, dest_lon, obstacle_dist)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            best_action_idx = torch.argmax(q_values).item()
            
        action_name = self.actions[best_action_idx]
        
        # In a fully trained environment, we follow exactly. Here we add heuristics
        # to ensure the untrained drone still generally approaches the target.
        base_step = 0.0001
        delta_lat = 0.0
        delta_lon = 0.0
        delta_alt = 0.0
        
        # Point towards target primarily
        dist_x = dest_lon - current_lon
        dist_y = dest_lat - current_lat
        angle = math.atan2(dist_y, dist_x)
        
        if action_name == "MAINTAIN_HEADING":
            delta_lat = math.sin(angle) * base_step
            delta_lon = math.cos(angle) * base_step
        elif action_name == "VEER_LEFT":
            delta_lat = math.sin(angle + 0.5) * base_step
            delta_lon = math.cos(angle + 0.5) * base_step
        elif action_name == "VEER_RIGHT":
            delta_lat = math.sin(angle - 0.5) * base_step
            delta_lon = math.cos(angle - 0.5) * base_step
        elif action_name == "CLIMB":
            delta_lat = math.sin(angle) * base_step * 0.5
            delta_lon = math.cos(angle) * base_step * 0.5
            delta_alt = 5.0
        elif action_name == "DESCEND":
            delta_lat = math.sin(angle) * base_step * 0.5
            delta_lon = math.cos(angle) * base_step * 0.5
            delta_alt = -2.0

        return action_name, delta_lat, delta_lon, delta_alt

# =======================================================================
# PHASE 2: True 3D Flight & Perception (Continuous PPO + Vision)
# =======================================================================

class VisionPPOActor(nn.Module):
    """
    Proximal Policy Optimization (PPO) Actor Network.
    Fuses 1D Depth/LiDAR scan (32 rays) with a 9-DOF flight state 
    to output continuous control signals (Pitch, Roll, Yaw, Thrust).
    """
    def __init__(self, depth_rays=32, physics_dim=9, action_dim=4):
        super(VisionPPOActor, self).__init__()
        
        # Convolutional feature extractor for Depth/LiDAR scan
        self.vision_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (depth_rays // 4), 64),
            nn.ReLU()
        )
        
        # Physics state feature extractor
        self.physics_net = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.ReLU()
        )
        
        # Fused decision layers
        self.actor_net = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Output actions in [-1.0, 1.0] range
        )
        
        # Log standard deviation for exploration in continuous space
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, depth_scan, physics_state):
        # depth_scan shape: (batch_size, 1, depth_rays)
        vis_features = self.vision_net(depth_scan)
        phys_features = self.physics_net(physics_state)
        
        fused = torch.cat([vis_features, phys_features], dim=1)
        mean_actions = self.actor_net(fused)
        
        std = self.log_std.exp().expand_as(mean_actions)
        return mean_actions, std


class VisionPPOCritic(nn.Module):
    """
    PPO Critic Network to estimate the Value Function V(s).
    """
    def __init__(self, depth_rays=32, physics_dim=9):
        super(VisionPPOCritic, self).__init__()
        
        self.vision_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (depth_rays // 4), 64),
            nn.ReLU()
        )
        
        self.physics_net = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.ReLU()
        )
        
        self.critic_net = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single scalar value
        )

    def forward(self, depth_scan, physics_state):
        # BUG 6 FIX: vis_features was never computed — caused NameError crash on every run.
        # The line `vis_features = self.vision_net(depth_scan)` was completely missing.
        vis_features = self.vision_net(depth_scan)
        phys_features = self.physics_net(physics_state)
        fused = torch.cat([vis_features, phys_features], dim=1)
        return self.critic_net(fused)

class VisionRLInferenceEngine:
    """
    Wrapper for the 3D Vision PPO Actor.
    Translates depth scans and 9-DOF physical data into continuous flight controls.
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        print(f"[Vision-RL] Initializing PyTorch PPO Actor on {self.device}...")
        
        self.model = VisionPPOActor(depth_rays=32, physics_dim=9, action_dim=4).to(self.device)
        
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/aegis_vision_v1-3d.pth'))
            
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
                print(f"[Vision-RL] SUCCESS: Loaded pre-trained PPO weights from {model_path}")
            except Exception as e:
                print(f"[Vision-RL] ERROR: Could not load weights: {e}")
        else:
            print(f"[Vision-RL] WARNING: Model weights not found at {model_path}")
            
        self.model.eval()

    def _normalize_state(self, current_lat, current_lon, dest_lat, dest_lon):
        # Rough distance (scaled to [-1, 1] range based on 20 meter bounds as in training)
        dist_x = (dest_lon - current_lon) * 111111.0 # approx meters
        dist_y = (dest_lat - current_lat) * 111111.0
        dist_z = 0.0 # assume flat target for now
        
        norm_x = max(-1.0, min(1.0, dist_x / 20.0))
        norm_y = max(-1.0, min(1.0, dist_y / 20.0))
        norm_z = max(-1.0, min(1.0, dist_z / 20.0))
        
        # Mocking Attitude and Velocity since we don't have full IMU stream here
        physics_state = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, norm_x, norm_y, norm_z]]).to(self.device)
        return physics_state

    def decide_next_action(self, current_lat, current_lon, dest_lat, dest_lon, depth_scan=None):
        """
        Runs the 32-ray depth scan and 9-DOF state through the PPO Actor.
        Returns continuous physical controls mapped to lat/lon/alt deltas.
        """
        physics_state = self._normalize_state(current_lat, current_lon, dest_lat, dest_lon)
        
        if depth_scan is None:
            depth_scan = torch.ones((1, 1, 32)).to(self.device) # Fallback to flat/safe depth scan
        else:
            depth_scan = depth_scan.to(self.device)
            
        with torch.no_grad():
            mean_actions, _ = self.model(depth_scan, physics_state)
            
        actions = mean_actions[0].cpu().numpy()
        pitch_cmd = actions[0] # [-1.0, 1.0]
        roll_cmd = actions[1]  # [-1.0, 1.0]
        yaw_cmd = actions[2]   # [-1.0, 1.0]
        thrust_cmd = actions[3] # [-1.0, 1.0]
        
        action_name = "PPO_CONTINUOUS_CTRL"
        if thrust_cmd > 0.5: action_name = "PPO_CLIMB_ACCEL"
        elif thrust_cmd < -0.5: action_name = "PPO_DESCEND"
        elif pitch_cmd > 0.5: action_name = "PPO_PITCH_FWD"
        elif roll_cmd > 0.5: action_name = "PPO_ROLL_RIGHT"
        elif roll_cmd < -0.5: action_name = "PPO_ROLL_LEFT"
        
        base_step = 0.0001
        
        # Point towards target primarily to calculate heading
        dist_x = dest_lon - current_lon
        dist_y = dest_lat - current_lat
        heading = math.atan2(dist_y, dist_x)
        
        # Thrust maps to Altitude delta. [-1, 1] mapped to [-2.0, 5.0]
        delta_alt = ((thrust_cmd + 1.0) / 2.0) * 7.0 - 2.0 
        
        # Pitch maps to forward velocity along heading
        forward_step = pitch_cmd * base_step
        
        # Roll maps to lateral velocity perpendicular to heading
        lateral_step = roll_cmd * base_step
        
        delta_lat = float(math.sin(heading) * forward_step + math.cos(heading) * lateral_step)
        delta_lon = float(math.cos(heading) * forward_step - math.sin(heading) * lateral_step)

        return action_name, delta_lat, delta_lon, float(delta_alt)
