import sys
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    Point3, Vec3, NodePath,
    CollisionNode, CollisionRay, CollisionBox, CollisionTraverser, CollisionHandlerQueue,
    DirectionalLight, AmbientLight, CardMaker, LColor
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from cognitive.rl_models import VisionPPOActor

# ── PANDA3D SIMULATOR ───────────────────────────────────────────────────
class PandaDroneSim(ShowBase):
    def __init__(self):
        super().__init__()
        
        # Enable basic shaders and shadows
        self.render.setShaderAuto()
        
        # Darker cinematic sky
        self.setBackgroundColor(0.2, 0.4, 0.6)
        
        self.NUM_RAYS = 32
        self.MAX_RANGE = 50.0
        
        # Load PyTorch Brain
        self.actor = VisionPPOActor(depth_rays=self.NUM_RAYS, physics_dim=10, action_dim=4)
        print("Loading AI Brain...")
        ckpt = torch.load("models/session3_final.pth", map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.eval()
        print("Brain Loaded!")
        
        # Physics / State
        self.dt = 0.05
        self.drone_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.drone_yaw = 0.0
        # Increased course length to 500m
        self.target_pos = np.array([0.0, 500.0, 2.0], dtype=np.float32)
        self.step_count = 0
        
        # Camera smoothing variables
        self.cam_pos = np.array([0.0, -15.0, 6.0], dtype=np.float32)
        
        self._setup_lighting()
        self._setup_world()
        self._setup_sensors()
        
        # Schedule AI loop
        self.taskMgr.add(self.update_ai, "update_ai_task")

    def _setup_lighting(self):
        # Ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
        self.render.attachNewNode(alight)
        
        # Sun (Directional Light with shadows)
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlight.setShadowCaster(True, 1024, 1024)
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -45, 0) # Angle the sun
        self.render.setLight(dlnp)

    def _create_drone_model(self):
        """Builds a cool quadcopter out of basic blocks"""
        drone_group = NodePath("drone_group")
        
        # Main body (black)
        body = drone_group.attachNewNode(CollisionNode('body'))
        body.node().addSolid(CollisionBox(Point3(0,0,0), 0.3, 0.4, 0.15))
        body.show()
        body.setColor(0.1, 0.1, 0.1, 1)
        
        # Arms and rotors (white)
        arm_positions = [(-0.4, 0.5), (0.4, 0.5), (-0.4, -0.5), (0.4, -0.5)]
        for (ax, ay) in arm_positions:
            arm = drone_group.attachNewNode(CollisionNode('arm'))
            arm.node().addSolid(CollisionBox(Point3(ax, ay, 0), 0.15, 0.15, 0.05))
            arm.show()
            arm.setColor(0.9, 0.9, 0.9, 1)
            
            # Rotor (red)
            rotor = drone_group.attachNewNode(CollisionNode('rotor'))
            rotor.node().addSolid(CollisionBox(Point3(ax, ay, 0.1), 0.2, 0.2, 0.02))
            rotor.show()
            rotor.setColor(1.0, 0.2, 0.2, 1)
            
        return drone_group

    def _setup_world(self):
        # Ground (receives shadows)
        cm = CardMaker('ground')
        cm.setFrame(-200, 200, -50, 600)
        ground = self.render.attachNewNode(cm.generate())
        ground.lookAt(0, 0, -1)
        ground.setColor(0.2, 0.6, 0.2, 1)
        
        # Drone
        self.drone = self._create_drone_model()
        self.drone.reparentTo(self.render)
        self.drone.setPos(*self.drone_pos)
        
        # Target visual
        target_box = CollisionBox(Point3(0,0,0), 2.0, 2.0, 20.0)
        target_node = CollisionNode('target_geom')
        target_node.addSolid(target_box)
        target = self.render.attachNewNode(target_node)
        target.show()
        target.setColor(0.0, 0.5, 1.0, 0.5) # Semi-transparent blue
        target.setPos(*self.target_pos)
        
        # Slalom Pillars (Physics Obstacles)
        self.moving_obstacles = []
        
        # 50 dense pillars
        for i in range(50):
            obs_y = 20.0 + i * 9.0 # Closer together!
            obs_x = (np.random.rand() - 0.5) * 30.0 # Wider spread
            width = 1.0 + np.random.rand() * 1.5
            
            box = CollisionBox(Point3(0,0,5.0), width/2, width/2, 5.0) # Taller pillars
            cnode = CollisionNode(f'pillar_{i}')
            cnode.addSolid(box)
            cnode.setIntoCollideMask(1)
            
            np_node = self.render.attachNewNode(cnode)
            np_node.setPos(obs_x, obs_y, 0)
            np_node.show()
            
            # Make 30% of them moving!
            if np.random.rand() < 0.3:
                np_node.setColor(1.0, 0.5, 0.0, 1) # Orange moving pillars
                self.moving_obstacles.append({
                    'node': np_node,
                    'base_x': obs_x,
                    'amplitude': 5.0 + np.random.rand() * 10.0,
                    'speed': 0.5 + np.random.rand() * 1.0,
                    'phase': np.random.rand() * np.pi * 2
                })
            else:
                np_node.setColor(0.8, 0.1, 0.1, 1) # Red stationary pillars

    def _setup_sensors(self):
        self.traverser = CollisionTraverser('traverser')
        self.queue = CollisionHandlerQueue()
        
        self.ray_np = self.drone.attachNewNode(CollisionNode('rays'))
        self.ray_np.node().setFromCollideMask(1)
        self.ray_np.node().setIntoCollideMask(0)
        
        self.rays = []
        fov_rad = np.pi # [-pi/2 to pi/2]
        start_angle = -fov_rad / 2
        angle_step = fov_rad / max(1, (self.NUM_RAYS - 1))
        
        for i in range(self.NUM_RAYS):
            angle = start_angle + i * angle_step
            dir_x = math.sin(angle)
            dir_y = math.cos(angle)
            
            ray = CollisionRay()
            ray.setOrigin(0, 0, 0)
            ray.setDirection(dir_x, dir_y, 0)
            self.ray_np.node().addSolid(ray)
            self.rays.append(ray)
            
        self.traverser.addCollider(self.ray_np, self.queue)

    def read_depth(self):
        self.traverser.traverse(self.render)
        depths = np.ones(self.NUM_RAYS, dtype=np.float32)
        hits = {}
        for i in range(self.queue.getNumEntries()):
            entry = self.queue.getEntry(i)
            solid_idx = self.ray_np.node().getSolids().index(entry.getFrom())
            dist = (entry.getSurfacePoint(self.render) - self.drone.getPos()).length()
            if solid_idx not in hits or dist < hits[solid_idx]:
                hits[solid_idx] = dist
                
        for i in range(self.NUM_RAYS):
            if i in hits:
                d = hits[i]
                depths[i] = min(d, self.MAX_RANGE) / self.MAX_RANGE
        return depths

    def update_ai(self, task):
        self.step_count += 1
        t = task.time
        
        # Update moving obstacles
        for m in self.moving_obstacles:
            node = m['node']
            new_x = m['base_x'] + math.sin(t * m['speed'] + m['phase']) * m['amplitude']
            node.setX(new_x)
            
        # 1. Read sensors
        depth_arr = self.read_depth()
        
        # 2. Physics logic (convert to AirSim Body Frame)
        dist_vec = self.target_pos - self.drone_pos
        airsim_dist = np.array([dist_vec[1], dist_vec[0], -dist_vec[2]])
        airsim_vel = np.array([self.drone_vel[1], self.drone_vel[0], -self.drone_vel[2]])
        
        ai_yaw = -self.drone_yaw
        ca, sa = np.cos(-ai_yaw), np.sin(-ai_yaw)
        R = np.array([[ca, -sa], [sa, ca]])
        
        body_dist_xy = R.dot(airsim_dist[:2])
        body_vel_xy = R.dot(airsim_vel[:2])
        
        norm_dist = np.zeros(3, dtype=np.float32)
        norm_dist[0] = np.clip(body_dist_xy[0] / 50.0, -1.0, 1.0)
        norm_dist[1] = np.clip(body_dist_xy[1] / 50.0, -1.0, 1.0)
        
        norm_vel = np.zeros(3, dtype=np.float32)
        norm_vel[0] = np.clip(body_vel_xy[0] / 15.0, -1.0, 1.0)
        norm_vel[1] = np.clip(body_vel_xy[1] / 15.0, -1.0, 1.0)
        norm_vel[2] = np.clip(airsim_vel[2] / 15.0, -1.0, 1.0)
        
        desired_physical_z = -2.0
        z_error = airsim_dist[2] - desired_physical_z
        perceived_horiz_dist = np.linalg.norm(norm_dist[:2]) * 50.0
        network_ideal_alt = perceived_horiz_dist * 0.364
        fake_altitude = network_ideal_alt - (z_error * 3.0)
        norm_dist[2] = np.clip(-fake_altitude / 50.0, -1.0, 1.0)
        
        attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        physics = np.concatenate([attitude, norm_vel, norm_dist, [0.0]]).astype(np.float32)
        
        depth_t = torch.FloatTensor(depth_arr).unsqueeze(0).unsqueeze(0)
        phys_t = torch.FloatTensor(physics).unsqueeze(0)
        
        with torch.no_grad():
            mean, _ = self.actor(depth_t, phys_t)
            action = torch.clamp(mean, -1.0, 1.0).numpy()[0]
            
        pitch = float(action[0])
        roll = float(action[1])
        yaw_rate = float(action[2])
        
        fwd_accel = pitch * 20.0
        right_accel = -roll * 20.0
        
        body_vel_xy[0] += fwd_accel * self.dt
        body_vel_xy[1] += right_accel * self.dt
        body_vel_xy *= 0.90
        
        R_inv = np.array([[ca, sa], [-sa, ca]])
        global_vel_xy = R_inv.dot(body_vel_xy)
        airsim_vel[:2] = global_vel_xy
        
        self.drone_vel[1] = airsim_vel[0]
        self.drone_vel[0] = airsim_vel[1]
        self.drone_yaw += -yaw_rate * 2.0 * self.dt
        
        self.drone_pos += self.drone_vel * self.dt
        self.drone.setPos(*self.drone_pos)
        self.drone.setH(math.degrees(self.drone_yaw))
        
        # Tilt the drone model visually based on acceleration
        self.drone.setP(-pitch * 30.0) # Pitch forward
        self.drone.setR(-roll * 30.0)  # Roll sideways
        
        if self.step_count % 50 == 0:
            speed = np.linalg.norm(self.drone_vel)
            dist_to_target = np.linalg.norm(self.target_pos - self.drone_pos)
            print(f"Step {self.step_count:>4} | X={self.drone_pos[0]:>6.1f}, Y={self.drone_pos[1]:>6.1f} | Speed={speed:>5.2f} | Dist={dist_to_target:>5.1f}")
        
        # Cinematic Camera Smoothing
        target_cam_x = self.drone_pos[0] - math.sin(self.drone_yaw) * 12.0
        target_cam_y = self.drone_pos[1] - math.cos(self.drone_yaw) * 12.0
        target_cam_z = self.drone_pos[2] + 4.0
        
        # LERP camera position for smoothness
        lerp = 5.0 * self.dt
        self.cam_pos[0] += (target_cam_x - self.cam_pos[0]) * lerp
        self.cam_pos[1] += (target_cam_y - self.cam_pos[1]) * lerp
        self.cam_pos[2] += (target_cam_z - self.cam_pos[2]) * lerp
        
        self.cam.setPos(*self.cam_pos)
        self.cam.lookAt(self.drone)
        
        if np.linalg.norm(self.target_pos - self.drone_pos) < 5.0:
            print("TARGET REACHED!")
            sys.exit(0)
            
        return Task.cont

if __name__ == "__main__":
    app = PandaDroneSim()
    app.run()
