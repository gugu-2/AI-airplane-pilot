"""
Aegis OS \u2014 Phase 2: Massively Parallel 6-DOF Physics + 1D LiDAR (Vision PPO)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import math
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import VisionPPOActor, VisionPPOCritic

class Vectorized6DOFEnv:
    def __init__(self, num_envs, device, num_rays=32):
        self.num_envs = num_envs
        self.device = device
        self.num_rays = num_rays
        self.dt = 0.05  # 20 Hz simulation
        
        # 6-DOF State
        self.pos = torch.zeros((num_envs, 3), device=device)       # x, y, z
        self.vel = torch.zeros((num_envs, 3), device=device)       # vx, vy, vz
        self.attitude = torch.zeros((num_envs, 3), device=device)  # pitch, roll, yaw
        self.ang_vel = torch.zeros((num_envs, 3), device=device)   # p, q, r
        
        # Goals
        self.target_pos = torch.zeros((num_envs, 3), device=device)
        
        # Obstacles (Simplified to spheres for fast batched raycasting)
        self.num_obs = 5
        self.obs_pos = torch.zeros((num_envs, self.num_obs, 3), device=device)
        self.obs_radius = torch.ones((num_envs, self.num_obs), device=device) * 2.0
        
        self.steps = torch.zeros(num_envs, device=device)
        
        # Pre-compute ray angles relative to heading
        # Ray angles spread from -60 deg to +60 deg horizontally
        fov = math.pi * (120.0 / 180.0)
        self.ray_angles = torch.linspace(-fov/2, fov/2, num_rays, device=device)
        
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        
    def reset(self, mask):
        num_reset = mask.sum().item()
        if num_reset == 0:
            return self._get_state()
            
        self.pos[mask] = torch.tensor([0.0, 0.0, 50.0], device=self.device)
        self.vel[mask] = 0.0
        self.attitude[mask] = 0.0
        self.ang_vel[mask] = 0.0
        
        self.target_pos[mask, 0] = torch.empty(num_reset, device=self.device).uniform_(-20.0, 20.0)
        self.target_pos[mask, 1] = torch.empty(num_reset, device=self.device).uniform_(-20.0, 20.0)
        self.target_pos[mask, 2] = torch.empty(num_reset, device=self.device).uniform_(50.0, 150.0)
        
        for i in range(self.num_obs):
            self.obs_pos[mask, i, 0] = torch.empty(num_reset, device=self.device).uniform_(-10.0, 10.0)
            self.obs_pos[mask, i, 1] = torch.empty(num_reset, device=self.device).uniform_(-10.0, 10.0)
            self.obs_pos[mask, i, 2] = torch.empty(num_reset, device=self.device).uniform_(50.0, 150.0)
        
        self.steps[mask] = 0
        return self._get_state()

    def _raycast(self):
        """
        Simulate a 1D LiDAR scanner using ray-sphere intersection.
        Returns a depth scan tensor of shape (num_envs, 1, num_rays).
        """
        # Drone headings: shape (num_envs, 1)
        yaw = self.attitude[:, 2].unsqueeze(1)
        
        # Absolute ray angles: shape (num_envs, num_rays)
        abs_ray_angles = yaw + self.ray_angles.unsqueeze(0)
        
        # Ray direction vectors: shape (num_envs, num_rays, 2)
        ray_dir_x = torch.cos(abs_ray_angles)
        ray_dir_y = torch.sin(abs_ray_angles)
        
        # Ray origins (drone pos XY): shape (num_envs, 1, 2)
        orig = self.pos[:, :2].unsqueeze(1)
        
        # Initialize depth scan to max distance (e.g. 50.0m)
        depth_scan = torch.full((self.num_envs, self.num_rays), 50.0, device=self.device)
        
        # Batched intersection
        for i in range(self.num_obs):
            obs_xy = self.obs_pos[:, i, :2].unsqueeze(1)
            obs_r = self.obs_radius[:, i].unsqueeze(1)
            
            L = obs_xy - orig
            L_x, L_y = L[..., 0], L[..., 1]
            
            tca = L_x * ray_dir_x + L_y * ray_dir_y
            d2 = L_x**2 + L_y**2 - tca**2
            
            r2 = obs_r**2
            hit_mask = (d2 <= r2) & (tca > 0.0)
            
            thc = torch.sqrt(torch.clamp(r2 - d2, min=0.0))
            t0 = tca - thc
            
            depth_scan = torch.where(hit_mask & (t0 < depth_scan), t0, depth_scan)
            
        # Normalize depth to [0, 1] where 1 is very close, 0 is far
        norm_depth = 1.0 - (depth_scan / 50.0)
        return norm_depth.unsqueeze(1)

    def _get_state(self):
        depth_scan = self._raycast()
        
        # Physics State (9-DOF): Pitch, Roll, Yaw, Vx, Vy, Vz, Target_Dist_X, Y, Z
        dist_vec = self.target_pos - self.pos
        norm_dist = torch.clamp(dist_vec / 20.0, -1.0, 1.0)
        norm_vel = torch.clamp(self.vel / 10.0, -1.0, 1.0)
        
        physics_state = torch.cat([self.attitude, norm_vel, norm_dist], dim=1)
        return depth_scan, physics_state
        
    def step(self, actions):
        """
        actions: (num_envs, 4) -> Pitch, Roll, Yaw Rate, Thrust
        """
        self.steps += 1
        
        pitch_cmd = actions[:, 0] * 0.5
        roll_cmd = actions[:, 1] * 0.5
        yaw_cmd = actions[:, 2] * 0.5
        thrust_cmd = (actions[:, 3] + 1.0) * 0.5 * 20.0 # Thrust [0, 20]
        
        self.attitude[:, 0] += (pitch_cmd - self.attitude[:, 0]) * 0.1
        self.attitude[:, 1] += (roll_cmd - self.attitude[:, 1]) * 0.1
        self.attitude[:, 2] += yaw_cmd * self.dt
        
        accel_x = torch.cos(self.attitude[:, 2]) * torch.sin(self.attitude[:, 0]) * thrust_cmd
        accel_y = torch.sin(self.attitude[:, 2]) * torch.sin(self.attitude[:, 0]) * thrust_cmd
        accel_z = torch.cos(self.attitude[:, 0]) * torch.cos(self.attitude[:, 1]) * thrust_cmd - 9.81
        
        drag = -0.1 * self.vel
        
        self.vel[:, 0] += (accel_x + drag[:, 0]) * self.dt
        self.vel[:, 1] += (accel_y + drag[:, 1]) * self.dt
        self.vel[:, 2] += (accel_z + drag[:, 2]) * self.dt
        
        self.pos += self.vel * self.dt
        
        dist_to_target = torch.norm(self.target_pos - self.pos, dim=1)
        
        collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_obs):
            dist_to_obs = torch.norm(self.obs_pos[:, i] - self.pos, dim=1)
            collision |= (dist_to_obs < self.obs_radius[:, i])
            
        rewards = -0.05 * torch.ones(self.num_envs, device=self.device)
        rewards -= torch.norm(actions, dim=1) * 0.01
        rewards += (10.0 / (dist_to_target + 1.0))  # Shaping: reward proximity to goal
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        goal_mask = dist_to_target < 2.0
        crash_mask = (self.pos[:, 2] < 0.5) | (self.pos[:, 2] > 300.0) | (torch.abs(self.pos[:, 0]) > 150.0) | (torch.abs(self.pos[:, 1]) > 150.0)
        # BUG 10 FIX: Increased max steps from 200 to 500 so the agent has
        # enough simulated time (25s at dt=0.05) to navigate and avoid obstacles.
        time_mask = self.steps > 500
        
        # Apply rewards with priority ordering (highest priority last)
        rewards = torch.where(crash_mask, torch.full_like(rewards, -100.0), rewards)
        dones |= crash_mask
        rewards = torch.where(collision, torch.full_like(rewards, -200.0), rewards)
        dones |= collision
        rewards = torch.where(time_mask, rewards, rewards)  # timeout: just let dones end it
        dones |= time_mask
        rewards = torch.where(goal_mask, torch.full_like(rewards, 200.0), rewards)  # goal is highest priority
        dones |= goal_mask
        
        # BUG 7 FIX: Call reset() BEFORE _get_state() so that the next_state
        # returned at episode boundaries reflects the fresh start, not the crashed state.
        if dones.any():
            self.reset(dones)  # resets internal pos/vel/attitude for done envs
        next_depth, next_phys = self._get_state()  # now reads fresh state for reset envs
            
        return next_depth, next_phys, rewards, dones, goal_mask.sum().item()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def train_ppo_3d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting MASSIVELY PARALLEL 3D PPO Training on: {device}")
    
    NUM_ENVS = 2048 if device.type == "cuda" else 32
    PPO_EPOCHS = 4
    # BUG 10 FIX: Increased from 128 to 512 steps per rollout.
    # At dt=0.05s: 128 steps = 6.4s simulated flight (not enough to navigate).
    # 512 steps = 25.6s, a realistic flight window for early training.
    PPO_STEPS = 512
    MAX_EPISODES = 500
    
    env = Vectorized6DOFEnv(num_envs=NUM_ENVS, device=device)
    
    actor = VisionPPOActor(depth_rays=32, physics_dim=9, action_dim=4).to(device)
    critic = VisionPPOCritic(depth_rays=32, physics_dim=9).to(device)
    
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    
    depth, phys = env._get_state()
    total_steps = 0
    
    for update in range(MAX_EPISODES):
        log_probs, values, states_d, states_p, actions, rewards, masks = [], [], [], [], [], [], []
        goals_this_update = 0  # BUG 9 FIX: Track successful goal reaches per update
        
        for _ in range(PPO_STEPS):
            states_d.append(depth)
            states_p.append(phys)
            
            with torch.no_grad():
                mean, std = actor(depth, phys)
                dist = Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                value = critic(depth, phys)
                
            # BUG 10 FIX: PPO_STEPS increased from 128 to 512.
            # 128 steps * 0.05s = only 6.4s simulated flight — not enough time to navigate.
            # 512 steps * 0.05s = 25.6s, giving the agent a meaningful flight window.
            next_depth, next_phys, reward, done, goals = env.step(action)
            goals_this_update += goals  # BUG 9 FIX: accumulate goal counts
            
            log_prob = dist.log_prob(action).sum(-1).unsqueeze(1)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward.unsqueeze(1))
            masks.append((1.0 - done.float()).unsqueeze(1))
            actions.append(action)
            
            depth = next_depth
            phys = next_phys
            
        with torch.no_grad():
            next_value = critic(depth, phys)
            
        returns = compute_gae(next_value, rewards, masks, values)
        
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states_d  = torch.cat(states_d)
        states_p  = torch.cat(states_p)
        actions   = torch.cat(actions)
        advantage = returns - values
        
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        for _ in range(PPO_EPOCHS):
            mean, std = actor(states_d, states_p)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1).unsqueeze(1)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - critic(states_d, states_p)).pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            
        total_steps += NUM_ENVS * PPO_STEPS
        
        avg_reward = torch.cat(rewards).mean().item()
        if update % 5 == 0:
            # BUG 9 FIX: Log goals reached per update for a meaningful success metric.
            # The old metric (avg step reward) was always ~-0.05 due to time penalty
            # and gave no information about whether the agent was actually learning.
            goal_rate = goals_this_update / max(1, NUM_ENVS)
            print(f"Update: {update:4d} | Global Steps: {total_steps:8d} | Avg Step Reward: {avg_reward:6.2f} | Goals/Env: {goal_rate:.2f} | Loss: {loss.item():6.2f}")
            
    print("\nPhase 2 3D Training Complete!")
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    # BUG 8 FIX: Corrected filename from aegis_vision_v1.pth to aegis_vision_v1-3d.pth.
    # The old name overwrote the 2D DQN model weights on every 3D training run.
    model_path = os.path.join(os.path.dirname(__file__), '../models/aegis_vision_v1-3d.pth')
    torch.save(actor.state_dict(), model_path)
    print(f"Saved 3D Vision weights to: {model_path}")

if __name__ == "__main__":
    train_ppo_3d()
