"""
Aegis OS  Checkpoint-Aware PPO Training for Free Google Colab (5-Hour T4 Limit)
==================================================================================

DESIGNED FOR: Google Colab Free Tier  T4 GPU, ~5 hour sessions
STRATEGY:     Saves weights to Google Drive every 25 updates (~every 12 minutes).
              Automatically resumes from the latest checkpoint if one exists.
              Stops itself 30 minutes before Colab would kill the session.

USAGE (in Colab):
    !python src/train_rl_session2.py \\
        --session 2 \\
        --checkpoint_dir /content/drive/MyDrive/aegis_checkpoints/ \\
        --max_hours 4.5

SESSION PLAN:
    Session 1: Basic 3D Navigation + Multi-Obstacle Avoidance
    Session 2: Vehicle Profiles (Drone vs. Fixed-Wing Physics)
    Session 3: Landing Scenarios (Pad, Ship, Road, Helipad)
    Session 4: Terrain Awareness (Perlin Noise Mountains)
    Session 5: Full Curriculum Mix + Fine-Tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import math
import os
import sys
import time
import argparse

#  Path Setup 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import VisionPPOActor, VisionPPOCritic


# 
# CHECKPOINT MANAGER
# Saves/loads all training state to Google Drive so sessions can be resumed.
# 

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, session: int):
        self.checkpoint_dir = checkpoint_dir
        self.session = session
        self.session_dir = os.path.join(checkpoint_dir, f"session_{session}")
        self.latest_dir = os.path.join(checkpoint_dir, "latest")
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.latest_dir, exist_ok=True)

    def save(self, actor, critic, optimizer, update: int, total_steps: int, best_reward: float):
        """Save a full training snapshot. Called every 25 updates (~12 min on T4)."""
        state = {
            'actor': actor.state_dict(),
            'critic': critic.state_dict(),
            'optimizer': optimizer.state_dict(),
            'update': update,
            'total_steps': total_steps,
            'best_reward': best_reward,
            'session': self.session,
            'timestamp': time.time()
        }

        # Numbered checkpoint in session folder
        session_path = os.path.join(self.session_dir, f"checkpoint_update_{update:05d}.pth")
        torch.save(state, session_path)

        # Overwrite the global "latest" file so the next session can find it
        latest_path = os.path.join(self.latest_dir, "latest.pth")
        torch.save(state, latest_path)

        print(f"   Checkpoint saved  update {update:,} | steps {total_steps:,} | {session_path}")
        return session_path

    def load_latest(self, actor, critic, optimizer, device):
        """
        Attempts to load the latest checkpoint. Returns (start_update, total_steps, best_reward).
        If no checkpoint exists, returns (0, 0, -inf) to start fresh.
        """
        latest_path = os.path.join(self.latest_dir, "latest.pth")
        if not os.path.exists(latest_path):
            print("    No checkpoint found. Starting from scratch.")
            return 0, 0, float('-inf')

        print(f"   Loading checkpoint from {latest_path}...")
        state = torch.load(latest_path, map_location=device)

        #  Network Surgery (Checkpoint Grafting) 
        # Session 1 trained on 9 physics dims. Session 2 uses 10 (adds vehicle_type).
        def graft_state_dict(model_state, old_state):
            for key, old_tensor in old_state.items():
                if key in model_state:
                    new_tensor = model_state[key]
                    if old_tensor.shape != new_tensor.shape:
                        if key == 'physics_net.0.weight' and old_tensor.shape[1] == 9 and new_tensor.shape[1] == 10:
                            # Graft 9 dims, initialize 10th dim to zero
                            new_tensor[:, :9] = old_tensor
                            new_tensor[:, 9] = 0.0
                            print(f"    [Surgery] Grafted {key}: expanded (64, 9) -> (64, 10)")
                        else:
                            print(f"    [Warning] Shape mismatch for {key}: {old_tensor.shape} != {new_tensor.shape}")
                    else:
                        new_tensor.copy_(old_tensor)
            return model_state

        actor.load_state_dict(graft_state_dict(actor.state_dict(), state['actor']))
        critic.load_state_dict(graft_state_dict(critic.state_dict(), state['critic']))

        prev_session = state.get('session', 1)
        if prev_session == self.session:
            optimizer.load_state_dict(state['optimizer'])
        else:
            print(f"    [Surgery] New session detected ({self.session} vs {prev_session}). Resetting optimizer state to avoid shape mismatches.")

        start_update = state.get('update', 0) + 1
        total_steps = state.get('total_steps', 0)
        best_reward = state.get('best_reward', float('-inf'))

        print(f"   Resumed from session {prev_session}, update {start_update - 1:,}, "
              f"steps {total_steps:,}")
        return start_update, total_steps, best_reward


# 
# SESSION 1 ENVIRONMENT: 3D Navigation + Multi-Obstacle Avoidance
# Teaches the AI to fly from point A to point B while avoiding 8 obstacles.
# 

class Session2Env:
    """
    Session 2 6-DOF flight environment with Vehicle Profiles.
    - Adds vehicle_type (0 = Drone, 1 = Fixed-Wing)
    - Enforces stall mechanics for fixed-wing aircraft
    """
    def __init__(self, num_envs: int, device: torch.device, num_rays: int = 32):
        self.num_envs = num_envs
        self.device = device
        self.num_rays = num_rays
        self.num_obs = 8
        self.dt = 0.05  # 20 Hz simulation

        # 6-DOF State
        self.pos = torch.zeros((num_envs, 3), device=device)
        self.vel = torch.zeros((num_envs, 3), device=device)
        self.attitude = torch.zeros((num_envs, 3), device=device)  # pitch, roll, yaw
        
        # Vehicle Profile (0.0 = Drone, 1.0 = Fixed-Wing)
        self.vehicle_type = torch.zeros(num_envs, device=device)

        # Goals and Obstacles
        self.target_pos = torch.zeros((num_envs, 3), device=device)
        self.obs_pos = torch.zeros((num_envs, self.num_obs, 3), device=device)
        self.obs_radius = torch.ones((num_envs, self.num_obs), device=device) * 2.5
        self.steps = torch.zeros(num_envs, device=device)

        # Pre-compute LiDAR ray angles (-60 to +60)
        fov = math.pi * (120.0 / 180.0)
        self.ray_angles = torch.linspace(-fov / 2, fov / 2, num_rays, device=device)

        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, mask: torch.Tensor):
        n = mask.sum().item()
        if n == 0:
            return self._get_state()

        # Spawn drone at varied altitudes to train altitude awareness
        self.pos[mask] = torch.stack([
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.empty(n, device=self.device).uniform_(10.0, 80.0)
        ], dim=1)
        self.vel[mask] = 0.0
        self.attitude[mask] = 0.0
        
        # Randomly assign Drone (0.0) or Fixed-Wing (1.0)
        self.vehicle_type[mask] = torch.randint(0, 2, (n,), device=self.device).float()

        # Random target in a 4040100m box
        self.target_pos[mask, 0] = torch.empty(n, device=self.device).uniform_(-20.0, 20.0)
        self.target_pos[mask, 1] = torch.empty(n, device=self.device).uniform_(-20.0, 20.0)
        self.target_pos[mask, 2] = torch.empty(n, device=self.device).uniform_(5.0, 120.0)

        # 8 random obstacles per environment
        for i in range(self.num_obs):
            self.obs_pos[mask, i, 0] = torch.empty(n, device=self.device).uniform_(-15.0, 15.0)
            self.obs_pos[mask, i, 1] = torch.empty(n, device=self.device).uniform_(-15.0, 15.0)
            self.obs_pos[mask, i, 2] = torch.empty(n, device=self.device).uniform_(5.0, 120.0)

        self.steps[mask] = 0
        return self._get_state()

    def _raycast(self) -> torch.Tensor:
        """1D LiDAR: 32-ray scan returning normalized depth [0=far, 1=very close]."""
        yaw = self.attitude[:, 2].unsqueeze(1)
        abs_angles = yaw + self.ray_angles.unsqueeze(0)
        ray_dir_x = torch.cos(abs_angles)
        ray_dir_y = torch.sin(abs_angles)
        orig = self.pos[:, :2].unsqueeze(1)
        depth_scan = torch.full((self.num_envs, self.num_rays), 50.0, device=self.device)

        for i in range(self.num_obs):
            obs_xy = self.obs_pos[:, i, :2].unsqueeze(1)
            obs_r = self.obs_radius[:, i].unsqueeze(1)
            L = obs_xy - orig
            tca = L[..., 0] * ray_dir_x + L[..., 1] * ray_dir_y
            d2 = L[..., 0] ** 2 + L[..., 1] ** 2 - tca ** 2
            hit = (d2 <= obs_r ** 2) & (tca > 0.0)
            thc = torch.sqrt(torch.clamp(obs_r ** 2 - d2, min=0.0))
            t0 = tca - thc
            depth_scan = torch.where(hit & (t0 < depth_scan), t0, depth_scan)

        return (1.0 - depth_scan / 50.0).unsqueeze(1)  # shape: (N, 1, 32)

    def _get_state(self):
        depth_scan = self._raycast()
        dist_vec = self.target_pos - self.pos
        norm_dist = torch.clamp(dist_vec / 25.0, -1.0, 1.0)
        norm_vel = torch.clamp(self.vel / 15.0, -1.0, 1.0)
        # Physics state is now 10-dimensional (9 + 1)
        physics_state = torch.cat([self.attitude, norm_vel, norm_dist, self.vehicle_type.unsqueeze(1)], dim=1)  # (N, 10)
        return depth_scan, physics_state

    def step(self, actions: torch.Tensor):
        """
        actions: (N, 4)  [Pitch, Roll, Yaw_rate, Thrust] all in [-1.0, 1.0]
        Returns: depth, physics, rewards, dones, num_goals_reached
        """
        self.steps += 1

        # Map actions to physical commands
        pitch_cmd = actions[:, 0] * 0.5      # max 0.5 rad pitch
        roll_cmd = actions[:, 1] * 0.5       # max 0.5 rad roll
        yaw_rate = actions[:, 2] * 1.0       # max 1.0 rad/s yaw
        thrust = (actions[:, 3] + 1.0) * 0.5 * 25.0  # thrust [0, 25] N/kg

        # Attitude dynamics (first-order lag filter)
        self.attitude[:, 0] += (pitch_cmd - self.attitude[:, 0]) * 0.15
        self.attitude[:, 1] += (roll_cmd - self.attitude[:, 1]) * 0.15
        self.attitude[:, 2] += yaw_rate * self.dt

        # Body-frame to world-frame accelerations
        cy = torch.cos(self.attitude[:, 2])
        sy = torch.sin(self.attitude[:, 2])
        cp = torch.cos(self.attitude[:, 0])
        sp = torch.sin(self.attitude[:, 0])
        cr = torch.cos(self.attitude[:, 1])

        accel_x = (cy * sp * cr + sy * self.attitude[:, 1]) * thrust
        accel_y = (sy * sp * cr - cy * self.attitude[:, 1]) * thrust
        accel_z = cp * cr * thrust - 9.81  # gravity cancellation

        # Linear drag (air resistance)
        drag = -0.12 * self.vel

        #  Stall Mechanics for Fixed-Wing (vehicle_type == 1.0) 
        speed = torch.norm(self.vel, dim=1)
        stall_mask = (self.vehicle_type == 1.0) & (speed < 10.0)
        # Apply massive gravity penalty during stall
        accel_z[stall_mask] -= 20.0

        self.vel[:, 0] += (accel_x + drag[:, 0]) * self.dt
        self.vel[:, 1] += (accel_y + drag[:, 1]) * self.dt
        self.vel[:, 2] += (accel_z + drag[:, 2]) * self.dt
        self.pos += self.vel * self.dt

        #  Compute Rewards 
        dist_to_target = torch.norm(self.target_pos - self.pos, dim=1)

        # Check collisions against all 8 obstacles
        pos_exp = self.pos.unsqueeze(1)                     # (N, 1, 3)
        dist_to_obs = torch.norm(self.obs_pos - pos_exp, dim=2)  # (N, 8)
        collision = (dist_to_obs < self.obs_radius).any(dim=1)

        # Boundary/crash detection
        out_of_bounds = (
            (self.pos[:, 2] < 0.5) |    # hit ground
            (self.pos[:, 2] > 350.0) |  # flew too high
            (torch.abs(self.pos[:, 0]) > 200.0) |
            (torch.abs(self.pos[:, 1]) > 200.0)
        )

        # Dense shaping: reward moving closer to target each step
        rewards = -0.05 * torch.ones(self.num_envs, device=self.device)
        rewards[stall_mask] -= 0.5  # Heavy penalty for stalling
        rewards += 5.0 / (dist_to_target + 1.0)           # shaped pull toward goal
        rewards -= torch.norm(actions, dim=1) * 0.005      # small energy cost

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Priority ordering (lowest first, highest overwrites)
        time_mask = self.steps > 500
        dones |= time_mask                                  # timeout: no extra penalty

        rewards = torch.where(out_of_bounds, torch.full_like(rewards, -100.0), rewards)
        dones |= out_of_bounds

        rewards = torch.where(collision, torch.full_like(rewards, -200.0), rewards)
        dones |= collision

        # Goal is highest priority  always overrides crash/timeout rewards
        goal_mask = dist_to_target < 2.5
        rewards = torch.where(goal_mask, torch.full_like(rewards, +300.0), rewards)
        dones |= goal_mask

        # BUG 7 FIX: Reset BEFORE reading next state
        if dones.any():
            self.reset(dones)
        next_depth, next_phys = self._get_state()

        return next_depth, next_phys, rewards, dones, goal_mask.sum().item()


# 
# GAE  Generalized Advantage Estimation
# 

def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    """
    Computes returns and advantages using GAE-.
    This reduces variance in gradient estimates compared to plain Monte Carlo.
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


# 
# MAIN TRAINING FUNCTION
# 

def train(session: int, checkpoint_dir: str, max_hours: float):
    """
    Main training loop. Checkpoint-aware. Stops gracefully before Colab timeout.

    Args:
        session:         Which session number (1-5). Used for curriculum stage.
        checkpoint_dir:  Path to Google Drive checkpoint directory.
        max_hours:       Stop training after this many hours (default 4.5).
                         Set to 4.5 to leave a 30-min buffer before Colab kills the session.
    """
    t_start = time.time()
    max_seconds = max_hours * 3600

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Aegis OS  Checkpoint-Aware PPO Training")
    print(f"  Session:   {session}")
    print(f"  Device:    {device}")
    if device.type == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:      {vram_total:.1f} GB")
    print(f"  Max time:  {max_hours:.1f} hours")
    print(f"  Saves to:  {checkpoint_dir}")
    print(f"{'='*70}\n")

    #  Hyperparameters 
    NUM_ENVS   = 2048 if device.type == "cuda" else 32
    PPO_STEPS  = 512         # 512 steps  0.05s = 25.6s per rollout
    PPO_EPOCHS = 4           # Re-use each rollout 4 times
    MAX_UPDATES = 1500        # Session 2 needs 1000 NEW updates (resumed at 500, so 500->1499).
    SAVE_EVERY  = 25         # Save checkpoint every 25 updates (~12 minutes on T4)
    CLIP_EPS    = 0.2
    GAMMA       = 0.99
    LAM         = 0.95
    LR          = 3e-4
    ENTROPY_COEF = 0.01      # Encourages exploration

    print(f"  Parallel Environments: {NUM_ENVS:,}")
    print(f"  Steps per rollout:     {PPO_STEPS:,}")
    print(f"  Steps per update:      {NUM_ENVS * PPO_STEPS:,}")
    print(f"  Max updates:           {MAX_UPDATES:,}")
    print(f"  Max env steps:         {NUM_ENVS * PPO_STEPS * MAX_UPDATES:,}\n")

    #  Build Models 
    actor  = VisionPPOActor(depth_rays=32, physics_dim=10, action_dim=4).to(device)
    critic = VisionPPOCritic(depth_rays=32, physics_dim=10).to(device)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=LR, eps=1e-5
    )

    #  Load Checkpoint 
    ckpt_mgr = CheckpointManager(checkpoint_dir, session)
    start_update, total_steps, best_reward = ckpt_mgr.load_latest(actor, critic, optimizer, device)

    #  Learning Rate Scheduler 
    # Linearly decay LR from LR to LR/10 over all updates
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=MAX_UPDATES - start_update
    )

    #  Environment 
    env = Session2Env(num_envs=NUM_ENVS, device=device)
    depth, phys = env._get_state()

    print(f"\n  Starting from update {start_update:,}. Training...\n")
    print(f"  {'Update':>7} | {'Steps':>12} | {'Avg Reward':>10} | {'Goals/Env':>9} | {'Loss':>8} | {'Elapsed':>8} | {'ETA':>8}")
    print(f"  {'-'*80}")

    #  Main Training Loop 
    for update in range(start_update, MAX_UPDATES):

        #  TIME GUARD: Stop 30 minutes before Colab kills us 
        elapsed = time.time() - t_start
        if elapsed > max_seconds:
            print(f"\n   Time limit reached ({max_hours:.1f}h). Saving final checkpoint...")
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)
            print(f"   Session {session} complete. Run Session {session + 1} next time!")
            break

        #  Collect Rollout 
        log_probs_buf, values_buf = [], []
        depths_buf, phys_buf     = [], []
        actions_buf, rewards_buf, masks_buf = [], [], []
        goals_this_update = 0

        actor.eval()
        critic.eval()
        with torch.no_grad():
            for _ in range(PPO_STEPS):
                depths_buf.append(depth)
                phys_buf.append(phys)

                mean, std = actor(depth, phys)
                dist = Normal(mean, std)
                action = torch.clamp(dist.sample(), -1.0, 1.0)
                value  = critic(depth, phys)

                next_depth, next_phys, reward, done, goals = env.step(action)
                goals_this_update += goals

                log_probs_buf.append(dist.log_prob(action).sum(-1, keepdim=True))
                values_buf.append(value)
                rewards_buf.append(reward.unsqueeze(1))
                masks_buf.append((~done).float().unsqueeze(1))
                actions_buf.append(action)

                depth, phys = next_depth, next_phys

            # Bootstrap next value for GAE
            next_value = critic(depth, phys)

        #  Compute Advantages 
        returns = compute_gae(next_value, rewards_buf, masks_buf, values_buf, GAMMA, LAM)
        returns    = torch.cat(returns).detach()
        log_probs  = torch.cat(log_probs_buf).detach()
        values_old = torch.cat(values_buf).detach()
        states_d   = torch.cat(depths_buf)
        states_p   = torch.cat(phys_buf)
        acts       = torch.cat(actions_buf)

        # Normalize advantages (reduces gradient variance)
        advantage = returns - values_old
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        #  PPO Update 
        actor.train()
        critic.train()
        total_loss = 0.0

        # Shuffle data into mini-batches for each epoch
        batch_size = NUM_ENVS * PPO_STEPS
        mini_batch = max(256, batch_size // 8)

        for epoch in range(PPO_EPOCHS):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch):
                idx = perm[start: start + mini_batch]

                mean_new, std_new = actor(states_d[idx], states_p[idx])
                dist_new = Normal(mean_new, std_new)
                new_log_probs = dist_new.log_prob(acts[idx]).sum(-1, keepdim=True)
                entropy = dist_new.entropy().mean()

                # Clipped PPO surrogate objective
                ratio = torch.exp(new_log_probs - log_probs[idx])
                surr1 = ratio * advantage[idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic MSE loss (clipped for stability)
                value_pred = critic(states_d[idx], states_p[idx])
                critic_loss = (returns[idx] - value_pred).pow(2).mean()

                # Combined PPO loss
                loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()

        scheduler.step()

        #  Logging 
        total_steps += NUM_ENVS * PPO_STEPS
        avg_reward   = torch.cat(rewards_buf).mean().item()
        goal_rate    = goals_this_update / max(1, NUM_ENVS)
        avg_loss     = total_loss / (PPO_EPOCHS * (batch_size // mini_batch))
        elapsed_m    = elapsed / 60.0
        updates_done = update - start_update + 1
        remaining_updates = MAX_UPDATES - update - 1
        time_per_update   = elapsed / max(1, updates_done)
        eta_minutes       = (remaining_updates * time_per_update) / 60.0

        if avg_reward > best_reward:
            best_reward = avg_reward

        if update % 5 == 0:
            print(f"  {update:>7,} | {total_steps:>12,} | {avg_reward:>10.2f} | "
                  f"{goal_rate:>9.2f} | {avg_loss:>8.4f} | "
                  f"{elapsed_m:>6.1f}m | {eta_minutes:>6.1f}m")

        #  Checkpoint 
        if (update + 1) % SAVE_EVERY == 0:
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)

    #  Final Save 
    print(f"\n{'='*70}")
    print(f"  Session {session} Training Complete!")
    print(f"  Total env steps this session: {total_steps:,}")
    print(f"  Best avg reward:              {best_reward:.2f}")
    print(f"  Elapsed:                      {(time.time() - t_start)/3600:.2f} hours")
    print(f"{'='*70}\n")
    ckpt_mgr.save(actor, critic, optimizer, MAX_UPDATES - 1, total_steps, best_reward)


# 
# ENTRY POINT
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis checkpoint-aware PPO training")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number (1-5). Controls which skills are trained.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/content/drive/MyDrive/aegis_checkpoints/",
                        help="Directory to save/load checkpoints (Google Drive path in Colab).")
    parser.add_argument("--max_hours", type=float, default=4.5,
                        help="Stop training after this many hours. Default 4.5 leaves a "
                             "30-min safety buffer before Colab's 5-hour limit.")
    args = parser.parse_args()

    train(
        session=args.session,
        checkpoint_dir=args.checkpoint_dir,
        max_hours=args.max_hours
    )
