"""
Aegis OS - Session 4: Obstacle Avoidance Training
===================================================
Input:  32 synthetic LiDAR rays + 10-DOF physics state
Goal:   Fly 500m to a target through a dense cylinder forest without crashing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import sys
import time
import argparse
import glob
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import VisionPPOActor, VisionPPOCritic


class CheckpointManager:
    def __init__(self, checkpoint_dir, session):
        self.session_dir = os.path.join(checkpoint_dir, f"session_{session}")
        self.latest_dir  = os.path.join(checkpoint_dir, "latest")
        self.session = session
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.latest_dir,  exist_ok=True)

    def save(self, actor, critic, optimizer, update, total_steps, best_reward):
        state = {
            'actor':       actor.state_dict(),
            'critic':      critic.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'update':      update,
            'total_steps': total_steps,
            'best_reward': best_reward,
            'session':     self.session,
            'timestamp':   time.time()
        }
        session_path = os.path.join(self.session_dir, f"checkpoint_update_{update:05d}.pth")
        torch.save(state, session_path)
        torch.save(state, os.path.join(self.latest_dir, "latest.pth"))
        print(f"   Checkpoint saved  update {update:,} | steps {total_steps:,} | {session_path}")
        return session_path

    def load_latest(self, actor, critic, optimizer, device):
        # First look for session 4 checkpoints
        session_pths = glob.glob(os.path.join(self.session_dir, "*.pth"))
        if session_pths:
            session_pths.sort(key=os.path.getmtime)
            load_path = session_pths[-1]
            resuming = True
        else:
            # Fall back to Session 3 final model
            possible = [
                os.path.join(os.path.dirname(self.latest_dir), "session4_final.pth"),
                os.path.join(self.latest_dir, "latest.pth")
            ]
            load_path = next((p for p in possible if os.path.exists(p)), None)
            resuming = False

        if not load_path or not os.path.exists(load_path):
            print("   No checkpoint found. Starting from scratch.")
            return 0, 0, float('-inf')

        print(f"   Loading checkpoint from {load_path}...")
        state = torch.load(load_path, map_location=device, weights_only=False)

        def graft_state_dict(model_state, old_state):
            for key, old_tensor in old_state.items():
                if key not in model_state:
                    continue
                new_tensor = model_state[key]
                if old_tensor.shape == new_tensor.shape:
                    new_tensor.copy_(old_tensor)
                elif key == 'physics_net.0.weight' and old_tensor.shape[1] == 9 and new_tensor.shape[1] == 10:
                    new_tensor[:, :9] = old_tensor
                    new_tensor[:, 9]  = 0.0
                    print(f"    [Surgery] Grafted {key}: expanded (64, 9) -> (64, 10)")
            return model_state

        actor.load_state_dict(graft_state_dict(actor.state_dict(), state['actor']))
        critic.load_state_dict(graft_state_dict(critic.state_dict(), state['critic']))

        prev_session = state.get('session', 4)
        total_steps  = state.get('total_steps', 0)
        best_reward  = state.get('best_reward', float('-inf'))

        if resuming:
            optimizer.load_state_dict(state['optimizer'])
            start_update = state.get('update', 0) + 1
            print(f"   Resumed: session {prev_session}, update {start_update - 1:,}, steps {total_steps:,}")
        else:
            start_update = 0
            best_reward  = float('-inf')
            print(f"   Loaded Session {prev_session} brain into Session 4.")
            print(f"   Prior steps: {total_steps:,}  |  Starting fresh update counter.")

        return start_update, total_steps, best_reward


class Session4ObstacleEnv:
    NUM_RAYS     = 32
    MAX_RANGE    = 50.0
    NUM_OBS      = 20
    OBS_RADIUS   = 1.5
    CRUISE_ALT   = 8.0
    DT           = 0.05
    MAX_STEPS    = 800

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device   = device

        angles = torch.linspace(0, 2 * math.pi, self.NUM_RAYS + 1, device=device)[:-1]
        self.ray_cos = torch.cos(angles)
        self.ray_sin = torch.sin(angles)

        self.pos          = torch.zeros(num_envs, 3, device=device)
        self.vel          = torch.zeros(num_envs, 3, device=device)
        self.attitude     = torch.zeros(num_envs, 3, device=device)
        self.target       = torch.zeros(num_envs, 3, device=device)
        self.obs_xy       = torch.zeros(num_envs, self.NUM_OBS, 2, device=device)
        self.steps        = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_dist    = torch.zeros(num_envs, device=device)

        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, mask):
        n = int(mask.sum().item())
        if n == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]

        self.pos[idx, 0] = (torch.rand(n, device=self.device) - 0.5) * 10.0
        self.pos[idx, 1] = (torch.rand(n, device=self.device) - 0.5) * 10.0
        self.pos[idx, 2] = self.CRUISE_ALT
        self.vel[idx]    = 0.0
        self.attitude[idx] = 0.0
        self.steps[idx]  = 0

        target_angle = torch.rand(n, device=self.device) * 2 * math.pi
        target_dist  = 400.0 + torch.rand(n, device=self.device) * 200.0
        self.target[idx, 0] = self.pos[idx, 0] + target_dist * torch.cos(target_angle)
        self.target[idx, 1] = self.pos[idx, 1] + target_dist * torch.sin(target_angle)
        self.target[idx, 2] = self.CRUISE_ALT

        spawn_x = self.pos[idx, 0].unsqueeze(1)
        spawn_y = self.pos[idx, 1].unsqueeze(1)
        r = 100.0 + torch.rand(n, self.NUM_OBS, device=self.device) * 500.0
        a = target_angle.unsqueeze(1) + (torch.rand(n, self.NUM_OBS, device=self.device) - 0.5) * 2.0
        self.obs_xy[idx, :, 0] = spawn_x + r * torch.cos(a)
        self.obs_xy[idx, :, 1] = spawn_y + r * torch.sin(a)

        self.prev_dist[idx] = torch.norm(self.target[idx, :2] - self.pos[idx, :2], dim=1)

    def _raycast(self):
        yaw = self.attitude[:, 2]
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)

        ray_dx = cos_yaw * self.ray_cos - sin_yaw * self.ray_sin
        ray_dy = sin_yaw * self.ray_cos + cos_yaw * self.ray_sin

        dx = self.obs_xy[:, :, 0] - self.pos[:, 0].unsqueeze(1)
        dy = self.obs_xy[:, :, 1] - self.pos[:, 1].unsqueeze(1)

        dx = dx.unsqueeze(2)
        dy = dy.unsqueeze(2)
        rdx = ray_dx.unsqueeze(1)
        rdy = ray_dy.unsqueeze(1)

        dot  = rdx * dx + rdy * dy
        c_sq = dx * dx + dy * dy
        disc = dot * dot - (c_sq - self.OBS_RADIUS ** 2)

        hit = disc >= 0
        t   = dot - torch.sqrt(disc.clamp(min=0))
        t   = torch.where(hit & (t > 0), t, torch.full_like(t, self.MAX_RANGE))
        depths, _ = t.min(dim=1)
        depths = depths.clamp(0, self.MAX_RANGE) / self.MAX_RANGE
        return depths.unsqueeze(1)

    def _get_state(self):
        depth_scan = self._raycast()
        dist_vec  = self.target - self.pos
        norm_dist = torch.clamp(dist_vec / 50.0, -1.0, 1.0)
        norm_vel  = torch.clamp(self.vel  / 15.0, -1.0, 1.0)
        vehicle_type = torch.zeros(self.num_envs, 1, device=self.device)
        physics = torch.cat([self.attitude, norm_vel, norm_dist, vehicle_type], dim=1)
        return depth_scan, physics

    def step(self, actions):
        self.steps += 1

        pitch_cmd = actions[:, 0] * 0.5
        roll_cmd  = actions[:, 1] * 0.5
        yaw_rate  = actions[:, 2] * 1.0
        thrust    = (actions[:, 3] + 1.0) * 0.5 * 22.0

        alt_error  = self.pos[:, 2] - self.CRUISE_ALT
        pitch_cmd  = pitch_cmd + torch.clamp(-alt_error * 0.15, -0.5, 0.5)

        self.attitude[:, 0] += (pitch_cmd - self.attitude[:, 0]) * 0.15
        self.attitude[:, 1] += (roll_cmd  - self.attitude[:, 1]) * 0.15
        self.attitude[:, 2] += yaw_rate * self.DT

        cy = torch.cos(self.attitude[:, 2])
        sy = torch.sin(self.attitude[:, 2])
        cp = torch.cos(self.attitude[:, 0])
        sp = torch.sin(self.attitude[:, 0])
        cr = torch.cos(self.attitude[:, 1])

        accel_x = (cy * sp * cr + sy * self.attitude[:, 1]) * thrust
        accel_y = (sy * sp * cr - cy * self.attitude[:, 1]) * thrust
        accel_z = cp * cr * thrust - 9.81

        drag = -0.12 * self.vel
        self.vel[:, 0] += (accel_x + drag[:, 0]) * self.DT
        self.vel[:, 1] += (accel_y + drag[:, 1]) * self.DT
        self.vel[:, 2] += (accel_z + drag[:, 2]) * self.DT
        self.pos       += self.vel * self.DT

        obs_dx = self.obs_xy[:, :, 0] - self.pos[:, 0].unsqueeze(1)
        obs_dy = self.obs_xy[:, :, 1] - self.pos[:, 1].unsqueeze(1)
        obs_dist = torch.sqrt(obs_dx**2 + obs_dy**2)
        collision = (obs_dist < self.OBS_RADIUS).any(dim=1)

        dist_xy   = torch.norm(self.target[:, :2] - self.pos[:, :2], dim=1)
        dist_3d   = torch.norm(self.target       - self.pos,         dim=1)
        reached   = dist_3d < 10.0

        oob = (
            (self.pos[:, 2] > 80.0) |
            (self.pos[:, 2] < 0.0)  |
            (torch.abs(self.pos[:, 0]) > 1200.0) |
            (torch.abs(self.pos[:, 1]) > 1200.0)
        )
        timeout = self.steps > self.MAX_STEPS

        rewards = torch.full((self.num_envs,), -0.05, device=self.device)
        dist_progress = self.prev_dist - dist_xy
        rewards += dist_progress * 0.5
        rewards += 3.0 / (dist_3d + 1.0)
        rewards -= torch.norm(actions, dim=1) * 0.003
        rewards = torch.where(collision, torch.full_like(rewards, -150.0), rewards)
        rewards = torch.where(reached, torch.full_like(rewards, +500.0), rewards)
        rewards = torch.where(oob, torch.full_like(rewards, -100.0), rewards)
        rewards = torch.where(timeout, torch.full_like(rewards, -50.0), rewards)

        self.prev_dist = dist_xy.clone()
        dones = collision | reached | oob | timeout
        n_reach = int(reached.sum().item())
        n_collision = int(collision.sum().item())

        if dones.any():
            self.reset(dones)

        next_depth, next_phys = self._get_state()
        return next_depth, next_phys, rewards, dones, n_reach, n_collision


def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values  = values + [next_value]
    gae     = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae   = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def train(session, checkpoint_dir, max_hours):
    t_start     = time.time()
    max_seconds = max_hours * 3600
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  Aegis OS  Session 4: Obstacle Avoidance")
    print(f"  Device:    {device}")
    if device.type == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:      {vram:.1f} GB")
    print(f"  Max time:  {max_hours:.2f} hours")
    print(f"  Saves to:  {checkpoint_dir}")
    print(f"{'='*70}\n")

    NUM_ENVS     = 2048 if device.type == "cuda" else 32
    PPO_STEPS    = 512
    PPO_EPOCHS   = 4
    MINI_BATCH   = 4096
    MAX_UPDATES  = 1500
    SAVE_EVERY   = 25
    CLIP_EPS     = 0.2
    GAMMA        = 0.99
    LAM          = 0.95
    LR           = 2e-4
    ENTROPY_COEF = 0.01

    actor  = VisionPPOActor(depth_rays=32, physics_dim=10, action_dim=4).to(device)
    critic = VisionPPOCritic(depth_rays=32, physics_dim=10).to(device)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LR, eps=1e-5
    )

    ckpt_mgr = CheckpointManager(checkpoint_dir, session)
    start_update, total_steps, best_reward = ckpt_mgr.load_latest(
        actor, critic, optimizer, device
    )

    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=max(1, MAX_UPDATES - start_update)
    )

    env = Session4ObstacleEnv(num_envs=NUM_ENVS, device=device)
    depth, phys = env._get_state()

    print(f"  Starting from update {start_update:,}. Training...\n")
    print(f"  {'Update':>7} | {'Steps':>12} | {'Avg Reward':>10} | {'Targets/Env':>12} | {'Crashes/Env':>12} | {'Loss':>8}")
    print(f"  {'-'*84}")

    last_update = start_update
    timed_out   = False

    for update in range(start_update, MAX_UPDATES):
        elapsed = time.time() - t_start
        if elapsed > max_seconds:
            print(f"\n   Time limit ({max_hours:.2f}h). Saving...")
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)
            last_update = update
            timed_out   = True
            break

        log_probs_buf, values_buf = [], []
        depths_buf, phys_buf     = [], []
        actions_buf, rewards_buf, masks_buf = [], [], []
        targets_this   = 0
        collisions_this = 0

        actor.eval()
        critic.eval()
        with torch.no_grad():
            for _ in range(PPO_STEPS):
                depths_buf.append(depth)
                phys_buf.append(phys)

                mean, std = actor(depth, phys)
                dist   = Normal(mean, std)
                action = torch.clamp(dist.sample(), -1.0, 1.0)
                value  = critic(depth, phys)

                next_depth, next_phys, reward, done, n_reach, n_crash = env.step(action)
                targets_this    += n_reach
                collisions_this += n_crash

                log_probs_buf.append(dist.log_prob(action).sum(-1, keepdim=True))
                values_buf.append(value)
                rewards_buf.append(reward.unsqueeze(1))
                masks_buf.append((~done).float().unsqueeze(1))
                actions_buf.append(action)
                depth, phys = next_depth, next_phys

            next_value = critic(depth, phys)

        returns    = compute_gae(next_value, rewards_buf, masks_buf, values_buf, GAMMA, LAM)
        returns    = torch.cat(returns).detach()
        log_probs  = torch.cat(log_probs_buf).detach()
        values_old = torch.cat(values_buf).detach()
        states_d   = torch.cat(depths_buf)
        states_p   = torch.cat(phys_buf)
        acts       = torch.cat(actions_buf)

        advantage  = returns - values_old
        advantage  = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor.train()
        critic.train()
        total_loss  = 0.0
        batch_size  = NUM_ENVS * PPO_STEPS

        for epoch in range(PPO_EPOCHS):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, MINI_BATCH):
                idx = perm[start: start + MINI_BATCH]
                mean_new, std_new = actor(states_d[idx], states_p[idx])
                dist_new      = Normal(mean_new, std_new)
                new_log_probs = dist_new.log_prob(acts[idx]).sum(-1, keepdim=True)
                entropy       = dist_new.entropy().mean()
                ratio  = torch.exp(new_log_probs - log_probs[idx])
                surr1  = ratio * advantage[idx]
                surr2  = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantage[idx]
                a_loss = -torch.min(surr1, surr2).mean()
                v_pred = critic(states_d[idx], states_p[idx])
                c_loss = (returns[idx] - v_pred).pow(2).mean()
                loss   = a_loss + 0.5 * c_loss - ENTROPY_COEF * entropy
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                optimizer.step()

        scheduler.step()

        total_steps   += NUM_ENVS * PPO_STEPS
        avg_reward     = torch.cat(rewards_buf).mean().item()
        avg_loss       = total_loss / (PPO_EPOCHS * (batch_size // MINI_BATCH))
        target_rate    = targets_this / NUM_ENVS
        crash_rate     = collisions_this / NUM_ENVS

        if avg_reward > best_reward:
            best_reward = avg_reward
        last_update = update

        if update % 5 == 0:
            print(f"  {update:>7,} | {total_steps:>12,} | {avg_reward:>10.2f} | "
                  f"{target_rate:>12.2f} | {crash_rate:>12.2f} | {avg_loss:>8.4f}")

        if (update + 1) % SAVE_EVERY == 0:
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)

    print(f"\n{'='*70}")
    if timed_out:
        print(f"  Session {session} Paused at update {last_update:,} / {MAX_UPDATES:,}")
    else:
        print(f"  Session {session} Training COMPLETE!")
        ckpt_mgr.save(actor, critic, optimizer, last_update, total_steps, best_reward)
    print(f"  Total env steps: {total_steps:,}")
    print(f"  Best avg reward: {best_reward:.2f}")
    print(f"  Elapsed:         {(time.time()-t_start)/3600:.2f} hours")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis OS Session 4 Obstacle Avoidance Training")
    parser.add_argument("--session", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/aegis_checkpoints/")
    parser.add_argument("--max_hours", type=float, default=5.1667)
    args = parser.parse_args()
    train(session=args.session, checkpoint_dir=args.checkpoint_dir, max_hours=args.max_hours)
