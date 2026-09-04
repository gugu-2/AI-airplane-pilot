"""
Aegis OS - Session 3: Autonomous Landing
==========================================

WHAT IS NEW IN SESSION 3:
    - Target is now a LANDING PAD on the GROUND (z = 0)
    - AI must descend from altitude and touch down SOFTLY
    - Soft landing  (vel_z > -3.0 m/s)  = +500 reward (SUCCESS)
    - Hard landing  (vel_z <= -3.0 m/s) = -100 reward (CRASH)
    - Missed pad    (landed outside pad) = -50 reward
    - Glide slope reward guides the AI toward ideal 20-degree descent angle
    - Keeps both vehicle types from Session 2 (Drone + Fixed-Wing)

NO NETWORK SURGERY NEEDED:
    - physics_dim = 10 (same as Session 2)
    - action_dim  = 4  (same as Session 2)
    - Session 2 brain loads directly with no weight reshaping

USAGE (in Colab):
    !python src/train_rl_session3.py \
        --session 3 \
        --checkpoint_dir /content/drive/MyDrive/aegis_checkpoints/ \
        --max_hours 4.5
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import VisionPPOActor, VisionPPOCritic


class CheckpointManager:
    def __init__(self, checkpoint_dir, session):
        self.checkpoint_dir = checkpoint_dir
        self.session = session
        self.session_dir = os.path.join(checkpoint_dir, f"session_{session}")
        self.latest_dir  = os.path.join(checkpoint_dir, "latest")
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
        # First look for session 3 checkpoints (resuming within S3)
        session_pths = glob.glob(os.path.join(self.session_dir, "*.pth"))
        if session_pths:
            session_pths.sort(key=os.path.getmtime)
            load_path = session_pths[-1]
            same_session = True
        else:
            load_path = os.path.join(self.latest_dir, "latest.pth")
            same_session = False

        if not os.path.exists(load_path):
            print("   No checkpoint found. Starting from scratch.")
            return 0, 0, float('-inf')

        print(f"   Loading checkpoint from {load_path}...")
        state = torch.load(load_path, map_location=device, weights_only=False)

        # Smart load: handles both Session 1 (9-dim) and Session 2 (10-dim) brains
        def graft_state_dict(model_state, old_state):
            for key, old_tensor in old_state.items():
                if key not in model_state:
                    continue
                new_tensor = model_state[key]
                if old_tensor.shape == new_tensor.shape:
                    new_tensor.copy_(old_tensor)
                elif key == 'physics_net.0.weight' and old_tensor.shape[1] == 9 and new_tensor.shape[1] == 10:
                    # Session 1 brain (9-dim) -> Session 3 needs 10-dim: expand safely
                    new_tensor[:, :9] = old_tensor
                    new_tensor[:, 9]  = 0.0
                    print(f"    [Surgery] Grafted {key}: expanded (64, 9) -> (64, 10)")
                else:
                    print(f"    [Warning] Shape mismatch for {key}: {old_tensor.shape} != {new_tensor.shape}")
            return model_state

        actor.load_state_dict(graft_state_dict(actor.state_dict(), state['actor']))
        critic.load_state_dict(graft_state_dict(critic.state_dict(), state['critic']))

        prev_session = state.get('session', 2)
        total_steps  = state.get('total_steps', 0)
        best_reward  = state.get('best_reward', float('-inf'))

        if same_session:
            optimizer.load_state_dict(state['optimizer'])
            start_update = state.get('update', 0) + 1
            print(f"   Resumed: session {prev_session}, update {start_update - 1:,}, steps {total_steps:,}")
        else:
            # KEY FIX: reset update counter when crossing session boundary
            start_update = 0
            best_reward  = float('-inf')
            print(f"   Loaded Session {prev_session} brain into Session 3.")
            print(f"   Prior steps: {total_steps:,}  |  Starting fresh update counter.")

        return start_update, total_steps, best_reward


class Session3Env:
    """
    3D Flight + Landing environment.
    - Target is a landing pad on the GROUND (z=0, pad radius=5m)
    - Touchdown detected when pos_z < 0.5m
    - Soft landing: descent rate > -3.0 m/s -> SUCCESS (+500)
    - Hard crash:   descent rate <= -3.0 m/s -> CRASH   (-100)
    - Missed pad:   touched down outside pad -> MISS    (-50)
    - Glide slope shaping guides toward ideal 20-degree descent
    - Vehicle types: 0=Drone, 1=Fixed-Wing (from Session 2)
    """
    def __init__(self, num_envs, device):
        self.num_envs   = num_envs
        self.device     = device
        self.dt         = 0.05
        self.obs_radius = 5.0
        self.pad_radius = 5.0
        self.soft_vz    = -3.0

        self.pos          = torch.zeros(num_envs, 3, device=device)
        self.vel          = torch.zeros(num_envs, 3, device=device)
        self.attitude     = torch.zeros(num_envs, 3, device=device)
        self.vehicle_type = torch.zeros(num_envs, 1, device=device)
        self.target_pos   = torch.zeros(num_envs, 3, device=device)
        self.obs_pos      = torch.zeros(num_envs, 4, 3, device=device)
        self.steps        = torch.zeros(num_envs, dtype=torch.long, device=device)

        mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.reset(mask)

    def reset(self, mask):
        n = int(mask.sum().item())
        if n == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]

        self.pos[idx, 0] = (torch.rand(n, device=self.device) - 0.5) * 80
        self.pos[idx, 1] = (torch.rand(n, device=self.device) - 0.5) * 80
        self.pos[idx, 2] = 20.0 + torch.rand(n, device=self.device) * 60.0

        self.vel[idx]          = torch.randn(n, 3, device=self.device) * 0.5
        self.attitude[idx]     = 0.0
        self.steps[idx]        = 0
        self.vehicle_type[idx] = (torch.rand(n, 1, device=self.device) > 0.5).float()

        self.target_pos[idx, 0] = (torch.rand(n, device=self.device) - 0.5) * 60.0
        self.target_pos[idx, 1] = (torch.rand(n, device=self.device) - 0.5) * 60.0
        self.target_pos[idx, 2] = 0.0

        for o in range(4):
            self.obs_pos[idx, o, 0] = (torch.rand(n, device=self.device) - 0.5) * 100.0
            self.obs_pos[idx, o, 1] = (torch.rand(n, device=self.device) - 0.5) * 100.0
            self.obs_pos[idx, o, 2] = torch.rand(n, device=self.device) * 40.0

    def _raycast(self):
        num_rays = 32
        angles   = torch.linspace(0, 2 * 3.14159, num_rays, device=self.device)
        ray_dirs = torch.stack([
            torch.cos(angles), torch.sin(angles),
            torch.zeros(num_rays, device=self.device)
        ], dim=1)

        pos_exp = self.pos.unsqueeze(1)
        depths  = []
        for r in range(num_rays):
            d = ray_dirs[r]
            min_dist = torch.full((self.num_envs,), 50.0, device=self.device)
            for o in range(4):
                to_obs = self.obs_pos[:, o, :] - pos_exp[:, 0, :]
                proj   = (to_obs * d).sum(-1)
                perp   = to_obs - proj.unsqueeze(-1) * d
                hit    = (perp.norm(dim=-1) < self.obs_radius) & (proj > 0)
                dist   = torch.where(hit, proj.clamp(min=0.1), torch.full_like(proj, 50.0))
                min_dist = torch.minimum(min_dist, dist)
            depths.append(min_dist)

        depth_scan = torch.stack(depths, dim=1) / 50.0
        return depth_scan.unsqueeze(1)

    def _get_state(self):
        depth_scan = self._raycast()
        dist_vec   = self.target_pos - self.pos
        norm_dist  = torch.clamp(dist_vec / 50.0, -1.0, 1.0)
        norm_vel   = torch.clamp(self.vel  / 15.0, -1.0, 1.0)
        physics    = torch.cat([self.attitude, norm_vel, norm_dist, self.vehicle_type], dim=1)
        return depth_scan, physics

    def step(self, actions):
        self.steps += 1

        pitch_cmd = actions[:, 0] * 0.5
        roll_cmd  = actions[:, 1] * 0.5
        yaw_rate  = actions[:, 2] * 1.0
        thrust    = (actions[:, 3] + 1.0) * 0.5 * 22.0

        is_fw  = (self.vehicle_type[:, 0] > 0.5)
        thrust = torch.where(is_fw, thrust.clamp(min=11.0), thrust)

        self.attitude[:, 0] += (pitch_cmd - self.attitude[:, 0]) * 0.15
        self.attitude[:, 1] += (roll_cmd  - self.attitude[:, 1]) * 0.15
        self.attitude[:, 2] += yaw_rate * self.dt

        cy = torch.cos(self.attitude[:, 2])
        sy = torch.sin(self.attitude[:, 2])
        cp = torch.cos(self.attitude[:, 0])
        sp = torch.sin(self.attitude[:, 0])
        cr = torch.cos(self.attitude[:, 1])

        accel_x = (cy * sp * cr + sy * self.attitude[:, 1]) * thrust
        accel_y = (sy * sp * cr - cy * self.attitude[:, 1]) * thrust
        accel_z = cp * cr * thrust - 9.81

        drag = -0.12 * self.vel
        self.vel[:, 0] += (accel_x + drag[:, 0]) * self.dt
        self.vel[:, 1] += (accel_y + drag[:, 1]) * self.dt
        self.vel[:, 2] += (accel_z + drag[:, 2]) * self.dt
        self.pos       += self.vel * self.dt

        altitude = self.pos[:, 2]
        pad_xy   = torch.norm(self.pos[:, :2] - self.target_pos[:, :2], dim=1)
        dist_3d  = torch.norm(self.target_pos - self.pos, dim=1)
        speed    = torch.norm(self.vel, dim=1)

        rewards = torch.full((self.num_envs,), -0.05, device=self.device)
        rewards += 3.0 / (dist_3d + 1.0)
        rewards -= torch.norm(actions, dim=1) * 0.005

        ideal_alt   = pad_xy * 0.364
        alt_error   = (altitude - ideal_alt).abs()
        rewards    += torch.exp(-alt_error / 15.0) * 0.8

        near_ground = (altitude < 15.0)
        rewards     = torch.where(near_ground, rewards - speed * 0.015, rewards)

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones |= (self.steps > 700)

        out_of_bounds = (
            (altitude > 400.0) |
            (torch.abs(self.pos[:, 0]) > 200.0) |
            (torch.abs(self.pos[:, 1]) > 200.0)
        )
        rewards = torch.where(out_of_bounds, torch.full_like(rewards, -100.0), rewards)
        dones  |= out_of_bounds

        pos_exp   = self.pos.unsqueeze(1)
        obs_dist  = torch.norm(self.obs_pos - pos_exp, dim=2)
        collision = (obs_dist < self.obs_radius).any(dim=1)
        rewards   = torch.where(collision, torch.full_like(rewards, -150.0), rewards)
        dones    |= collision

        touched = (altitude < 0.5)
        on_pad  = touched & (pad_xy < self.pad_radius)
        off_pad = touched & (pad_xy >= self.pad_radius)
        soft_ok = on_pad & (self.vel[:, 2] > self.soft_vz)
        too_fast= on_pad & (self.vel[:, 2] <= self.soft_vz)

        rewards = torch.where(soft_ok,  torch.full_like(rewards, +500.0), rewards)
        rewards = torch.where(too_fast, torch.full_like(rewards, -100.0), rewards)
        rewards = torch.where(off_pad,  torch.full_like(rewards,  -50.0), rewards)
        dones  |= touched

        n_soft = int(soft_ok.sum().item())
        if dones.any():
            self.reset(dones)

        next_depth, next_phys = self._get_state()
        return next_depth, next_phys, rewards, dones, n_soft


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
    print(f"  Aegis OS  Session 3: Autonomous Landing")
    print(f"  Device:    {device}")
    if device.type == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:      {vram:.1f} GB")
    print(f"  Max time:  {max_hours:.1f} hours")
    print(f"  Saves to:  {checkpoint_dir}")
    print(f"{'='*70}\n")

    NUM_ENVS     = 2048 if device.type == "cuda" else 32
    PPO_STEPS    = 512
    PPO_EPOCHS   = 4
    MAX_UPDATES  = 1500
    SAVE_EVERY   = 25
    CLIP_EPS     = 0.2
    GAMMA        = 0.99
    LAM          = 0.95
    LR           = 2e-4
    ENTROPY_COEF = 0.01

    print(f"  Parallel Environments: {NUM_ENVS:,}")
    print(f"  Steps per update:      {NUM_ENVS * PPO_STEPS:,}")
    print(f"  Max new updates:       {MAX_UPDATES:,}\n")

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
        total_iters=MAX_UPDATES - start_update
    )

    env = Session3Env(num_envs=NUM_ENVS, device=device)
    depth, phys = env._get_state()

    print(f"  Starting from update {start_update:,}. Training...\n")
    print(f"  {'Update':>7} | {'Steps':>12} | {'Avg Reward':>10} | {'Landings/Env':>12} | {'Loss':>8} | {'Elapsed':>8} | {'ETA':>8}")
    print(f"  {'-'*84}")

    last_update  = start_update
    timed_out    = False

    for update in range(start_update, MAX_UPDATES):
        elapsed = time.time() - t_start
        if elapsed > max_seconds:
            print(f"\n   Time limit ({max_hours:.1f}h). Saving...")
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)
            print(f"   Session {session} paused at update {update:,}. Run again to continue!")
            last_update = update
            timed_out   = True
            break

        log_probs_buf, values_buf = [], []
        depths_buf, phys_buf     = [], []
        actions_buf, rewards_buf, masks_buf = [], [], []
        landings_this = 0

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

                next_depth, next_phys, reward, done, n_land = env.step(action)
                landings_this += n_land

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
        mini_batch  = max(256, batch_size // 8)

        for epoch in range(PPO_EPOCHS):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch):
                idx = perm[start: start + mini_batch]
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
        land_rate      = landings_this / max(1, NUM_ENVS)
        avg_loss       = total_loss / (PPO_EPOCHS * (batch_size // mini_batch))
        elapsed_m      = elapsed / 60.0
        updates_done   = update - start_update + 1
        time_per_upd   = elapsed / max(1, updates_done)
        eta_m          = ((MAX_UPDATES - update - 1) * time_per_upd) / 60.0

        if avg_reward > best_reward:
            best_reward = avg_reward
        last_update = update

        if update % 5 == 0:
            print(f"  {update:>7,} | {total_steps:>12,} | {avg_reward:>10.2f} | "
                  f"{land_rate:>12.2f} | {avg_loss:>8.4f} | "
                  f"{elapsed_m:>6.1f}m | {eta_m:>6.1f}m")

        if (update + 1) % SAVE_EVERY == 0:
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)

    print(f"\n{'='*70}")
    if timed_out:
        print(f"  Session {session} Paused at update {last_update:,} / {MAX_UPDATES:,}")
        print(f"  Resume by running this script again - it will continue from update {last_update+1:,}")
    else:
        print(f"  Session {session} Training COMPLETE!")
        # Only do a final save when fully done to avoid the fake update-1499 bug
        ckpt_mgr.save(actor, critic, optimizer, last_update, total_steps, best_reward)
    print(f"  Total env steps: {total_steps:,}")
    print(f"  Best avg reward: {best_reward:.2f}")
    print(f"  Elapsed:         {(time.time()-t_start)/3600:.2f} hours")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis OS Session 3 Landing Training")
    parser.add_argument("--session", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/content/drive/MyDrive/aegis_checkpoints/")
    parser.add_argument("--max_hours", type=float, default=4.5)
    args = parser.parse_args()
    train(session=args.session, checkpoint_dir=args.checkpoint_dir, max_hours=args.max_hours)
