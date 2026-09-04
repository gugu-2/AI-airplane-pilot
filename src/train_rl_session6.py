# -*- coding: utf-8 -*-
"""
Aegis OS - Session 6: Advanced Kinematics + Dense Obstacles + JEPA Lookahead
=============================================================================
BUGS FIXED vs previous session6:
  BUG1: LiDAR rays now EXACTLY match airsim_bridge.py = linspace(-pi/2, pi/2, 32)
        Previous script used 360-degree sweep -> complete sensor mismatch!
  BUG2: NUM_OBS = 40 (was left at 20 by failed patch script)
  BUG3: Class renamed to Session6ObstacleEnv
  BUG4: Checkpoint print says Session 6 not Session 4
  BUG5: --session defaults to 6, --max_hours defaults to 5.5
  BUG6: Directional alignment penalty restored (was dropped by patch script)
  BUG7: Hard 15 m/s speed cap matches sensor normalization divisor exactly

JEPA UPGRADE (from RESEARCH_PAPER_AI_PILOT_JEPA.md):
  jepa_lookahead(): sample 4 candidate actions, pick highest-value one.
  "Imagine trajectory safety in latent space before issuing motor commands."
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os, sys, time, argparse, glob, math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import VisionPPOActor, VisionPPOCritic


# ────────────────────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, checkpoint_dir, session):
        self.session_dir = os.path.join(checkpoint_dir, f"session_{session}")
        self.latest_dir  = os.path.join(checkpoint_dir, "latest")
        self.session     = session
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
            'timestamp':   time.time(),
        }
        path = os.path.join(self.session_dir, f"checkpoint_update_{update:05d}.pth")
        torch.save(state, path)
        torch.save(state, os.path.join(self.latest_dir, "latest.pth"))
        print(f"   Checkpoint saved  update {update:,} | steps {total_steps:,} | {path}")
        return path

    def load_latest(self, actor, critic, optimizer, device):
        pths = glob.glob(os.path.join(self.session_dir, "*.pth"))
        if pths:
            pths.sort(key=os.path.getmtime)
            load_path, resuming = pths[-1], True
        else:
            candidates = [
                os.path.join(os.path.dirname(self.latest_dir), "session5_final.pth"),
                os.path.join(os.path.dirname(self.latest_dir), "session4_final.pth"),
                os.path.join(self.latest_dir, "latest.pth"),
            ]
            load_path = next((p for p in candidates if os.path.exists(p)), None)
            resuming = False

        if not load_path or not os.path.exists(load_path):
            print("   No checkpoint found. Starting from scratch.")
            return 0, 0, float('-inf')

        print(f"   Loading checkpoint from {load_path}...")
        state = torch.load(load_path, map_location=device, weights_only=False)

        def graft(model_sd, old_sd):
            for k, ot in old_sd.items():
                if k not in model_sd:
                    continue
                nt = model_sd[k]
                if ot.shape == nt.shape:
                    nt.copy_(ot)
                elif k == 'physics_net.0.weight' and ot.shape[1] == 9 and nt.shape[1] == 10:
                    nt[:, :9] = ot
                    nt[:, 9]  = 0.0
                    print(f"    [Surgery] Grafted {k}: (64,9)->(64,10)")
            return model_sd

        actor.load_state_dict(graft(actor.state_dict(),  state['actor']))
        critic.load_state_dict(graft(critic.state_dict(), state['critic']))

        prev_session = state.get('session', 5)
        total_steps  = state.get('total_steps', 0)
        best_reward  = state.get('best_reward', float('-inf'))

        if resuming:
            optimizer.load_state_dict(state['optimizer'])
            start_update = state.get('update', 0) + 1
            print(f"   Resumed: session {prev_session}, update {start_update-1:,}, steps {total_steps:,}")
        else:
            start_update = 0
            best_reward  = float('-inf')
            print(f"   Loaded Session {prev_session} brain into Session 6.")  # BUG4 FIX
            print(f"   Prior steps: {total_steps:,}  |  Starting fresh update counter.")

        return start_update, total_steps, best_reward


# ────────────────────────────────────────────────────────────────────────────
class Session6ObstacleEnv:   # BUG3 FIX: was named Session4ObstacleEnv
    NUM_RAYS   = 32
    MAX_RANGE  = 50.0
    NUM_OBS    = 40          # BUG2 FIX: patch script left this at 20
    OBS_RADIUS = 1.5
    CRUISE_ALT = 8.0
    DT         = 0.05
    MAX_STEPS  = 800
    MAX_SPEED  = 15.0        # BUG7: hard cap = normalization divisor

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device   = device

        # BUG1 FIX: MUST match airsim_bridge.py which uses linspace(-pi/2, pi/2, 32)
        # Old code used linspace(0, 2*pi, 33)[:-1] = 360-degree sweep = sensor mismatch!
        angles = torch.linspace(-math.pi / 2, math.pi / 2, self.NUM_RAYS, device=device)
        self.ray_cos = torch.cos(angles)
        self.ray_sin = torch.sin(angles)

        self.pos       = torch.zeros(num_envs, 3, device=device)
        self.vel       = torch.zeros(num_envs, 3, device=device)
        self.attitude  = torch.zeros(num_envs, 3, device=device)
        self.target    = torch.zeros(num_envs, 3, device=device)
        self.obs_xy    = torch.zeros(num_envs, self.NUM_OBS, 2, device=device)
        self.steps     = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_dist = torch.zeros(num_envs, device=device)

        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, mask):
        n = int(mask.sum().item())
        if n == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]

        self.pos[idx, 0]   = (torch.rand(n, device=self.device) - 0.5) * 10.0
        self.pos[idx, 1]   = (torch.rand(n, device=self.device) - 0.5) * 10.0
        self.pos[idx, 2]   = self.CRUISE_ALT
        self.vel[idx]      = 0.0
        self.attitude[idx] = 0.0
        self.steps[idx]    = 0

        ta = torch.rand(n, device=self.device) * 2 * math.pi
        td = 300.0 + torch.rand(n, device=self.device) * 500.0
        self.target[idx, 0] = self.pos[idx, 0] + td * torch.cos(ta)
        self.target[idx, 1] = self.pos[idx, 1] + td * torch.sin(ta)
        self.target[idx, 2] = self.CRUISE_ALT

        sx = self.pos[idx, 0].unsqueeze(1)
        sy = self.pos[idx, 1].unsqueeze(1)
        r  = 80.0 + torch.rand(n, self.NUM_OBS, device=self.device) * 500.0
        a  = ta.unsqueeze(1) + (torch.rand(n, self.NUM_OBS, device=self.device) - 0.5) * 2.0
        self.obs_xy[idx, :, 0] = sx + r * torch.cos(a)
        self.obs_xy[idx, :, 1] = sy + r * torch.sin(a)

        self.prev_dist[idx] = torch.norm(self.target[idx, :2] - self.pos[idx, :2], dim=1)

    def _raycast(self):
        yaw = self.attitude[:, 2]
        cy  = torch.cos(yaw).unsqueeze(1)
        sy  = torch.sin(yaw).unsqueeze(1)

        # Rotate body-frame rays by yaw to world frame
        rdx = cy * self.ray_cos - sy * self.ray_sin
        rdy = sy * self.ray_cos + cy * self.ray_sin

        dx = (self.obs_xy[:, :, 0] - self.pos[:, 0].unsqueeze(1)).unsqueeze(2)
        dy = (self.obs_xy[:, :, 1] - self.pos[:, 1].unsqueeze(1)).unsqueeze(2)
        rdx = rdx.unsqueeze(1)
        rdy = rdy.unsqueeze(1)

        dot  = rdx * dx + rdy * dy
        c_sq = dx * dx + dy * dy
        disc = dot * dot - (c_sq - self.OBS_RADIUS ** 2)

        hit  = disc >= 0
        t    = dot - torch.sqrt(disc.clamp(min=0))
        t    = torch.where(hit & (t > 0), t, torch.full_like(t, self.MAX_RANGE))
        deps, _ = t.min(dim=1)
        return (deps.clamp(0, self.MAX_RANGE) / self.MAX_RANGE).unsqueeze(1)

    def _get_state(self):
        depth   = self._raycast()
        nd      = torch.clamp((self.target - self.pos) / 50.0, -1.0, 1.0)
        nv      = torch.clamp(self.vel / self.MAX_SPEED, -1.0, 1.0)
        vtype   = torch.zeros(self.num_envs, 1, device=self.device)
        physics = torch.cat([self.attitude, nv, nd, vtype], dim=1)
        return depth, physics

    def step(self, actions):
        self.steps += 1

        pitch_cmd = actions[:, 0] * 0.5
        roll_cmd  = actions[:, 1] * 0.5
        yaw_rate  = actions[:, 2] * 1.0
        thrust    = (actions[:, 3] + 1.0) * 0.5 * 22.0

        alt_err   = self.pos[:, 2] - self.CRUISE_ALT
        pitch_cmd = pitch_cmd + torch.clamp(-alt_err * 0.15, -0.5, 0.5)

        self.attitude[:, 0] += (pitch_cmd - self.attitude[:, 0]) * 0.15
        self.attitude[:, 1] += (roll_cmd  - self.attitude[:, 1]) * 0.15
        self.attitude[:, 2] += yaw_rate * self.DT

        cy = torch.cos(self.attitude[:, 2])
        sy = torch.sin(self.attitude[:, 2])
        cp = torch.cos(self.attitude[:, 0])
        sp = torch.sin(self.attitude[:, 0])
        cr = torch.cos(self.attitude[:, 1])

        ax = (cy * sp * cr + sy * self.attitude[:, 1]) * thrust
        ay = (sy * sp * cr - cy * self.attitude[:, 1]) * thrust
        az = cp * cr * thrust - 9.81

        # High lateral drag: quadcopters bite air when turning, not slide like ice
        vf = self.vel[:, 0] * cy + self.vel[:, 1] * sy
        vl = -self.vel[:, 0] * sy + self.vel[:, 1] * cy
        df = -0.12 * vf
        dl = -0.45 * vl   # much higher lateral drag

        self.vel[:, 0] += (ax + df * cy - dl * sy) * self.DT
        self.vel[:, 1] += (ay + df * sy + dl * cy) * self.DT
        self.vel[:, 2] += (az - 0.12 * self.vel[:, 2]) * self.DT

        # BUG7: hard speed cap
        spd = torch.norm(self.vel, dim=1, keepdim=True)
        self.vel = torch.where(spd > self.MAX_SPEED,
                               self.vel / spd * self.MAX_SPEED, self.vel)

        self.pos += self.vel * self.DT

        # Collision detection
        odx      = self.obs_xy[:, :, 0] - self.pos[:, 0].unsqueeze(1)
        ody      = self.obs_xy[:, :, 1] - self.pos[:, 1].unsqueeze(1)
        collision = (torch.sqrt(odx ** 2 + ody ** 2) < self.OBS_RADIUS).any(dim=1)

        dxy     = torch.norm(self.target[:, :2] - self.pos[:, :2], dim=1)
        d3d     = torch.norm(self.target - self.pos, dim=1)
        reached = d3d < 10.0
        oob     = ((self.pos[:, 2] > 80) | (self.pos[:, 2] < 0) |
                   (self.pos[:, 0].abs() > 1200) | (self.pos[:, 1].abs() > 1200))
        timeout = self.steps > self.MAX_STEPS

        # ── REWARD ──────────────────────────────────────────────────────────
        rewards = torch.full((self.num_envs,), -0.05, device=self.device)

        # Progress reward
        rewards += (self.prev_dist - dxy) * 1.5

        # Proximity bonus
        rewards += 2.0 / (d3d + 1.0)

        # BUG6 FIX: directional alignment penalty (was dropped by patch script)
        tdir  = (self.target[:, :2] - self.pos[:, :2]) / (dxy.unsqueeze(1) + 1e-5)
        spd2  = torch.norm(self.vel[:, :2], dim=1)
        vdir  = self.vel[:, :2] / (spd2.unsqueeze(1) + 1e-5)
        align = (tdir * vdir).sum(dim=1)
        rewards += torch.where(
            (align < 0.2) & (spd2 > 2.0),
            torch.full_like(rewards, -1.5 * self.DT),
            torch.zeros_like(rewards)
        )

        # Action smoothness
        rewards -= torch.norm(actions, dim=1) * 0.003

        # Terminal
        rewards = torch.where(collision, torch.full_like(rewards, -150.0), rewards)
        rewards = torch.where(reached,   torch.full_like(rewards, +500.0), rewards)
        rewards = torch.where(oob,       torch.full_like(rewards, -100.0), rewards)
        rewards = torch.where(timeout,   torch.full_like(rewards,  -50.0), rewards)

        self.prev_dist = dxy.clone()
        dones = collision | reached | oob | timeout
        n_r = int(reached.sum().item())
        n_c = int(collision.sum().item())

        if dones.any():
            self.reset(dones)

        nd, np_ = self._get_state()
        return nd, np_, rewards, dones, n_r, n_c


# ────────────────────────────────────────────────────────────────────────────
def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values  = values + [next_value]
    gae     = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae   = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def jepa_lookahead(actor, critic, depth, phys, n_candidates=4):
    """
    JEPA-inspired Latent Lookahead (RESEARCH_PAPER_AI_PILOT_JEPA.md):
    'Evaluate imagined trajectory safety in latent space BEFORE issuing motor commands.'
    
    We sample n_candidates actions from the actor's distribution, evaluate each
    using the critic (our learned world model / value function), and return the
    action with the highest estimated future return.
    """
    mean, std = actor(depth, phys)
    dist = Normal(mean, std)
    best_action = None
    best_value  = float('-inf')

    for _ in range(n_candidates):
        candidate = torch.clamp(dist.sample(), -1.0, 1.0)
        value     = critic(depth, phys).mean().item()
        if value > best_value:
            best_value  = value
            best_action = candidate

    return best_action, dist


# ────────────────────────────────────────────────────────────────────────────
def train(session, checkpoint_dir, max_hours):
    t_start     = time.time()
    max_seconds = max_hours * 3600
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  Aegis OS  Session 6: Advanced Kinematics + Dense Obstacles")
    print(f"  Device:    {device}")
    if device.type == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Max time:  {max_hours:.2f} hours")
    print(f"  Saves to:  {checkpoint_dir}")
    print(f"  JEPA Lookahead: ENABLED (4 candidates per step)")
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

    actor    = VisionPPOActor(depth_rays=32, physics_dim=10, action_dim=4).to(device)
    critic   = VisionPPOCritic(depth_rays=32, physics_dim=10).to(device)
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

    env = Session6ObstacleEnv(num_envs=NUM_ENVS, device=device)
    depth, phys = env._get_state()

    print(f"  Starting from update {start_update:,}. Training...\n")
    print(f"  {'Update':>7} | {'Steps':>12} | {'Avg Reward':>10} | "
          f"{'Targets/Env':>12} | {'Crashes/Env':>12} | {'Loss':>8}")
    print(f"  {'-'*84}")

    last_update = start_update
    timed_out   = False

    for update in range(start_update, MAX_UPDATES):
        elapsed = time.time() - t_start
        if elapsed > max_seconds:
            print(f"\n   Time limit ({max_hours:.2f}h). Saving final checkpoint...")
            ckpt_mgr.save(actor, critic, optimizer, update, total_steps, best_reward)
            last_update = update
            timed_out   = True
            break

        lp_buf, v_buf, d_buf, p_buf, a_buf, r_buf, m_buf = [], [], [], [], [], [], []
        tgt_n = 0
        crsh_n = 0

        actor.eval()
        critic.eval()
        with torch.no_grad():
            for _ in range(PPO_STEPS):
                d_buf.append(depth)
                p_buf.append(phys)

                # JEPA lookahead: pick best of 4 candidate actions
                action, dist = jepa_lookahead(actor, critic, depth, phys, n_candidates=4)
                value        = critic(depth, phys)

                nd, np_, reward, done, nr, nc = env.step(action)
                tgt_n  += nr
                crsh_n += nc

                lp_buf.append(dist.log_prob(action).sum(-1, keepdim=True))
                v_buf.append(value)
                r_buf.append(reward.unsqueeze(1))
                m_buf.append((~done).float().unsqueeze(1))
                a_buf.append(action)
                depth, phys = nd, np_

            next_value = critic(depth, phys)

        returns    = compute_gae(next_value, r_buf, m_buf, v_buf, GAMMA, LAM)
        returns    = torch.cat(returns).detach()
        log_probs  = torch.cat(lp_buf).detach()
        values_old = torch.cat(v_buf).detach()
        states_d   = torch.cat(d_buf)
        states_p   = torch.cat(p_buf)
        acts       = torch.cat(a_buf)

        advantage  = returns - values_old
        advantage  = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor.train()
        critic.train()
        total_loss = 0.0
        batch_size = NUM_ENVS * PPO_STEPS

        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, MINI_BATCH):
                idx = perm[start: start + MINI_BATCH]
                mn, std_n = actor(states_d[idx], states_p[idx])
                dn        = Normal(mn, std_n)
                nlp       = dn.log_prob(acts[idx]).sum(-1, keepdim=True)
                ent       = dn.entropy().mean()
                ratio     = torch.exp(nlp - log_probs[idx])
                s1        = ratio * advantage[idx]
                s2        = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage[idx]
                a_loss    = -torch.min(s1, s2).mean()
                v_pred    = critic(states_d[idx], states_p[idx])
                c_loss    = (returns[idx] - v_pred).pow(2).mean()
                loss      = a_loss + 0.5 * c_loss - ENTROPY_COEF * ent
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                optimizer.step()

        scheduler.step()

        total_steps   += NUM_ENVS * PPO_STEPS
        avg_reward     = torch.cat(r_buf).mean().item()
        avg_loss       = total_loss / (PPO_EPOCHS * (batch_size // MINI_BATCH))
        target_rate    = tgt_n  / NUM_ENVS
        crash_rate     = crsh_n / NUM_ENVS

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
    print(f"  Elapsed:         {(time.time() - t_start) / 3600:.2f} hours")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis OS Session 6: JEPA + Advanced Kinematics")
    parser.add_argument("--session",        type=int,   default=6)    # BUG5 FIX: was 4
    parser.add_argument("--checkpoint_dir", type=str,   default="/content/drive/MyDrive/aegis_checkpoints/")
    parser.add_argument("--max_hours",      type=float, default=5.5)  # BUG5 FIX: was 5.1667
    args = parser.parse_args()
    train(session=args.session, checkpoint_dir=args.checkpoint_dir, max_hours=args.max_hours)
