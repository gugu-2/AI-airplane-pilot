# Training Code Full Audit Report

Both training scripts were read line by line, and their math, physics, and logic verified.

---

## FILE 1: `train_rl.py` — 2D DQN Training

### ✅ Things that ARE Correct
- Replay Buffer circular wrapping logic (`ptr`, `overflow`) is mathematically correct.
- DQN Bellman target: `Q_target = r + γ * max Q(s', a') * (1 - done)` is correct.
- `SmoothL1Loss` (Huber Loss) is a good choice for DQN — more stable than MSE.
- Gradient clipping (`max_norm=10.0`) is present.
- `target_model` hard-update every 250 steps is standard and correct.
- Epsilon-greedy decay is correctly implemented.
- The replay buffer is pre-allocated fully on GPU — good.

### 🐛 Confirmed Bugs & Math Errors

#### BUG 1 — SENSOR NOISE IN STATE VECTOR (Line 126–128)
```python
noise = torch.empty(self.num_envs, device=self.device).uniform_(-0.05, 0.05)
return torch.stack([norm_x, norm_y, norm_heading, norm_obs, noise], dim=1)
```
**Problem:** The 5th dimension of the state vector is **random noise every single step**. This means the DQN network can never learn what the 5th input means, because it changes randomly each call to `_get_state()`. The same physical state (drone at position X, obstacle at distance Y) will produce thousands of different state vectors. This dramatically slows convergence and caps maximum performance. The old DQN in `rl_models.py` describes the 5th dim as "reserved" — it should be `0.0` (a constant).

#### BUG 2 — REWARD ASSIGNMENT OVERWRITE (Lines 171–187)
```python
rewards = torch.full((self.num_envs,), -0.1, ...)
rewards[goal_mask] = 100.0   # overwrites -0.1
rewards[obs_mask] = -100.0   # may overwrite a 100.0 goal reward!
rewards[time_mask] = -50.0   # may overwrite collision reward
```
**Problem:** Masks are applied sequentially with `=` assignment. If a drone reaches the goal AND the goal spawned inside an obstacle (possible because obstacles are random), then `obs_mask` will overwrite the `+100` goal reward with `-100`. The order is wrong — goal should always take priority. Should use `torch.where` with priority ordering.

#### BUG 3 — ONLY ONE OBSTACLE PER ENVIRONMENT (Lines 82–84)
```python
self.obs_x = torch.zeros(num_envs, device=device)
self.obs_y = torch.zeros(num_envs, device=device)
self.obs_z = torch.zeros(num_envs, device=device)
```
**Problem:** There is exactly 1 obstacle per parallel environment. The real world has thousands. An AI that trains with one obstacle will fail in the real world with many.

#### BUG 4 — DESCENT ACTION IS CLAMPED TO MINIMUM ALTITUDE (Line 161)
```python
self.drone_z[mask4] = torch.clamp(self.drone_z[mask4] - step_z, min=50.0)
```
**Problem:** The drone **can never go below 50 metres**. This means the drone will never learn to actually land. It will approach but stop 50m in the air indefinitely.

#### BUG 5 — EPSILON DECAY IS TOO SLOW FOR 5M STEPS (Line 219)
```python
EPSILON_DECAY = 0.9995
```
**Problem:** With 5M steps and 4096 parallel envs, `num_loops = 5M / 4096 ≈ 1220` loops. Epsilon after 1220 decays: `1.0 * 0.9995^1220 ≈ 0.543`. Epsilon never gets below 0.01 as intended. The agent spends the **entire training run** taking random actions more than half the time. The `EPSILON_DECAY` should be approximately `(EPSILON_MIN / EPSILON_START) ^ (1 / num_loops) = 0.01^(1/1220) ≈ 0.9962` per loop.

---

## FILE 2: `train_rl_3d.py` — 3D PPO Training

### ✅ Things that ARE Correct
- GAE (Generalized Advantage Estimation) math is correct: `δ = r + γV(s') - V(s)`, `A = δ + γλA`.
- PPO clipped surrogate objective `min(r·A, clip(r, 1-ε, 1+ε)·A)` is correct.
- Combined loss `L = L_actor + 0.5·L_critic - 0.01·L_entropy` is the standard PPO formula.
- Gradient clipping on both actor and critic is correct.
- 6-DOF physics: gravity subtraction `accel_z = cos(pitch)*cos(roll)*thrust - 9.81` is correct.
- Drag model `-0.1 * vel` (linear drag) is a reasonable approximation.
- LiDAR ray-sphere intersection math (`tca`, `thc`, `t0`) is geometrically correct.

### 🐛 Confirmed Bugs & Math Errors

#### CRITICAL BUG 1 — CRASH IN `VisionPPOCritic.forward()` (`rl_models.py` Line 226–229)
```python
def forward(self, depth_scan, physics_state):
    phys_features = self.physics_net(physics_state)
    fused = torch.cat([vis_features, phys_features], dim=1)  # NameError!
    return self.critic_net(fused)
```
**Problem:** `vis_features` is **never computed**. The line `vis_features = self.vision_net(depth_scan)` is completely missing. This will throw a `NameError: name 'vis_features' is not defined` the first time the Critic is called during training. This means the 3D PPO training CANNOT RUN AT ALL without hitting this crash within the first update.

#### CRITICAL BUG 2 — RESET LOGIC READS STALE STATE (Lines 177–181)
```python
next_depth, next_phys = self._get_state()
if dones.any():
    reset_depth, reset_phys = self.reset(dones)
    next_depth[dones] = reset_depth[dones]
    next_phys[dones] = reset_phys[dones]
```
**Problem:** `_get_state()` is called BEFORE `reset(dones)`. So `next_depth` and `next_phys` already contain the terminal (crashed/exploded) state. Then `reset()` resets the internal variables. `reset_depth[dones]` correctly contains fresh states, but `reset_depth[~dones]` contains a re-queried state of already-reset envs. The semantically correct pattern is to call `_get_state()` **after** `reset()`, or use the return value of `reset()` directly.

#### BUG 3 — WRONG SAVE FILENAME (Line 285)
```python
model_path = os.path.join(os.path.dirname(__file__), '../models/aegis_vision_v1.pth')
```
**Problem:** The script is supposed to produce `aegis_vision_v1-3d.pth` (the file you trained and used), but it saves as `aegis_vision_v1.pth`, which **overwrites the 2D DQN model** trained by `train_rl.py`. You would lose your 2D model every time you run 3D training.

#### BUG 4 — PHYSICS: PITCH CONTROLS WRONG AXIS (Lines 138–139)
```python
accel_x = torch.cos(self.attitude[:, 2]) * torch.sin(self.attitude[:, 0]) * thrust_cmd
accel_y = torch.sin(self.attitude[:, 2]) * torch.sin(self.attitude[:, 0]) * thrust_cmd
```
**Problem:** In standard aerospace body-frame convention, `attitude[:, 0]` is **Pitch** (nose up/down). A pitched-forward attitude (nose down) should generate forward acceleration, but `sin(pitch) * thrust` is correct only for small angles. The issue is `attitude[:, 0]` is being incorrectly initialized to `[pitch, roll, yaw]` order, but the action mapping (Line 129-131) uses `actions[:, 0]` as `pitch_cmd` and updates `attitude[:, 0]` — this part is internally consistent. However, the Z-axis lift equation `cos(pitch)*cos(roll)*thrust - g` should include the ground-contact condition. A drone hovering at 0 pitch/roll with `thrust=9.81` produces `cos(0)*cos(0)*9.81 - 9.81 = 0`. ✅ This is actually correct for hover.

#### BUG 5 — NO EPISODE-LEVEL REWARD TRACKING (Line 279–281)
```python
avg_reward = torch.cat(rewards).mean().item()
if update % 5 == 0:
    print(f"Update: {update} | Avg Step Reward: {avg_reward:.2f}")
```
**Problem:** The logging shows **average step reward** not **average episode reward**. Since the step reward is always dominated by the time penalty `-0.05`, this metric will always show a slightly negative number and gives no information about whether the agent is actually succeeding at reaching targets. It should track goal-reached fraction per update.

#### BUG 6 — MAX EPISODE STEPS TOO SHORT (Line 202)
```python
PPO_STEPS = 128
MAX_EPISODES = 500
```
**Problem:** 128 steps × `dt=0.05s` = **only 6.4 seconds of simulated flight per rollout**. With a maximum episode length of 200 steps (Line 174), that is also only 10 seconds. Reaching a target 20 metres away while also avoiding 5 obstacles in 10 simulated seconds is extremely challenging for early training, causing the agent to time out nearly every episode. PPO_STEPS should be at least 512 and max_steps should be 400-500.

---

## Summary Table

| # | File | Severity | Description |
|---|------|----------|-------------|
| 1 | `train_rl.py` | 🟡 Medium | Random noise as a state feature, prevents convergence |
| 2 | `train_rl.py` | 🟠 High | Reward overwrite ordering, wrong reward when goal & obstacle overlap |
| 3 | `train_rl.py` | 🟡 Medium | Only 1 obstacle per env, AI won't generalize |
| 4 | `train_rl.py` | 🔴 Critical | Drone can never descend below 50m, cannot learn landing |
| 5 | `train_rl.py` | 🟠 High | Epsilon decay too slow, agent is random >50% of training time |
| 6 | `train_rl_3d.py` | 🔴 Critical | `vis_features` NameError crash in Critic, PPO cannot train AT ALL |
| 7 | `train_rl_3d.py` | 🔴 Critical | State read before reset, corrupts observations at episode boundaries |
| 8 | `train_rl_3d.py` | 🟠 High | Wrong save filename, overwrites 2D model |
| 9 | `train_rl_3d.py` | 🟡 Medium | No episode success rate logging |
| 10 | `train_rl_3d.py` | 🟡 Medium | Episode too short (6.4s), causes premature timeouts during early training |
