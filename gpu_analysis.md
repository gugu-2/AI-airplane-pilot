# GPU Usage, Training Time & Advanced Scenario Architecture

---

## Part 1: Why Your T4 Only Used 2GB — The Exact Numbers

The T4 has **15GB VRAM**. Here is a precise breakdown of what each script actually allocates:

### 2D DQN (`train_rl.py`) — was using ~0.3GB

| Component | Calculation | Size |
|-----------|-------------|------|
| `DeepQNetwork` (×2, online + target) | 4 tiny linear layers = 26K params × 4 bytes × 2 | **0.2 MB** |
| Replay Buffer (2M transitions) | 2M × 5 floats × 4 bytes × 4 arrays | **160 MB** |
| Env state (4096 envs) | 4096 × 5 × 4 bytes | **0.08 MB** |
| Batch tensors | 16384 × 5 × 4 bytes | **0.3 MB** |
| PyTorch CUDA overhead | Fixed cost | ~300 MB |
| **TOTAL** | | **~0.5 GB** |

### 3D PPO (`train_rl_3d.py`) — was using ~1.5GB

| Component | Calculation | Size |
|-----------|-------------|------|
| `VisionPPOActor` + `VisionPPOCritic` | 72K params × 4 bytes | **0.3 MB** |
| Rollout buffer (2048 envs × 128 steps) | 2048 × 128 × (32+9) floats × 4 bytes | **43 MB** |
| Stacked training tensors (PPO update) | 2048 × 128 = 262K samples | ~100 MB |
| PyTorch CUDA overhead | Fixed cost | ~300 MB |
| **TOTAL** | | **~1.5 GB** |

> [!WARNING]
> The network itself is only **0.3 MB** — the size of a small JPEG photo.
> A T4-class GPU is a **matrix multiplication supercomputer**. Feeding it a 26,000-parameter network 
> is like hiring a Formula 1 driver to go to the grocery store.

### With the Bug Fixes (PPO_STEPS 128→512):
- Rollout buffer: 2048 × 512 × 41 floats × 4 bytes ≈ **172 MB**
- Stacked training tensors: ~400 MB
- **New estimated total: ~1.8–2.5 GB** — still massively under-utilizing the GPU.

---

## Part 2: Honest Training Time Estimates

### Current Fixed Scripts on T4:

| Script | Steps | GPU % Used | Estimated Time |
|--------|-------|-----------|----------------|
| `train_rl.py` (2D DQN) | 5M global steps | ~5% | **30–60 minutes** |
| `train_rl_3d.py` (3D PPO) | 500 updates × 512 steps × 2048 envs | ~15% | **3–5 hours** |

The 3D script takes longer because PPO is more computationally expensive per-sample than DQN — it re-evaluates the policy multiple times (PPO_EPOCHS=4) on the same rollout. But 85% of the GPU is still wasted.

---

## Part 3: Why the Current Architecture Cannot Handle Your Scenarios

You are asking the right questions. Here is why the current models **fundamentally cannot** handle the scenarios you described:

### Scenario 1: Drone landing on a skyscraper helipad
- **Problem:** The current environment spawns spherical obstacles in an empty void. There are no flat surfaces, no altitude awareness of ground-level objects, no concept of a "landing zone" at height.
- **What's needed:** A 3D bounding box environment with a designated `landing_pad` region at a target altitude. The AI needs a landing reward shaped to require both horizontal precision (land *on* the pad) and vertical descent control.

### Scenario 2: Fixed-wing airplane landing on a dirt road through a forest
- **Problem:** The current physics assume a drone (vertical hover). A fixed-wing airplane **stalls below ~50 km/h** and cannot hover. It must maintain forward airspeed, descend at a specific glide slope angle (~3°), and have space to roll out.
- **What's needed:** A separate `FixedWingPhysics` class with aerodynamic lift equations: `L = 0.5 * ρ * v² * Cl * A`. Without forward velocity, the plane falls. The AI must learn a completely different control strategy.

### Scenario 3: Airplane landing on a moving ship at sea
- **Problem:** The current environment has static targets. A ship moves at ~25 km/h in a random direction on the water. The landing zone itself is moving, rolling, and pitching with the waves.
- **What's needed:** A `MovingTarget` class that updates `target_pos` every step using a velocity vector. The landing pad must have a `normal_vector` (representing the ship's deck tilt) and the aircraft needs to match the platform's velocity before touchdown.

### Scenario 4: Crash landing in Norwegian mountains
- **Problem:** The environment has no terrain. "Ground" is a flat plane at z=0. Mountains don't exist.
- **What's needed:** A **Perlin noise heightmap** (procedurally generated terrain) so every episode spawns a different mountain range. The AI must find the flattest patch, judge slope angle, and commit to a controlled landing before fuel runs out. A "crash landing reward" must be shaped: landing at 5 m/s on a 5° slope = survival. Landing at 30 m/s on a cliff = catastrophe.

### Scenario 5: Drone avoiding trees on either side of a road vs. airplane avoiding skyscrapers
- **Problem:** A single AI model cannot differentiate between a drone weaving through 2m-tall obstacles and a 747 avoiding 300m skyscrapers. Their physics are completely different.
- **What's needed:** A **Vehicle Profile Vector** injected into the neural network, e.g.:
  `[mass_kg, wingspan_m, max_speed_ms, min_speed_ms, is_vtol, max_climb_rate, max_descent_rate]`
  The network learns that `is_vtol=True, wingspan=0.5` means hover and weave, while `is_vtol=False, wingspan=30` means maintain 80m/s and give wide berths.

---

## Part 4: What a Proper Architecture Would Look Like

To train an AI that generalizes across all these scenarios, here is what is actually needed:

### Network Architecture Upgrade
```
Current:  26,000 parameters, 1D LiDAR (32 rays), 9 physics dims
Advanced: 2,000,000+ parameters, 2D Depth Map (64×64), 20+ physics dims + vehicle profile + LSTM temporal memory
```

### Memory Estimate for Advanced System (T4 15GB):
| Component | Size |
|-----------|------|
| ResNet-18 vision backbone (64×64 depth) | ~11M params = **44 MB** |
| LSTM temporal memory (512 hidden) | ~4M params = **16 MB** |
| Vehicle profile embedding | ~0.1M params |
| Rollout buffer: 2048 envs × 512 steps | ~800 MB |
| Mixed-precision (fp16) training | Halves model memory |
| **Estimated TOTAL** | **~6–10 GB** (proper GPU saturation) |

### Training Time Estimate for Advanced System:
| Scenario Set | Episodes | T4 Time |
|--------------|----------|---------|
| Drone basic navigation + obstacle avoidance | 10K updates | ~2 hrs |
| Fixed-wing approach + runway landing | 10K updates | ~2 hrs |
| Moving ship landing | 15K updates | ~3 hrs |
| Mountain terrain crash landing | 10K updates | ~2 hrs |
| Full curriculum (all scenarios combined) | 50K updates | **~10–14 hours** |

> [!IMPORTANT]
> **Curriculum Learning** is critical. You cannot train on all scenarios at once from the beginning.
> The AI must master easy navigation first (update 0–10K), then add obstacles (10K–20K), 
> then add complex terrain (20K–50K). This is exactly how human pilots are trained.
> Throwing a student into a mountain storm landing on Day 1 means they learn nothing.

---

## Summary: What I Recommend Building

| Stage | What to Build | GPU Usage | Colab Time |
|-------|--------------|-----------|------------|
| **Stage 1** (now) | Fix bugs ✅ + run current scripts | ~2 GB | 4–5 hrs |
| **Stage 2** | Add vehicle profiles + 8 obstacle types | ~4 GB | 6–8 hrs |
| **Stage 3** | Add terrain heightmap + 2D depth vision (ResNet) | ~8 GB | 10–14 hrs |
| **Stage 4** | Add LSTM + moving targets + full curriculum | ~12 GB | 24–48 hrs (Colab Pro) |

Would you like me to start writing Stage 2 — the Vehicle Profile system?
