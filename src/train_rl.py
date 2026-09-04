"""
Aegis OS \u2014 Massively Parallel GPU RL Training Script
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import DeepQNetwork

class GPUReplayBuffer:
    def __init__(self, capacity, state_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate on GPU
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        
    def push(self, states, actions, rewards, next_states, dones):
        batch_size = states.size(0)
        
        # If batch fits entirely before wrapping
        end_idx = self.ptr + batch_size
        if end_idx <= self.capacity:
            self.states[self.ptr:end_idx] = states
            self.actions[self.ptr:end_idx] = actions.unsqueeze(1)
            self.rewards[self.ptr:end_idx] = rewards.unsqueeze(1)
            self.next_states[self.ptr:end_idx] = next_states
            self.dones[self.ptr:end_idx] = dones.unsqueeze(1)
        else:
            # Wrap around
            overflow = end_idx - self.capacity
            fit = batch_size - overflow
            
            self.states[self.ptr:self.capacity] = states[:fit]
            self.actions[self.ptr:self.capacity] = actions[:fit].unsqueeze(1)
            self.rewards[self.ptr:self.capacity] = rewards[:fit].unsqueeze(1)
            self.next_states[self.ptr:self.capacity] = next_states[:fit]
            self.dones[self.ptr:self.capacity] = dones[:fit].unsqueeze(1)
            
            self.states[0:overflow] = states[fit:]
            self.actions[0:overflow] = actions[fit:].unsqueeze(1)
            self.rewards[0:overflow] = rewards[fit:].unsqueeze(1)
            self.next_states[0:overflow] = next_states[fit:]
            self.dones[0:overflow] = dones[fit:].unsqueeze(1)
            
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

# BUG 3 FIX: Increased obstacle count from 1 to 8 per environment.
# One spherical obstacle does not represent a real-world airspace.
NUM_OBSTACLES = 8

class VectorizedGPUEnv:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.num_obs = NUM_OBSTACLES
        
        self.drone_x = torch.zeros(num_envs, device=device)
        self.drone_y = torch.zeros(num_envs, device=device)
        self.drone_z = torch.ones(num_envs, device=device) * 50.0
        
        self.target_x = torch.zeros(num_envs, device=device)
        self.target_y = torch.zeros(num_envs, device=device)
        self.target_z = torch.zeros(num_envs, device=device)
        
        # BUG 3 FIX: Store (num_envs, num_obs) arrays instead of flat (num_envs,)
        self.obs_x = torch.zeros((num_envs, self.num_obs), device=device)
        self.obs_y = torch.zeros((num_envs, self.num_obs), device=device)
        self.obs_z = torch.zeros((num_envs, self.num_obs), device=device)
        self.obs_r = torch.ones((num_envs, self.num_obs), device=device) * 1.5  # 1.5m radius
        
        self.steps = torch.zeros(num_envs, device=device)
        
        # Initialize all envs
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        
    def reset(self, mask):
        num_reset = mask.sum().item()
        if num_reset == 0:
            return self._get_state()
            
        self.drone_x[mask] = 0.0
        self.drone_y[mask] = 0.0
        # BUG 4 FIX: Allow spawning and landing near ground (5m) not locked at 50m
        self.drone_z[mask] = 5.0
        
        self.target_x[mask] = torch.empty(num_reset, device=self.device).uniform_(-15.0, 15.0)
        self.target_y[mask] = torch.empty(num_reset, device=self.device).uniform_(-15.0, 15.0)
        self.target_z[mask] = torch.empty(num_reset, device=self.device).uniform_(0.0, 100.0)
        
        # BUG 3 FIX: Populate all 8 obstacles per reset env
        for i in range(self.num_obs):
            self.obs_x[mask, i] = torch.empty(num_reset, device=self.device).uniform_(-12.0, 12.0)
            self.obs_y[mask, i] = torch.empty(num_reset, device=self.device).uniform_(-12.0, 12.0)
            self.obs_z[mask, i] = torch.empty(num_reset, device=self.device).uniform_(0.0, 120.0)
        
        self.steps[mask] = 0
        return self._get_state()
        
    def _get_state(self):
        dist_x = self.target_x - self.drone_x
        dist_y = self.target_y - self.drone_y
        
        norm_x = torch.clamp(dist_x / 20.0, -1.0, 1.0)
        norm_y = torch.clamp(dist_y / 20.0, -1.0, 1.0)
        
        heading = torch.atan2(dist_y, dist_x)
        norm_heading = heading / 3.14159265
        
        # BUG 3 FIX: Find closest obstacle across all 8 obstacles per env.
        # obs_x shape: (num_envs, num_obs). Expand drone position to broadcast.
        drone_x_exp = self.drone_x.unsqueeze(1)
        drone_y_exp = self.drone_y.unsqueeze(1)
        drone_z_exp = self.drone_z.unsqueeze(1)
        all_obs_dist = torch.sqrt(
            (self.obs_x - drone_x_exp)**2 +
            (self.obs_y - drone_y_exp)**2 +
            (self.obs_z - drone_z_exp)**2
        )  # shape: (num_envs, num_obs)
        min_obs_dist = all_obs_dist.min(dim=1).values  # shape: (num_envs,)
        norm_obs = torch.clamp(1.0 / (min_obs_dist + 0.1), 0.0, 1.0)
        
        # BUG 1 FIX: Removed random noise. The 5th dim was noise which prevented the
        # network from ever converging. Using a constant 0.0 (reserved slot) instead.
        reserved = torch.zeros(self.num_envs, device=self.device)
        
        return torch.stack([norm_x, norm_y, norm_heading, norm_obs, reserved], dim=1)
        
    def step(self, actions):
        self.steps += 1
        
        dist_x = self.target_x - self.drone_x
        dist_y = self.target_y - self.drone_y
        angle = torch.atan2(dist_y, dist_x)
        
        step_xy = 0.5
        step_z = 2.0
        
        # Action 0: FWD
        mask0 = (actions == 0)
        self.drone_x[mask0] += torch.cos(angle[mask0]) * step_xy
        self.drone_y[mask0] += torch.sin(angle[mask0]) * step_xy
        
        # Action 1: LEFT
        mask1 = (actions == 1)
        self.drone_x[mask1] += torch.cos(angle[mask1] + 0.5) * step_xy
        self.drone_y[mask1] += torch.sin(angle[mask1] + 0.5) * step_xy
        
        # Action 2: RIGHT
        mask2 = (actions == 2)
        self.drone_x[mask2] += torch.cos(angle[mask2] - 0.5) * step_xy
        self.drone_y[mask2] += torch.sin(angle[mask2] - 0.5) * step_xy
        
        # Action 3: CLIMB
        mask3 = (actions == 3)
        self.drone_z[mask3] = torch.clamp(self.drone_z[mask3] + step_z, max=1000.0)
        
        # Action 4: DESCEND
        # BUG 4 FIX: Removed min=50.0 clamp. Drone was locked above 50m and could
        # never learn to land. Now allows descent to ground level (0m).
        mask4 = (actions == 4)
        self.drone_z[mask4] = torch.clamp(self.drone_z[mask4] - step_z, min=0.0)
        
        target_dist = torch.sqrt((self.target_x - self.drone_x)**2 + 
                                 (self.target_y - self.drone_y)**2 +
                                 (self.target_z - self.drone_z)**2)
        
        # BUG 3 FIX: Check collision against all 8 obstacles
        drone_x_exp = self.drone_x.unsqueeze(1)
        drone_y_exp = self.drone_y.unsqueeze(1)
        drone_z_exp = self.drone_z.unsqueeze(1)
        all_obs_dist = torch.sqrt(
            (self.obs_x - drone_x_exp)**2 +
            (self.obs_y - drone_y_exp)**2 +
            (self.obs_z - drone_z_exp)**2
        )  # (num_envs, num_obs)
        # Collision if drone is within any obstacle's radius
        obs_collision = (all_obs_dist < self.obs_r).any(dim=1)
                              
        rewards = torch.full((self.num_envs,), -0.1, device=self.device)
        # Shaping reward: small pull toward the target each step
        rewards -= target_dist * 0.005
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # BUG 2 FIX: Apply rewards with strict priority ordering using boolean masks
        # so that a goal reached inside an obstacle still gets the +100 reward.
        # Priority: Goal > Collision > Timeout (highest priority applied last with |=)
        
        # 1. Timeout (lowest priority)
        time_mask = self.steps > 200
        rewards = torch.where(time_mask, torch.full_like(rewards, -30.0), rewards)
        dones |= time_mask
        
        # 2. Collision (medium priority, overwrites timeout)
        rewards = torch.where(obs_collision, torch.full_like(rewards, -100.0), rewards)
        dones |= obs_collision
        
        # 3. Goal reached (highest priority, always overwrites collision/timeout)
        goal_mask = target_dist < 1.5
        rewards = torch.where(goal_mask, torch.full_like(rewards, 200.0), rewards)
        dones |= goal_mask
        
        next_states = self._get_state()
        
        # Auto-reset completed environments
        if dones.any():
            reset_states = self.reset(dones)
            # Update next_states with the reset states so the next iteration starts fresh
            next_states[dones] = reset_states[dones]
            
        return next_states, rewards, dones

def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting MASSIVELY PARALLEL RL Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Tuned to fit beautifully inside a 10GB-15GB VRAM limit
        NUM_ENVS = 4096
        REPLAY_BUFFER_SIZE = 2000000
        BATCH_SIZE = 16384
    else:
        # Fallback parameters for CPU testing
        NUM_ENVS = 64
        REPLAY_BUFFER_SIZE = 50000
        BATCH_SIZE = 256
        
    TOTAL_GLOBAL_STEPS = 5_000_000
    GAMMA = 0.99
    LR = 0.0005
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    # BUG 5 FIX: Recalculate epsilon decay correctly.
    # num_loops = TOTAL_GLOBAL_STEPS / NUM_ENVS = 5_000_000 / 4096 ≈ 1220 loops.
    # We want: EPSILON_START * DECAY^num_loops = EPSILON_MIN
    # => DECAY = (EPSILON_MIN / EPSILON_START)^(1 / num_loops)
    # => DECAY = 0.01^(1/1220) ≈ 0.9962
    # The old value 0.9995 only reached ~0.54 after all loops (agent was 54% random).
    EPSILON_DECAY = 0.9962
    
    TARGET_UPDATE_FREQ = 250
    
    online_model = DeepQNetwork(state_dim=5, action_dim=5).to(device)
    target_model = DeepQNetwork(state_dim=5, action_dim=5).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(online_model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    
    replay_buffer = GPUReplayBuffer(capacity=REPLAY_BUFFER_SIZE, state_dim=5, device=device)
    env = VectorizedGPUEnv(num_envs=NUM_ENVS, device=device)
    
    epsilon = EPSILON_START
    
    print(f"Pre-filling replay buffer with {NUM_ENVS * 10} transitions...")
    states = env._get_state()
    for _ in range(10):
        actions = torch.randint(0, 5, (NUM_ENVS,), device=device)
        next_states, rewards, dones = env.step(actions)
        replay_buffer.push(states, actions, rewards, next_states, dones.float())
        states = next_states
        
    print("Buffer primed. Engaging high-speed training loop...\n")
    
    optim_steps = 0
    total_rewards = 0.0
    episodes_completed = 0
    
    steps_per_loop = NUM_ENVS
    num_loops = TOTAL_GLOBAL_STEPS // steps_per_loop
    
    for loop_idx in range(num_loops):
        # 1. Collect experiences
        if torch.rand(1).item() < epsilon:
            actions = torch.randint(0, 5, (NUM_ENVS,), device=device)
        else:
            with torch.no_grad():
                q_vals = online_model(states)
                actions = torch.argmax(q_vals, dim=1)
                
        next_states, rewards, dones = env.step(actions)
        replay_buffer.push(states, actions, rewards, next_states, dones.float())
        
        total_rewards += rewards.sum().item()
        episodes_completed += dones.sum().item()
        
        states = next_states
        
        # 2. Optimize Model
        for _ in range(2): 
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(BATCH_SIZE)
            
            q_values = online_model(states_b).gather(1, actions_b).squeeze(1)
            
            with torch.no_grad():
                next_q = target_model(next_states_b).max(1)[0]
                target_q = rewards_b.squeeze(1) + GAMMA * next_q * (1 - dones_b.squeeze(1))
                
            loss = loss_fn(q_values, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=10.0)
            optimizer.step()
            optim_steps += 1
            
            if optim_steps % TARGET_UPDATE_FREQ == 0:
                target_model.load_state_dict(online_model.state_dict())
                
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Logging
        if loop_idx % 50 == 0:
            avg_reward = total_rewards / max(1, episodes_completed)
            global_steps = loop_idx * NUM_ENVS
            print(f"Global Steps: {global_steps:9d}/{TOTAL_GLOBAL_STEPS} | "
                  f"Avg Reward/Ep: {avg_reward:7.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Optim Steps: {optim_steps}")
            total_rewards = 0.0
            episodes_completed = 0

    print("\nTraining Complete!")
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '../models/aegis_pilot_v1.pth')
    torch.save(online_model.state_dict(), model_path)
    print(f"Saved trained weights to: {model_path}")

if __name__ == "__main__":
    train_agent()
