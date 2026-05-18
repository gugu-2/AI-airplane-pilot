"""
Aegis OS \u2014 RL Training Script
Fixed: Replay Buffer + Target Network (Fix #5)
Fixed: Actions 3/4 CLIMB/DESCEND implemented in 2D env as altitude (Fix #17)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import sys
from collections import deque

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import DeepQNetwork

# =======================================================================
# Fix #5: Experience Replay Buffer
# Without this, training on sequential correlated samples causes Q-values
# to oscillate wildly and the model never converges.
# =======================================================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# =======================================================================
# Fix #17: 2D Kinematic Environment with CLIMB/DESCEND actions
# Previous version left actions 3 and 4 unimplemented (40% dead action space)
# =======================================================================
class Fast2DEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 50.0  # Altitude (meters) \u2014 starts at minimum cruise altitude
        
        # Target is randomly placed in 3D
        self.target_x = random.uniform(-10.0, 10.0)
        self.target_y = random.uniform(-10.0, 10.0)
        self.target_z = random.uniform(50.0, 150.0)
        
        # Obstacle is randomly placed
        self.obs_x = random.uniform(-5.0, 5.0)
        self.obs_y = random.uniform(-5.0, 5.0)
        self.obs_z = random.uniform(50.0, 120.0)
        
        self.steps = 0
        return self._get_state()
        
    def _get_state(self):
        dist_x = self.target_x - self.drone_x
        dist_y = self.target_y - self.drone_y
        
        norm_x = max(-1.0, min(1.0, dist_x / 20.0))
        norm_y = max(-1.0, min(1.0, dist_y / 20.0))
        
        heading = math.atan2(dist_y, dist_x)
        norm_heading = heading / math.pi
        
        obs_dist = math.sqrt((self.obs_x - self.drone_x)**2 + 
                             (self.obs_y - self.drone_y)**2 + 
                             (self.obs_z - self.drone_z)**2)
        norm_obs = max(0.0, min(1.0, 1.0 / (obs_dist + 0.1)))
        
        noise = random.uniform(-0.05, 0.05)
        
        return [norm_x, norm_y, norm_heading, norm_obs, noise]
        
    def step(self, action_idx):
        self.steps += 1
        
        dist_x = self.target_x - self.drone_x
        dist_y = self.target_y - self.drone_y
        angle = math.atan2(dist_y, dist_x)
        
        step_xy = 0.5
        step_z = 2.0  # Meters per altitude step
        
        # Fix #17: All 5 actions are now fully implemented
        if action_idx == 0:    # FWD \u2014 move toward target
            self.drone_x += math.cos(angle) * step_xy
            self.drone_y += math.sin(angle) * step_xy
        elif action_idx == 1:  # LEFT \u2014 veer left
            self.drone_x += math.cos(angle + 0.5) * step_xy
            self.drone_y += math.sin(angle + 0.5) * step_xy
        elif action_idx == 2:  # RIGHT \u2014 veer right
            self.drone_x += math.cos(angle - 0.5) * step_xy
            self.drone_y += math.sin(angle - 0.5) * step_xy
        elif action_idx == 3:  # CLIMB \u2014 gain altitude
            self.drone_z = min(self.drone_z + step_z, 1000.0)
        elif action_idx == 4:  # DESCEND \u2014 lose altitude (env floor = 50m MIN)
            self.drone_z = max(self.drone_z - step_z, 50.0)
            
        # 3D distance reward calculation
        target_dist = math.sqrt((self.target_x - self.drone_x)**2 + 
                                (self.target_y - self.drone_y)**2 +
                                (self.target_z - self.drone_z)**2)
        obs_dist = math.sqrt((self.obs_x - self.drone_x)**2 + 
                             (self.obs_y - self.drone_y)**2 +
                             (self.obs_z - self.drone_z)**2)
        
        reward = -0.1  # Step penalty (incentivizes efficiency)
        done = False
        
        if target_dist < 1.5:
            reward = 100.0   # Goal reached
            done = True
        elif obs_dist < 1.0:
            reward = -100.0  # Collision
            done = True
        elif self.steps > 150:
            reward = -50.0   # Fuel exhausted
            done = True
            
        return self._get_state(), reward, done


def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting High-Speed RL Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- Hyperparameters ---
    EPOCHS = 10000          # More epochs = better convergence
    BATCH_SIZE = 64         # Standard DQN batch size
    GAMMA = 0.99            # Discount factor
    LR = 0.0005             # Adam learning rate
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.997
    REPLAY_BUFFER_SIZE = 50000
    MIN_REPLAY_SIZE = 1000  # Don't start training until buffer has this many samples
    TARGET_UPDATE_FREQ = 100 # Steps between target network sync
    
    # Fix #5: Online model + frozen Target network
    online_model = DeepQNetwork(state_dim=5, action_dim=5).to(device)
    target_model = DeepQNetwork(state_dim=5, action_dim=5).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()  # Target network is never trained directly
    
    optimizer = optim.Adam(online_model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()  # Huber Loss is more stable than MSE for DQN
    
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    env = Fast2DEnv()
    
    epsilon = EPSILON_START
    total_steps = 0
    
    print(f"Pre-filling replay buffer ({MIN_REPLAY_SIZE} random transitions)...")
    while len(replay_buffer) < MIN_REPLAY_SIZE:
        state = env.reset()
        while True:
            action = random.randint(0, 4)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            if done:
                break
    print("Replay buffer ready. Starting training...\n")
    
    for epoch in range(EPOCHS):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(device)
                    action = torch.argmax(online_model(state_t)).item()
                    
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))
            total_reward += reward
            total_steps += 1
            state = next_state
            
            # Fix #5: Sample a RANDOM mini-batch (breaks correlation)
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(BATCH_SIZE)
            states_b = states_b.to(device)
            actions_b = actions_b.to(device)
            rewards_b = rewards_b.to(device)
            next_states_b = next_states_b.to(device)
            dones_b = dones_b.to(device)
            
            # Compute current Q-values
            q_values = online_model(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values using FROZEN target network
            with torch.no_grad():
                next_q = target_model(next_states_b).max(1)[0]
                target_q = rewards_b + GAMMA * next_q * (1 - dones_b)
            
            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Fix #5: Periodically sync target network with online model
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_model.load_state_dict(online_model.state_dict())
            
            if done:
                break
                
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}/{EPOCHS} | Reward: {total_reward:8.1f} | "
                  f"Epsilon: {epsilon:.3f} | Buffer: {len(replay_buffer)}")

    print("\nTraining Complete!")
    
    # Save the trained weights
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '../models/aegis_pilot_v1.pth')
    torch.save(online_model.state_dict(), model_path)
    print(f"Saved trained weights to: {model_path}")

if __name__ == "__main__":
    train_agent()
