import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys

# Import the RL Agent from our cognitive module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/cognitive')))
from rl_models import FlightRLAgent

class FlightEnvironment:
    """
    A simulated physics environment to train the AI Brain.
    The AI must learn to fly straight and avoid crashing into randomly generated obstacles.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.alt = 500.0 # Cruising altitude
        self.obstacle_dist = random.uniform(50, 200) # Random obstacle ahead
        return np.array([self.x, self.y, self.alt, self.obstacle_dist], dtype=np.float32)

    def step(self, action):
        """
        Actions: 0=Maintain, 1=Climb, 2=Descend, 3=Left, 4=Right
        """
        reward = 0
        done = False
        
        # Simulate Physics
        self.x += 10.0 # Plane constantly moves forward
        self.obstacle_dist -= 10.0 # Obstacle gets closer
        
        if action == 1: self.alt += 10.0
        elif action == 2: self.alt -= 10.0
        elif action == 3: self.y -= 10.0
        elif action == 4: self.y += 10.0
        
        # Calculate Reward
        # 1. Heavily penalize crashing (Altitude < 0 or hitting the obstacle)
        if self.alt < 0.0:
            reward = -1000
            done = True
        elif self.obstacle_dist <= 0 and abs(self.y) < 15 and abs(self.alt - 500) < 15:
            # Hit the obstacle
            reward = -500
            done = True
        else:
            # 2. Reward smooth flight at cruising altitude (500)
            alt_error = abs(500 - self.alt)
            reward += (50 - alt_error) # Positive reward if close to 500, negative if far
            
            # 3. Reward evading an obstacle
            if self.obstacle_dist <= 20 and (abs(self.y) > 15 or abs(self.alt - 500) > 15):
                reward += 100 # Successfully evaded!
                
        # End episode after 50 steps
        if self.x >= 500:
            done = True
            
        next_state = np.array([self.x, self.y, self.alt, self.obstacle_dist], dtype=np.float32)
        return next_state, reward, done

def train_agent(episodes=1000):
    print("==================================================")
    print("INITIATING REINFORCEMENT LEARNING: DEEP Q-NETWORK")
    print("==================================================")
    
    env = FlightEnvironment()
    agent = FlightRLAgent()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    gamma = 0.99 # Discount factor
    epsilon = 1.0 # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    for ep in range(episodes):
        state = env.reset()
        state_tensor = torch.tensor(state)
        total_reward = 0
        
        while True:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = random.randint(0, 4) # Explore
            else:
                with torch.no_grad():
                    q_values = agent(state_tensor)
                    action = torch.argmax(q_values).item() # Exploit
                    
            # Take Action in Environment
            next_state, reward, done = env.step(action)
            next_state_tensor = torch.tensor(next_state)
            
            # Simple Q-Learning Update
            q_values = agent(state_tensor)
            next_q_values = agent(next_state_tensor)
            
            target_q = q_values.clone()
            target_q[action] = reward + (gamma * torch.max(next_q_values) * (1 - int(done)))
            
            # Backpropagation
            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += reward
            state_tensor = next_state_tensor
            
            if done:
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (ep + 1) % 100 == 0:
            print(f"[Training] Epoch {ep+1:4d}/{episodes} | Total Reward: {total_reward:6.1f} | Epsilon: {epsilon:.2f} | Loss: {loss.item():.2f}")
            
    # Save the trained brain
    os.makedirs('models', exist_ok=True)
    torch.save(agent.state_dict(), 'models/flight_rl_model.pth')
    print("\nSUCCESS: Trained neural network saved to 'models/flight_rl_model.pth'")
    print("The ROS 2 Cognitive Planner Node will now use these optimized weights for flight.")

if __name__ == "__main__":
    train_agent(episodes=1000)
