import torch.nn as nn

class FlightRLAgent(nn.Module):
    """
    A lightweight Neural Network for Reinforcement Learning Path Planning.
    Takes current state [x, y, alt, obstacle_dist] and outputs the optimal 
    action (e.g., Pitch up, pitch down, bank left, bank right, maintain).
    """
    def __init__(self, state_dim=4, action_dim=5):
        super(FlightRLAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.out(x)
