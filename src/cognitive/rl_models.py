import torch
import torch.nn as nn
import math
import random
import os

class DeepQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for Autonomous Path Planning.
    Takes a 5-dimensional state vector and outputs Q-values for 5 discrete actions.
    """
    def __init__(self, state_dim=5, action_dim=5):
        super(DeepQNetwork, self).__init__()
        
        # 3 Hidden layers for complex non-linear spatial reasoning
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class RLInferenceEngine:
    """
    Wrapper for the PyTorch Neural Network.
    Translates physical drone GPS and Semantic Map data into tensors, runs inference,
    and translates the raw tensor output back into physical GPS maneuvers.
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Cognitive-RL] Initializing PyTorch DQN on {self.device}...")
        
        self.model = DeepQNetwork(state_dim=5, action_dim=5).to(self.device)
        
        # Look for the default trained weights if no path is provided
        if model_path is None:
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/aegis_pilot_v1.pth'))
            if os.path.exists(default_path):
                model_path = default_path
                
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                print(f"[Cognitive-RL] SUCCESS: Loaded pre-trained weights from {model_path}")
                self.untrained_mode = False
            except Exception as e:
                print(f"[Cognitive-RL] WARNING: Could not load weights: {e}")
                self.untrained_mode = True
        else:
            print("[Cognitive-RL] WARNING: No pre-trained weights found. Using rule-based heuristic fallback.")
            self.untrained_mode = True
                
        self.model.eval()
        
        # Action space mapping
        self.actions = {
            0: "MAINTAIN_HEADING",
            1: "VEER_LEFT",
            2: "VEER_RIGHT",
            3: "CLIMB",
            4: "DESCEND"
        }

    def _normalize_state(self, current_lat, current_lon, dest_lat, dest_lon, obstacle_dist):
        """Converts raw physics data into normalized tensor inputs (-1.0 to 1.0).
        B3 FIX: Removed random noise from state vector.
        The same physical state must always produce the same Q-values (determinism).
        """
        dist_x = dest_lon - current_lon
        dist_y = dest_lat - current_lat
        norm_x = max(-1.0, min(1.0, dist_x / 0.005))
        norm_y = max(-1.0, min(1.0, dist_y / 0.005))
        heading = math.atan2(dist_y, dist_x)
        norm_heading = heading / math.pi
        # Obstacle proximity (0 = far, 1 = extremely close)
        norm_obs = max(0.0, min(1.0, 1.0 / (obstacle_dist + 0.1)))
        state_vector = [norm_x, norm_y, norm_heading, norm_obs, 0.0]  # 5th dim = reserved
        return torch.FloatTensor(state_vector).to(self.device)

    def decide_next_action(self, current_lat, current_lon, dest_lat, dest_lon, obstacle_dist=100.0):
        """
        Runs the state through the Neural Network and returns a physical waypoint delta.
        B3 FIX: When no trained model is loaded, falls back to a deterministic rule-based
                heuristic (always face the target, climb if obstacle close) instead of
                random noise-driven inference.
        """
        # Rule-based fallback for untrained model
        if self.untrained_mode:
            dist_x = dest_lon - current_lon
            dist_y = dest_lat - current_lat
            angle = math.atan2(dist_y, dist_x)
            base_step = 0.0001
            if obstacle_dist < 20.0:
                return "CLIMB", math.sin(angle) * base_step * 0.5, math.cos(angle) * base_step * 0.5, 5.0
            return "MAINTAIN_HEADING", math.sin(angle) * base_step, math.cos(angle) * base_step, 0.0

        state_tensor = self._normalize_state(current_lat, current_lon, dest_lat, dest_lon, obstacle_dist)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            best_action_idx = torch.argmax(q_values).item()
            
        action_name = self.actions[best_action_idx]
        
        # In a fully trained environment, we follow exactly. Here we add heuristics
        # to ensure the untrained drone still generally approaches the target.
        base_step = 0.0001
        delta_lat = 0.0
        delta_lon = 0.0
        delta_alt = 0.0
        
        # Point towards target primarily
        dist_x = dest_lon - current_lon
        dist_y = dest_lat - current_lat
        angle = math.atan2(dist_y, dist_x)
        
        if action_name == "MAINTAIN_HEADING":
            delta_lat = math.sin(angle) * base_step
            delta_lon = math.cos(angle) * base_step
        elif action_name == "VEER_LEFT":
            delta_lat = math.sin(angle + 0.5) * base_step
            delta_lon = math.cos(angle + 0.5) * base_step
        elif action_name == "VEER_RIGHT":
            delta_lat = math.sin(angle - 0.5) * base_step
            delta_lon = math.cos(angle - 0.5) * base_step
        elif action_name == "CLIMB":
            delta_lat = math.sin(angle) * base_step * 0.5
            delta_lon = math.cos(angle) * base_step * 0.5
            delta_alt = 5.0
        elif action_name == "DESCEND":
            delta_lat = math.sin(angle) * base_step * 0.5
            delta_lon = math.cos(angle) * base_step * 0.5
            delta_alt = -2.0

        return action_name, delta_lat, delta_lon, delta_alt
