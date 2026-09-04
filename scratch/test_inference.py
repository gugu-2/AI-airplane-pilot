import torch
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from cognitive.rl_models import VisionPPOActor

ckpt = torch.load("models/session4_final.pth", map_location='cpu', weights_only=False)
phys_dim = ckpt['actor']['physics_net.0.weight'].shape[1]
actor = VisionPPOActor(depth_rays=32, physics_dim=phys_dim, action_dim=4)
actor.load_state_dict(ckpt['actor'])
actor.eval()

depth_arr = np.ones(32, dtype=np.float32)
depth_tensor = torch.tensor(depth_arr, dtype=torch.float32).view(1, 1, 32)

attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Pitch, Roll, Yaw
norm_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
norm_dist = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
physics = np.concatenate([attitude, norm_vel, norm_dist, [0.0]])
phys_tensor = torch.tensor(physics, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    action_mean, _ = actor(depth_tensor, phys_tensor)

action = action_mean.squeeze(0).numpy()
print("Action: ", action)
print("Pitch cmd: ", action[0] * 0.25)
print("Roll cmd: ", -action[1] * 0.35)
print("Yaw rate: ", action[2] * 1.0)
