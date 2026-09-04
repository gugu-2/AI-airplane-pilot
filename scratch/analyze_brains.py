import torch
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from cognitive.rl_models import VisionPPOActor, VisionPPOCritic

def evaluate_brain(pth_path, desc):
    if not os.path.exists(pth_path):
        return None
    
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    phys_dim = ckpt['actor'].get('physics_net.0.weight').shape[1]
    
    actor = VisionPPOActor(depth_rays=32, physics_dim=phys_dim, action_dim=4)
    critic = VisionPPOCritic(depth_rays=32, physics_dim=phys_dim)
    
    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])
    
    actor.eval()
    critic.eval()
    
    # Generate 1000 random states
    torch.manual_seed(42)
    depth_t = torch.ones(1000, 1, 32) * 0.5  # mid-range obstacles
    phys_t = torch.zeros(1000, phys_dim)
    
    # Vary distance and velocity
    phys_t[:, 3:6] = (torch.rand(1000, 3) - 0.5) * 2.0  # Velocity
    phys_t[:, 6:9] = (torch.rand(1000, 3) - 0.5) * 2.0  # Distance
    
    with torch.no_grad():
        values = critic(depth_t, phys_t)
        mean, std = actor(depth_t, phys_t)
        
    value_mean = values.mean().item()
    confidence = (1.0 / (std.mean().item() + 1e-8)) # Lower std = higher confidence
    action_magnitude = torch.norm(mean, dim=1).mean().item()
    
    # Calculate weight norms
    l2_norm = sum(p.norm(2).item()**2 for p in actor.parameters())**0.5
    
    print(f"\n--- {desc} ---")
    print(f"Total Steps Trained: {ckpt.get('total_steps', 0):,}")
    print(f"Average Predicted Value (Q-Score): {value_mean:.2f}")
    print(f"Decision Confidence (1/std): {confidence:.2f}")
    print(f"Action Magnitude (Aggressiveness): {action_magnitude:.2f}")
    print(f"Neural Synapse L2 Norm: {l2_norm:.2f}")

evaluate_brain("models/session3_final.pth", "Session 3 Brain")
evaluate_brain("models/session4_final.pth", "Session 4 Brain")
