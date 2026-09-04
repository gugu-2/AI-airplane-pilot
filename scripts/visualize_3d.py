import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import time

# Add src to path so we can import the environment and model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_rl_checkpointed import Session1Env
from train_rl_session2 import Session2Env
from cognitive.rl_models import VisionPPOActor

def main():
    # Force CPU for local visualization due to local PyTorch architecture mismatch
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Find the latest model - check root models/, then session subdirs, then latest/
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    pth_files = (
        glob.glob(os.path.join(model_dir, '*.pth')) +
        glob.glob(os.path.join(model_dir, 'session_*', '*.pth')) +
        glob.glob(os.path.join(model_dir, 'latest', '*.pth'))
    )
    # Exclude non-checkpoint files (old v1 models trained with different architectures)
    pth_files = [f for f in pth_files if 'aegis_pilot_v1' not in f and 'aegis_vision' not in f and 'flight_rl_model' not in f]

    if not pth_files:
        print("\nCould not find any checkpoint .pth files in the 'models/' folder.")
        print("Please move the 'checkpoint_update_00499.pth' file you downloaded into the 'models/' folder.")
        return

    # Sort by modification time to find the newest downloaded model
    pth_files.sort(key=os.path.getmtime)
    latest_model = pth_files[-1]
    print(f"Loading AI Brain: {os.path.basename(latest_model)}")

    # Load the learned weights to inspect dimensions
    checkpoint = torch.load(latest_model, map_location=device, weights_only=False)
    phys_dim = checkpoint['actor']['physics_net.0.weight'].shape[1]

    # Dynamically select environment based on what the brain was trained on
    if phys_dim == 10:
        print("-> Detected Session 2 Brain (10-dim with Vehicle Profiles)")
        env = Session2Env(num_envs=1, device=device)
    else:
        print("-> Detected Session 1 Brain (9-dim Drone)")
        env = Session1Env(num_envs=1, device=device)
        
    actor = VisionPPOActor(depth_rays=32, physics_dim=phys_dim, action_dim=4).to(device)

    actor.load_state_dict(checkpoint['actor'])
    actor.eval() # Set to evaluation mode

    # Set up interactive Matplotlib 3D
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    reset_mask = torch.ones(1, dtype=torch.bool, device=device)
    depth, phys = env.reset(reset_mask)
    
    path_x, path_y, path_z = [], [], []

    print("\nTaking off... Watch the 3D window!")
    
    try:
        # Run the simulation for 1000 steps
        for step in range(1000):
            # The AI looks at the state and lidar, and decides how to move
            # We use action_mean directly for deterministic "best" flight (no random exploration)
            with torch.no_grad():
                action_mean, _ = actor(depth, phys)
                action = torch.clamp(action_mean, -1.0, 1.0)

            # The physics engine simulates 1 step forward based on the AI's action
            next_depth, next_phys, reward, done, goals = env.step(action)
            depth = next_depth
            phys = next_phys

            # Extract the 3D position of the drone
            pos = env.pos[0].cpu().numpy()
            path_x.append(pos[0])
            path_y.append(pos[1])
            path_z.append(pos[2])

            # Update the screen every 2 steps to keep the animation smooth
            if step % 2 == 0:
                ax.clear()
                
                # Setup 3D bounds (Camera follows the vehicle)
                ax.set_xlim(pos[0] - 40, pos[0] + 40)
                ax.set_ylim(pos[1] - 40, pos[1] + 40)
                ax.set_zlim(max(0, pos[2] - 40), pos[2] + 40)
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.set_zlabel("Altitude (meters)")
                title = "Aegis OS - Flight View (Camera Follow)"
                if phys_dim == 10:
                    vtype = "Airplane" if phys[0, 9].item() > 0.5 else "Drone"
                    title = f"Aegis OS - Session 2 ({vtype})"
                ax.set_title(title)
                
                # Plot the Goal (Green Star)
                goal = env.target_pos[0].cpu().numpy()
                ax.scatter(goal[0], goal[1], goal[2], color='green', marker='*', s=300, label="Goal")
                
                # Plot the Obstacles (Red Spheres)
                obstacles = env.obs_pos[0].cpu().numpy()
                for obs in obstacles:
                    ax.scatter(obs[0], obs[1], obs[2], color='red', marker='o', s=200, alpha=0.3)

                # Plot the drone's flight path (Blue Line)
                ax.plot(path_x, path_y, path_z, color='blue', linewidth=2, label="Flight Path")
                
                # Plot the drone itself (Blue Triangle)
                ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='^', s=150, label="Drone")

                ax.legend()
                plt.pause(0.02) # Briefly pause to render the frame

            if done[0]:
                if reward[0].item() > 100:
                    print(f"Goal Reached Successfully at step {step}!")
                else:
                    print(f"Collided or Out of Bounds at step {step}.")
                
                # Reset for a new episode
                depth, phys = env.reset(reset_mask)
                path_x, path_y, path_z = [], [], []
                plt.pause(1.5) # Pause to let you see what happened before restarting

    except KeyboardInterrupt:
        print("\nStopping visualization.")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
