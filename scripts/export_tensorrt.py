#!/usr/bin/env python3
"""
export_tensorrt.py
This script exports the standard PyTorch YOLOv8 model into an optimized NVIDIA TensorRT engine.
Run this script directly on the Jetson edge computer to compile the model for its specific GPU architecture.
"""
import os

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package is not installed. Run 'pip install ultralytics'")
    exit(1)

def export_model():
    print("==================================================")
    print("Initiating TensorRT Compilation for Edge AI...")
    print("==================================================")
    
    # 1. Load the base PyTorch model
    model_name = 'yolov8n.pt'
    print(f"[1/3] Loading base model: {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    # 2. Export to TensorRT
    print("[2/3] Exporting to TensorRT Engine (this may take 5-15 minutes on edge hardware)...")
    print("      Optimizing for FP16 (Half precision) to maximize FPS and reduce thermal load.")
    
    try:
        # Note: 'format=engine' requires TensorRT to be installed on the host system.
        # half=True forces FP16 precision, which is highly recommended for Jetson architectures.
        # dynamic=False creates a static batch size which is faster for single-camera drone feeds.
        exported_path = model.export(
            format='engine', 
            device='0', # Use GPU 0
            half=True, 
            dynamic=False,
            imgsz=640 # Ensure image size matches perception input
        )
        print(f"[3/3] Export successful! Optimized engine saved to: {exported_path}")
        print("\nSUCCESS: The AI Brain will automatically detect and utilize the .engine file on next boot.")
    except Exception as e:
        print(f"\n[ERROR] Export failed. Please ensure NVIDIA TensorRT is installed via JetPack.")
        print(f"Error details: {e}")
        print("\nNote: Running this on Windows requires Windows-specific TensorRT binaries.")
        print("For simulation, continue using the .pt file.")

if __name__ == "__main__":
    export_model()
