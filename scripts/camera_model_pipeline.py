#!/usr/bin/env python3

"""
Final Camera + Model Inference Script
Camera → Preprocessing → Neural Network → Motor Coordinates

Usage: uv run python scripts/final_camera_inference.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
import json
from pathlib import Path

# Model Definition
class MobileNetMiniTransformer(nn.Module):
    def __init__(self, chunk_size=20, state_dim=9, action_dim=9, hidden_dim=256):
        super().__init__()
        self.chunk_size = chunk_size

        # MobileNet vision backbone
        self.vision_backbone = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.vision_backbone.classifier = nn.Identity()

        # Projections
        self.vision_proj = nn.Linear(576, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)

        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, image, state):
        batch_size = image.shape[0]

        # Vision + state encoding
        vision_features = self.vision_backbone(image)
        vision_embed = self.vision_proj(vision_features).unsqueeze(1)
        state_embed = self.state_proj(state).unsqueeze(1)

        # Transformer
        encoder_input = torch.cat([state_embed, vision_embed], dim=1)
        memory = self.encoder(encoder_input)

        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        action_features = self.decoder(queries, memory)
        actions = self.action_head(action_features)

        return actions

def load_model():
    """Load trained model"""
    model = MobileNetMiniTransformer()
    model_path = "deployment/lekiwi_mobilenet_model.pth"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"🧠 Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def preprocess_frame(frame):
    """Preprocess camera frame for model"""
    # Resize and convert
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame_norm = (frame_float - mean) / std

    # To tensor
    frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0).float()
    return frame_tensor

def get_robot_state():
    """Simulate current robot state (9 DOF)"""
    # In real robot: read from encoders
    t = time.time() * 0.1
    state = np.array([
        0.1 * np.sin(t),         # Arm joints 1-6
        0.2 * np.cos(t * 1.1),
        0.15 * np.sin(t * 0.8),
        0.1 * np.cos(t * 1.3),
        0.05 * np.sin(t * 1.5),
        0.2 * np.cos(t * 0.7),
        0.0, 0.0, 0.0            # Base velocities
    ], dtype=np.float32)

    return torch.from_numpy(state).unsqueeze(0)

def main():
    print("🤖 Final Camera + Model Inference")
    print("=" * 40)

    # Load model
    model = load_model()
    if model is None:
        return

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Camera opened successfully")
    print("📹 Running inference... Press 'q' to quit")

    frame_count = 0
    start_time = time.time()

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Preprocess
        image_tensor = preprocess_frame(frame)
        state_tensor = get_robot_state()

        # Model inference
        inference_start = time.time()
        with torch.no_grad():
            actions = model(image_tensor, state_tensor)
        inference_time = time.time() - inference_start

        # Get motor coordinates
        motor_coords = actions[0, 0].cpu().numpy()  # First action step

        # Display results
        fps = frame_count / (time.time() - start_time)

        # Overlay info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Motor coordinates (first 3 DOF)
        motor_text = f"Motor: [{motor_coords[0]:.2f}, {motor_coords[1]:.2f}, {motor_coords[2]:.2f}...]"
        cv2.putText(frame, motor_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

        # Robot state (first 3 DOF)
        state_vals = state_tensor[0].numpy()
        state_text = f"State: [{state_vals[0]:.2f}, {state_vals[1]:.2f}, {state_vals[2]:.2f}...]"
        cv2.putText(frame, state_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Status
        status = "READY" if inference_time < 0.06 else "SLOW"
        color = (0, 255, 0) if status == "READY" else (0, 255, 255)
        cv2.putText(frame, status, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("LeKiwi Camera + Model Inference", frame)

        # Print motor coordinates every 30 frames
        if frame_count % 30 == 0:
            print(f"🎯 Frame {frame_count}: Motor coords = {motor_coords.round(3)}")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"\n📊 Final Stats:")
    print(f"  Frames: {frame_count}")
    print(f"  Runtime: {total_time:.1f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print("✅ Camera + Model pipeline working!")

if __name__ == "__main__":
    main()