#!/usr/bin/env python3

"""
Simple test script for our trained MobileNet + MiniTransformer model
Tests the forward pass: image + state → action coordinates
"""

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2

# Add notebook directory to path to import model
sys.path.append('../notebooks')

class MobileNetMiniTransformer(nn.Module):
    """
    LeKiwi Policy: MobileNet vision + MiniTransformer temporal reasoning

    Input: Image [B, 3, 224, 224] + State [B, 9]
    Output: Actions [B, chunk_size, 9]
    """

    def __init__(self,
                 chunk_size=20,
                 state_dim=9,
                 action_dim=9,
                 hidden_dim=256,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 num_heads=4):
        super().__init__()

        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Vision backbone: MobileNetV3-Small (ImageNet pretrained)
        self.vision_backbone = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Remove classifier, use as feature extractor
        self.vision_backbone.classifier = nn.Identity()
        vision_features = 576  # MobileNetV3-Small output

        # Project vision features to hidden dim
        self.vision_proj = nn.Linear(vision_features, hidden_dim)

        # Project robot state to hidden dim
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # MiniTransformer encoder (combines vision + state)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Learnable action queries for decoder
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))

        # MiniTransformer decoder (generates action sequence)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Action head: project to motor coordinates
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, image, state):
        """
        Forward pass: image + state → action sequence

        Args:
            image: [B, 3, 224, 224] Camera input
            state: [B, 9] Current robot state (6 arm + 3 base)

        Returns:
            actions: [B, chunk_size, 9] Predicted action sequence
        """
        batch_size = image.shape[0]

        # Vision encoding
        vision_features = self.vision_backbone(image)  # [B, 576]
        vision_embed = self.vision_proj(vision_features)  # [B, 256]
        vision_embed = vision_embed.unsqueeze(1)  # [B, 1, 256]

        # State encoding
        state_embed = self.state_proj(state)  # [B, 256]
        state_embed = state_embed.unsqueeze(1)  # [B, 1, 256]

        # Combine vision + state for encoder
        encoder_input = torch.cat([state_embed, vision_embed], dim=1)  # [B, 2, 256]

        # Encoder: contextual understanding
        memory = self.encoder(encoder_input)  # [B, 2, 256]

        # Decoder: generate action sequence
        # Expand action queries for batch
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, chunk_size, 256]

        # Decode actions
        action_features = self.decoder(queries, memory)  # [B, chunk_size, 256]

        # Project to action space
        actions = self.action_head(action_features)  # [B, chunk_size, 9]

        return actions


def load_model(model_path, config_path=None):
    """Load the trained model"""

    # Load config if available
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"📋 Config loaded: {config.get('total_parameters', 'unknown')} parameters")

    # Create model
    model = MobileNetMiniTransformer()

    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(f"🧠 Model loaded: {model.count_parameters():,} parameters")
    return model


def create_test_image():
    """Create a test image similar to what camera would see"""
    # Create a colorful test pattern
    image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    # Add some recognizable features
    cv2.rectangle(image, (100, 100), (200, 200), (255, 100, 100), -1)  # Red square
    cv2.circle(image, (400, 300), 50, (100, 255, 100), -1)  # Green circle
    cv2.putText(image, 'LeKiwi Test', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image


def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize to 224x224
    image_resized = cv2.resize(image, (224, 224))

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_float - mean) / std

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, 224, 224]

    return image_tensor


def test_model_forward():
    """Test the complete forward pass"""
    print("🧪 Testing MobileNet + MiniTransformer Model Forward Pass")
    print("=" * 60)

    # Paths
    model_path = "deployment/lekiwi_mobilenet_model.pth"
    config_path = "deployment/model_config.json"

    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return

    # Load model
    model = load_model(model_path, config_path)

    # Create test inputs
    print("\n📸 Creating test inputs...")

    # Test image
    test_image_raw = create_test_image()
    test_image = preprocess_image(test_image_raw)
    print(f"  Image shape: {test_image.shape}")

    # Test robot state (9 DOF: 6 arm + 3 base)
    test_state = torch.randn(1, 9)  # Random state
    print(f"  State shape: {test_state.shape}")
    print(f"  State values: {test_state[0].numpy().round(3)}")

    # Forward pass
    print("\n🚀 Running forward pass...")
    start_time = time.time()

    with torch.no_grad():
        predicted_actions = model(test_image, test_state)

    inference_time = time.time() - start_time

    # Results
    print(f"\n✅ Forward pass completed!")
    print(f"  Inference time: {inference_time*1000:.1f}ms")
    print(f"  Output shape: {predicted_actions.shape}")
    print(f"  Expected shape: [1, 20, 9] (batch, time_steps, dof)")

    # Show predicted actions
    actions = predicted_actions[0].numpy()  # Remove batch dimension

    print(f"\n🎯 Predicted Actions (Motor Coordinates):")
    print(f"  Action sequence length: {actions.shape[0]} steps")
    print(f"  DOF per action: {actions.shape[1]}")

    # Show first few action steps
    print(f"\n📊 First 5 Action Steps:")
    for i in range(min(5, actions.shape[0])):
        arm_actions = actions[i, :6].round(3)
        base_actions = actions[i, 6:].round(3)
        print(f"  Step {i+1:2d}: Arm={arm_actions} Base={base_actions}")

    # Show action statistics
    print(f"\n📈 Action Statistics:")
    print(f"  Arm actions (DOF 1-6):")
    for i in range(6):
        values = actions[:, i]
        print(f"    DOF {i+1}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")

    print(f"  Base actions (DOF 7-9):")
    for i in range(3):
        values = actions[:, 6+i]
        print(f"    Base {i+1}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")

    # Test multiple inferences
    print(f"\n⚡ Performance Test (100 inferences)...")
    times = []

    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(test_image, test_state)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    max_freq = 1000 / avg_time

    print(f"  Average inference: {avg_time:.1f}±{std_time:.1f}ms")
    print(f"  Max frequency: {max_freq:.1f}Hz")
    print(f"  Target: 16Hz (62.5ms budget)")

    if avg_time < 62.5:
        print(f"  ✅ Performance target met!")
    else:
        print(f"  ⚠️  Too slow for 16Hz target")

    return model, predicted_actions


if __name__ == "__main__":
    test_model_forward()