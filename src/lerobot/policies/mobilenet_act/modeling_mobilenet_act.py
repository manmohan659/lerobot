#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MobileNet ACT Policy Implementation

LeKiwi robot policy using MobileNet vision encoder + MiniTransformer
Optimized for edge deployment on Raspberry Pi 5 with real-time control
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision import transforms

from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.policies.mobilenet_act.configuration_mobilenet_act import MobileNetACTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


class MobileNetBackbone(nn.Module):
    """
    MobileNet V3 Small backbone for efficient vision encoding

    Features:
    - ImageNet pretrained weights
    - Lightweight architecture (2.5M parameters)
    - Optimized for mobile deployment
    - Global average pooling output
    """

    def __init__(self, config: MobileNetACTConfig):
        super().__init__()

        # Load pretrained MobileNet V3 Small
        if config.pretrained_backbone_weights:
            weights = getattr(torchvision.models, config.pretrained_backbone_weights)
            self.backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = torchvision.models.mobilenet_v3_small(weights=None)

        # Remove classifier, use as feature extractor
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Output dimension
        self.feature_dim = config.vision_feature_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Extract features from input images

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Visual features [B, feature_dim]
        """
        features = self.backbone(x)  # [B, 576]
        return features


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequences
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:x.size(0), :]


class MobileNetACTPolicy(PreTrainedPolicy):
    """
    MobileNet ACT Policy for LeKiwi Robot

    Architecture:
    - MobileNet V3 Small vision encoder (ImageNet pretrained)
    - MiniTransformer for temporal reasoning and multimodal fusion
    - Action chunking for smooth motion generation
    - Optimized for Raspberry Pi 5 edge deployment

    Input:
    - RGB images: [B, 3, 224, 224]
    - Robot state: [B, 9] (6 arm + 3 base)

    Output:
    - Action sequence: [B, chunk_size, 9]
    """

    config_class = MobileNetACTConfig
    name = "mobilenet_act"

    def __init__(
        self,
        config: MobileNetACTConfig,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        super().__init__(config, dataset_stats)

        self.config = config

        # Vision backbone
        self.vision_backbone = MobileNetBackbone(config)

        # Project vision features to hidden dimension
        self.vision_proj = nn.Linear(
            config.vision_feature_dim,
            config.hidden_dim
        )

        # Project robot state to hidden dimension
        self.state_proj = nn.Linear(
            config.state_dim,
            config.hidden_dim
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            max_len=config.chunk_size + 10  # Extra space for encoder inputs
        )

        # Transformer encoder (processes combined vision + state)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )

        # Learnable action queries for decoder
        self.action_queries = nn.Parameter(
            torch.randn(config.chunk_size, config.hidden_dim) * 0.02
        )

        # Transformer decoder (generates action sequence)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )

        # Action head: project to action space
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.action_dim)
        )

        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(config.input_image_size),
            transforms.Normalize(
                mean=config.image_mean,
                std=config.image_std
            )
        ])

        # Normalization layers
        if dataset_stats is not None:
            self.normalize = Normalize(
                dataset_stats["action"]["mean"],
                dataset_stats["action"]["std"],
            )
            self.unnormalize = Unnormalize(
                dataset_stats["action"]["mean"],
                dataset_stats["action"]["std"],
            )
        else:
            self.normalize = None
            self.unnormalize = None

        # Action buffer for temporal consistency
        self.action_buffer = deque(maxlen=config.chunk_size)

        # Initialize weights
        self._initialize_weights()

        print(f"MobileNet ACT Policy initialized:")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"  Vision parameters: {self.count_vision_parameters():,}")
        print(f"  Policy parameters: {self.count_policy_parameters():,}")
        print(f"  Target inference time: <{config.max_inference_time_ms}ms")

    def _initialize_weights(self):
        """Initialize weights for transformer and projection layers"""
        for module in [self.vision_proj, self.state_proj, self.action_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

        # Initialize action queries
        nn.init.normal_(self.action_queries, std=0.02)

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_vision_parameters(self) -> int:
        """Count vision backbone parameters"""
        return sum(p.numel() for p in self.vision_backbone.parameters() if p.requires_grad)

    def count_policy_parameters(self) -> int:
        """Count policy network parameters (excluding vision)"""
        return self.count_parameters() - self.count_vision_parameters()

    def preprocess_images(self, images: Tensor) -> Tensor:
        """
        Preprocess input images

        Args:
            images: Raw images [B, 3, H, W] in range [0, 1]

        Returns:
            processed: Preprocessed images [B, 3, 224, 224]
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Apply normalization and resizing
        processed = torch.stack([
            self.image_transform(img) for img in images
        ])

        return processed

    def encode_observations(self, images: Tensor, states: Tensor) -> Tensor:
        """
        Encode observations (images + states) into contextual features

        Args:
            images: Input images [B, 3, 224, 224]
            states: Robot states [B, state_dim]

        Returns:
            memory: Encoded observations [B, 2, hidden_dim]
        """
        batch_size = images.shape[0]

        # Vision encoding
        visual_features = self.vision_backbone(images)  # [B, 576]
        visual_embed = self.vision_proj(visual_features)  # [B, hidden_dim]
        visual_embed = visual_embed.unsqueeze(1)  # [B, 1, hidden_dim]

        # State encoding
        state_embed = self.state_proj(states)  # [B, hidden_dim]
        state_embed = state_embed.unsqueeze(1)  # [B, 1, hidden_dim]

        # Combine vision + state
        encoder_input = torch.cat([state_embed, visual_embed], dim=1)  # [B, 2, hidden_dim]

        # Add positional encoding
        encoder_input = encoder_input.transpose(0, 1)  # [2, B, hidden_dim]
        encoder_input = self.pos_encoding(encoder_input)
        encoder_input = encoder_input.transpose(0, 1)  # [B, 2, hidden_dim]

        # Transformer encoder
        memory = self.encoder(encoder_input)  # [B, 2, hidden_dim]

        return memory

    def decode_actions(self, memory: Tensor, batch_size: int) -> Tensor:
        """
        Decode action sequence from encoded observations

        Args:
            memory: Encoded observations [B, 2, hidden_dim]
            batch_size: Batch size

        Returns:
            actions: Predicted actions [B, chunk_size, action_dim]
        """
        # Prepare action queries
        queries = self.action_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, chunk_size, hidden_dim]

        # Add positional encoding to queries
        queries = queries.transpose(0, 1)  # [chunk_size, B, hidden_dim]
        queries = self.pos_encoding(queries)
        queries = queries.transpose(0, 1)  # [B, chunk_size, hidden_dim]

        # Transformer decoder
        action_features = self.decoder(queries, memory)  # [B, chunk_size, hidden_dim]

        # Project to action space
        actions = self.action_head(action_features)  # [B, chunk_size, action_dim]

        return actions

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for training

        Args:
            batch: Training batch containing:
                - images: [B, 3, H, W]
                - states: [B, state_dim]
                - actions: [B, action_dim] (target)

        Returns:
            output: Dictionary containing:
                - action: Predicted actions [B, chunk_size, action_dim]
                - loss: Training loss
        """
        # Extract inputs
        images = batch[self.config.camera_keys[0]]  # [B, 3, H, W]
        states = batch["observation.state"]  # [B, state_dim]
        target_actions = batch["action"]  # [B, action_dim]

        # Preprocess images
        images = self.preprocess_images(images)

        # Forward pass
        memory = self.encode_observations(images, states)
        predicted_actions = self.decode_actions(memory, images.shape[0])

        # Compute loss (using first predicted action vs target)
        pred_action = predicted_actions[:, 0, :]  # [B, action_dim]

        # Apply normalization if available
        if self.normalize is not None:
            pred_action = self.normalize(pred_action)
            target_actions = self.normalize(target_actions)

        # MSE loss for continuous action prediction
        loss = F.mse_loss(pred_action, target_actions)

        # Apply unnormalization for output
        if self.unnormalize is not None:
            predicted_actions = self.unnormalize(predicted_actions)

        return {
            "action": predicted_actions,
            "loss": loss
        }

    @torch.no_grad()
    def select_action(self, observation: Dict[str, Tensor]) -> Tensor:
        """
        Select action for deployment/inference

        Args:
            observation: Single observation containing:
                - image: [3, H, W] or [1, 3, H, W]
                - state: [state_dim] or [1, state_dim]

        Returns:
            action: Next action to execute [action_dim]
        """
        self.eval()

        # Extract and prepare inputs
        image = observation[self.config.camera_keys[0]]
        state = observation["observation.state"]

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Preprocess
        image = self.preprocess_images(image)

        # Forward pass
        memory = self.encode_observations(image, state)
        predicted_actions = self.decode_actions(memory, 1)

        # Get first action from sequence
        action = predicted_actions[0, 0]  # [action_dim]

        # Apply unnormalization
        if self.unnormalize is not None:
            action = self.unnormalize(action.unsqueeze(0)).squeeze(0)

        # Update action buffer for temporal consistency
        self.action_buffer.append(action.cpu())

        # Apply action smoothing if buffer is full
        if len(self.action_buffer) > 1:
            # Simple exponential moving average
            alpha = 0.7
            smoothed_action = alpha * action + (1 - alpha) * self.action_buffer[-2].to(action.device)
            action = smoothed_action

        # Apply safety constraints
        if self.config.action_clip_range is not None:
            action = torch.clamp(
                action,
                self.config.action_clip_range[0],
                self.config.action_clip_range[1]
            )

        return action

    def reset(self):
        """Reset policy state (clear action buffer)"""
        self.action_buffer.clear()

    def get_arm_action(self, action: Tensor) -> Tensor:
        """Extract arm actions from full action vector"""
        return action[self.config.arm_action_indices]

    def get_base_action(self, action: Tensor) -> Tensor:
        """Extract base actions from full action vector"""
        return action[self.config.base_action_indices]

    def set_inference_mode(self, use_mixed_precision: bool = True):
        """Optimize model for inference"""
        self.eval()

        if use_mixed_precision and torch.cuda.is_available():
            # Enable mixed precision inference
            for module in self.modules():
                if hasattr(module, 'half'):
                    module.half()

        if self.config.compile_model and hasattr(torch, 'compile'):
            # Compile model for faster inference (PyTorch 2.0+)
            self.forward = torch.compile(self.forward)
            self.select_action = torch.compile(self.select_action)

    def save_pretrained(self, save_directory: str):
        """Save model and configuration"""
        import os
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

        # Save configuration
        self.config.save_pretrained(save_directory)

        # Save additional info
        info = {
            "model_type": self.name,
            "total_parameters": self.count_parameters(),
            "vision_parameters": self.count_vision_parameters(),
            "policy_parameters": self.count_policy_parameters()
        }

        import json
        with open(save_path / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load pretrained model"""
        from pathlib import Path

        model_path = Path(model_name_or_path)

        # Load configuration
        config = MobileNetACTConfig.from_pretrained(model_name_or_path)

        # Create model
        model = cls(config, **kwargs)

        # Load weights if available
        weights_path = model_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)

        return model