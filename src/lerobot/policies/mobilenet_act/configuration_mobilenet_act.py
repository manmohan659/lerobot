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

"""MobileNet ACT Policy Configuration

Configuration for LeKiwi robot policy using MobileNet vision + MiniTransformer
Optimized for edge deployment on Raspberry Pi 5
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lerobot.policies.pretrained import PreTrainedPolicy


@dataclass
class MobileNetACTConfig:
    """Configuration class for MobileNet ACT Policy

    Designed for LeKiwi robot with:
    - 6 DOF arm joints
    - 3 DOF mobile base (omni wheels)
    - Single camera input
    - Edge deployment optimization
    """

    # Model architecture
    chunk_size: int = 20
    """Number of action steps to predict in sequence"""

    state_dim: int = 9
    """Robot state dimension (6 arm + 3 base)"""

    action_dim: int = 9
    """Action space dimension (6 arm + 3 base)"""

    hidden_dim: int = 256
    """Hidden dimension for transformer layers"""

    # Vision encoder settings
    vision_backbone: str = "mobilenet_v3_small"
    """Vision backbone architecture"""

    pretrained_backbone_weights: str = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    """Pretrained weights for vision backbone"""

    vision_feature_dim: int = 576
    """Output dimension of MobileNet V3 Small"""

    freeze_backbone: bool = False
    """Whether to freeze vision backbone during training"""

    # Transformer settings
    num_encoder_layers: int = 2
    """Number of transformer encoder layers"""

    num_decoder_layers: int = 2
    """Number of transformer decoder layers"""

    num_heads: int = 4
    """Number of attention heads"""

    feedforward_dim: int = 512
    """Feedforward network dimension (hidden_dim * 2)"""

    dropout: float = 0.1
    """Dropout rate for transformer layers"""

    # Input preprocessing
    input_image_size: List[int] = field(default_factory=lambda: [224, 224])
    """Input image size [height, width]"""

    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    """ImageNet normalization mean"""

    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    """ImageNet normalization std"""

    # Training settings
    learning_rate: float = 1e-4
    """Base learning rate"""

    backbone_lr_ratio: float = 0.1
    """Learning rate ratio for backbone vs policy (backbone_lr = lr * ratio)"""

    weight_decay: float = 1e-4
    """Weight decay for optimizer"""

    # Normalization
    normalize_action: bool = True
    """Whether to normalize actions"""

    normalize_state: bool = True
    """Whether to normalize states"""

    # Loss settings
    action_loss_weight: float = 1.0
    """Weight for action prediction loss"""

    # Camera settings (for LeKiwi)
    camera_keys: List[str] = field(default_factory=lambda: ["observation.image"])
    """Keys for camera observations in dataset"""

    # Edge deployment optimization
    use_mixed_precision: bool = True
    """Use mixed precision training/inference"""

    compile_model: bool = False
    """Compile model for faster inference (PyTorch 2.0+)"""

    # Hardware specific
    target_fps: int = 16
    """Target control frequency for real-time deployment"""

    max_inference_time_ms: float = 60.0
    """Maximum allowed inference time in milliseconds"""

    # Dataset specific
    episode_length: Optional[int] = None
    """Maximum episode length (None for variable length)"""

    # LeKiwi specific configurations
    arm_action_indices: List[int] = field(default_factory=lambda: list(range(6)))
    """Indices for arm joint actions"""

    base_action_indices: List[int] = field(default_factory=lambda: [6, 7, 8])
    """Indices for base movement actions"""

    # Safety settings
    action_clip_range: Optional[List[float]] = None
    """Action clipping range [min, max]"""

    max_action_change: Optional[float] = None
    """Maximum allowed action change between steps"""

    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_encoder_layers > 0, "num_encoder_layers must be positive"
        assert self.num_decoder_layers > 0, "num_decoder_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert len(self.input_image_size) == 2, "input_image_size must be [height, width]"
        assert len(self.image_mean) == 3, "image_mean must have 3 values (RGB)"
        assert len(self.image_std) == 3, "image_std must have 3 values (RGB)"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be between 0 and 1"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.backbone_lr_ratio > 0, "backbone_lr_ratio must be positive"
        assert self.target_fps > 0, "target_fps must be positive"
        assert self.max_inference_time_ms > 0, "max_inference_time_ms must be positive"

        # Validate LeKiwi specific settings
        assert len(self.arm_action_indices) == 6, "LeKiwi arm must have 6 DOF"
        assert len(self.base_action_indices) == 3, "LeKiwi base must have 3 DOF"
        assert max(self.arm_action_indices + self.base_action_indices) < self.action_dim, \
            "Action indices must be within action_dim"

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"

        # Set feedforward_dim if not explicitly set
        if self.feedforward_dim == 512 and self.hidden_dim != 256:
            self.feedforward_dim = self.hidden_dim * 2

    @property
    def model_name(self) -> str:
        """Model identifier for logging/saving"""
        return "mobilenet_act"

    @property
    def estimated_parameters(self) -> int:
        """Rough estimate of model parameters"""
        # MobileNetV3-Small: ~2.5M parameters
        # MiniTransformer: ~5M parameters
        # Projections: ~0.5M parameters
        return 8_000_000  # Approximately 8M total parameters

    @property
    def memory_requirements_mb(self) -> float:
        """Estimated memory requirements in MB"""
        # Model parameters: ~32MB (8M * 4 bytes)
        # Activations: ~50MB (batch_size=1, conservative estimate)
        # Overhead: ~18MB
        return 100.0  # Conservative estimate for Pi 5

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MobileNetACTConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str):
        """Save configuration to directory"""
        import json
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_file = save_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> 'MobileNetACTConfig':
        """Load configuration from pretrained model"""
        import json
        from pathlib import Path

        config_path = Path(model_name_or_path) / "config.json"
        if not config_path.exists():
            # Try to load from hub or fallback to default
            return cls()

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)