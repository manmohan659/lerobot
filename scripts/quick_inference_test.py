#!/usr/bin/env python3

"""
Quick Inference Test - Isolate the performance issue
"""

import torch
import time
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def test_pure_inference():
    print("🧪 Pure Inference Test - No Camera")

    # Load model
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to('cpu')

    # Fix normalization comprehensively
    print("🔧 Fixing all normalization buffers...")

    fixed_count = 0
    # Fix all normalize_inputs buffers
    if hasattr(policy, 'normalize_inputs'):
        for attr_name in dir(policy.normalize_inputs):
            if attr_name.startswith('buffer_'):
                buffer = getattr(policy.normalize_inputs, attr_name)
                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    buffer.mean.fill_(0.0)
                    buffer.std.fill_(1.0)
                    fixed_count += 1
                    print(f"   ✅ Fixed normalize_inputs.{attr_name}")

    # Fix all normalize_targets buffers
    if hasattr(policy, 'normalize_targets'):
        for attr_name in dir(policy.normalize_targets):
            if attr_name.startswith('buffer_'):
                buffer = getattr(policy.normalize_targets, attr_name)
                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    buffer.mean.fill_(0.0)
                    buffer.std.fill_(1.0)
                    fixed_count += 1
                    print(f"   ✅ Fixed normalize_targets.{attr_name}")

    # Fix all unnormalize_outputs buffers
    if hasattr(policy, 'unnormalize_outputs'):
        for attr_name in dir(policy.unnormalize_outputs):
            if attr_name.startswith('buffer_'):
                buffer = getattr(policy.unnormalize_outputs, attr_name)
                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    buffer.mean.fill_(0.0)
                    buffer.std.fill_(1.0)
                    fixed_count += 1
                    print(f"   ✅ Fixed unnormalize_outputs.{attr_name}")

    print(f"   ✅ Fixed {fixed_count} normalization buffers total")

    print("🚀 Running 50 pure inference tests...")

    times = []

    for i in range(50):
        # Create dummy observation
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_state = torch.zeros(1, 6)

        observation = {
            "observation.image": dummy_image,
            "observation.state": dummy_state,
            "task": f"test task {i}"
        }

        # Time inference
        start = time.time()
        with torch.no_grad():
            action = policy.select_action(observation)
        inference_time = time.time() - start

        times.append(inference_time * 1000)  # Convert to ms

        # Clean up
        del dummy_image, dummy_state, action

        print(f"Frame {i+1:2d}: {inference_time*1000:7.1f}ms", end="")

        # Flag slow ones
        if inference_time * 1000 > 1000:
            print(" ⚠️  SLOW!")
        else:
            print()

    print(f"\n📊 Results:")
    print(f"   Average: {np.mean(times):.1f}ms")
    print(f"   Min: {np.min(times):.1f}ms")
    print(f"   Max: {np.max(times):.1f}ms")
    print(f"   Std: {np.std(times):.1f}ms")

    # Find outliers
    outliers = [t for t in times if t > np.mean(times) + 2*np.std(times)]
    if outliers:
        print(f"   Outliers: {len(outliers)} frames > {np.mean(times) + 2*np.std(times):.1f}ms")
        print(f"   Outlier values: {[f'{t:.1f}' for t in outliers[:5]]}")

if __name__ == "__main__":
    test_pure_inference()