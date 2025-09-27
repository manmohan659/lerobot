#!/usr/bin/env python3

"""
Simple ACT Test - Based on Official Example
Following examples/2_evaluate_pretrained_policy.py pattern
"""

import torch
import numpy as np
import time
from lerobot.policies.act.modeling_act import ACTPolicy

def test_act_inference():
    """Test ACT inference following official pattern"""

    print("🧪 Simple ACT Test (Official Pattern)")
    print("=" * 50)

    # Load ACT model
    print("📥 Loading ACT model...")
    pretrained_policy_path = "lerobot/act_aloha_sim_transfer_cube_human"
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)

    device = torch.device("cpu")
    policy = policy.to(device)
    policy.eval()

    print(f"✅ Model loaded: {pretrained_policy_path}")
    print(f"🖥️ Device: {device}")

    # Check what the model expects
    print("\n🔍 Model Configuration:")
    print(f"   Input features: {policy.config.input_features}")
    print(f"   Output features: {policy.config.output_features}")

    # Reset the policy (important!)
    policy.reset()

    print("\n🎯 Running inference tests...")

    # Based on the model's input features, create proper observations
    times = []

    for i in range(10):
        # Create observation following official pattern
        # The model expects "observation.images.top" based on the error we saw

        # Create dummy image (following official example format)
        image = torch.randn(3, 480, 640)  # Channel first format
        image = image.to(device).unsqueeze(0)  # Add batch dimension

        # Create dummy state (ALOHA typically has 14 state dimensions)
        state = torch.randn(14)
        state = state.to(device).unsqueeze(0)  # Add batch dimension

        # Create observation dict (using correct key from error message)
        observation = {
            "observation.images.top": image,
            "observation.state": state,
        }

        # Time the inference
        start_time = time.time()

        with torch.inference_mode():
            action = policy.select_action(observation)

        inference_time = time.time() - start_time
        times.append(inference_time * 1000)

        # Get action values
        action_numpy = action.squeeze(0).cpu().numpy()

        print(f"Test {i+1:2d}: {inference_time*1000:6.1f}ms → Action shape: {action.shape}, Sample: [{', '.join(f'{x:.3f}' for x in action_numpy[:6])}]")

    # Stats
    print(f"\n📊 Performance:")
    print(f"   Average: {np.mean(times):.1f}ms")
    print(f"   Min: {np.min(times):.1f}ms")
    print(f"   Max: {np.max(times):.1f}ms")

    if np.mean(times) < 100:
        print("✅ EXCELLENT performance!")
    elif np.mean(times) < 500:
        print("✅ GOOD performance!")
    else:
        print("⚠️ SLOW performance")

    print("\n🎉 ACT inference test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_act_inference()
        if success:
            print("\n✅ Ready to adapt for your robot!")
            print("Next: Create observation dict with your camera and robot state")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()