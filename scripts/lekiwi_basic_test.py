#!/usr/bin/env python3

"""
LeKiwi Basic Test - Minimal working example
Just test if we can load model and do basic inference without robot
"""

import torch
import numpy as np
import time
from lerobot.policies.act.modeling_act import ACTPolicy

def test_basic_inference():
    """Test basic model loading and inference"""

    print("🧪 LeKiwi Basic Inference Test")
    print("=" * 50)

    # Load a pretrained ACT model (try different available models)
    print("📥 Loading ACT policy...")

    models_to_try = [
        "lerobot/act_aloha_sim_transfer_cube_human",  # From the docs
        "lerobot/act_aloha_sim_insertion_human",      # From test files
    ]

    policy = None
    for model_id in models_to_try:
        try:
            print(f"   Trying {model_id}...")
            policy = ACTPolicy.from_pretrained(model_id)
            print(f"✅ Model loaded successfully: {model_id}")
            break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    if policy is None:
        print("❌ No working model found!")
        return False

    device = torch.device("cpu")
    policy = policy.to(device)
    policy.eval()

    print(f"🖥️ Device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Check what features the model expects
    print(f"🔍 Model expects these image features: {policy.config.image_features}")
    print(f"🔍 Model expects these state features: {policy.config.state_features}")

    # Create dummy observation matching model expectations
    print("\n🎯 Testing inference with dummy data...")

    # Create observation based on what the model actually expects
    dummy_observation = {}

    # Add expected image features
    for img_key in policy.config.image_features:
        dummy_observation[img_key] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"   Added image: {img_key}")

    # Add expected state features
    for state_key in policy.config.state_features:
        # Use appropriate state size - check the model's expected dimensions
        state_dim = 14  # Default ALOHA dual arm
        dummy_observation[state_key] = np.random.uniform(-1, 1, state_dim).astype(np.float32)
        print(f"   Added state: {state_key} (dim: {state_dim})")

    # Test multiple inferences for timing
    times = []

    for i in range(10):
        # Convert observation to PyTorch format (like control_utils.predict_action)
        obs_torch = {}

        for key, value in dummy_observation.items():
            obs_torch[key] = torch.from_numpy(value)

            if "image" in key:
                # Convert to float32 in [0,1], channel first, add batch dim
                obs_torch[key] = obs_torch[key].float() / 255.0
                obs_torch[key] = obs_torch[key].permute(2, 0, 1)  # HWC -> CHW

            obs_torch[key] = obs_torch[key].unsqueeze(0)  # Add batch dimension
            obs_torch[key] = obs_torch[key].to(device)

        # Add task
        obs_torch["task"] = "Pick up the red cup"

        # Time inference
        start_time = time.time()

        with torch.no_grad():
            action = policy.select_action(obs_torch)

        inference_time = time.time() - start_time
        times.append(inference_time * 1000)  # Convert to ms

        # Get action values
        action_values = action.squeeze(0).cpu().numpy()

        print(f"Inference {i+1:2d}: {inference_time*1000:6.1f}ms → Action: [{', '.join(f'{x:6.3f}' for x in action_values[:6])}]")

    # Stats
    print(f"\n📊 Inference Statistics:")
    print(f"   Average: {np.mean(times):.1f}ms")
    print(f"   Min: {np.min(times):.1f}ms")
    print(f"   Max: {np.max(times):.1f}ms")
    print(f"   Std: {np.std(times):.1f}ms")

    if np.mean(times) < 100:
        print("✅ Performance: EXCELLENT")
    elif np.mean(times) < 500:
        print("✅ Performance: GOOD")
    else:
        print("⚠️ Performance: SLOW")

    print("\n🎉 Basic inference test completed!")
    return True

def main():
    success = test_basic_inference()

    if success:
        print("\n✅ Ready for robot connection!")
        print("Next steps:")
        print("1. Update robot IP in lekiwi_simple_inference.py")
        print("2. Ensure LeKiwi robot is running and connected")
        print("3. Run: uv run python scripts/lekiwi_simple_inference.py")
    else:
        print("\n❌ Basic test failed - fix model loading first")

if __name__ == "__main__":
    main()