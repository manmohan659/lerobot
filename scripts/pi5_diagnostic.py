#!/usr/bin/env python3

"""
Pi 5 Complete Diagnostic Script
Tests all components individually to isolate issues
"""

import time
import sys
import os
import subprocess
import traceback

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def test_system_info():
    print_header("SYSTEM INFORMATION")

    # Basic system info
    stdout, stderr = run_command("uname -a")
    print(f"System: {stdout}")

    stdout, stderr = run_command("free -h")
    print(f"Memory:\n{stdout}")

    stdout, stderr = run_command("df -h /")
    print(f"Disk:\n{stdout}")

    # Python info
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

def test_camera():
    print_header("CAMERA TEST")

    try:
        import cv2
        print("✅ OpenCV imported successfully")

        # Test camera access
        print("📷 Testing camera access...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Camera 0 failed, trying camera 1...")
            cap = cv2.VideoCapture(1)

        if cap.isOpened():
            print("✅ Camera opened successfully")

            # Test frame capture
            print("📸 Testing frame capture...")
            ret, frame = cap.read()

            if ret:
                print(f"✅ Frame captured: {frame.shape}, dtype: {frame.dtype}")
                print(f"   Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

                # Test multiple frames
                frame_count = 0
                start_time = time.time()

                for i in range(10):
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                    time.sleep(0.1)

                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"✅ Captured {frame_count}/10 frames in {elapsed:.1f}s = {fps:.1f} FPS")

            else:
                print("❌ Failed to capture frame")

            cap.release()
        else:
            print("❌ Cannot open any camera")

        # Check camera devices
        stdout, stderr = run_command("ls /dev/video*")
        print(f"Available video devices: {stdout}")

    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        print(traceback.format_exc())

def test_basic_imports():
    print_header("PYTHON IMPORTS TEST")

    imports_to_test = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
    ]

    for module, name in imports_to_test:
        try:
            start_time = time.time()
            imported = __import__(module)
            import_time = time.time() - start_time

            version = getattr(imported, '__version__', 'unknown')
            print(f"✅ {name}: v{version} (imported in {import_time:.2f}s)")

            # Special checks
            if module == 'torch':
                print(f"   PyTorch backend: {imported.get_default_dtype()}")
                print(f"   Available devices: CPU")

        except ImportError as e:
            print(f"❌ {name}: Import failed - {e}")
        except Exception as e:
            print(f"⚠️  {name}: Import succeeded but error - {e}")

def test_lerobot_imports():
    print_header("LEROBOT IMPORTS TEST")

    try:
        print("📦 Testing LeRobot policy import...")
        start_time = time.time()

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import_time = time.time() - start_time

        print(f"✅ SmolVLAPolicy imported in {import_time:.2f}s")

    except ImportError as e:
        print(f"❌ LeRobot import failed: {e}")
        print("   Trying alternative import path...")
        try:
            sys.path.append('/home/manmohan/Workspaces/lerobot/src')
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            print("✅ Alternative import path worked")
        except Exception as e2:
            print(f"❌ Alternative import also failed: {e2}")
    except Exception as e:
        print(f"❌ LeRobot import error: {e}")
        print(traceback.format_exc())

def test_model_loading():
    print_header("MODEL LOADING TEST")

    try:
        # Import required modules
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import torch

        print("🧠 Testing SmolVLA model loading...")
        print("⚠️  This may take 1-2 minutes on first run...")

        # Monitor memory before
        stdout, stderr = run_command("free -h | grep Mem")
        print(f"Memory before: {stdout}")

        start_time = time.time()

        # Load model
        print("📥 Loading SmolVLA policy...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")

        # Monitor memory after
        stdout, stderr = run_command("free -h | grep Mem")
        print(f"Memory after: {stdout}")

        # Test simple inference
        print("🧪 Testing simple inference...")

        # Create dummy inputs
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_state = torch.zeros(1, 6)

        observation = {
            "observation.image": dummy_image,
            "observation.state": dummy_state,
            "task": "test task"
        }

        inference_start = time.time()

        try:
            with torch.no_grad():
                action = policy.select_action(observation)

            inference_time = time.time() - inference_start
            print(f"✅ Inference successful in {inference_time*1000:.1f}ms")
            print(f"   Action shape: {action.shape}")
            print(f"   Action sample: {action.flatten()[:3].numpy().round(3)}")

            # Test multiple inferences
            print("⚡ Testing multiple inferences...")
            times = []

            for i in range(5):
                start = time.time()
                with torch.no_grad():
                    action = policy.select_action(observation)
                times.append(time.time() - start)
                print(f"   Inference {i+1}: {times[-1]*1000:.1f}ms")

            avg_time = sum(times) / len(times) * 1000
            print(f"✅ Average inference: {avg_time:.1f}ms")

            if avg_time < 500:
                print("✅ Performance: GOOD")
            elif avg_time < 1000:
                print("⚠️  Performance: ACCEPTABLE")
            else:
                print("❌ Performance: TOO SLOW")

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            print(traceback.format_exc())

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(traceback.format_exc())

def test_integration():
    print_header("INTEGRATION TEST")

    try:
        import cv2
        import torch
        import numpy as np
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        print("🔗 Testing camera + model integration...")

        # Quick model load (should be cached)
        print("📥 Loading model (cached)...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

        # Fix normalization (simplified)
        if hasattr(policy, 'normalize_inputs'):
            if hasattr(policy.normalize_inputs, 'buffer_observation_state'):
                policy.normalize_inputs.buffer_observation_state.mean.fill_(0.0)
                policy.normalize_inputs.buffer_observation_state.std.fill_(1.0)

        print("📷 Opening camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Camera not available for integration test")
            return

        print("🎯 Running 5 camera + inference cycles...")

        for i in range(5):
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Frame {i+1} capture failed")
                continue

            # Preprocess
            image_resized = cv2.resize(frame, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Dummy robot state
            state_tensor = torch.zeros(1, 6)

            # Create observation
            observation = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
                "task": f"test instruction {i+1}"
            }

            # Inference
            start_time = time.time()
            with torch.no_grad():
                action = policy.select_action(observation)
            inference_time = time.time() - start_time

            print(f"   Cycle {i+1}: {inference_time*1000:.1f}ms → {action.flatten()[:3].numpy().round(3)}")

        cap.release()
        print("✅ Integration test completed")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(traceback.format_exc())

def main():
    print("🚀 Pi 5 Complete Diagnostic Script")
    print("This will test all components systematically...")

    # Run all tests
    test_system_info()
    test_camera()
    test_basic_imports()
    test_lerobot_imports()
    test_model_loading()
    test_integration()

    print_header("DIAGNOSTIC COMPLETE")
    print("📋 Summary:")
    print("   1. Check each section above for ✅ (pass) or ❌ (fail)")
    print("   2. Copy this entire output for analysis")
    print("   3. Main issues will be highlighted in red sections")
    print("\n🎯 Key metrics to check:")
    print("   - Camera FPS should be >5")
    print("   - Model loading should be <120s")
    print("   - Inference should be <1000ms")
    print("   - Memory usage should not exceed available RAM")

if __name__ == "__main__":
    main()