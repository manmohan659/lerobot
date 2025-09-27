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
        import platform
        from pathlib import Path
        print("✅ OpenCV imported successfully")

        # Use the same robust camera detection as main branch
        MAX_OPENCV_INDEX = 60

        print("📷 Testing camera access with extended search...")

        camera_attempts = []

        if platform.system() == "Linux":
            # Scan /dev/video* devices first
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            for path in possible_paths:
                camera_attempts.append((str(path), f"Video device {path.name}"))

        # Then try indices 0 to MAX_OPENCV_INDEX
        for i in range(min(MAX_OPENCV_INDEX, 20)):  # Limit to 20 for diagnostic
            camera_attempts.append((i, f"Camera index {i}"))

        print(f"   Scanning {len(camera_attempts)} possible camera locations...")

        working_camera = None
        for camera_id, desc in camera_attempts:
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    # Test frame capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"   ✅ {desc} working! Frame: {frame.shape}")
                        working_camera = cap
                        break
                    else:
                        cap.release()
                else:
                    if cap:
                        cap.release()
            except Exception:
                continue

        if working_camera:
            print("✅ Camera opened successfully")

            # Test frame capture
            print("📸 Testing frame capture...")
            ret, frame = working_camera.read()

            if ret:
                print(f"✅ Frame captured: {frame.shape}, dtype: {frame.dtype}")
                print(f"   Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

                # Test multiple frames
                frame_count = 0
                start_time = time.time()

                for i in range(10):
                    ret, frame = working_camera.read()
                    if ret:
                        frame_count += 1
                    time.sleep(0.1)

                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"✅ Captured {frame_count}/10 frames in {elapsed:.1f}s = {fps:.1f} FPS")

            else:
                print("❌ Failed to capture frame")

            working_camera.release()
        else:
            print("❌ Cannot open any camera after extended search")

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

        print("📷 Opening camera with robust detection...")

        # Use same detection logic
        import platform
        from pathlib import Path

        camera_attempts = []

        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            for path in possible_paths[:3]:  # Just first 3 for integration test
                camera_attempts.append((str(path), f"Video device {path.name}"))

        for i in range(3):  # Just first 3 indices for integration test
            camera_attempts.append((i, f"Camera index {i}"))

        cap = None
        for camera_id, desc in camera_attempts:
            try:
                test_cap = cv2.VideoCapture(camera_id)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        cap = test_cap
                        print(f"   Using {desc}")
                        break
                    else:
                        test_cap.release()
                else:
                    if test_cap:
                        test_cap.release()
            except Exception:
                continue

        if not cap:
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