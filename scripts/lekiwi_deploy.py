#!/usr/bin/env python3

"""
LeKiwi Real-time Deployment Script

Raspberry Pi 5 inference pipeline:
Camera → MobileNet → MiniTransformer → Motor Commands

Target: 16Hz control frequency (~60ms total latency)
Memory: <1GB RAM usage
Hardware: Raspberry Pi 5 + SO-101 arms + mobile base

Usage:
    python lekiwi_deploy.py --model_path ../models/lekiwi_mobilenet_model.pth
"""

import argparse
import json
import time
import threading
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# LeRobot imports
import sys
sys.path.append('../src')

try:
    from lerobot.robots.lekiwi.lekiwi import LeKiwi
    from lerobot.policies.mobilenet_act.modeling_mobilenet_act import MobileNetACTPolicy
    from lerobot.policies.mobilenet_act.configuration_mobilenet_act import MobileNetACTConfig
    LEROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LeRobot imports failed: {e}")
    print("Running in simulation mode")
    LEROBOT_AVAILABLE = False


class FakeLeKiwi:
    """Fake LeKiwi robot for testing without hardware"""

    def __init__(self):
        self.current_state = np.zeros(9)  # 6 arm + 3 base
        print("🤖 Fake LeKiwi robot initialized")

    def get_observation(self):
        """Get fake observation"""
        return {
            "observation.state": torch.tensor(self.current_state, dtype=torch.float32)
        }

    def set_action(self, action):
        """Execute fake action"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Simulate robot movement
        self.current_state = action
        print(f"🎮 Executing action: {action[:3].round(3)} (showing first 3 DOF)")

    def disconnect(self):
        """Fake disconnect"""
        print("🔌 Fake robot disconnected")


class FakeCamera:
    """Fake camera for testing without hardware"""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        print(f"📷 Fake camera initialized ({width}x{height})")

    def read(self):
        """Generate fake camera frame"""
        # Create colorful test pattern
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        # Add some structure
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
        cv2.circle(frame, (400, 300), 50, (0, 255, 0), -1)  # Green circle

        return True, frame

    def release(self):
        """Fake release"""
        print("📷 Fake camera released")


class PerformanceMonitor:
    """Monitor inference performance and system metrics"""

    def __init__(self, window_size=50):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_print = 0

    def update(self, inference_time: float, total_time: float):
        """Update performance metrics"""
        self.inference_times.append(inference_time * 1000)  # Convert to ms
        self.total_times.append(total_time * 1000)

        if total_time > 0:
            fps = 1.0 / total_time
            self.fps_history.append(fps)

        # Print stats every 5 seconds
        if time.time() - self.last_print > 5.0:
            self.print_stats()
            self.last_print = time.time()

    def print_stats(self):
        """Print performance statistics"""
        if not self.inference_times:
            return

        inf_mean = np.mean(self.inference_times)
        inf_std = np.std(self.inference_times)
        total_mean = np.mean(self.total_times)
        fps_mean = np.mean(self.fps_history) if self.fps_history else 0

        runtime = time.time() - self.start_time

        print(f"\n📊 Performance Stats (runtime: {runtime:.1f}s)")
        print(f"  Inference: {inf_mean:.1f}±{inf_std:.1f}ms")
        print(f"  Total cycle: {total_mean:.1f}ms")
        print(f"  Control freq: {fps_mean:.1f}Hz")
        print(f"  Target: 16Hz (62.5ms)")

        if inf_mean > 60:
            print("  ⚠️  High inference time!")
        if fps_mean < 10:
            print("  ⚠️  Low control frequency!")


class LeKiwiController:
    """Real-time LeKiwi robot controller"""

    def __init__(self,
                 model_path: str,
                 camera_id: int = 0,
                 target_fps: int = 16,
                 use_fake_hardware: bool = False):

        self.target_fps = target_fps
        self.target_dt = 1.0 / target_fps
        self.use_fake_hardware = use_fake_hardware or not LEROBOT_AVAILABLE

        print(f"🚀 Initializing LeKiwi Controller")
        print(f"  Target frequency: {target_fps}Hz")
        print(f"  Model path: {model_path}")
        print(f"  Camera ID: {camera_id}")
        print(f"  Fake hardware: {self.use_fake_hardware}")

        # Load model
        self.model = self._load_model(model_path)
        self.device = torch.device('cpu')  # Use CPU for Pi 5
        self.model.to(self.device)
        self.model.eval()

        # Initialize hardware
        self.camera = self._init_camera(camera_id)
        self.robot = self._init_robot()

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()

        # Threading for concurrent processing
        self.frame_queue = Queue(maxsize=2)
        self.action_queue = Queue(maxsize=2)
        self.running = False

        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("✅ LeKiwi Controller initialized")

    def _load_model(self, model_path: str) -> MobileNetACTPolicy:
        """Load trained MobileNet model"""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load config
        config_path = model_path.parent / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            print(f"📋 Loaded model config: {config_dict.get('total_parameters', 'unknown')} parameters")

        # Create model with default config
        config = MobileNetACTConfig()
        model = MobileNetACTPolicy(config)

        # Load weights
        if model_path.suffix == '.pth':
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            # Try loading full model
            model = torch.load(model_path, map_location='cpu')

        print(f"🧠 Model loaded: {model.count_parameters():,} parameters")
        return model

    def _init_camera(self, camera_id: int):
        """Initialize camera"""
        if self.use_fake_hardware:
            return FakeCamera()

        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            print(f"⚠️  Could not open camera {camera_id}, using fake camera")
            return FakeCamera()

        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        print(f"📷 Camera initialized: {camera_id}")
        return camera

    def _init_robot(self):
        """Initialize LeKiwi robot"""
        if self.use_fake_hardware:
            return FakeLeKiwi()

        try:
            robot = LeKiwi()
            robot.connect()
            print("🤖 LeKiwi robot connected")
            return robot
        except Exception as e:
            print(f"⚠️  Could not connect to robot: {e}")
            print("Using fake robot")
            return FakeLeKiwi()

    def preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess camera frame for model input"""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image_tensor = self.image_transform(frame)
        return image_tensor.unsqueeze(0)  # Add batch dimension

    def camera_thread(self):
        """Camera capture thread"""
        print("📷 Camera thread started")

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("⚠️  Failed to read camera frame")
                continue

            # Add to queue (drop old frames if queue is full)
            try:
                self.frame_queue.put_nowait(frame)
            except:
                # Queue full, drop old frame
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass

        print("📷 Camera thread stopped")

    def inference_thread(self):
        """Model inference thread"""
        print("🧠 Inference thread started")

        while self.running:
            try:
                # Get latest frame
                frame = self.frame_queue.get(timeout=0.1)

                # Preprocess
                start_time = time.time()
                image = self.preprocess_image(frame)

                # Get robot state
                obs = self.robot.get_observation()
                state = obs["observation.state"].unsqueeze(0)  # Add batch dimension

                # Model inference
                with torch.no_grad():
                    observation = {
                        "observation.image": image,
                        "observation.state": state
                    }
                    action = self.model.select_action(observation)

                inference_time = time.time() - start_time

                # Add to action queue
                try:
                    self.action_queue.put_nowait((action, inference_time))
                except:
                    # Queue full, drop old action
                    try:
                        self.action_queue.get_nowait()
                        self.action_queue.put_nowait((action, inference_time))
                    except:
                        pass

            except Empty:
                continue
            except Exception as e:
                print(f"⚠️  Inference error: {e}")
                continue

        print("🧠 Inference thread stopped")

    def control_loop(self):
        """Main control loop"""
        print("🎮 Control loop started")

        last_time = time.time()

        while self.running:
            cycle_start = time.time()

            try:
                # Get latest action
                action, inference_time = self.action_queue.get(timeout=0.1)

                # Execute action
                self.robot.set_action(action)

                # Update performance metrics
                total_time = time.time() - cycle_start
                self.perf_monitor.update(inference_time, total_time)

                # Maintain target frequency
                elapsed = time.time() - last_time
                if elapsed < self.target_dt:
                    time.sleep(self.target_dt - elapsed)
                last_time = time.time()

            except Empty:
                # No action available, continue
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Control error: {e}")
                continue

        print("🎮 Control loop stopped")

    def run(self, duration: Optional[float] = None):
        """Run the controller"""
        print(f"🚀 Starting LeKiwi controller...")
        if duration:
            print(f"  Running for {duration} seconds")
        else:
            print("  Running until Ctrl+C")

        self.running = True

        # Start threads
        camera_thread = threading.Thread(target=self.camera_thread, daemon=True)
        inference_thread = threading.Thread(target=self.inference_thread, daemon=True)

        camera_thread.start()
        inference_thread.start()

        try:
            if duration:
                # Run for specified duration
                self.control_loop_timed(duration)
            else:
                # Run until interrupted
                self.control_loop()

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.stop()

    def control_loop_timed(self, duration: float):
        """Run control loop for specified duration"""
        start_time = time.time()

        while self.running and (time.time() - start_time) < duration:
            try:
                # Get latest action
                action, inference_time = self.action_queue.get(timeout=0.1)

                # Execute action
                self.robot.set_action(action)

                # Update performance metrics
                total_time = time.time() - start_time
                self.perf_monitor.update(inference_time, total_time)

            except Empty:
                continue
            except Exception as e:
                print(f"⚠️  Control error: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"✅ Completed {elapsed:.1f}s run")

    def stop(self):
        """Stop the controller"""
        print("🛑 Stopping controller...")
        self.running = False

        # Final performance stats
        self.perf_monitor.print_stats()

        # Cleanup
        time.sleep(0.5)  # Allow threads to finish
        self.camera.release()
        self.robot.disconnect()

        print("✅ Controller stopped")


def main():
    parser = argparse.ArgumentParser(description="LeKiwi Real-time Deployment")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../deployment/lekiwi_mobilenet_model.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Target control frequency"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (None for infinite)"
    )
    parser.add_argument(
        "--fake_hardware",
        action="store_true",
        help="Use fake hardware for testing"
    )

    args = parser.parse_args()

    print("🤖 LeKiwi Real-time Deployment")
    print("=" * 50)

    # Create controller
    controller = LeKiwiController(
        model_path=args.model_path,
        camera_id=args.camera_id,
        target_fps=args.fps,
        use_fake_hardware=args.fake_hardware
    )

    # Run controller
    try:
        controller.run(duration=args.duration)
    except Exception as e:
        print(f"💥 Error: {e}")
        controller.stop()

    print("👋 Goodbye!")


if __name__ == "__main__":
    main()