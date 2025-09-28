#!/usr/bin/env python3
"""
LeKiwi Robot Motor Configuration Checker for Raspberry Pi
Tests all arm and base motors to verify proper configuration.
"""

import json
import logging
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# LeKiwi Robot Motor Configuration
LEKIWI_MOTOR_CONFIG = {
    # ARM MOTORS (6 motors)
    "arm_shoulder_pan": {"id": 1, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "arm"},
    "arm_shoulder_lift": {"id": 2, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "arm"},
    "arm_elbow_flex": {"id": 3, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "arm"},
    "arm_wrist_flex": {"id": 4, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "arm"},
    "arm_wrist_roll": {"id": 5, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "arm"},
    "arm_gripper": {"id": 6, "model": "sts3215", "mode": MotorNormMode.RANGE_0_100, "type": "arm"},
    # BASE MOTORS (3 motors)
    "base_left_wheel": {"id": 7, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "base"},
    "base_back_wheel": {"id": 8, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "base"},
    "base_right_wheel": {"id": 9, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100, "type": "base"},
}

# Default USB port for LeKiwi on Raspberry Pi
DEFAULT_USB_PORT = "/dev/ttyACM0"

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def print_status(message: str, status: str):
    """Print status with color coding"""
    status_symbols = {
        "PASS": "✅",
        "FAIL": "❌", 
        "WARN": "⚠️",
        "INFO": "ℹ️"
    }
    symbol = status_symbols.get(status, "•")
    print(f"{symbol} {message}")

def detect_usb_port():
    """Detect available USB ports for LeKiwi"""
    print_header("USB PORT DETECTION")
    
    possible_ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
    available_ports = []
    
    for port in possible_ports:
        port_path = Path(port)
        if port_path.exists():
            available_ports.append(port)
            print_status(f"Found USB device at {port}", "PASS")
    
    if not available_ports:
        print_status("No USB devices found", "FAIL")
        print("   Common LeKiwi ports: /dev/ttyACM0, /dev/ttyUSB0")
        return None
    
    # Use the first available port (typically /dev/ttyACM0)
    selected_port = available_ports[0]
    print_status(f"Using port: {selected_port}", "INFO")
    return selected_port

def check_calibration_file(robot_id: str = "my_lekiwi"):
    """Check if calibration file exists for LeKiwi robot"""
    print_header("CALIBRATION FILE CHECK")
    
    calibration_file = Path.home() / f".cache/huggingface/lerobot/calibration/robots/lekiwi/{robot_id}.json"
    
    if not calibration_file.exists():
        print_status(f"Calibration file NOT found at {calibration_file}", "WARN")
        print("   This is normal for a new robot - calibration will be created on first run")
        return None
    
    try:
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        print_status(f"Calibration file found at {calibration_file}", "PASS")
        
        # Check if all expected motors are in calibration
        expected_motors = set(LEKIWI_MOTOR_CONFIG.keys())
        calibrated_motors = set(calibration_data.keys())
        
        if expected_motors == calibrated_motors:
            print_status("All motors have calibration data", "PASS")
        else:
            missing = expected_motors - calibrated_motors
            extra = calibrated_motors - expected_motors
            if missing:
                print_status(f"Missing calibration for: {missing}", "WARN")
            if extra:
                print_status(f"Extra calibration data for: {extra}", "INFO")
        
        # Print calibration summary
        print("\nCalibration Summary:")
        print("ARM MOTORS:")
        for motor_name, cal_data in calibration_data.items():
            if motor_name.startswith("arm_"):
                motor_id = cal_data.get('id', 'N/A')
                homing_offset = cal_data.get('homing_offset', 'N/A')
                range_min = cal_data.get('range_min', 'N/A')
                range_max = cal_data.get('range_max', 'N/A')
                print(f"  {motor_name:20} | ID: {motor_id:2} | Offset: {homing_offset:6} | Range: {range_min:4}-{range_max:4}")
        
        print("BASE MOTORS:")
        for motor_name, cal_data in calibration_data.items():
            if motor_name.startswith("base_"):
                motor_id = cal_data.get('id', 'N/A')
                homing_offset = cal_data.get('homing_offset', 'N/A')
                range_min = cal_data.get('range_min', 'N/A')
                range_max = cal_data.get('range_max', 'N/A')
                print(f"  {motor_name:20} | ID: {motor_id:2} | Offset: {homing_offset:6} | Range: {range_min:4}-{range_max:4}")
        
        return calibration_data
        
    except json.JSONDecodeError as e:
        print_status(f"Invalid JSON in calibration file: {e}", "FAIL")
        return None
    except Exception as e:
        print_status(f"Error reading calibration file: {e}", "FAIL")
        return None

def test_individual_motors(usb_port: str):
    """Test each motor individually"""
    print_header("INDIVIDUAL MOTOR TESTS")
    
    results = {"arm": {}, "base": {}}
    
    # Test ARM motors first
    print("\n🦾 TESTING ARM MOTORS:")
    for motor_name, config in LEKIWI_MOTOR_CONFIG.items():
        if config["type"] != "arm":
            continue
            
        print(f"\nTesting {motor_name} (ID: {config['id']})...")
        
        # Create single motor configuration
        motors = {motor_name: Motor(config['id'], config['model'], config['mode'])}
        
        try:
            bus = FeetechMotorsBus(port=usb_port, motors=motors)
            bus.connect()
            
            # Test basic communication
            position = bus.read("Present_Position", motor_name, normalize=False)
            print_status(f"{motor_name} position: {position}", "PASS")
            
            # Test multiple reads for stability
            positions = []
            for i in range(3):
                pos = bus.read("Present_Position", motor_name, normalize=False)
                positions.append(pos)
                time.sleep(0.1)
            
            # Check for consistent readings
            if len(set(positions)) <= 2:  # Allow for minor variations
                print_status(f"{motor_name} readings stable: {positions}", "PASS")
                results["arm"][motor_name] = {"status": "PASS", "position": position}
            else:
                print_status(f"{motor_name} readings unstable: {positions}", "WARN")
                results["arm"][motor_name] = {"status": "WARN", "position": position}
            
            bus.disconnect()
            
        except Exception as e:
            print_status(f"{motor_name} communication failed: {e}", "FAIL")
            results["arm"][motor_name] = {"status": "FAIL", "error": str(e)}
    
    # Test BASE motors
    print("\n🛞 TESTING BASE MOTORS:")
    for motor_name, config in LEKIWI_MOTOR_CONFIG.items():
        if config["type"] != "base":
            continue
            
        print(f"\nTesting {motor_name} (ID: {config['id']})...")
        
        # Create single motor configuration
        motors = {motor_name: Motor(config['id'], config['model'], config['mode'])}
        
        try:
            bus = FeetechMotorsBus(port=usb_port, motors=motors)
            bus.connect()
            
            # Test basic communication
            position = bus.read("Present_Position", motor_name, normalize=False)
            print_status(f"{motor_name} position: {position}", "PASS")
            
            # Test velocity reading (base motors use velocity mode)
            try:
                velocity = bus.read("Present_Speed", motor_name, normalize=False)
                print_status(f"{motor_name} velocity: {velocity}", "PASS")
            except:
                print_status(f"{motor_name} velocity read failed (normal if not moving)", "INFO")
            
            results["base"][motor_name] = {"status": "PASS", "position": position}
            bus.disconnect()
            
        except Exception as e:
            print_status(f"{motor_name} communication failed: {e}", "FAIL")
            results["base"][motor_name] = {"status": "FAIL", "error": str(e)}
    
    return results

def test_all_motors_sync(usb_port: str):
    """Test all motors together using sync operations"""
    print_header("ALL MOTORS SYNC TEST")
    
    try:
        # Create all motors
        motors = {
            name: Motor(config['id'], config['model'], config['mode'])
            for name, config in LEKIWI_MOTOR_CONFIG.items()
        }
        
        bus = FeetechMotorsBus(port=usb_port, motors=motors)
        bus.connect()
        
        print_status("Connected to all motors", "PASS")
        
        # Test sync read for arm motors
        arm_motors = [name for name, config in LEKIWI_MOTOR_CONFIG.items() if config["type"] == "arm"]
        arm_positions = bus.sync_read("Present_Position", arm_motors, normalize=False)
        
        print_status("Arm motors sync read successful", "PASS")
        print("\n🦾 ARM MOTOR POSITIONS:")
        for motor_name, position in arm_positions.items():
            print(f"  {motor_name:20}: {position:6}")
        
        # Test sync read for base motors
        base_motors = [name for name, config in LEKIWI_MOTOR_CONFIG.items() if config["type"] == "base"]
        base_positions = bus.sync_read("Present_Position", base_motors, normalize=False)
        
        print_status("Base motors sync read successful", "PASS")
        print("\n🛞 BASE MOTOR POSITIONS:")
        for motor_name, position in base_positions.items():
            print(f"  {motor_name:20}: {position:6}")
        
        # Test stability with multiple sync reads
        print("\nTesting sync read stability...")
        for i in range(3):
            all_positions = bus.sync_read("Present_Position", list(LEKIWI_MOTOR_CONFIG.keys()), normalize=False)
            arm_pos = [all_positions[name] for name in arm_motors]
            base_pos = [all_positions[name] for name in base_motors]
            print(f"  Read {i+1} - Arm: {arm_pos}, Base: {base_pos}")
            time.sleep(0.2)
        
        bus.disconnect()
        print_status("All motors sync test completed", "PASS")
        return True
        
    except Exception as e:
        print_status(f"Sync test failed: {e}", "FAIL")
        return False

def test_lekiwi_robot_class(usb_port: str, robot_id: str = "my_lekiwi"):
    """Test using the actual LeKiwi robot class"""
    print_header("LEKIWI ROBOT CLASS TEST")
    
    try:
        # Create LeKiwi configuration
        config = LeKiwiConfig(port=usb_port, id=robot_id)
        
        # Initialize LeKiwi robot
        robot = LeKiwi(config)
        
        print_status("LeKiwi robot instance created", "PASS")
        
        # Test connection
        robot.connect()
        print_status("LeKiwi robot connected successfully", "PASS")
        
        # Test getting observation (current state)
        observation = robot.get_observation()
        print_status("Observation retrieval successful", "PASS")
        
        print("\nCurrent Robot State:")
        print("🦾 ARM POSITIONS:")
        for key, value in observation.items():
            if key.startswith("arm_") and key.endswith(".pos"):
                print(f"  {key:25}: {value:8.4f}")
        
        print("🛞 BASE VELOCITIES:")
        for key, value in observation.items():
            if key.endswith(".vel"):
                print(f"  {key:25}: {value:8.4f}")
        
        # Test calibration status
        if robot.is_calibrated:
            print_status("Robot is properly calibrated", "PASS")
        else:
            print_status("Robot needs calibration", "WARN")
            print("   Run robot.calibrate() to calibrate motors")
        
        # Test motor configuration
        print(f"\nMotor Configuration:")
        print(f"  Total motors: {len(robot.bus.motors)}")
        print(f"  Arm motors: {len(robot.arm_motors)}")
        print(f"  Base motors: {len(robot.base_motors)}")
        
        robot.disconnect()
        print_status("LeKiwi robot disconnected", "PASS")
        return True
        
    except Exception as e:
        print_status(f"LeKiwi robot test failed: {e}", "FAIL")
        print(f"   Error details: {str(e)}")
        return False

def main():
    """Main function to run all tests"""
    print_header("LEKIWI ROBOT MOTOR CONFIGURATION CHECKER")
    print("This script tests all LeKiwi robot motors (6 arm + 3 base motors)")
    
    # Detect USB port
    usb_port = detect_usb_port()
    if not usb_port:
        print("\n❌ Cannot proceed without USB connection")
        return
    
    print(f"USB Port: {usb_port}")
    
    # Check calibration file
    robot_id = "my_lekiwi"  # Default robot ID
    calibration_data = check_calibration_file(robot_id)
    
    # Run motor tests
    individual_results = test_individual_motors(usb_port)
    sync_ok = test_all_motors_sync(usb_port)
    robot_class_ok = test_lekiwi_robot_class(usb_port, robot_id)
    
    # Summary
    print_header("SUMMARY")
    
    arm_passed = sum(1 for result in individual_results["arm"].values() if result["status"] == "PASS")
    base_passed = sum(1 for result in individual_results["base"].values() if result["status"] == "PASS")
    total_arm_motors = len([m for m, c in LEKIWI_MOTOR_CONFIG.items() if c["type"] == "arm"])
    total_base_motors = len([m for m, c in LEKIWI_MOTOR_CONFIG.items() if c["type"] == "base"])
    
    print(f"🦾 Arm Motor Tests: {arm_passed}/{total_arm_motors} passed")
    print(f"🛞 Base Motor Tests: {base_passed}/{total_base_motors} passed")
    print(f"🔄 Sync Test: {'PASS' if sync_ok else 'FAIL'}")
    print(f"🤖 LeKiwi Robot Class Test: {'PASS' if robot_class_ok else 'FAIL'}")
    print(f"📋 Calibration File: {'FOUND' if calibration_data else 'MISSING (will be created)'}")
    
    total_passed = arm_passed + base_passed
    total_motors = total_arm_motors + total_base_motors
    
    if total_passed == total_motors and sync_ok and robot_class_ok:
        print_status("🎉 ALL TESTS PASSED - LeKiwi robot is ready for operation!", "PASS")
        print("\nYou can now run:")
        print("  python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi")
    else:
        print_status("⚠️ Some tests failed - check individual results above", "WARN")
        if not robot_class_ok:
            print("\n🔧 Troubleshooting tips:")
            print("  1. Make sure all motor connections are secure")
            print("  2. Check power supply to motors")
            print("  3. Verify USB cable connection")
            print("  4. Try running: sudo chmod 666 /dev/ttyACM*")

if __name__ == "__main__":
    main()
