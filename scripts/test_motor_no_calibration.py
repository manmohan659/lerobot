#!/usr/bin/env python3

"""
Direct Motor Test - No Calibration Required
Test basic motor communication without calibration step
"""

import time
from lerobot.motors.feetech import FeetechMotorsBus

def test_motors_basic():
    print("🔧 Basic Motor Communication Test")
    print("=" * 50)

    # Motor configuration for LeKiwi arm
    motors = {
        1: {"name": "arm_shoulder_pan", "model": "sts3215"},
        2: {"name": "arm_shoulder_lift", "model": "sts3215"},
        3: {"name": "arm_elbow_flex", "model": "sts3215"},
        4: {"name": "arm_wrist_flex", "model": "sts3215"},
        5: {"name": "arm_wrist_roll", "model": "sts3215"},
        6: {"name": "arm_gripper", "model": "sts3215"},
    }

    print("🔌 Connecting to motor bus...")
    try:
        # Create motor bus
        bus = FeetechMotorsBus(
            port="/dev/ttyACM0",
            motors=motors,
            baudrate=1000000
        )

        # Connect without handshake to skip calibration
        bus.connect(handshake=False)
        print("✅ Motor bus connected (no calibration)!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Test each motor individually
    print("\n🔍 Testing individual motors...")

    for motor_id, motor_info in motors.items():
        print(f"\n📍 Testing Motor {motor_id} ({motor_info['name']}):")

        try:
            # Try to ping motor
            model_num = bus.ping(motor_id)
            print(f"   ✅ Ping successful - Model: {model_num}")

            # Try to read current position
            current_pos = bus.read("Present_Position", motor_id)
            print(f"   📊 Current position: {current_pos}")

            # Try small movement (±50 from current position)
            target_pos = current_pos + 50
            print(f"   🎯 Moving to position: {target_pos}")

            bus.write("Goal_Position", motor_id, target_pos)
            time.sleep(1.0)

            # Read new position
            new_pos = bus.read("Present_Position", motor_id)
            print(f"   📍 New position: {new_pos}")

            # Return to original
            bus.write("Goal_Position", motor_id, current_pos)
            time.sleep(1.0)

            final_pos = bus.read("Present_Position", motor_id)
            print(f"   🏠 Returned to: {final_pos}")

        except Exception as e:
            print(f"   ❌ Motor {motor_id} failed: {e}")
            continue

    # Test all motors together
    print("\n🎯 Testing coordinated movement...")
    try:
        # Read all current positions
        current_positions = {}
        for motor_id in motors.keys():
            try:
                pos = bus.read("Present_Position", motor_id)
                current_positions[motor_id] = pos
                print(f"   Motor {motor_id}: {pos}")
            except Exception as e:
                print(f"   ❌ Could not read motor {motor_id}: {e}")

        if current_positions:
            # Move all motors slightly
            print("\n   🔄 Moving all motors +30...")
            for motor_id, current_pos in current_positions.items():
                try:
                    bus.write("Goal_Position", motor_id, current_pos + 30)
                except Exception as e:
                    print(f"   ❌ Could not move motor {motor_id}: {e}")

            time.sleep(2.0)

            # Return all to original
            print("   🏠 Returning all to original positions...")
            for motor_id, current_pos in current_positions.items():
                try:
                    bus.write("Goal_Position", motor_id, current_pos)
                except Exception as e:
                    print(f"   ❌ Could not return motor {motor_id}: {e}")

            time.sleep(1.0)
            print("   ✅ Coordinated movement test complete!")

    except Exception as e:
        print(f"   ❌ Coordinated movement failed: {e}")

    # Disconnect
    try:
        bus.disconnect()
        print("\n✅ Disconnected safely")
    except Exception as e:
        print(f"\n⚠️  Disconnect error: {e}")

    print("\n🏁 Basic motor test complete!")

if __name__ == "__main__":
    test_motors_basic()