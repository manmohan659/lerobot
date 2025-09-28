#!/usr/bin/env python3

"""
Direct Motor Test - Send hardcoded coordinates to LeKiwi motors
Bypasses vision pipeline to test basic motor communication
"""

import time
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

def test_motors():
    print("🔧 Direct Motor Test - LeKiwi")
    print("=" * 50)

    # Create robot config
    print("📋 Creating robot configuration...")
    robot_config = LeKiwiConfig(port="/dev/ttyACM0")
    robot = LeKiwi(robot_config)

    # Connect to robot
    print("🔌 Connecting to motors...")
    try:
        robot.connect()
        print("✅ Motors connected successfully!")
    except Exception as e:
        print(f"❌ Motor connection failed: {e}")
        return

    # Test coordinates (safe small movements)
    test_actions = [
        {
            "arm_shoulder_pan.pos": 0.0,
            "arm_shoulder_lift.pos": 0.0,
            "arm_elbow_flex.pos": 0.0,
            "arm_wrist_flex.pos": 0.0,
            "arm_wrist_roll.pos": 0.0,
            "arm_gripper.pos": 0.0,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        },
        {
            "arm_shoulder_pan.pos": 0.1,
            "arm_shoulder_lift.pos": 0.1,
            "arm_elbow_flex.pos": 0.1,
            "arm_wrist_flex.pos": 0.1,
            "arm_wrist_roll.pos": 0.1,
            "arm_gripper.pos": 0.1,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        },
        {
            "arm_shoulder_pan.pos": -0.1,
            "arm_shoulder_lift.pos": -0.1,
            "arm_elbow_flex.pos": -0.1,
            "arm_wrist_flex.pos": -0.1,
            "arm_wrist_roll.pos": -0.1,
            "arm_gripper.pos": -0.1,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }
    ]

    print("🎯 Testing motor movements...")
    print("⚠️  Make sure robot has space to move!")

    for i, action in enumerate(test_actions):
        print(f"\n📍 Test {i+1}/3: Sending action...")
        print(f"   Arm positions: [{action['arm_shoulder_pan.pos']:.1f}, {action['arm_shoulder_lift.pos']:.1f}, {action['arm_elbow_flex.pos']:.1f}]")

        try:
            # Send action
            robot.send_action(action)
            print("✅ Action sent successfully!")

            # Wait and get observation
            time.sleep(2.0)

            try:
                obs = robot.get_observation()
                state = obs.get("observation.state", "No state data")
                print(f"📊 Current state: {state}")
            except Exception as e:
                print(f"⚠️  Could not read state: {e}")

        except Exception as e:
            print(f"❌ Action failed: {e}")
            break

        print(f"⏱️  Waiting 3 seconds...")
        time.sleep(3.0)

    # Return to home
    print("\n🏠 Returning to home position...")
    try:
        home_action = {k: 0.0 for k in test_actions[0].keys()}
        robot.send_action(home_action)
        print("✅ Returned to home")
    except Exception as e:
        print(f"⚠️  Could not return home: {e}")

    # Disconnect
    try:
        robot.disconnect()
        print("✅ Disconnected safely")
    except Exception as e:
        print(f"⚠️  Disconnect error: {e}")

if __name__ == "__main__":
    test_motors()