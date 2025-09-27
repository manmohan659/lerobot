#!/usr/bin/env python3
"""
Motor Debug Script to diagnose motor4 issue
"""

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

def test_motor4():
    """Test motor4 individually to see if it returns valid position data"""

    # Configure motor4 only
    motors = {
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    }

    # Create bus with only motor4
    bus = FeetechMotorsBus(
        port="/dev/tty.usbmodem5A680119971",  # Your port
        motors=motors,
    )

    try:
        print("Connecting to motor4 only...")
        bus.connect()

        print("Reading position from motor4...")
        position = bus.read("Present_Position", "wrist_flex", normalize=False)
        print(f"Motor4 position: {position}")

        print("Testing sync_read with motor4 only...")
        positions = bus.sync_read("Present_Position", ["wrist_flex"], normalize=False)
        print(f"Sync read result: {positions}")

        print("Testing multiple reads...")
        for i in range(5):
            pos = bus.read("Present_Position", "wrist_flex", normalize=False)
            print(f"Read {i+1}: {pos}")

        bus.disconnect()
        print("✅ Motor4 communication successful")

    except Exception as e:
        print(f"❌ Error with motor4: {e}")
        bus.disconnect()

if __name__ == "__main__":
    test_motor4()