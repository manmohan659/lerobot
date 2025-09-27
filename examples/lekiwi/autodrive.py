#!/usr/bin/env python

import os
import time
from typing import List

import cv2

from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.utils.utils import log_say

from examples.lekiwi.agent.detector_client import DetectorClient
from examples.lekiwi.agent.llm import IntentParser
from examples.lekiwi.agent.navigator import Navigator
from examples.lekiwi.agent.picker import Picker
from examples.lekiwi.agent.returner import Returner
from examples.lekiwi.agent.types import Detection
from examples.lekiwi.agent.voice import stt_to_text
from examples.lekiwi.agent.logging_utils import setup_json_logger, log_event, get_session_id


def main() -> None:
    # Config from env
    remote_ip = os.environ.get("LEKIWI_PI_IP", "127.0.0.1")
    robot_id = os.environ.get("LEKIWI_ROBOT_ID", "my_lekiwi")
    detector_url = os.environ.get("DETECTOR_URL", "http://127.0.0.1:8080")

    sid = get_session_id()
    logger = setup_json_logger(
        "autodrive",
        log_file=os.environ.get("MAC_LOG_FILE"),
        node="mac",
        session_id=sid,
    )

    client_timeout = int(os.environ.get("LEKIWI_CONNECT_TIMEOUT_S", "30"))
    robot = LeKiwiClient(
        LeKiwiClientConfig(remote_ip=remote_ip, id=robot_id, connect_timeout_s=client_timeout)
    )
    robot.connect()

    detector = DetectorClient(detector_url)
    navigator = Navigator(img_width=640, img_height=480)
    picker = Picker()
    returner = Returner()
    intent_parser = IntentParser()

    _init_rerun(session_name="lekiwi_autodrive")

    print("[agent] Ready. Say your command after the beep or type it.")
    try:
        log_say("Ready for command", play_sounds=True)
    except Exception:
        pass
    text = stt_to_text()
    intent = intent_parser.parse_intent(text)
    target_label = intent.object if (intent.object and intent.object.lower() != "object") else "tissue"
    print(f"[agent] Intent parsed: task={intent.task}, object={intent.object}")
    try:
        log_say(f"I will {intent.task} {target_label}", play_sounds=True)
    except Exception:
        pass
    log_event(logger, "intent", text=text, task=intent.task, object=intent.object)

    # Simple heading placeholder
    home_heading_deg = 0.0
    returner.set_home_heading(home_heading_deg)

    print("Starting loop...")
    state = "SEARCH"
    FPS = 15
    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            frame_bgr = obs["front"]  # OpenCV BGR

            detections: List[Detection] = []
            try:
                detections = detector.detect(frame_bgr, [target_label])
            except Exception:
                detections = []

            if state in ("SEARCH", "APPROACH"):
                action = navigator.compute_action(detections)
                print(f"[agent] action: {action}")
                robot.send_action(action)
                log_event(logger, "action", **action)
                # Transition
                if len(detections) > 0:
                    det = max(detections, key=lambda d: d.score)
                    area_ratio = (det.bbox.w * det.bbox.h) / float(640 * 480)
                    if area_ratio > 0.05:
                        state = "PICK"
                else:
                    state = "SEARCH"

            elif state == "PICK":
                ok = picker.run_pick(robot)
                log_event(logger, "pick_done", success=ok)
                state = "RETURN" if ok else "SEARCH"

            elif state == "RETURN":
                action = returner.step(current_heading_deg=0.0)
                robot.send_action(action)
                log_event(logger, "return_action", **action)
                # Stop after short return for demo
                time.sleep(2.0)
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                state = "DONE"

            elif state == "DONE":
                break

            log_rerun_data(observation=obs, action=action)
            # pace
            dt = time.perf_counter() - t0
            time.sleep(max(1.0 / FPS - dt, 0.0))

    finally:
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


if __name__ == "__main__":
    main()


