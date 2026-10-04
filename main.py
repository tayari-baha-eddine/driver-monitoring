"""Driver Monitoring System - Full pipeline with Risk Engine.

Pipeline:
  webcam -> face -> eyes + head pose + phone + recognition
         -> drowsiness + distraction
         -> RiskEngine (weighted fusion + EMA smoothing)
         -> TemporalStateEngine (hysteresis + confirmation)
         -> alerts + logger + HUD
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np
import torch

from database.logger import Event, EventLogger
from intelligence.distraction import DistractionDetector
from intelligence.driver_state import DriverState
from intelligence.drowsiness import DrowsinessDetector
from intelligence.risk_engine import RiskEngine, TemporalStateEngine
from intelligence.scoring import AttentionScorer
from monitoring.alert_manager import AlertManager
from monitoring.performance import PerformanceMonitor
from vision.eye_analysis import EyeAnalyzer, EyeMetrics, EyeState
from vision.face_detection import FaceDetector
from vision.face_recognition import FaceRecognizer
from vision.head_pose import HeadPose, HeadDirection, HeadPoseEstimator
from vision.phone_detection import PhoneDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Driver Monitoring System")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-phone", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-recognition", action="store_true")
    p.add_argument("--alpha", type=float, default=0.20,
                   help="EMA smoothing factor (0.05 slow - 0.5 fast)")
    p.add_argument("--min-dwell", type=float, default=1.0,
                   help="Minimum dwell time in a state (seconds)")
    return p.parse_args()


# ============================================================
# HUD
# ============================================================

def _risk_bar(frame, x: int, y: int, w: int, h: int,
              value: float, color: tuple[int, int, int]) -> None:
    """Draw a horizontal risk bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill_w = int(w * min(100.0, max(0.0, value)) / 100.0)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)


def draw_hud(
    frame: np.ndarray,
    state_result,
    scores,
    drowsy_res,
    distract_res,
    perf_stats,
    driver_name: str = "unknown",
    driver_sim: float = 0.0,
) -> None:
    h, w = frame.shape[:2]

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Driver identity
    driver_text = f"Driver: {driver_name}" if driver_name != "unknown" else "Driver: Unknown"
    if driver_sim > 0 and driver_name != "unknown":
        driver_text += f"  (sim {driver_sim:.2f})"
    (tw, _), _ = cv2.getTextSize(driver_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, driver_text,
                (max(20, w // 2 - tw // 2), 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (100, 255, 255) if driver_name != "unknown" else (160, 160, 160), 2)

    # State (big, colored)
    color = AlertManager.color_for(DriverState(state_result.state))
    state_label = state_result.state
    if state_result.pending_state:
        state_label += f"  ({state_result.pending_state} "
        state_label += f"{int(state_result.pending_progress * 100)}%)"

    cv2.putText(frame, state_label,
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    if state_result.reason:
        cv2.putText(frame, state_result.reason,
                    (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Risk bar (smoothed)
    risk = state_result.risk_smoothed
    if risk >= 70:
        risk_color = (0, 0, 255)
    elif risk >= 45:
        risk_color = (0, 140, 255)
    elif risk >= 20:
        risk_color = (0, 200, 255)
    else:
        risk_color = (0, 200, 0)

    cv2.putText(frame, f"RISK", (20, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    _risk_bar(frame, 70, 138, 180, 12, risk, risk_color)
    cv2.putText(frame, f"{risk:5.1f}", (260, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Scores (right)
    cv2.putText(frame, f"Attention {scores.attention:5.1f}",
                (w - 340, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Vigilance {scores.vigilance:5.1f}",
                (w - 340, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Risk: {scores.risk_level.value}",
                (w - 340, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Bottom: performance
    cv2.putText(
        frame,
        f"FPS {perf_stats.fps:5.1f} | Lat {perf_stats.latency_ms:5.0f}ms | "
        f"CPU {perf_stats.cpu_percent:3.0f}% | RAM {perf_stats.ram_mb:4.0f}MB",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    cv2.putText(
        frame,
        f"Drow {drowsy_res.score:5.1f} | Dist {distract_res.score:5.1f}",
        (w - 260, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    # Border
    thickness = 8 if state_result.state == "CRITICAL" else 3
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error("Webcam inaccessible")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if torch.cuda.is_available():
        logger.info(f"Device: GPU {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device: CPU")

    # ----- Detectors -----
    face_det     = FaceDetector()
    eye_an       = EyeAnalyzer()
    hp_est       = HeadPoseEstimator()
    phone_det    = None if args.no_phone else PhoneDetector()
    drowsy_det   = DrowsinessDetector()
    distract_det = DistractionDetector()
    scorer       = AttentionScorer()
    perf         = PerformanceMonitor()
    alerts       = None if args.no_audio else AlertManager()
    event_log    = EventLogger()

    # ----- Risk & temporal engines -----
    risk_engine   = RiskEngine(ema_alpha=args.alpha)
    state_engine  = TemporalStateEngine(min_dwell_seconds=args.min_dwell)

    logger.info(f"EMA alpha = {args.alpha}  (higher = faster reaction)")
    logger.info(f"Min dwell = {args.min_dwell}s")

    # ----- Recognition -----
    recognizer = None
    if not args.no_recognition:
        try:
            recognizer = FaceRecognizer()
            logger.info(f"Recognition enabled: {len(recognizer.db)} driver(s)")
        except Exception as e:
            logger.warning(f"Recognition disabled: {e}")

    logger.info("Pipeline started. Press ECHAP to quit.")

    current_driver = "unknown"
    current_similarity = 0.0
    frame_count = 0
    RECOGNITION_INTERVAL = 30

    try:
        while True:
            with perf.stage("total"):
                ok, frame = cap.read()
                if not ok:
                    continue
                perf.tick()
                frame_count += 1

                face_result = face_det.process(frame)

                eye_metrics = None
                pose = None

                if face_result is not None:
                    with perf.stage("eyes"):
                        eye_metrics = eye_an.analyze(face_result.landmarks)
                    with perf.stage("head"):
                        pose = hp_est.estimate(face_result.landmarks, frame.shape)

                    if recognizer is not None and frame_count % RECOGNITION_INTERVAL == 0:
                        with perf.stage("recognize"):
                            emb = recognizer.compute_embedding(frame, face_result.bbox)
                            if emb is not None:
                                name, sim = recognizer.recognize(emb)
                                current_similarity = sim
                                if name is not None:
                                    current_driver = name
                                elif sim < 0.5:
                                    current_driver = "unknown"

                phones = []
                if phone_det is not None:
                    with perf.stage("phone"):
                        phones = phone_det.detect(frame)
                        phone_det.annotate(frame, phones)

                if eye_metrics is None:
                    eye_metrics = EyeMetrics(0.0, 0.0, 0.0, EyeState.UNKNOWN)
                if pose is None:
                    pose = HeadPose(0.0, 0.0, 0.0, HeadDirection.FRONT)

                # ----- Intelligence -----
                with perf.stage("reasoning"):
                    drowsy_res = drowsy_det.update(eye_metrics)
                    distract_res = distract_det.update(pose, bool(phones))
                    risk_res = risk_engine.update(
                        drowsy_res, distract_res, pose, bool(phones))
                    state_res = state_engine.update(
                        face_result is not None, drowsy_res, distract_res, risk_res)
                    scores = scorer.update(
                        DriverState(state_res.state), drowsy_res, distract_res)

                if alerts is not None:
                    alerts.trigger(DriverState(state_res.state))

                # ----- Logging -----
                perf_snapshot = perf.stats()
                event = Event(
                    ts=time.time(),
                    state=state_res.state,
                    reason=state_res.reason,
                    attention=scores.attention,
                    vigilance=scores.vigilance,
                    drowsiness=drowsy_res.score,
                    distraction=distract_res.score,
                    risk_instant=state_res.risk_instant,
                    risk_smoothed=state_res.risk_smoothed,
                    ear=eye_metrics.ear_avg,
                    mar=eye_metrics.mar,
                    yaw=pose.yaw,
                    pitch=pose.pitch,
                    roll=pose.roll,
                    phone=int(bool(phones)),
                    fps=perf_snapshot.fps,
                    latency_ms=perf_snapshot.latency_ms,
                    driver=current_driver,
                )
                event_log.log(event)

                draw_hud(frame, state_res, scores, drowsy_res, distract_res,
                         perf_snapshot, current_driver, current_similarity)

                cv2.imshow("Driver Monitoring", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_det.close()
        if alerts is not None:
            alerts.close()
        logger.info(f"Total state transitions: {state_engine.total_transitions}")
        logger.info(f"Last driver recognized: {current_driver}")
        logger.info(f"Logger stats: {event_log.stats()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())