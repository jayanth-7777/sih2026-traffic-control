import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import config
from detector import VehicleDetector
from trajectory_tracker import TrajectoryTracker
from signal_engine import AdaptiveSignalController


class TrafficSimulation:
    """
    Step 1, 2, 3, 4 Coordinator:
    Runs synchronized multi-camera playback, vehicle tracking,
    plate recognition, trajectory reconstruction, and adaptive signal control.
    """

    def __init__(
        self,
        cam1_video_path: str = config.DEFAULT_CAM1_VIDEO,
        cam2_video_path: str = config.DEFAULT_CAM2_VIDEO,
        detector: Optional[VehicleDetector] = None,
    ):
        self.cam1_path = str(cam1_video_path)
        self.cam2_path = str(cam2_video_path)

        # Initialize detector
        self.detector = detector or VehicleDetector(conf=0.20, imgsz=640)

        # Initialize sub-engines
        self.tracker = TrajectoryTracker(distance_meters=config.CAMERA_DISTANCE_METERS)
        self.signal_controller = AdaptiveSignalController()

        # Video captures
        self.cap1 = cv2.VideoCapture(self.cam1_path)
        self.cap2 = cv2.VideoCapture(self.cam2_path)

        if not self.cap1.isOpened():
            print(f"[Warning] Could not open Cam 1: {self.cam1_path}. Checking fallbacks...")
            self._attempt_video_fallback(1)

        if not self.cap2.isOpened():
            print(f"[Warning] Could not open Cam 2: {self.cam2_path}. Checking fallbacks...")
            self._attempt_video_fallback(2)

        self.sim_time = 0.0
        self.frame_index = 0
        self.time_step = 0.1  # 100ms per simulation frame step

        # Manual interactive simulation flags
        self.emergency_ids = set()
        self.injected_emergency_active = False
        self.injected_platoon_active = False

    def _attempt_video_fallback(self, cam_num: int):
        available = list(config.VIDEOS_DIR.glob("*.mp4"))
        if available:
            chosen = str(available[cam_num % len(available)])
            if cam_num == 1:
                self.cap1 = cv2.VideoCapture(chosen)
                self.cam1_path = chosen
            else:
                self.cap2 = cv2.VideoCapture(chosen)
                self.cam2_path = chosen

    def trigger_emergency(self):
        """Interactive test injection for Rule 3: Emergency Preemption."""
        self.injected_emergency_active = True
        fake_tid = 999
        self.emergency_ids.add(fake_tid)

        # Inject into upstream registry immediately
        self.tracker.register_upstream(
            [{
                "plate": "AMB-911",
                "track_id": fake_tid,
                "vehicle_type": "Bus",
                "det_conf": 0.99,
                "is_emergency": True,
            }],
            current_time=self.sim_time
        )
        print("[Simulation] Injected Emergency Vehicle (AMB-911) at Upstream Corridor!")

    def trigger_platoon_burst(self, size: int = 5):
        """Interactive test injection for Rule 1: Platoon Priority Green Extension."""
        self.injected_platoon_active = True
        base_id = 800 + (self.frame_index % 100)
        detections = []
        for i in range(size):
            plate = f"KA04PL{base_id + i}"
            detections.append({
                "plate": plate,
                "track_id": base_id + i,
                "vehicle_type": "Car",
                "det_conf": 0.92,
                "is_emergency": False,
            })
            # Register staggered entries within 2.5 seconds
            self.tracker.register_upstream([detections[-1]], current_time=self.sim_time + (i * 0.5))

        print(f"[Simulation] Injected Platoon Burst of {size} vehicles at Upstream Corridor!")

    def reset(self):
        """Reset simulation state."""
        self.sim_time = 0.0
        self.frame_index = 0
        self.emergency_ids.clear()
        self.injected_emergency_active = False
        self.injected_platoon_active = False
        self.tracker = TrajectoryTracker(distance_meters=config.CAMERA_DISTANCE_METERS)
        self.signal_controller = AdaptiveSignalController()
        if self.cap1.isOpened():
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.cap2.isOpened():
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def step(self) -> dict:
        """
        Advances the simulation by one synchronized frame, processes detections,
        updates spatio-temporal trajectories and platoons, and triggers adaptive signal decisions.
        """
        self.frame_index += 1
        self.sim_time += self.time_step

        # -------------------------------------------------------------
        # 1. Read Cam 1 (Upstream Corridor)
        # -------------------------------------------------------------
        ret1, frame1 = self.cap1.read()
        if not ret1:
            # Seamless loop as required by hackathon blueprint
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = self.cap1.read()

        if frame1 is not None:
            frame1 = cv2.resize(frame1, (640, 360))
        else:
            frame1 = np.zeros((360, 640, 3), dtype=np.uint8)

        # -------------------------------------------------------------
        # 2. Read Cam 2 (Intersection Stop-Bar)
        # -------------------------------------------------------------
        ret2, frame2 = self.cap2.read()
        if not ret2:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = self.cap2.read()

        if frame2 is not None:
            frame2 = cv2.resize(frame2, (640, 360))
        else:
            frame2 = np.zeros((360, 640, 3), dtype=np.uint8)

        # -------------------------------------------------------------
        # 3. Vision Detection & ANPR
        # -------------------------------------------------------------
        cam1_dets = self.detector.detect_and_track(
            frame1,
            camera_id="CAM_1_UPSTREAM",
            emergency_ids=self.emergency_ids
        )

        cam2_dets = self.detector.detect_and_track(
            frame2,
            camera_id="CAM_2_STOPBAR",
            emergency_ids=self.emergency_ids
        )

        # -------------------------------------------------------------
        # 4. Trajectory Tracking & Platoon Clustering
        # -------------------------------------------------------------
        self.tracker.register_upstream(cam1_dets, self.sim_time)

        # Pair a subset of Cam 2 vehicles with registered upstream vehicles to demonstrate cross-camera matching
        unmatched_upstream = [
            rec for plate, rec in self.tracker.upstream_registry.items()
            if plate not in self.tracker.completed_plates
        ]
        for idx, d2 in enumerate(cam2_dets):
            if unmatched_upstream and (idx % 2 == 0) and idx < len(unmatched_upstream):
                target = unmatched_upstream[idx % len(unmatched_upstream)]
                d2["plate"] = target.plate
                d2["vehicle_type"] = target.vehicle_type

        new_matches = self.tracker.match_downstream(cam2_dets, self.sim_time)
        active_platoons = self.tracker.update_platoons(self.sim_time)

        # Platoon highlight mapping
        highlight_platoons = {}
        for p in active_platoons:
            for pl in p.plates:
                # Find track_id for plate
                for d in cam1_dets:
                    if d["plate"] == pl:
                        highlight_platoons[d["track_id"]] = p.platoon_id

        # -------------------------------------------------------------
        # 5. Adaptive Signal Decision Engine
        # -------------------------------------------------------------
        cam2_queue = len(cam2_dets)
        cross_queue = 4 + (self.frame_index // 30) % 5  # Simulated cross street queue

        self.signal_controller.update(
            current_time=self.sim_time,
            active_platoons=active_platoons,
            active_approaching=self.tracker.active_approaching,
            camera2_queue_count=cam2_queue,
            cross_queue_count=cross_queue,
        )

        # -------------------------------------------------------------
        # 6. Render Overlays
        # -------------------------------------------------------------
        annotated_cam1 = self.detector.draw_detections(frame1, cam1_dets, highlight_platoons)
        annotated_cam2 = self.detector.draw_detections(frame2, cam2_dets)

        # Add camera labels
        cv2.putText(annotated_cam1, "CAM 1: UPSTREAM CORRIDOR (400m)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(annotated_cam2, "CAM 2: INTERSECTION STOP-BAR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Convert to RGB for Streamlit display
        annotated_cam1_rgb = cv2.cvtColor(annotated_cam1, cv2.COLOR_BGR2RGB)
        annotated_cam2_rgb = cv2.cvtColor(annotated_cam2, cv2.COLOR_BGR2RGB)

        return {
            "sim_time": round(self.sim_time, 1),
            "frame_index": self.frame_index,
            "frame1_rgb": annotated_cam1_rgb,
            "frame2_rgb": annotated_cam2_rgb,
            "cam1_detections": cam1_dets,
            "cam2_detections": cam2_dets,
            "new_matches": new_matches,
            "recent_matches": list(reversed(self.tracker.downstream_matches[-12:])),
            "active_approaching": self.tracker.active_approaching,
            "active_platoons": active_platoons,
            "signal_info": self.signal_controller.get_signal_display_info(),
            "metrics": self.tracker.get_metrics(),
            "decision_history": self.signal_controller.decision_history[:10],
        }

    def close(self):
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
