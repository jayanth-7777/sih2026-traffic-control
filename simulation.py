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

    def _generate_synthetic_traffic_frame(self, camera_id: str, frame_idx: int) -> np.ndarray:
        """
        Procedural animated urban traffic corridor generator.
        Activates automatically whenever local video files are absent, unreadable,
        or codec-restricted on cloud deployments. Ensures the live demo never fails.
        """
        h, w = 360, 640
        frame = np.full((h, w, 3), (35, 38, 44), dtype=np.uint8)  # Road asphalt

        # Road boundaries & sidewalks
        road_x1, road_x2 = 60, 580
        cv2.rectangle(frame, (road_x1, 0), (road_x2, h), (44, 49, 58), -1)
        cv2.line(frame, (road_x1, 0), (road_x1, h), (240, 240, 240), 3)
        cv2.line(frame, (road_x2, 0), (road_x2, h), (240, 240, 240), 3)

        # 3 Lanes with animated dashed lane markings moving downward
        lane_w = (road_x2 - road_x1) // 3
        scroll = (frame_idx * 7) % 40

        for lane_i in (1, 2):
            lx = road_x1 + lane_i * lane_w
            for y in range(-40 + scroll, h + 40, 40):
                cv2.line(frame, (lx, y), (lx, y + 22), (230, 230, 230), 2)

        # If Stop-Bar camera: render thick intersection stop line & pedestrian crosswalk
        stop_y = int(h * 0.72)
        if "STOPBAR" in camera_id or "CAM_2" in camera_id:
            cv2.line(frame, (road_x1, stop_y), (road_x2, stop_y), (255, 255, 255), 5)
            cv2.putText(frame, "STOP LINE", (road_x1 + 10, stop_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for zx in range(road_x1 + 10, road_x2 - 20, 28):
                cv2.rectangle(frame, (zx, stop_y + 10), (zx + 16, stop_y + 42), (220, 220, 220), -1)

        # Moving vehicles with distinct colors, windows, headlights, and plates
        palette = [
            (210, 45, 45),   # Red car
            (45, 120, 220),  # Blue sedan
            (230, 230, 230), # Silver hatchback
            (70, 75, 85),    # Dark SUV
            (35, 175, 95),   # Green taxi
            (235, 190, 30),  # Yellow cab
        ]

        for lane_idx in range(3):
            lane_cx = road_x1 + int((lane_idx + 0.5) * lane_w)
            base_speed = 6 + (lane_idx * 2)

            for v_idx in range(2):
                seed = lane_idx * 10 + v_idx
                raw_y = (frame_idx * base_speed + v_idx * 190) % (h + 130) - 65

                # Cam 2 queue deceleration near stop line if red
                if ("STOPBAR" in camera_id or "CAM_2" in camera_id) and raw_y > (stop_y - 80):
                    raw_y = min(stop_y - 35 - (v_idx * 55), raw_y)

                is_bus = (seed % 4 == 0)
                vw = 44 if not is_bus else 52
                vh = 68 if not is_bus else 110
                vx = lane_cx - vw // 2
                vy = int(raw_y)

                vcolor = palette[seed % len(palette)]

                # Vehicle body & outline
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), vcolor, -1)
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (15, 15, 15), 2)

                # Windshield (front/back)
                cv2.rectangle(frame, (vx + 4, vy + 10), (vx + vw - 4, vy + 22), (40, 50, 60), -1)
                cv2.rectangle(frame, (vx + 4, vy + vh - 20), (vx + vw - 4, vy + vh - 9), (40, 50, 60), -1)

                # Headlights (bottom facing) and tail lights
                cv2.circle(frame, (vx + 8, vy + vh - 3), 3, (0, 0, 240), -1)
                cv2.circle(frame, (vx + vw - 8, vy + vh - 3), 3, (0, 0, 240), -1)
                cv2.circle(frame, (vx + 8, vy + 3), 3, (210, 240, 255), -1)
                cv2.circle(frame, (vx + vw - 8, vy + 3), 3, (210, 240, 255), -1)

                # License plate white badge
                cv2.rectangle(frame, (vx + 8, vy + vh - 8), (vx + vw - 8, vy + vh - 2), (255, 255, 255), -1)

        return frame

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

        if frame1 is not None and ret1:
            frame1 = cv2.resize(frame1, (640, 360))
            self.using_synthetic_cam1 = False
        else:
            frame1 = self._generate_synthetic_traffic_frame("CAM_1_UPSTREAM", self.frame_index)
            self.using_synthetic_cam1 = True

        # -------------------------------------------------------------
        # 2. Read Cam 2 (Intersection Stop-Bar)
        # -------------------------------------------------------------
        ret2, frame2 = False, None
        if self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if not ret2:
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = self.cap2.read()

        if frame2 is not None and ret2:
            frame2 = cv2.resize(frame2, (640, 360))
            self.using_synthetic_cam2 = False
        else:
            frame2 = self._generate_synthetic_traffic_frame("CAM_2_STOPBAR", self.frame_index)
            self.using_synthetic_cam2 = True

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
            "using_synthetic_cam1": getattr(self, "using_synthetic_cam1", False),
            "using_synthetic_cam2": getattr(self, "using_synthetic_cam2", False),
        }

    def close(self):
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
