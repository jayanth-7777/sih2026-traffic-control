import os
from pathlib import Path

# System base directory
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Datasets and Videos (checks local bundled videos/ first for portability)
LOCAL_VIDEOS = BASE_DIR / "videos"
EXTERNAL_VIDEOS = Path(r"C:\sih2026-1\videos")
VIDEOS_DIR = LOCAL_VIDEOS if (LOCAL_VIDEOS.exists() and any(LOCAL_VIDEOS.glob("*.mp4"))) else EXTERNAL_VIDEOS
DEFAULT_CAM1_VIDEO = str(VIDEOS_DIR / "traffic5.mp4")
DEFAULT_CAM2_VIDEO = str(VIDEOS_DIR / "traffice7.mp4")

# YOLO Models
YOLO_NANO_PATH = str(MODELS_DIR / "yolov8n.pt")
YOLO_SMALL_PATH = str(MODELS_DIR / "yolov8s.pt")

# Fallback paths if not yet copied to models/
FALLBACK_NANO_PATH = r"C:\sih2026-1\phase2\yolov8n.pt"
FALLBACK_SMALL_PATH = r"C:\sih2026-1\phase2\yolov8s.pt"

# Camera & Spatial Setup (STEP 1)
CAMERA_DISTANCE_METERS = 400.0  # 400 meters between Upstream Corridor and Intersection Stop-Bar
SPEED_LIMIT_KMH = 60.0          # Flag speeding > 60 km/h
DELAY_LIMIT_SEC = 180.0         # Flag extreme delay > 3 minutes (180s)

# Trajectory & Platoon Formation (STEP 3)
PLATOON_ARRIVAL_WINDOW_SEC = 3.0  # Vehicles arriving within 3s form a platoon
PLATOON_MIN_SIZE = 4              # 4+ vehicles trigger platoon priority
PLATOON_MAX_ETA_TRIGGER = 8.0     # Trigger extension if platoon ETA < 8s

# Adaptive Traffic Signal Control Engine (STEP 4)
DEFAULT_GREEN_TIME = 25.0         # Base green duration in seconds
DEFAULT_AMBER_TIME = 4.0          # Safe amber clearance in seconds
DEFAULT_RED_TIME = 20.0           # Base red duration in seconds
GREEN_EXTENSION_TIME = 7.0        # Rule 1: Platoon priority green extension (5-10s)
QUEUE_BALANCE_RATIO = 2.0         # Rule 2: Dynamic balancing ratio (2x queue threshold)
QUEUE_CRITICAL_COUNT = 6          # Stop-bar queue count threshold for congestion warning

# COCO vehicle classes used by YOLO:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Idle fuel consumption savings factor:
# Average idling car consumes ~0.8 Liters/hour.
# 24% reduction in intersection idling time translates to measurable fuel saved.
FUEL_CONSUMPTION_LITER_PER_SEC = 0.8 / 3600.0
IDLE_REDUCTION_FACTOR = 0.24
