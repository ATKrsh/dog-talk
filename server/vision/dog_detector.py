"""
Dog Talk - Vision Pipeline: Dog Detection + Pose Estimation
Uses YOLOv8-pose for detecting dogs and extracting body keypoints.
Falls back to standard YOLOv8 detection if pose model unavailable.
"""
import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DogKeypoints:
    """24 keypoints for a detected dog."""
    nose: Optional[tuple] = None
    left_eye: Optional[tuple] = None
    right_eye: Optional[tuple] = None
    left_ear_base: Optional[tuple] = None
    right_ear_base: Optional[tuple] = None
    left_ear_tip: Optional[tuple] = None
    right_ear_tip: Optional[tuple] = None
    neck: Optional[tuple] = None
    left_shoulder: Optional[tuple] = None
    right_shoulder: Optional[tuple] = None
    left_elbow: Optional[tuple] = None
    right_elbow: Optional[tuple] = None
    left_front_paw: Optional[tuple] = None
    right_front_paw: Optional[tuple] = None
    spine_mid: Optional[tuple] = None
    left_hip: Optional[tuple] = None
    right_hip: Optional[tuple] = None
    left_knee: Optional[tuple] = None
    right_knee: Optional[tuple] = None
    left_rear_paw: Optional[tuple] = None
    right_rear_paw: Optional[tuple] = None
    tail_base: Optional[tuple] = None
    tail_mid: Optional[tuple] = None
    tail_tip: Optional[tuple] = None

    # Confidence scores for each keypoint
    confidences: dict = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Convert keypoints to numpy array [24, 3] (x, y, confidence)."""
        points = [
            self.nose, self.left_eye, self.right_eye,
            self.left_ear_base, self.right_ear_base,
            self.left_ear_tip, self.right_ear_tip,
            self.neck, self.left_shoulder, self.right_shoulder,
            self.left_elbow, self.right_elbow,
            self.left_front_paw, self.right_front_paw,
            self.spine_mid, self.left_hip, self.right_hip,
            self.left_knee, self.right_knee,
            self.left_rear_paw, self.right_rear_paw,
            self.tail_base, self.tail_mid, self.tail_tip
        ]
        result = np.zeros((24, 3))
        for i, pt in enumerate(points):
            if pt is not None:
                result[i, 0] = pt[0]
                result[i, 1] = pt[1]
                result[i, 2] = pt[2] if len(pt) > 2 else 1.0
        return result


@dataclass
class DogDetection:
    """A detected dog with bounding box and keypoints."""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Optional[DogKeypoints] = None
    breed_estimate: Optional[str] = None

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def size(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)


class DogDetector:
    """
    Detects dogs and extracts pose keypoints from images.
    Uses YOLOv8-pose with custom dog keypoint weights,
    falling back to standard YOLO detection + estimated keypoints.
    """

    # COCO class ID for 'dog' in standard YOLO
    DOG_CLASS_ID = 16

    # Keypoint names mapping for the dog-pose model
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye",
        "left_ear_base", "right_ear_base",
        "left_ear_tip", "right_ear_tip",
        "neck", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_front_paw", "right_front_paw",
        "spine_mid", "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_rear_paw", "right_rear_paw",
        "tail_base", "tail_mid", "tail_tip"
    ]

    def __init__(self, model_path: str = None, confidence: float = 0.5,
                 pose_confidence: float = 0.3, device: str = "auto"):
        self.confidence_threshold = confidence
        self.pose_confidence = pose_confidence
        self.device = device
        self.model = None
        self.pose_model = None
        self.using_pose = False
        self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        """Load YOLO model - try pose model first, fall back to detection."""
        try:
            from ultralytics import YOLO
            
            # Try to load custom dog-pose model
            if model_path:
                try:
                    self.model = YOLO(model_path)
                    self.using_pose = True
                    logger.info(f"Loaded custom dog-pose model from {model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Could not load custom pose model: {e}")

            # Fall back to standard YOLOv8 detection (detects dogs, but no pose)
            try:
                self.model = YOLO("yolov8n.pt")
                self.using_pose = False
                logger.info("Loaded YOLOv8n detection fallback (no pose estimation)")
            except Exception as e:
                logger.warning(f"Could not load YOLOv8n detection: {e}")
                # Last resort fallback
                try:
                    self.model = YOLO("yolov8n-pose.pt")
                    self.using_pose = True
                    logger.info("Loaded YOLOv8n-pose (human keypoints, last resort fallback)")
                except Exception:
                    logger.error("No YOLO model could be loaded.")
                    raise

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def detect(self, frame: np.ndarray) -> list[DogDetection]:
        """
        Detect dogs in a frame and extract keypoints.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            
        Returns:
            List of DogDetection objects
        """
        if self.model is None:
            return []

        detections = []

        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device if self.device != "auto" else None
            )

            for result in results:
                if result.boxes is None:
                    continue

                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Filter for dogs only (class 16 in COCO)
                    # In pose models, there's only one class, so we check differently
                    if not self.using_pose and cls_id != self.DOG_CLASS_ID:
                        continue

                    bbox = tuple(box.xyxy[0].cpu().numpy().astype(int))

                    # Extract keypoints if available
                    keypoints = None
                    if self.using_pose and result.keypoints is not None:
                        try:
                            kp_data = result.keypoints[i].data[0].cpu().numpy()
                            keypoints = self._parse_keypoints(kp_data, bbox)
                        except (IndexError, AttributeError) as e:
                            logger.debug(f"Could not extract keypoints: {e}")

                    # If no keypoints from model, estimate from bounding box
                    if keypoints is None:
                        keypoints = self._estimate_keypoints(bbox)

                    detection = DogDetection(
                        bbox=bbox,
                        confidence=conf,
                        keypoints=keypoints
                    )
                    detections.append(detection)

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return detections

    def _parse_keypoints(self, kp_data: np.ndarray, bbox: tuple) -> DogKeypoints:
        """Parse raw keypoint data into DogKeypoints."""
        keypoints = DogKeypoints()
        
        # Map available keypoints (may have fewer than 24)
        num_kps = min(len(kp_data), len(self.KEYPOINT_NAMES))
        
        for i in range(num_kps):
            x, y = float(kp_data[i, 0]), float(kp_data[i, 1])
            conf = float(kp_data[i, 2]) if kp_data.shape[1] > 2 else 1.0
            
            if conf >= self.pose_confidence and x > 0 and y > 0:
                name = self.KEYPOINT_NAMES[i]
                setattr(keypoints, name, (x, y, conf))
                keypoints.confidences[name] = conf

        return keypoints

    def _estimate_keypoints(self, bbox: tuple) -> DogKeypoints:
        """
        Estimate approximate keypoint positions from bounding box.
        Uses anatomical proportions for a typical dog.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Rough anatomical proportions (assuming side view)
        keypoints = DogKeypoints()
        low_conf = 0.3  # Low confidence since these are estimates
        
        # Head region (front ~20% of bbox)
        head_x = x1 + w * 0.15
        head_y = y1 + h * 0.25
        keypoints.nose = (x1 + w * 0.05, head_y, low_conf)
        keypoints.left_eye = (head_x - w * 0.02, head_y - h * 0.05, low_conf)
        keypoints.right_eye = (head_x + w * 0.02, head_y - h * 0.05, low_conf)
        keypoints.left_ear_tip = (head_x - w * 0.05, y1 + h * 0.05, low_conf)
        keypoints.right_ear_tip = (head_x + w * 0.05, y1 + h * 0.05, low_conf)
        
        # Neck
        keypoints.neck = (x1 + w * 0.25, y1 + h * 0.30, low_conf)
        
        # Spine
        keypoints.spine_mid = (x1 + w * 0.50, y1 + h * 0.25, low_conf)
        
        # Front legs
        keypoints.left_shoulder = (x1 + w * 0.30, y1 + h * 0.45, low_conf)
        keypoints.left_front_paw = (x1 + w * 0.28, y2 - h * 0.05, low_conf)
        
        # Rear legs
        keypoints.left_hip = (x1 + w * 0.75, y1 + h * 0.40, low_conf)
        keypoints.left_rear_paw = (x1 + w * 0.73, y2 - h * 0.05, low_conf)
        
        # Tail
        keypoints.tail_base = (x1 + w * 0.85, y1 + h * 0.30, low_conf)
        keypoints.tail_tip = (x2 - w * 0.02, y1 + h * 0.15, low_conf)
        
        return keypoints
