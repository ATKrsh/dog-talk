"""
Dog Talk - Body Language Analysis
Extracts geometric features from dog keypoints to determine body language signals.
"""
import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass, field

from .dog_detector import DogKeypoints

logger = logging.getLogger(__name__)


@dataclass
class BodyLanguageSignals:
    """Extracted body language signals from keypoints."""
    # Tail
    tail_position: str = "unknown"  # high, mid, low, tucked
    tail_movement: str = "unknown"  # fast_wag, slow_wag, stiff, none, circular
    tail_confidence: float = 0.0

    # Ears
    ear_position: str = "unknown"  # forward, slightly_forward, natural, slightly_back, flat, asymmetric
    ear_confidence: float = 0.0

    # Posture
    posture: str = "unknown"  # play_bow, standing_relaxed, standing_alert, forward_lean, cowering, belly_up, freeze
    posture_confidence: float = 0.0

    # Hackles (estimated from body shape)
    hackles_raised: bool = False
    hackles_confidence: float = 0.0

    # Overall body tension
    body_tension: float = 0.0  # 0.0 = loose/relaxed, 1.0 = stiff/tense

    # Head
    head_tilt: float = 0.0  # degrees
    head_height: str = "neutral"  # high, neutral, low

    # Weight distribution
    weight_forward: float = 0.5  # 0.0 = all back, 1.0 = all forward

    # Raw features for ML
    raw_features: dict = field(default_factory=dict)


class BodyLanguageAnalyzer:
    """
    Analyzes dog body keypoints to extract body language signals.
    Uses geometric relationships between keypoints and temporal analysis
    for movement detection.
    """

    def __init__(self):
        self._prev_keypoints: Optional[np.ndarray] = None
        self._prev_tail_positions: list = []
        self._max_history = 10  # frames of history for movement analysis

    def analyze(self, keypoints: DogKeypoints) -> BodyLanguageSignals:
        """
        Analyze keypoints to produce body language signals.
        
        Args:
            keypoints: DogKeypoints from the detector
            
        Returns:
            BodyLanguageSignals with all detected signals
        """
        signals = BodyLanguageSignals()
        kp_array = keypoints.to_array()

        # Analyze each body part
        signals.tail_position, signals.tail_confidence = self._analyze_tail_position(keypoints)
        signals.tail_movement = self._analyze_tail_movement(keypoints)
        signals.ear_position, signals.ear_confidence = self._analyze_ears(keypoints)
        signals.posture, signals.posture_confidence = self._analyze_posture(keypoints)
        signals.body_tension = self._calculate_body_tension(keypoints)
        signals.head_height = self._analyze_head_height(keypoints)
        signals.weight_forward = self._analyze_weight_distribution(keypoints)

        # Extract raw features for ML classifier
        signals.raw_features = self._extract_features(keypoints)

        # Save for temporal analysis
        self._prev_keypoints = kp_array
        if keypoints.tail_tip:
            self._prev_tail_positions.append(keypoints.tail_tip)
            if len(self._prev_tail_positions) > self._max_history:
                self._prev_tail_positions.pop(0)

        return signals

    def _analyze_tail_position(self, kp: DogKeypoints) -> tuple[str, float]:
        """Determine tail position relative to the spine."""
        if not kp.tail_base or not kp.spine_mid:
            return "unknown", 0.0

        spine_y = kp.spine_mid[1]
        tail_base_y = kp.tail_base[1]
        
        if kp.tail_tip:
            tail_tip_y = kp.tail_tip[1]
        else:
            tail_tip_y = tail_base_y

        # Calculate tail height relative to spine
        # Note: in image coordinates, Y increases downward
        avg_tail_y = (tail_base_y + tail_tip_y) / 2
        spine_to_tail = spine_y - avg_tail_y  # positive = tail above spine

        # Get body height for normalization
        if kp.left_front_paw and kp.spine_mid:
            body_height = abs(kp.left_front_paw[1] - kp.spine_mid[1])
        elif kp.left_rear_paw and kp.spine_mid:
            body_height = abs(kp.left_rear_paw[1] - kp.spine_mid[1])
        else:
            body_height = 100  # fallback

        if body_height == 0:
            body_height = 1

        normalized_height = spine_to_tail / body_height

        # Check for tucked tail (tail tip near or between rear legs)
        if kp.tail_tip and kp.left_hip:
            hip_y = kp.left_hip[1]
            if tail_tip_y > hip_y + body_height * 0.3:
                return "tucked", 0.80

        # Classify
        if normalized_height > 0.4:
            return "high", 0.75
        elif normalized_height > 0.1:
            return "mid", 0.70
        elif normalized_height > -0.2:
            return "low", 0.70
        else:
            return "tucked", 0.75

    def _analyze_tail_movement(self, kp: DogKeypoints) -> str:
        """Detect tail movement from temporal analysis."""
        if not kp.tail_tip or len(self._prev_tail_positions) < 3:
            return "unknown"

        # Calculate lateral movement of tail tip
        recent_positions = self._prev_tail_positions[-5:]
        x_positions = [p[0] for p in recent_positions]
        
        if len(x_positions) < 3:
            return "unknown"

        # Calculate variance in X (lateral movement)
        x_var = np.var(x_positions)
        x_range = max(x_positions) - min(x_positions)

        # Check for direction changes (oscillation = wagging)
        direction_changes = 0
        for i in range(2, len(x_positions)):
            prev_dir = x_positions[i - 1] - x_positions[i - 2]
            curr_dir = x_positions[i] - x_positions[i - 1]
            if prev_dir * curr_dir < 0:
                direction_changes += 1

        # Classify movement
        if x_range < 5:  # Very little movement
            return "stiff" if self._calculate_body_tension(kp) > 0.6 else "none"
        elif direction_changes >= 3 and x_range > 30:
            return "fast_wag"
        elif direction_changes >= 2 and x_range > 15:
            return "slow_wag"
        elif direction_changes >= 3 and x_range > 50:
            # Check for circular motion (helicopter wag)
            y_positions = [p[1] for p in recent_positions]
            y_range = max(y_positions) - min(y_positions)
            if y_range > x_range * 0.5:
                return "circular"
            return "fast_wag"
        else:
            return "slow_wag"

    def _analyze_ears(self, kp: DogKeypoints) -> tuple[str, float]:
        """Determine ear position."""
        # Check if we have ear keypoints
        has_ear_tips = kp.left_ear_tip is not None or kp.right_ear_tip is not None
        has_ear_bases = kp.left_ear_base is not None or kp.right_ear_base is not None

        if not has_ear_tips and not has_ear_bases:
            return "unknown", 0.0

        # Use available ear data
        if kp.left_ear_tip and kp.right_ear_tip and kp.nose:
            left_tip = kp.left_ear_tip
            right_tip = kp.right_ear_tip
            nose = kp.nose

            # Check asymmetry
            left_y = left_tip[1]
            right_y = right_tip[1]
            if abs(left_y - right_y) > 20:
                return "asymmetric", 0.60

            # Average ear tip Y relative to nose
            avg_ear_y = (left_y + right_y) / 2
            
            # Forward/back: compare ear tip X to nose X
            avg_ear_x = (left_tip[0] + right_tip[0]) / 2
            nose_x = nose[0]
            
            ear_nose_diff_y = nose[1] - avg_ear_y  # positive = ears above nose
            
            # Check ear spread (wider = more forward/alert)
            ear_spread = abs(left_tip[0] - right_tip[0])
            head_width = ear_spread * 1.5 if ear_spread > 0 else 50

            if ear_nose_diff_y > 15:
                return "forward", 0.75
            elif ear_nose_diff_y > 5:
                return "slightly_forward", 0.65
            elif ear_nose_diff_y > -5:
                return "natural", 0.70
            elif ear_nose_diff_y > -20:
                return "slightly_back", 0.65
            else:
                return "flat", 0.75

        # Fallback: estimate from available data
        return "natural", 0.40

    def _analyze_posture(self, kp: DogKeypoints) -> tuple[str, float]:
        """Determine overall body posture."""
        if not kp.spine_mid:
            return "unknown", 0.0

        # Check for play bow: front low, rear high
        if kp.left_shoulder and kp.left_hip:
            front_y = kp.left_shoulder[1]
            rear_y = kp.left_hip[1]
            height_diff = front_y - rear_y  # positive = front lower (play bow)
            
            if kp.left_front_paw:
                front_paw_to_shoulder = abs(kp.left_front_paw[1] - kp.left_shoulder[1])
                if height_diff > 30 and front_paw_to_shoulder < 40:
                    return "play_bow", 0.85

        # Check for cowering: body low to ground
        if kp.spine_mid and kp.left_front_paw:
            spine_height = kp.left_front_paw[1] - kp.spine_mid[1]
            if spine_height < 30:  # Very low
                return "cowering", 0.75

        # Check for forward lean
        weight = self._analyze_weight_distribution(kp)
        if weight > 0.7:
            return "forward_lean", 0.70
        
        # Check body tension for freeze vs relaxed
        tension = self._calculate_body_tension(kp)
        if tension > 0.7:
            return "stiff_freeze", 0.70

        # Check if standing alert vs relaxed
        if kp.neck and kp.spine_mid:
            neck_height = kp.spine_mid[1] - kp.neck[1]
            if neck_height > 20:
                return "standing_alert", 0.65
            else:
                return "standing_relaxed", 0.70

        return "standing_relaxed", 0.50

    def _calculate_body_tension(self, kp: DogKeypoints) -> float:
        """
        Estimate body tension from keypoint positions.
        Higher values = more tense/stiff.
        """
        tension_score = 0.0
        checks = 0

        # Straight spine = more tense
        if kp.neck and kp.spine_mid and kp.tail_base:
            neck = np.array(kp.neck[:2])
            spine = np.array(kp.spine_mid[:2])
            tail = np.array(kp.tail_base[:2])
            
            # Calculate spine curvature
            v1 = spine - neck
            v2 = tail - spine
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            straightness = abs(cos_angle)
            tension_score += straightness
            checks += 1

        # Legs spread evenly = stable/tense stance
        if kp.left_front_paw and kp.right_front_paw:
            front_spread = abs(kp.left_front_paw[0] - kp.right_front_paw[0])
            if front_spread > 50:
                tension_score += 0.3
            checks += 1

        if checks == 0:
            return 0.5

        return min(tension_score / checks, 1.0)

    def _analyze_head_height(self, kp: DogKeypoints) -> str:
        """Determine head height relative to body."""
        if not kp.nose or not kp.spine_mid:
            return "neutral"

        diff = kp.spine_mid[1] - kp.nose[1]  # positive = head above spine

        if diff > 30:
            return "high"
        elif diff > -10:
            return "neutral"
        else:
            return "low"

    def _analyze_weight_distribution(self, kp: DogKeypoints) -> float:
        """
        Calculate weight distribution (forward vs backward).
        Returns 0.0 (all back) to 1.0 (all forward).
        """
        if not kp.spine_mid:
            return 0.5

        front_x_sum = 0
        front_count = 0
        rear_x_sum = 0
        rear_count = 0

        for attr in ['left_front_paw', 'right_front_paw', 'left_shoulder', 'right_shoulder']:
            pt = getattr(kp, attr, None)
            if pt:
                front_x_sum += pt[0]
                front_count += 1

        for attr in ['left_rear_paw', 'right_rear_paw', 'left_hip', 'right_hip']:
            pt = getattr(kp, attr, None)
            if pt:
                rear_x_sum += pt[0]
                rear_count += 1

        if front_count == 0 or rear_count == 0:
            return 0.5

        front_avg = front_x_sum / front_count
        rear_avg = rear_x_sum / rear_count
        spine_x = kp.spine_mid[0]

        # How far spine is from center of front/rear
        total_length = abs(front_avg - rear_avg)
        if total_length == 0:
            return 0.5

        forward_shift = (spine_x - rear_avg) / total_length
        return max(0.0, min(1.0, forward_shift))

    def _extract_features(self, kp: DogKeypoints) -> dict:
        """Extract numerical features for ML classifier."""
        features = {}
        kp_array = kp.to_array()

        # Calculate all pairwise angles and distances
        valid_mask = kp_array[:, 2] > 0.1

        # Basic features
        features['num_visible_keypoints'] = int(np.sum(valid_mask))
        
        # Spine angle
        if kp.neck and kp.spine_mid and kp.tail_base:
            features['spine_angle'] = self._angle_between(
                kp.neck[:2], kp.spine_mid[:2], kp.tail_base[:2]
            )

        # Tail angle relative to spine
        if kp.tail_base and kp.tail_tip and kp.spine_mid:
            features['tail_angle'] = self._angle_between(
                kp.spine_mid[:2], kp.tail_base[:2], kp.tail_tip[:2]
            )

        # Head angle
        if kp.nose and kp.neck and kp.spine_mid:
            features['head_angle'] = self._angle_between(
                kp.nose[:2], kp.neck[:2], kp.spine_mid[:2]
            )

        # Body height ratio
        if kp.spine_mid and kp.left_front_paw:
            features['body_height_ratio'] = abs(
                kp.left_front_paw[1] - kp.spine_mid[1]
            ) / max(abs(kp.left_front_paw[0] - kp.spine_mid[0]), 1)

        return features

    @staticmethod
    def _angle_between(p1: tuple, p2: tuple, p3: tuple) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return float(np.degrees(angle))

    def reset(self):
        """Reset temporal state (call when switching to a different dog)."""
        self._prev_keypoints = None
        self._prev_tail_positions = []
