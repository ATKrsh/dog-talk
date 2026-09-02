"""
Dog Talk - Knowledge Base
Loads and queries the curated dog behavior knowledge base.
"""
import json
import os
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BehaviorMatch:
    """A matched behavior pattern from the knowledge base."""
    pattern_id: str
    name: str
    emoji: str
    emotion: str
    confidence: float
    interpretation: str
    predicted_action: str
    action_confidence: float
    warning_level: str
    safe_to_approach: bool
    matched_signals: list


class DogKnowledgeBase:
    """
    Manages the curated dog behavior knowledge base.
    Matches observed signals to known behavior patterns.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.body_language = {}
        self.vocalizations = {}
        self.predictions = {}
        self.emotion_labels = {}
        self.warning_levels = {}
        self._load_data()

    def _load_data(self):
        """Load all knowledge base JSON files."""
        try:
            body_path = os.path.join(self.data_dir, "body_language.json")
            if os.path.exists(body_path):
                with open(body_path, 'r', encoding='utf-8') as f:
                    self.body_language = json.load(f)
                logger.info(f"Loaded body language DB: {len(self.body_language.get('tail_signals', {}))} tail signals")

            vocab_path = os.path.join(self.data_dir, "vocalizations.json")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.vocalizations = json.load(f)
                logger.info(f"Loaded vocalizations DB: {len(self.vocalizations.get('vocalization_types', {}))} types")

            pred_path = os.path.join(self.data_dir, "predictions.json")
            if os.path.exists(pred_path):
                with open(pred_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.predictions = data.get("signal_combinations", [])
                    self.emotion_labels = data.get("emotion_labels", {})
                    self.warning_levels = data.get("warning_levels", {})
                logger.info(f"Loaded predictions DB: {len(self.predictions)} patterns")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")

    def match_behavior(self, tail_position: str, tail_movement: str,
                       ear_position: str, posture: str,
                       vocalization: str, hackles_raised: bool = False) -> Optional[BehaviorMatch]:
        """
        Match observed signals to known behavior patterns.
        
        Returns the best matching pattern with confidence adjustment.
        """
        if not self.predictions:
            return None

        best_match = None
        best_score = 0.0

        observed = {
            "tail": f"{tail_position}_{tail_movement}" if tail_movement != "unknown" else tail_position,
            "ears": ear_position,
            "posture": posture,
            "vocalization": vocalization,
            "hackles": "raised_full" if hackles_raised else "not_raised"
        }

        for pattern in self.predictions:
            score, matched = self._score_pattern(pattern, observed)
            if score > best_score:
                best_score = score
                pred = pattern.get("prediction", {})
                best_match = BehaviorMatch(
                    pattern_id=pattern.get("id", "unknown"),
                    name=pattern.get("name", "Unknown"),
                    emoji=pattern.get("emoji", "❓"),
                    emotion=pattern.get("emotion", "unknown"),
                    confidence=min(score * pattern.get("base_confidence", 0.5), 1.0),
                    interpretation=pattern.get("interpretation", ""),
                    predicted_action=pred.get("action", "Behavior unclear"),
                    action_confidence=pred.get("confidence", 0.5) * score,
                    warning_level=pred.get("warning_level", "none"),
                    safe_to_approach=pred.get("safe_to_approach", True),
                    matched_signals=matched
                )

        return best_match

    def _score_pattern(self, pattern: dict, observed: dict) -> tuple[float, list]:
        """Score how well observed signals match a pattern."""
        signals = pattern.get("signals", {})
        required = pattern.get("required_signals", [])
        
        total_weight = 0
        match_weight = 0
        matched_signals = []

        # Check each signal category
        for category, possible_values in signals.items():
            if category == "body" or category == "eyes" or category == "mouth":
                continue  # Skip categories we can't directly observe from keypoints

            observed_value = observed.get(category, "unknown")
            if observed_value == "unknown":
                continue

            weight = 2.0 if category in ["posture", "tail"] else 1.5 if category == "vocalization" else 1.0
            total_weight += weight

            # Check if observed value matches any of the pattern's possible values
            if isinstance(possible_values, list):
                for pv in possible_values:
                    if pv in observed_value or observed_value in pv:
                        match_weight += weight
                        matched_signals.append(f"{category}:{observed_value}")
                        break
            elif isinstance(possible_values, str):
                if possible_values in observed_value or observed_value in possible_values:
                    match_weight += weight
                    matched_signals.append(f"{category}:{observed_value}")

        if total_weight == 0:
            return 0.0, []

        # Check required signals
        required_met = True
        for req in required:
            req_parts = req.split(":")
            if len(req_parts) == 2:
                req_cat = req_parts[0]
                req_vals = req_parts[1].split("|")
                obs_val = observed.get(req_cat, "unknown")
                if not any(rv in obs_val or obs_val in rv for rv in req_vals):
                    required_met = False
                    break

        if not required_met:
            return 0.0, []

        score = match_weight / total_weight
        return score, matched_signals

    def get_tail_meaning(self, position: str, movement: str) -> dict:
        """Get meaning of a specific tail signal."""
        tail_signals = self.body_language.get("tail_signals", {})
        key = f"{position}_{movement}"
        return tail_signals.get(key, {
            "label": f"{position.title()} tail",
            "description": "Unknown tail signal",
            "emotions": ["unknown"],
            "confidence": 0.3
        })

    def get_ear_meaning(self, position: str) -> dict:
        """Get meaning of ear position."""
        ear_signals = self.body_language.get("ear_signals", {})
        # Map position to key
        position_map = {
            "forward": "forward_erect",
            "slightly_forward": "slightly_forward",
            "natural": "relaxed_natural",
            "slightly_back": "slightly_back",
            "flat": "pinned_flat",
            "asymmetric": "one_forward_one_back"
        }
        key = position_map.get(position, position)
        return ear_signals.get(key, {
            "label": f"{position.title()} ears",
            "description": "Unknown ear position",
            "emotions": ["unknown"],
            "confidence": 0.3
        })

    def get_vocalization_meaning(self, voc_type: str) -> dict:
        """Get meaning of a vocalization type."""
        voc_types = self.vocalizations.get("vocalization_types", {})
        return voc_types.get(voc_type, {
            "label": voc_type.replace("_", " ").title(),
            "meaning": "Unknown vocalization",
            "emotions": ["unknown"],
            "confidence": 0.3
        })

    def get_emotion_info(self, emotion: str) -> dict:
        """Get display info for an emotion label."""
        return self.emotion_labels.get(emotion, {
            "color": "#9E9E9E",
            "icon": "❓",
            "valence": "neutral"
        })

    def get_warning_info(self, level: str) -> dict:
        """Get display info for a warning level."""
        return self.warning_levels.get(level, {
            "color": "#9E9E9E",
            "description": "Unknown warning level"
        })
