"""
Dog Talk - Behavior Engine
Multimodal fusion of vision and audio signals to produce final analysis.
"""
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from vision.body_language import BodyLanguageSignals
from audio.sound_classifier import VocalizationResult
from .knowledge_base import DogKnowledgeBase, BehaviorMatch
from .llm_interpreter import LLMInterpreter

logger = logging.getLogger(__name__)


@dataclass
class EmotionScore:
    """A single emotion with confidence score."""
    name: str
    confidence: float
    emoji: str
    color: str


@dataclass
class AnalysisResult:
    """Complete analysis result sent back to the app."""
    dog_detected: bool = False
    
    # Emotions (sorted by confidence)
    emotions: list = field(default_factory=list)
    
    # Body language breakdown
    body_language: dict = field(default_factory=dict)
    
    # Vocalization
    vocalization: dict = field(default_factory=dict)
    
    # Natural language interpretation
    interpretation: str = ""
    
    # Behavior prediction
    prediction: dict = field(default_factory=dict)
    
    # Metadata
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": "analysis",
            "dog_detected": self.dog_detected,
            "emotions": [
                {"name": e.name, "confidence": round(e.confidence, 2),
                 "emoji": e.emoji, "color": e.color}
                for e in self.emotions
            ],
            "body_language": self.body_language,
            "vocalization": self.vocalization,
            "interpretation": self.interpretation,
            "prediction": self.prediction,
            "timestamp": self.timestamp,
            "processing_time_ms": round(self.processing_time_ms, 1)
        }


class BehaviorEngine:
    """
    Fuses vision and audio signals to produce a complete behavior analysis.
    Uses the knowledge base for signal interpretation and the LLM for
    natural language generation.
    """

    # Default emotion emojis and colors
    EMOTION_DISPLAY = {
        "playful": ("🎾", "#4CAF50"),
        "extremely_happy": ("🥰", "#FFD700"),
        "happy": ("😊", "#66BB6A"),
        "excited": ("🤩", "#FFC107"),
        "content": ("😌", "#81C784"),
        "calm": ("😊", "#81C784"),
        "relaxed": ("😌", "#A5D6A7"),
        "curious": ("🤔", "#42A5F5"),
        "alert": ("⚡", "#FFA726"),
        "focused": ("👀", "#FFA726"),
        "anxious": ("😰", "#EF5350"),
        "fearful": ("😨", "#E53935"),
        "stressed": ("😣", "#F44336"),
        "aggressive": ("⚠️", "#B71C1C"),
        "warning": ("⚠️", "#F44336"),
        "submissive": ("🙏", "#9575CD"),
        "frustrated": ("😤", "#FF7043"),
        "protective": ("🛡️", "#5C6BC0"),
        "pain": ("🤕", "#D32F2F"),
        "dominant": ("👑", "#FF8F00"),
        "lonely": ("😢", "#78909C"),
        "unknown": ("❓", "#9E9E9E"),
    }

    def __init__(self, knowledge_base: DogKnowledgeBase,
                 llm_interpreter: Optional[LLMInterpreter] = None):
        self.kb = knowledge_base
        self.llm = llm_interpreter
        self._emotion_history: list = []
        self._max_history = 30

    def analyze(self, body_signals: Optional[BodyLanguageSignals],
                vocal_result: Optional[VocalizationResult],
                image_base64: str = None) -> AnalysisResult:
        """
        Produce a complete behavior analysis from vision + audio signals.
        
        Args:
            body_signals: Body language signals from vision pipeline
            vocal_result: Vocalization result from audio pipeline
            image_base64: Optional base64 image for LLM vision
            
        Returns:
            Complete AnalysisResult
        """
        start_time = time.time()
        result = AnalysisResult(timestamp=time.time())

        # Determine if a dog is actually detected (seen or heard)
        has_dog_vision = body_signals is not None
        has_dog_audio = vocal_result is not None and vocal_result.is_dog_sound

        if not has_dog_vision and not has_dog_audio:
            result.dog_detected = False
            result.interpretation = "No dog detected in the frame. Point your camera at a dog to start analyzing!"
            return result

        result.dog_detected = True

        # Extract signal strings
        tail_pos = body_signals.tail_position if body_signals else "unknown"
        tail_mov = body_signals.tail_movement if body_signals else "unknown"
        ear_pos = body_signals.ear_position if body_signals else "unknown"
        posture = body_signals.posture if body_signals else "unknown"
        voc_type = vocal_result.vocalization_type if vocal_result else "no_vocalization"
        hackles = body_signals.hackles_raised if body_signals else False

        # Match against knowledge base
        behavior_match = self.kb.match_behavior(
            tail_position=tail_pos,
            tail_movement=tail_mov,
            ear_position=ear_pos,
            posture=posture,
            vocalization=voc_type,
            hackles_raised=hackles
        )

        # Build emotion scores
        result.emotions = self._compute_emotions(
            body_signals, vocal_result, behavior_match
        )

        # Build body language breakdown
        result.body_language = self._build_body_language_dict(
            body_signals, tail_pos, tail_mov, ear_pos, posture
        )

        # Build vocalization info
        result.vocalization = self._build_vocalization_dict(vocal_result, voc_type)

        # Build prediction
        result.prediction = self._build_prediction(behavior_match)

        # Generate interpretation (LLM or knowledge base fallback)
        kb_interpretation = behavior_match.interpretation if behavior_match else ""
        
        if self.llm and self.llm.available:
            result.interpretation = self.llm.interpret(
                body_language=result.body_language,
                vocalization=result.vocalization,
                knowledge_interpretation=kb_interpretation,
                image_base64=image_base64
            )
        else:
            result.interpretation = kb_interpretation or self._generate_basic_interpretation(
                result.body_language, result.vocalization, result.emotions
            )

        result.processing_time_ms = (time.time() - start_time) * 1000

        # Update history
        self._emotion_history.append(result.emotions)
        if len(self._emotion_history) > self._max_history:
            self._emotion_history.pop(0)

        return result

    def _compute_emotions(self, body_signals: Optional[BodyLanguageSignals],
                          vocal_result: Optional[VocalizationResult],
                          behavior_match: Optional[BehaviorMatch]) -> list[EmotionScore]:
        """Compute emotion scores from all available signals."""
        emotion_scores = {}

        # From knowledge base match
        if behavior_match:
            emotion = behavior_match.emotion
            conf = behavior_match.confidence
            emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), conf)

        # From body language signals
        if body_signals:
            # Tail signals
            tail_info = self.kb.get_tail_meaning(
                body_signals.tail_position, body_signals.tail_movement
            )
            for emotion in tail_info.get("emotions", []):
                base_conf = tail_info.get("confidence", 0.5) * body_signals.tail_confidence
                emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), base_conf)

            # Ear signals
            ear_info = self.kb.get_ear_meaning(body_signals.ear_position)
            for emotion in ear_info.get("emotions", []):
                base_conf = ear_info.get("confidence", 0.5) * body_signals.ear_confidence
                emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), base_conf * 0.8)

        # From vocalization
        if vocal_result and vocal_result.is_dog_sound:
            voc_info = self.kb.get_vocalization_meaning(vocal_result.vocalization_type)
            for emotion in voc_info.get("emotions", []):
                base_conf = voc_info.get("confidence", 0.5) * vocal_result.confidence
                emotion_scores[emotion] = max(emotion_scores.get(emotion, 0), base_conf)

        # Normalize and sort
        if not emotion_scores:
            emotion_scores["unknown"] = 0.5

        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}

        # Convert to EmotionScore objects
        emotions = []
        for emotion, confidence in sorted(emotion_scores.items(), key=lambda x: -x[1]):
            emoji, color = self.EMOTION_DISPLAY.get(emotion, ("❓", "#9E9E9E"))
            emotions.append(EmotionScore(
                name=emotion.replace("_", " ").title(),
                confidence=confidence,
                emoji=emoji,
                color=color
            ))

        return emotions[:5]  # Top 5 emotions

    def _build_body_language_dict(self, body_signals: Optional[BodyLanguageSignals],
                                   tail_pos: str, tail_mov: str,
                                   ear_pos: str, posture: str) -> dict:
        """Build the body language breakdown dictionary."""
        tail_info = self.kb.get_tail_meaning(tail_pos, tail_mov)
        ear_info = self.kb.get_ear_meaning(ear_pos)
        
        posture_meanings = {
            "play_bow": "Inviting play — front down, rear up!",
            "standing_relaxed": "Standing comfortably, at ease",
            "standing_alert": "Standing tall, focused on something",
            "forward_lean": "Leaning forward — assertive or about to move",
            "cowering": "Making themselves small — feeling scared",
            "belly_up": "Showing belly — trust or submission",
            "stiff_freeze": "Frozen still — processing or warning",
            "sitting_relaxed": "Sitting comfortably",
            "lying_relaxed": "Lying down, very comfortable",
        }

        return {
            "tail": {
                "position": tail_pos,
                "movement": tail_mov,
                "meaning": tail_info.get("description", ""),
                "label": tail_info.get("label", tail_pos.title())
            },
            "ears": {
                "position": ear_pos,
                "meaning": ear_info.get("description", ""),
                "label": ear_info.get("label", ear_pos.title())
            },
            "posture": {
                "stance": posture,
                "meaning": posture_meanings.get(posture, posture.replace("_", " ").title()),
                "label": posture.replace("_", " ").title()
            },
            "hackles": {
                "raised": body_signals.hackles_raised if body_signals else False
            },
            "body_tension": body_signals.body_tension if body_signals else 0.0,
            "head_height": body_signals.head_height if body_signals else "neutral",
            "weight_forward": body_signals.weight_forward if body_signals else 0.5
        }

    def _build_vocalization_dict(self, vocal_result: Optional[VocalizationResult],
                                  voc_type: str) -> dict:
        """Build vocalization info dictionary."""
        voc_info = self.kb.get_vocalization_meaning(voc_type)
        
        return {
            "type": voc_type,
            "label": voc_info.get("label", voc_type.replace("_", " ").title()),
            "meaning": voc_info.get("meaning", ""),
            "confidence": vocal_result.confidence if vocal_result else 0.0,
            "is_dog_sound": vocal_result.is_dog_sound if vocal_result else False
        }

    def _build_prediction(self, behavior_match: Optional[BehaviorMatch]) -> dict:
        """Build behavior prediction dictionary."""
        if behavior_match:
            warning_info = self.kb.get_warning_info(behavior_match.warning_level)
            return {
                "action": behavior_match.predicted_action,
                "confidence": round(behavior_match.action_confidence, 2),
                "warning_level": behavior_match.warning_level,
                "warning_color": warning_info.get("color", "#4CAF50"),
                "warning_description": warning_info.get("description", ""),
                "safe_to_approach": behavior_match.safe_to_approach
            }
        
        return {
            "action": "Continue observing for more signals...",
            "confidence": 0.3,
            "warning_level": "none",
            "warning_color": "#4CAF50",
            "warning_description": "Insufficient data for prediction",
            "safe_to_approach": True
        }

    def _generate_basic_interpretation(self, body_language: dict,
                                        vocalization: dict,
                                        emotions: list) -> str:
        """Generate basic text interpretation without LLM."""
        parts = []
        
        if emotions:
            top_emotion = emotions[0]
            parts.append(f"This dog appears to be feeling {top_emotion.name.lower()}.")

        tail = body_language.get("tail", {})
        if tail.get("meaning"):
            parts.append(f"Their tail is {tail.get('label', '').lower()} — {tail['meaning'].lower()}.")

        ears = body_language.get("ears", {})
        if ears.get("meaning"):
            parts.append(f"Ears are {ears.get('label', '').lower()}.")

        posture = body_language.get("posture", {})
        if posture.get("meaning"):
            parts.append(posture["meaning"])

        voc = vocalization
        if voc.get("is_dog_sound") and voc.get("meaning"):
            parts.append(voc["meaning"])

        return " ".join(parts) if parts else "Analyzing dog behavior..."

    def reset(self):
        """Reset analysis state."""
        self._emotion_history = []
