"""
Dog Talk - Audio Pipeline: Sound Classification
Uses YAMNet for audio feature extraction and classifies dog vocalizations.
"""
import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VocalizationResult:
    """Result from vocalization classification."""
    vocalization_type: str  # e.g., "excited_bark", "growl_warning"
    confidence: float
    top_classes: list  # [(class_name, confidence), ...]
    is_dog_sound: bool
    raw_embeddings: Optional[np.ndarray] = None


class SoundClassifier:
    """
    Classifies dog vocalizations using YAMNet embeddings + custom classifier.
    Falls back to YAMNet's built-in dog sound classes if custom model unavailable.
    """

    # YAMNet class indices related to dogs
    YAMNET_DOG_CLASSES = {
        67: "bark",
        68: "yip",
        69: "howl",
        70: "bow-wow",
        71: "growling",
        72: "whimper_(dog)"
    }

    # Our custom vocalization classes
    VOCALIZATION_CLASSES = [
        "alert_bark", "excited_bark", "demand_bark", "fearful_bark",
        "aggressive_bark", "lonely_bark", "whine_attention", "whine_anxious",
        "whimper_pain", "growl_warning", "growl_play", "howl",
        "yelp", "sigh", "panting_stressed", "no_vocalization"
    ]

    # Mapping from YAMNet classes to our classes (fallback)
    YAMNET_TO_CUSTOM = {
        "bark": "alert_bark",
        "yip": "excited_bark",
        "howl": "howl",
        "bow-wow": "excited_bark",
        "growling": "growl_warning",
        "whimper_(dog)": "whine_anxious"
    }

    def __init__(self, custom_model_path: str = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.yamnet_model = None
        self.custom_classifier = None
        self.class_names = None
        self._load_models(custom_model_path)

    def _load_models(self, custom_model_path: str = None):
        """Load YAMNet and optional custom classifier."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            logger.info("Loading YAMNet model from TensorFlow Hub...")
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            
            # Load class names
            class_map_path = self.yamnet_model.class_map_path().numpy().decode("utf-8")
            import csv
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self.class_names = [row["display_name"] for row in reader]
            
            logger.info("YAMNet loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load YAMNet: {e}")
            logger.info("Audio classification will use basic analysis")

        # Try to load custom fine-tuned classifier
        if custom_model_path:
            try:
                import joblib
                self.custom_classifier = joblib.load(custom_model_path)
                logger.info(f"Loaded custom sound classifier from {custom_model_path}")
            except Exception as e:
                logger.warning(f"Could not load custom classifier: {e}")

    def classify(self, audio_data: np.ndarray) -> VocalizationResult:
        """
        Classify audio data for dog vocalizations.
        
        Args:
            audio_data: Audio samples as float32 numpy array (16kHz mono)
            
        Returns:
            VocalizationResult with classification
        """
        if len(audio_data) == 0:
            return VocalizationResult(
                vocalization_type="no_vocalization",
                confidence=1.0,
                top_classes=[("no_vocalization", 1.0)],
                is_dog_sound=False
            )

        # Check if raw audio is mostly silence
        audio_data = audio_data.astype(np.float32)
        raw_rms = np.sqrt(np.mean(audio_data ** 2))
        logger.info(f"Audio raw RMS: {raw_rms:.5f}")
        if raw_rms < 0.015:  # Noise floor threshold (~ -36dB)
            return VocalizationResult(
                vocalization_type="no_vocalization",
                confidence=0.95,
                top_classes=[("no_vocalization", 0.95)],
                is_dog_sound=False
            )

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Use YAMNet if available
        if self.yamnet_model is not None:
            return self._classify_with_yamnet(audio_data)
        
        # Fallback: basic spectral analysis
        return self._classify_basic(audio_data)

    def _classify_with_yamnet(self, audio_data: np.ndarray) -> VocalizationResult:
        """Classify using YAMNet embeddings."""
        import tensorflow as tf
        
        try:
            # Run YAMNet
            scores, embeddings, spectrogram = self.yamnet_model(audio_data)
            scores = scores.numpy()
            embeddings = embeddings.numpy()

            # Average scores across time frames
            avg_scores = np.mean(scores, axis=0)

            # Check for dog-related sounds
            dog_score = 0.0
            best_dog_class = "no_vocalization"
            
            for class_idx, class_name in self.YAMNET_DOG_CLASSES.items():
                if class_idx < len(avg_scores):
                    score = avg_scores[class_idx]
                    if score > dog_score:
                        dog_score = score
                        best_dog_class = class_name

            is_dog = dog_score > 0.1

            # If custom classifier is available, use it
            if self.custom_classifier is not None and is_dog:
                avg_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
                try:
                    pred_proba = self.custom_classifier.predict_proba(avg_embedding)[0]
                    pred_class_idx = np.argmax(pred_proba)
                    pred_confidence = pred_proba[pred_class_idx]
                    
                    top_indices = np.argsort(pred_proba)[::-1][:5]
                    top_classes = [
                        (self.VOCALIZATION_CLASSES[i], float(pred_proba[i]))
                        for i in top_indices
                    ]
                    
                    return VocalizationResult(
                        vocalization_type=self.VOCALIZATION_CLASSES[pred_class_idx],
                        confidence=float(pred_confidence),
                        top_classes=top_classes,
                        is_dog_sound=True,
                        raw_embeddings=avg_embedding[0]
                    )
                except Exception as e:
                    logger.warning(f"Custom classifier failed: {e}")

            # Fallback to YAMNet's own classification
            if is_dog:
                mapped_class = self.YAMNET_TO_CUSTOM.get(best_dog_class, "alert_bark")
                
                # Get top-5 overall classes for context
                top_indices = np.argsort(avg_scores)[::-1][:5]
                top_classes = []
                for idx in top_indices:
                    if idx < len(self.class_names):
                        top_classes.append((self.class_names[idx], float(avg_scores[idx])))
                
                return VocalizationResult(
                    vocalization_type=mapped_class,
                    confidence=float(dog_score),
                    top_classes=top_classes,
                    is_dog_sound=True,
                    raw_embeddings=np.mean(embeddings, axis=0) if embeddings.size > 0 else None
                )

            # Not a dog sound
            top_idx = np.argmax(avg_scores)
            return VocalizationResult(
                vocalization_type="no_vocalization",
                confidence=1.0 - float(dog_score),
                top_classes=[
                    (self.class_names[i] if i < len(self.class_names) else f"class_{i}",
                     float(avg_scores[i]))
                    for i in np.argsort(avg_scores)[::-1][:3]
                ],
                is_dog_sound=False
            )

        except Exception as e:
            logger.error(f"YAMNet classification error: {e}")
            return self._classify_basic(audio_data)

    def _classify_basic(self, audio_data: np.ndarray) -> VocalizationResult:
        """
        Basic spectral analysis fallback when YAMNet is unavailable.
        Uses simple frequency and energy features.
        """
        from scipy import signal as scipy_signal

        # Compute power spectral density
        freqs, psd = scipy_signal.welch(audio_data, fs=self.sample_rate, nperseg=1024)

        # Find dominant frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        # Energy in different bands
        low_energy = np.sum(psd[(freqs >= 50) & (freqs < 500)])
        mid_energy = np.sum(psd[(freqs >= 500) & (freqs < 2000)])
        high_energy = np.sum(psd[(freqs >= 2000) & (freqs < 8000)])
        total_energy = low_energy + mid_energy + high_energy + 1e-10

        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        # RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))

        # Simple classification based on spectral features
        if rms < 0.02:
            return VocalizationResult(
                vocalization_type="no_vocalization",
                confidence=0.80,
                top_classes=[("no_vocalization", 0.80)],
                is_dog_sound=False
            )
        elif low_ratio > 0.6 and rms > 0.1:
            vtype = "growl_warning"
            conf = 0.55
        elif high_ratio > 0.5 and rms > 0.15:
            vtype = "excited_bark"
            conf = 0.50
        elif dominant_freq > 1000 and rms > 0.05:
            vtype = "whine_anxious"
            conf = 0.45
        elif dominant_freq < 300 and rms > 0.1:
            vtype = "howl"
            conf = 0.45
        else:
            return VocalizationResult(
                vocalization_type="no_vocalization",
                confidence=0.90,
                top_classes=[("no_vocalization", 0.90)],
                is_dog_sound=False
            )

        return VocalizationResult(
            vocalization_type=vtype,
            confidence=conf,
            top_classes=[(vtype, conf), ("no_vocalization", 1.0 - conf)],
            is_dog_sound=True
        )
