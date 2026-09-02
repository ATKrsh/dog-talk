"""
Dog Talk - Model Download Script
Downloads pre-trained models and prepares them for the server.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_yolo_pose():
    """Download YOLOv8n-pose model (will be fine-tuned for dogs)."""
    logger.info("Downloading YOLOv8n-pose model...")
    try:
        from ultralytics import YOLO
        
        # Download the base pose model (human keypoints)
        # This will be used as-is until custom dog-pose model is trained
        model = YOLO("yolov8n-pose.pt")
        model_path = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
        
        # Also download standard detection model for dog detection
        det_model = YOLO("yolov8n.pt")
        
        logger.info(f"✅ YOLOv8n-pose downloaded")
        logger.info(f"✅ YOLOv8n detection downloaded")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download YOLO model: {e}")
        return False


def download_yamnet():
    """Download YAMNet model from TensorFlow Hub."""
    logger.info("Downloading YAMNet model...")
    try:
        import tensorflow_hub as hub
        
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        logger.info("✅ YAMNet downloaded and cached")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download YAMNet: {e}")
        return False


def create_placeholder_classifiers():
    """
    Create placeholder classifiers that will be replaced
    after training on real data.
    """
    logger.info("Creating placeholder classifiers...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        # Pose classifier placeholder
        # Will classify body language from keypoint features
        pose_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # Train on tiny synthetic data just so it can run
        X_pose = np.random.randn(100, 10)
        y_pose = np.random.choice([
            "relaxed", "alert", "playful", "fearful",
            "aggressive", "submissive", "curious", "anxious"
        ], size=100)
        pose_clf.fit(X_pose, y_pose)
        pose_path = os.path.join(MODELS_DIR, "pose_classifier.joblib")
        joblib.dump(pose_clf, pose_path)
        logger.info(f"✅ Placeholder pose classifier saved to {pose_path}")

        # Sound classifier placeholder
        sound_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        X_sound = np.random.randn(100, 1024)  # YAMNet embedding size
        y_sound = np.random.choice([
            "alert_bark", "excited_bark", "demand_bark", "fearful_bark",
            "aggressive_bark", "lonely_bark", "whine_attention", "whine_anxious",
            "whimper_pain", "growl_warning", "growl_play", "howl",
            "yelp", "sigh", "panting_stressed", "no_vocalization"
        ], size=100)
        sound_clf.fit(X_sound, y_sound)
        sound_path = os.path.join(MODELS_DIR, "sound_classifier.joblib")
        joblib.dump(sound_clf, sound_path)
        logger.info(f"✅ Placeholder sound classifier saved to {sound_path}")

        return True
    except Exception as e:
        logger.error(f"❌ Failed to create classifiers: {e}")
        return False


def main():
    logger.info("=" * 50)
    logger.info("🐕 Dog Talk - Model Download")
    logger.info("=" * 50)

    results = {
        "YOLOv8 Pose": download_yolo_pose(),
        "YAMNet": download_yamnet(),
        "Classifiers": create_placeholder_classifiers(),
    }

    logger.info("\n" + "=" * 50)
    logger.info("Download Summary:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {status} {name}")
    
    all_ok = all(results.values())
    if all_ok:
        logger.info("\n🎉 All models ready! You can start the server.")
    else:
        logger.warning("\n⚠️ Some models failed to download.")
        logger.warning("The server will still run with available models.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
