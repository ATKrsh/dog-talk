"""
Dog Talk Server Configuration
"""
import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Main server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Vision
    yolo_model_path: str = ""
    pose_classifier_path: str = os.path.join(os.path.dirname(__file__), "models", "pose_classifier.joblib")
    detection_confidence: float = 0.5
    pose_confidence: float = 0.3
    input_size: int = 640

    def __post_init__(self):
        # Dynamically locate the latest trained dog pose model
        base_dir = os.path.join(os.path.dirname(__file__), "runs", "pose")
        self.yolo_model_path = os.path.join(os.path.dirname(__file__), "runs", "pose", "dog_talk_pose_model", "weights", "best.pt")
        if os.path.exists(base_dir):
            candidates = []
            for d in os.listdir(base_dir):
                if d.startswith("dog_talk_pose_model"):
                    path = os.path.join(base_dir, d, "weights", "best.pt")
                    if os.path.exists(path):
                        suffix = d[len("dog_talk_pose_model"):]
                        num = 0
                        if suffix.startswith("-"):
                            try:
                                num = int(suffix[1:])
                            except ValueError:
                                pass
                        candidates.append((num, path))
            if candidates:
                self.yolo_model_path = max(candidates, key=lambda x: x[0])[1]
    
    # Audio
    yamnet_model_url: str = "https://tfhub.dev/google/yamnet/1"
    sound_classifier_path: str = os.path.join(os.path.dirname(__file__), "models", "sound_classifier.joblib")
    audio_sample_rate: int = 16000
    audio_chunk_duration: float = 1.0  # seconds
    
    # LLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"
    llm_enabled: bool = True
    llm_timeout: float = 10.0  # seconds
    
    # Performance
    max_fps: int = 10  # Max frames per second to process
    frame_skip: int = 3  # Process every Nth frame for analysis
    max_clients: int = 5
    
    # Knowledge base
    knowledge_base_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "dog_behavior_dataset"
    )


@dataclass
class ModelPaths:
    """Paths to all model files."""
    base_dir: str = os.path.join(os.path.dirname(__file__), "models")
    
    @property
    def yolo_pose(self) -> str:
        return os.path.join(self.base_dir, "yolov8n-pose-dog.pt")
    
    @property
    def pose_classifier(self) -> str:
        return os.path.join(self.base_dir, "pose_classifier.joblib")
    
    @property
    def sound_classifier(self) -> str:
        return os.path.join(self.base_dir, "sound_classifier.joblib")


# Global config instance
config = ServerConfig()
model_paths = ModelPaths()
