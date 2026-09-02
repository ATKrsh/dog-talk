# Model Files

This directory holds trained model files. They are **not** tracked in git because they are large binary files.

## How to get the models

### Option 1: Download pre-trained (recommended)
```bash
cd server
python models/download_models.py
```

### Option 2: Train from scratch
```bash
cd data/training
python train_pose_classifier.py
python train_sound_classifier.py
python evaluate_models.py
```

## Model Files

| File | Description | Size |
|------|-------------|------|
| `yolov8n-pose-dog.pt` | YOLOv8-nano pose model for dog keypoints | ~12MB |
| `pose_classifier.joblib` | Body language classifier (from keypoint features) | ~1MB |
| `sound_classifier.joblib` | Dog vocalization classifier (YAMNet embeddings → class) | ~2MB |
