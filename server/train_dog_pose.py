import sys
import os
import time
import urllib.request
import zipfile
import re

class RollingFileLogger:
    def __init__(self, filepath, max_history=20, total_epochs=50):
        self.filepath = filepath
        self.max_history = max_history
        self.total_epochs = total_epochs
        self.terminal = sys.stdout
        self.history = []
        self.start_time = None
        self.estimated_remaining = "Calculating..."
        self._write_file()

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode('ascii', 'ignore').decode('ascii'))

        if not message:
            return

        # Track when training actually starts to calculate elapsed time accurately
        if "Starting training for" in message and self.start_time is None:
            self.start_time = time.time()

        parts = message.split('\r')
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            if i > 0 and self.history:
                self.history[-1] = part
            else:
                self.history.append(part)

            # Try to calculate remaining time
            self._update_estimate(part)

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self._write_file()

    def _update_estimate(self, line):
        if self.start_time is None:
            return

        # Clean line to make regex matching robust
        clean_line = line.replace('[K', '').strip()
        
        # Look for patterns like "1/50" followed by VRAM usage "2.18G"
        # Format of YOLO progress bar: "   1/50      2.35G   ...   640: 10% ━━━ 42/424"
        epoch_match = re.search(r'(\d+)/(\d+)\s+\d+\.?\d*G', clean_line)
        percent_match = re.search(r'(\d+)%\s+.*?\d+/\d+', clean_line)
        
        if epoch_match:
            try:
                curr_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                
                percent = 0
                if percent_match:
                    percent = int(percent_match.group(1))
                
                # Calculate fractional progress (e.g. 1.45 out of 50)
                progress = (curr_epoch - 1) + (percent / 100.0)
                
                if progress > 0.01:
                    elapsed = time.time() - self.start_time
                    total_estimated = elapsed / progress
                    remaining = total_estimated * (total_epochs - progress)
                    
                    # Convert remaining seconds to hh:mm:ss format
                    hours = int(remaining // 3600)
                    minutes = int((remaining % 3600) // 60)
                    seconds = int(remaining % 60)
                    
                    if hours > 0:
                        self.estimated_remaining = f"{hours}h {minutes}m {seconds}s"
                    else:
                        self.estimated_remaining = f"{minutes}m {seconds}s"
            except Exception:
                pass

    def _write_file(self):
        header = [
            "==================================================",
            "🐕 Dog Talk - Custom Model Training (GPU)",
            "==================================================",
            "This file updates automatically. No scrolling needed!",
            f"ESTIMATED TIME REMAINING: {self.estimated_remaining}",
            "==================================================\n"
        ]
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(header) + "\n")
                clean_history = []
                for line in self.history:
                    clean_line = line.replace('[K', '').strip()
                    if clean_line:
                        clean_history.append(clean_line)
                f.write("\n".join(clean_history) + "\n")
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = f"\rDownloading dataset: {percent:5.1f}% [{read_so_far / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB]"
        sys.stdout.write(s)
    else:
        sys.stdout.write(f"\rDownloading dataset: {read_so_far / 1024 / 1024:.1f} MB")

def ensure_dataset():
    dest_dir = "C:\\Dump\\datasets"
    extract_path = os.path.join(dest_dir, "dog-pose")
    
    # Check if images/train folder exists
    train_images_path = os.path.join(extract_path, "images", "train")
    if os.path.isdir(train_images_path):
        print("Dataset already exists and is validated.")
        return

    print("Dataset not found or incomplete. Downloading now...")
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "dog-pose.zip")
    url = "https://ultralytics.com/assets/dog-pose.zip"
    
    urllib.request.urlretrieve(url, zip_path, download_progress)
    print("\nExtraction started...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    print(f"Extraction completed to {extract_path}")
    try:
        os.remove(zip_path)
    except:
        pass

def main():
    progress_file = "training_progress.txt"
    sys.stdout = RollingFileLogger(progress_file, max_history=15, total_epochs=50)
    sys.stderr = sys.stdout

    try:
        ensure_dataset()
        
        import torch
        from ultralytics import YOLO

        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"\nCUDA Available: {torch.cuda.is_available()}")
        print(f"Using device: {'GPU (CUDA) 🚀' if device == '0' else 'CPU 🐢'}")
        print("==================================================\n")

        # Load the base model
        model = YOLO("yolov8n-pose.pt")

        print("Starting training on the Dog Pose dataset...")
        results = model.train(
            data="dog-pose.yaml", 
            epochs=50, 
            imgsz=640,
            device=device,
            patience=10,
            name="dog_talk_pose_model"
        )

        print("\nTraining Complete!")
        print(f"Your new custom dog model is saved at: {results.save_dir}/weights/best.pt")

    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main()
