import sys
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import time
import os

class RedirectStdout:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        # Update UI every chunk to avoid freezing, but safe via after
        self.text_widget.after(10, self._update_text, string)

    def _update_text(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

def run_training():
    try:
        from ultralytics import YOLO
        import torch

        print("==================================================")
        print("🐕 Dog Talk - Custom Model Training Setup (GPU)")
        print("==================================================")
        
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Using device: {'GPU (CUDA) 🚀' if device == '0' else 'CPU 🐢'}")
        print("==================================================\n")

        # Load the base nano pose model
        model = YOLO("yolov8n-pose.pt")

        print("Starting training... (Grab a coffee, this will take a while!)")
        
        # Training YOLO in a separate thread inside Tkinter
        results = model.train(
            data="dog-pose.yaml", 
            epochs=50, 
            imgsz=640,
            device=device,
            patience=10,
            name="dog_talk_pose_model"
        )

        print("\n✅ Training Complete!")
        print(f"Your new custom dog model is saved at: {results.save_dir}/weights/best.pt")
        print("You can close this window now.")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")

def main():
    root = tk.Tk()
    root.title("Dog Talk - Live AI Training Progress")
    root.geometry("800x600")
    root.configure(bg="#1e1e1e")

    label = tk.Label(root, text="Training AI Brain (Dog Anatomy)", font=("Helvetica", 16, "bold"), bg="#1e1e1e", fg="#ffffff")
    label.pack(pady=10)

    # Scrolled text widget for console output
    st = ScrolledText(root, bg="#000000", fg="#00ff00", font=("Consolas", 10))
    st.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    # Redirect stdout and stderr
    redirector = RedirectStdout(st)
    sys.stdout = redirector
    sys.stderr = redirector

    # Run training in a background thread so UI stays responsive
    threading.Thread(target=run_training, daemon=True).start()

    root.mainloop()

if __name__ == "__main__":
    main()
