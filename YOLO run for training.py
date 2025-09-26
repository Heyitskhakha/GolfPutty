import os
import torch
from ultralytics import YOLO

# Ensure the main guard for multiprocessing
def train_model():
    # Check if the dataset configuration file exists
    print(os.path.exists("C:/Users/skha/PycharmProjects/GolfPutty/datasets/config.yaml"))

    # Check if CUDA is available
    print(torch.cuda.is_available())

    # Load the pre-trained model (choose from yolov8n.pt, yolov8s.pt, etc.)
    model = YOLO("yolov8s-obb.pt")
    device = "cuda"
    model.to(device)  # Move the model to GPU if available

    # Train the model
    model.train(
        data="C:/Users/skha/PycharmProjects/GolfPutty/datasets/config.yaml",
        epochs=1000,
        imgsz=640,
        project="lego_training",
        name="Red Lego",
        batch=32,
    )

if __name__ == '__main__':

    train_model()
