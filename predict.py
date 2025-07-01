from ultralytics import YOLO

# Load your custom model
model = YOLO("Yolo/runs/detect/train/weights/bestv8_better.pt")  # Replace with your actual model path


# Run prediction on an image
results = model("Yolo/data/test_for_prediction", save=True) 