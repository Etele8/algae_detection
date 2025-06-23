from ultralytics import YOLO


# Load model (can start from pretrained one)
model = YOLO("yolov11n.pt")  # or yolov11s.pt for small, yolov11m.pt for medium

# Train
model.train(data="Yolo/combined_dataset/data.yaml", epochs=50, imgsz=640, batch=16, patience=5)
# terminal commnad: yolo train model=yolo11l.pt data=D:/intezet/Bogi/Yolo/combined_dataset/data.yaml epochs=40 imgsz=640 batch=16 patience=5
# yolo detect predict model="D:\intezet\Bogi\Yolo\combined_dataset\runs\detect\train\weights\best.pt" source=D:/intezet/Bogi/test_for_prediction imgsz=640 save=True
