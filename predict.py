from ultralytics import YOLO


model = YOLO("D:/intezet/Bogi/Yolo/runs/detect/weights/bestv8m_1024.pt")
results = model("D:/intezet/Bogi/Yolo/test", save=True, imgsz=1024, iou=0.5, conf=0.1)