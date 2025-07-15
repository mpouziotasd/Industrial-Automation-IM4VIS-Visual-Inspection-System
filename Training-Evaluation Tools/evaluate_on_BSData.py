from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
models_list = ['yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt', 
               'yolo11m-seg.pt', 'yolo11l-seg.pt', 'yolo11x-seg.pt',
               'yolov12m-seg.pt', 'yolov12l-seg.pt', 'yolov12x-seg.pt']

for model_name in models_list:
    model = YOLO(model_name)
    model_name = model_name.split('.')[0]
    weights_path = f"runs/segment/{model_name}-BSData/weights/best.pt"
    model = YOLO(weights_path)
    metrics = model.val()
    print("====================================")
    print(f"Model Name: {model_name}")
    print(f"mAP50-95: {metrics.box.map}")  # mAP50-95
    print(f"mAP50: {metrics.box.map50}")  # mAP50
    print("====================================")