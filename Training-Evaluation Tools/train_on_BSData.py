from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
models_list = ['yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt', 
               'yolo11m-seg.pt', 'yolo11l-seg.pt', 'yolo11x-seg.pt',
               'yolov12m-seg.pt', 'yolov12l-seg.pt', 'yolov12x-seg.pt']

for model_name in models_list:
        model = YOLO(model_name) 
        model_name = model_name.split('.')[0]
        model.train(
                data='/home/mpouziotasd/Documents/Personal/industry 4.0/Visual Inspection System/datasets/BSData/data.yaml',
                epochs=80,
                batch=0.8,
                name=f'{model_name}-BSData'
                )