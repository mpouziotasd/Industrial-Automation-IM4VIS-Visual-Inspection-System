from ultralytics import YOLO

models_list = ['yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 
               'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt', 
               'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt']

for model_name in models_list:
    model = YOLO(model_name) 
    model_name = model_name.split('.')[0]
    results = model.train(data="medical-pills.yaml", epochs=30, imgsz=640, name=f'{model_name}-MedicalPills', batch=0.95)