from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
models_list = ['yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 
               'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
               'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt']

results = []

for model_name in models_list:
    base_name = model_name.split('.')[0]
    weights_path = f"runs/detect/{base_name}-MedicalPills/weights/best.pt"
    model = YOLO(weights_path)
    metrics = model.val()
    results.append((base_name, metrics.box.map50, metrics.box.map))

print(f"\n{'Model':<15} {'mAP50':>10} {'mAP50-95':>12}")
print("-" * 40)
for name, map50, map5095 in results:
    print(f"{name:<15} {map50:>10.4f} {map5095:>12.4f}")
