from ultralytics import YOLO
import os

model_path = "models/yolov8m-seg.pt"  # Path to the pre-trained YOLOv8 model
test_path = "../../datasets/BSData/test/images"
out_dir = "predict_outputs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

images = os.listdir(test_path)

def load_model(model_path):
    model = YOLO(model_path)
    return model

model = load_model(model_path)


for image in images:
    img_src = f"{test_path}/{image}"
    results = model(img_src, save=True, project=out_dir, exist_ok=True)[0]
    
    