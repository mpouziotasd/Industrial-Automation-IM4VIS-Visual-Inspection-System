from ultralytics import YOLO
import numpy as np
import cv2 as cv
import os
from utils.img_utils import draw_polygons, get_mask_area, calculate_roughness, extract_mask, draw_text

input_dir = "data/defects/"

out_dir = "results"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(f"{out_dir}/imgs/"):
    os.mkdir(f"{out_dir}/imgs/")

model = YOLO("models/yolov12x-BSData.pt")
imgs_src = os.listdir(input_dir)

def process_image(img):
    results = model(img)[0]
    annotated_img = img.copy()

    inspection_info = {}
    if results.masks:
        masks = results.masks.xy
        num_masks = 0
        roughness_sum = 0
        area_sum = 0
        num_masks = len(masks)
        for i, mask in enumerate(masks):
            mask = np.array(mask, np.int32)
            annotated_img = draw_polygons(img, mask)
            area = get_mask_area(mask, img.shape)
            defect_crop = extract_mask(img, mask)
            roughness = calculate_roughness(defect_crop)

            roughness_sum += roughness
            area_sum += area
        draw_text(annotated_img, f"Area: {area_sum} pxls")
        draw_text(annotated_img, f"Roughness {roughness_sum:.2f}", position=(20, 55))
        inspection_info = {
                    'num_defects': num_masks,
                    'defect_area': area_sum,
                    'defect_roughness': roughness_sum
        }
    else:
        inspection_info = "No Defects Detected"
    return annotated_img, inspection_info

detection_info = {}

for it, img_src in enumerate(imgs_src):
    file_name = img_src.split('.')[0]
    print(f"Processing Image: [{img_src}]")
    img_path = f"{input_dir}{img_src}"
    img_cv = cv.imread(img_path)
    annotated_img, inspection_info = process_image(img_cv)
    detection_info[file_name] = inspection_info
    out_img_path = f"results/imgs/{img_src}"
    
    cv.imwrite(out_img_path, annotated_img)

import json
with open(f"{out_dir}/results.json", 'w') as f:
    json.dump(detection_info, f, indent=2)
        