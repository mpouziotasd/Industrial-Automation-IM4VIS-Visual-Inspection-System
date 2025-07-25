import argparse
import os
import cv2
import numpy as np
import time
from ultralytics import solutions
from utils.model_utilities import load_model, detect
from utils.img_utils import draw_text, draw_data, draw_counting_region

def main():
    parser = argparse.ArgumentParser(description="Object Counting System")
    parser.add_argument("--video", required=True, 
                        help="Path to input video file")
    parser.add_argument("--model", required=True, 
                        help="Model name (without .pt extension)")
    parser.add_argument("--output", 
                        help="Path to output video file (optional)")
    parser.add_argument("--classes", type=int, nargs='+', default=[0], 
                        help="List of class IDs to detect (default: 0)")
    
    args = parser.parse_args()

    cap = None
    model = None
    counter = None
    region_points = []
    prev_time = time.time()
    writer = None
    print(args.output)
    try:
        cap = cv2.VideoCapture(args.video)
        

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {args.video}")

        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")
        model = load_model(args.model)[0]

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame from video")
        
        region_points, _ = draw_counting_region(frame)
        if region_points is None:
            print("Region selection canceled. Exiting.")
            return

        counter = solutions.ObjectCounter(
            region=region_points,
            model=args.model,
            classes=args.classes,
            show=False,
            show_conf=False,
            show_labels=False,
            verbose=False,
            show_in=False,
            show_out=False,
            line_width=2
        )

        if args.output:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        print("Starting object counting. Press 'q' to exit...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            delta = current_time - prev_time
            prev_time = current_time
            delay_ms = delta * 1000
            
            results = detect(model, frame, device="0")[0]
            
            counter_results = counter(frame.copy())
            print(counter_results, type(counter_results))

            bboxes = results.boxes.xyxy
            clss = results.boxes.cls
            if bboxes is not None:
                processed_frame = draw_data(frame, bboxes, clss, region_points)
                draw_text(frame, text=f"Delay: {delay_ms:.2f}ms")
                try:
                    draw_text(frame, text=f'Count: {counter_results.in_count}', position=(20, 80))
                except:
                    draw_text(frame, text=f'Count: 0', position=(20, 50))
                
            else:
                processed_frame = frame
            
            if writer:
                writer.write(processed_frame)
                print("WRITING...")
            else:
                cv2.imshow("Object Counting", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()