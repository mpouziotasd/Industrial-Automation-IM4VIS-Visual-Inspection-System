import streamlit as st
import cv2 as cv
import os
import numpy as np
import time
from ultralytics import solutions
from utils.model_utilities import load_model, detect
from utils.img_utils import draw_delay, draw_data, draw_counting_region


st.set_page_config(layout="wide")

# Session state initialization
for key, val in {
    "cap": None,
    "frozen_frame": None,
    "is_frozen": False,
    "model_path": None,
    "model": None,
    "counter": None,
    "current_mode": "Inference",
    "vid_path": None,
    "prev_time": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def load_video(vid_path):
    if st.session_state.cap is None or st.session_state.vid_path != vid_path:
        release_camera()
        cap = cv.VideoCapture(vid_path)
        if not cap.isOpened():
            st.error("Unable to load video")
            return False
        st.session_state.cap = cap
        st.session_state.vid_path = vid_path
    return True

def reset_states():
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.counter = None
    st.session_state.running = False
    st.session_state.vid_path = None
    st.session_state.points = []
    st.session_state.region_submitted = False

def release_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        st.session_state.vid_path = None


def object_counting():
    st.title("Object Counting")
    parameter_col, display_col = st.columns([1, 3])

    with parameter_col:
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        video_files = [f for f in os.listdir("videos/") if f.lower().endswith((".mp4", ".avi", ".mov"))]
        selected_video = st.selectbox("Or Select Video", video_files) if not uploaded_video else None

        if uploaded_video:
            vid_path = f"temp_{uploaded_video.name}"
            with open(vid_path, "wb") as f:
                f.write(uploaded_video.read())
        elif selected_video:
            vid_path = os.path.join("videos", selected_video)
        else:
            st.warning("Upload or select a video.")
            return

        models = [file.split('.')[0] for file in os.listdir('models/') if file.endswith('.pt')]
        model_name = st.selectbox("Select Detection Model", models)
        if model_name != st.session_state.model_path:
            st.session_state.model_path = f"models/{model_name}.pt"
            st.session_state.model = load_model(f"models/{model_name}.pt")[0]

    if not load_video(vid_path):
        return

    if st.button("Freeze/Unfreeze"):
        st.session_state.is_frozen = not st.session_state.is_frozen

    st.session_state.running = st.session_state.get("running", False)

    if st.button("Begin Detection"):
        st.session_state.running = True

    frame_placeholder = display_col.empty()
    region_points = []
    if st.session_state.running and not st.session_state.is_frozen:
        if st.session_state.prev_time is None:
            st.session_state.prev_time = time.time()

        while st.session_state.cap and st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if not st.session_state.counter:
                region_points, _ = draw_counting_region(frame)
                
                if region_points is not None:
                    st.session_state.counter = solutions.ObjectCounter(
                        region=region_points,
                        model=st.session_state.model_path,
                        classes=[0],
                        show=False,
                        show_conf=False,
                        show_labels=False,
                        verbose=False,
                        show_in=True,
                        show_out=False,
                        line_width=2
                    )
                else:
                    st.warning("Region selection canceled")
                    reset_states()
                    return
                
            if not ret:
                st.warning("Video playback complete.")
                release_camera()

                break

            current_time = time.time()
            delta = current_time - st.session_state.prev_time
            st.session_state.prev_time = current_time
            delay_ms = delta * 1000
            

            results = detect(st.session_state.model, frame, device="0")[0]
            bboxes = results.boxes.xyxy
            counter = st.session_state.counter
            counter_results = counter(frame)
            
            clss = results.boxes.cls
            if bboxes is not None:
                draw_delay(frame, text=f"delay: {delay_ms:.2f}")
                
                processed_frame = draw_data(frame, bboxes, clss, region_points)
            else:
                processed_frame = frame

            st.session_state.frozen_frame = processed_frame

            frame_placeholder.image(processed_frame, channels="BGR")
    reset_states()

def main():
    st.sidebar.title("Select Solution")
    mode = st.sidebar.radio("Choose a Solution", ["Object Counting", "Quality Inspection"])
    if mode == "Object Counting":
        object_counting()

if __name__ == "__main__":
    main()
