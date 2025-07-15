from ultralytics import YOLO
import streamlit as st
import numpy as np

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
st.set_page_config(page_title="BSData Segmentation Inspection", layout="wide")
st.title("🔍 BSData Mask Quality Inspection")

model = YOLO("weights/yolov8x-seg-BSData.pt")

def process_image(img):
    results = model(img)[0]
    masks = results.masks.data.cpu().numpy() if results.masks else []
    annotated_img = results.plot()

    inspection_info = []

    for i, mask in enumerate(masks):
        binary_mask = (mask * 255).astype(np.uint8)
        roughness = calculate_roughness(binary_mask)
        area = get_mask_area(binary_mask)
        inspection_info.append({
            'index': i + 1,
            'area_px': area,
            'roughness': roughness
        })

    return annotated_img, inspection_info

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    st.image(img, caption="Input Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Processing..."):
            annotated, info = process_image(img)
            st.image(annotated, caption="Segmented Mask", use_column_width=True)
            st.subheader("Inspection Results")
            for obj in info:
                st.markdown(f"**Object {obj['index']}**: Area = `{obj['area_px']}` px, Roughness = `{obj['roughness']}`")
