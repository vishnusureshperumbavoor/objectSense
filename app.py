import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

st.set_page_config(
    page_title="Object Sense",
    page_icon="🔎"
)

def main():
    st.title("Object Detection App by VSP")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        
        # Display original image on the right side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, channels="BGR", caption="Human Vision")
        
        # Detect objects using yolov5
        # Convert OpenCV image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Perform object detection using YOLOv5
        results = model(image_pil)

        # Annotated image
        annotated_image = results.render()

        # Display annotated image on the left side
        with col2:
            st.image(annotated_image, channels="RGB", caption="AI Vision")

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) ## yolov5
    # model = YOLO('yolov8n.pt')
    main()
