import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import glob
from PIL import Image
import tempfile
from predict_function import predict_with_yolo_combine_model, draw_boxes_on_image, combine_predictions_with_wbf, predict_with_yolo, predict_with_sahi
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import time

models = {
    "YOLO V11m": YOLO(r"./model/yolo11m_best.pt"), 
    "YOLO V8m": YOLO(r"./model/yolo8m_best.pt"),
    "YOLO V5m": YOLO(r"./model/yolo5m_best.pt"),
}

sahi_models = {
    "SAHI YOLO V11m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"./model/yolo11m_best.pt",  
        confidence_threshold=0.5,
        device="cpu"
    ),
    "SAHI YOLO V8m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"./model/yolo8m_best.pt",
        confidence_threshold=0.5,
        device="cpu"
    ),
    "SAHI YOLO V5m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"./model/yolo5m_best.pt",
        confidence_threshold=0.5,
        device="cpu"
    ),
}


categories = ["no helmet", "helmet"]
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    body{
        font-family: 'Georgia', serif;
    }
    h1 {
        text-align: center;
        color: #FF8C00;
        font-family: 'Georgia', serif;
    }
    
    .stButton button {
        font-family: 'Georgia', serif;
        font-weight: bold;
    }
            
    .stSubheader {
        font-family: 'Georgia', serif;
        font-weight: bold;
        font-size: 20px;
    }

    .stSelectbox div[data-baseweb="select"] {
        font-family: 'Georgia', serif;
    }
    </style>

    <h1> Helmet Detection App 🌠 </h1>
""", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])


conf_threshold = st.slider("Confidence Threshold:", min_value=0.01, max_value=1.0, value=0.5, step=0.01)


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    st.subheader("Select Prediction Method")
    prediction_method = st.selectbox(
        "Choose Prediction Method:",
        ["Single YOLO Model", "Combined YOLO Models with WBF", "SAHI"]
    )

    if prediction_method == "Single YOLO Model":
        selected_model = st.selectbox("Select YOLO Model:", list(models.keys()))
        if st.button("Run Prediction"):
            model = models[selected_model]
            
            # Start time
            start_time = time.time()
            annotated_image = predict_with_yolo(model, image_np, conf_threshold)
            # End time
            end_time = time.time()

            elapsed_time = end_time - start_time
            st.write(f"Time taken for {selected_model}: {elapsed_time:.2f} seconds")
            st.image(annotated_image, caption=f"Results using {selected_model}")

    elif prediction_method == "Combined YOLO Models with WBF":
        if st.button("Run Combined Prediction"):
            models_predictions = []
            # Start time
            start_time = time.time()
            for model_name, model in models.items():
                bboxes, scores, labels = predict_with_yolo_combine_model(model, image_np, conf_threshold)
                models_predictions.append((bboxes, scores, labels))

            weights = [0.4, 0.4, 0.2]  
            fused_boxes, fused_scores, fused_labels = combine_predictions_with_wbf(
                models_predictions, image_np.shape, weights
            )
            annotated_image = draw_boxes_on_image(
                image_np, fused_boxes, fused_scores, fused_labels, categories, conf_threshold
            )
            # End time
            end_time = time.time()

            elapsed_time = end_time - start_time
            st.write(f"Time taken for Combined YOLO Models: {elapsed_time:.2f} seconds")
            st.image(annotated_image, caption="Combined Model Results")

    elif prediction_method == "SAHI":
        if st.button("Run SAHI Prediction"):
            sahi_columns = st.columns(len(sahi_models))
            for col, (model_name, sahi_model) in zip(sahi_columns, sahi_models.items()):
                with col:
                    st.write(model_name)
                    
                    # Start time
                    start_time = time.time()
                    annotated_image = predict_with_sahi(image_np, sahi_model, categories)
                    # End time
                    end_time = time.time()

                    elapsed_time = end_time - start_time
                    st.write(f"Time taken for {model_name}: {elapsed_time:.2f} seconds")
                    
                    annotated_image_pil = Image.fromarray(annotated_image)
                    st.image(annotated_image_pil, caption=f"{model_name}")



if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.subheader("Original Video")
    st.video(video_path)

    st.subheader("Detection on Video")
    selected_models = st.multiselect("Select YOLO Models for Video:", list(models.keys()), default=list(models.keys()))

    if st.button("Process Video"):
        selected_model_objects = [models[model_name] for model_name in selected_models]
        weights = [0.3, 0.4, 0.4] 

        cap = cv2.VideoCapture(video_path)
        output_frames = []

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            models_predictions = []
            for model in selected_model_objects:
                bboxes, scores, labels = predict_with_yolo_combine_model(model, frame_rgb, conf_threshold)
                models_predictions.append((bboxes, scores, labels))

            fused_boxes, fused_scores, fused_labels = combine_predictions_with_wbf(
                models_predictions, frame_rgb.shape, weights
            )

            annotated_frame = draw_boxes_on_image(
                frame_rgb, fused_boxes, fused_scores, fused_labels, categories, 0.3
            )

            stframe.image(annotated_frame, channels="RGB")

            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            output_frames.append(annotated_frame_bgr)

        cap.release()

        if output_frames:
            height, width, _ = output_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = "output_video.mp4"
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            for frame in output_frames:
                out.write(frame)
            out.release()

            st.subheader("Processed Video")
            st.video(output_path)
