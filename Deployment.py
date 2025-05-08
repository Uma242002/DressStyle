import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="Indian Dressing Style Classifier", layout="wide")

# Class dictionary
di = {
    'AndhraPradeshDressingStyle': 0,
    'JammuKashmirDressingStyle': 1,
    'KeralaDressingStyle': 2,
    'MaharashtraDressingStyle': 3,
    'SikkimDressingStyle': 4
}
idx_to_label = {v: k for k, v in di.items()}

@st.cache_resource
def load_models():
    try:
        yolo = YOLO(r"OD\best.pt")  # Your YOLO model path
        cnn = load_model(r"Classification\cnn.keras")  # Your CNN model path
        return yolo, cnn
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def classify(image, cnn_model):
    try:
        img = image / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = cnn_model.predict(img)[0]
        return prediction
    except Exception as e:
        st.warning(f"Prediction Error: {e}")
        return None

# App title and instructions
st.title("State Wise Dressing Style Classifier")
st.write("Upload an image or video to detect dressing style using YOLO + CNN models.")

# Load models
yolo_model, cnn_model = load_models()

# Input type
media_type = st.radio("Select Input Type:", ("Image", "Video"))


# IMAGE PROCESSING
# IMAGE PROCESSING
if media_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image and yolo_model and cnn_model:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Failed to decode the uploaded image. Please try a different file.")
            st.stop()

        results = yolo_model(image)[0]
        predicted_classes = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            try:
                cropped_resized = cv2.resize(cropped, (256, 256))
            except:
                continue  # skip if resize fails

            prediction = classify(cropped_resized, cnn_model)
            if prediction is not None:
                class_id = np.argmax(prediction)
                confidence = prediction[class_id]
                class_name = idx_to_label.get(class_id, "Unknown")

                label = f"{class_name} ({confidence:.2f})"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                predicted_classes.append(label)

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image_rgb, (225, 225))  # Resize to fixed size
            st.image(resized_image, caption="Prediction Result", width=800)
        except Exception as e:
            st.error(f"Error displaying image: {e}")

        if predicted_classes:
            st.subheader("Predicted Classes:")
            for cls in predicted_classes:
                st.write(cls)


# VIDEO PROCESSING
elif media_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video and yolo_model and cnn_model:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        predicted_video_classes = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2]
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    continue
                cropped_resized = cv2.resize(cropped, (256, 256))

                prediction = classify(cropped_resized, cnn_model)
                if prediction is not None:
                    class_id = np.argmax(prediction)
                    confidence = prediction[class_id]
                    class_name = idx_to_label.get(class_id, "Unknown")

                    label = f"{class_name} ({confidence:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    predicted_video_classes.append(label)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)

        cap.release()

        st.subheader("Predicted Classes in Video:")
        if predicted_video_classes:
            for cls in predicted_video_classes[-10:]:  # last 10 predictions
                st.write(cls)
