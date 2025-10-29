import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# ---------------------------------------------------
# Load CNN model
# ---------------------------------------------------
MODEL_PATH = "face_reg_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------
# Load Haar Cascade for face detection
# ---------------------------------------------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------
st.set_page_config(page_title="Face Emotion Recognition", page_icon="😊", layout="centered")
st.title("😊 Face Emotion Recognition App")
st.write("Upload an image or take a photo to detect faces and predict emotions with confidence levels.")

# Add reset button
if st.button("🔄 Try Another Image"):
    st.session_state.clear()
    st.rerun()

# ---------------------------------------------------
# Option to upload OR take photo
# ---------------------------------------------------
option = st.radio("Choose Input Method:", ("📁 Upload Image", "📷 Open Camera"))

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Read uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        img = None

elif option == "📷 Open Camera":
    captured_image = st.camera_input("Take a photo using your webcam")
    if captured_image:
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        img = None

# ---------------------------------------------------
# Face Detection & Emotion Prediction
# ---------------------------------------------------
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Please try again or use a clearer image.")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded/Captured Image", use_container_width=True)
    else:
        emotion_results = []
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)

            # Predict emotion
            pred = model.predict(face_expanded)
            emotion_idx = np.argmax(pred)
            confidence = float(np.max(pred))
            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            emotion = emotions[emotion_idx]

            # Draw bounding box + label
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} ({confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            emotion_results.append((emotion, confidence))

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Detected Faces & Emotions",
                 use_container_width=True)

        # Display emotion confidence bars
        st.subheader("📊 Emotion Confidence Scores")
        for i, (emotion, conf) in enumerate(emotion_results, 1):
            st.write(f"**Face {i}: {emotion} ({conf*100:.1f}%)**")
            st.progress(conf)
