import tempfile
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyrebase
import os
from PIL import Image

# Firebase configuration and initialization
firebaseConfig = {
    "apiKey": "AIzaSyCPjnpMd--BcI58wMx7B29eUAS4C7EhAEA",
    "authDomain": "deepfake-detection-app.firebaseapp.com",
    "projectId": "deepfake-detection-app",
    "storageBucket": "deepfake-detection-app.appspot.com",
    "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
    "appId": "1:1056432748440:web:9d762352d0d91fde581ded",
    "databaseURL": "https://deepfake-detection-app.firebaseio.com"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth_service = firebase.auth()


def signup(email, password, confirm_password):
    if password != confirm_password:
        st.error("Passwords do not match!")
        return None
    try:
        user = auth_service.create_user_with_email_and_password(email, password)
        st.success("Account created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")


def login(email, password):
    try:
        user = auth_service.sign_in_with_email_and_password(email, password)
        st.success("Login successful!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")


IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
BATCH_SIZE = 8

model = keras.models.load_model("best_modelk_FINAL1_MINI_PROJECT.h5")


def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    frames = np.array(frames)
    if frames.ndim == 4 and frames.shape[1:] == (IMG_SIZE, IMG_SIZE, 3):
        return frames
    else:
        st.warning(f"Warning: Frame shape {frames.shape} from video {path} does not match expected shape.")
        return None

def crop_center_square(frame):
    y, x, _ = frame.shape
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()


def predict_video(video_path):
    frames = load_video(video_path)
    if frames is None:
        st.warning("Error loading video or video format unsupported.")
        return None

    temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    video_length = frames.shape[0]
    length = min(MAX_SEQ_LENGTH, video_length)

    for j in range(length):
        temp_frame_features[0, j, :] = feature_extractor.predict(np.expand_dims(frames[j], axis=0))
    temp_frame_mask[0, :length] = 1  

    temp_frame_features = temp_frame_features[:, :MAX_SEQ_LENGTH, :]
    temp_frame_mask = temp_frame_mask[:, :MAX_SEQ_LENGTH]

    predictions = model.predict([temp_frame_features, temp_frame_mask])

    fake_probability = predictions[0][0]
    confidence_score = fake_probability * 100 if fake_probability >= 0.5 else (1 - fake_probability) * 100
    prediction_label = "Fake" if fake_probability >= 0.5 else "Real"
    return prediction_label, confidence_score

st.set_page_config(page_title="Deepfake Detection App", page_icon="🔍", layout="wide")


if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

def toggle_dark_mode():
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]




theme = """
    <style>
        .stApp {{
            background-color: #1c1c1c;
            color: #f9f9f9;
        }}
        .sidebar .sidebar-content {{
            background-color: #262626;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #ff6700;
        }}
        .stButton > button {{
            background-color: #ff6700;
            color: #ffffff;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }}
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #262626;
            color: #f9f9f9;
            text-align: center;
            padding: 10px 0;
            border-top: 1px solid #ff6700;
        }}
        .content {{
            margin-bottom: 50px; /* Space for footer */
        }}
    </style>
    """ if st.session_state["dark_mode"] else """
    <style>
        .stApp {{
            background-color: #f9f9f9;
            color: #333;
        }}
        .sidebar .sidebar-content {{
            background-color: #ffffff;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #ff6700;
        }}
        .stButton > button {{
            background-color: #ff6700;
            color: #ffffff;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }}
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            color: #333;
            text-align: center;
            padding: 10px 0;
            border-top: 1px solid #ff6700;
        }}
        .content {{
            margin-bottom: 50px; /* Space for footer */
        }}
    </style>
    """
st.markdown(theme, unsafe_allow_html=True)

st.title("🔍 Deepfake Detection App")

menu = ["Home", "Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.header("What are Deepfakes?")
    st.write("""
        Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. 
        The term is a combination of "deep learning" and "fake." Deepfake technology uses advanced machine learning techniques 
        to create hyper-realistic videos that can be used to impersonate others, leading to significant implications for privacy, security, and trust.
    """)

    st.write("### Why is Deepfake Detection Important?")
    st.write("""
        As deepfake technology evolves, it becomes increasingly important to identify and combat the spread of misleading or harmful media.
        This app aims to help users detect whether a video is real or manipulated, contributing to the fight against misinformation and preserving trust in digital content.
    """)

    st.image(r"public/Leonardo_Phoenix_A_surreal_and_futuristic_digital_artwork_depi_3.jpg", use_container_width=True)

elif choice == "Signup":
    st.header("Create an Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        signup(email, password, confirm_password)

elif choice == "Login":
    st.header("Login to Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login(email, password)
        if user:
            st.session_state["user"] = user

if "user" in st.session_state:
    st.header("Upload Video for Detection")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        st.video(temp_file.name)

        if st.button("Detect"):
            prediction, confidence = predict_video(temp_file.name)
            if prediction and confidence is not None:
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: 20px;">
                        <h3>Prediction: <span style="color: {'#e74c3c' if prediction == 'Fake' else '#2ecc71'};">{prediction}</span></h3>
                        <h4>Confidence Score: <span style="color: #3498db;">{confidence:.2f}%</span></h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("An error occurred during video analysis.")


# st.markdown("<div class='footer'>© 2024 Deepfake Detection App</div>", unsafe_allow_html=True)