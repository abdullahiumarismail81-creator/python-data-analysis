
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Lung Disease Classifier",
    layout="centered"
)

CLASS_NAMES = ["Lung Opacity", "Normal", "Viral Pneumonia"]
CLASS_INFO = {
    "Normal": {
        "color": "#22c55e",
        "description": "No signs of lung disease detected. The X-ray appears normal."
    },
    "Lung Opacity": {
        "color": "#f97316",
        "description": "Lung opacity detected. This may indicate fluid, infection, or other abnormalities in the lung tissue."
    },
    "Viral Pneumonia": {
        "color": "#ef4444",
        "description": "Signs consistent with viral pneumonia detected. Please consult a medical professional immediately."
    }
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_disease.keras")
    return model

def enhance_image(image):
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    hue = image[:, :, 0]
    saturation = image[:, :, 1]
    value = image[:, :, 2]
    value = np.clip(value * 1.25, 0, 255)
    image[:, :, 2] = value
    return image

def preprocess_image(pil_image):
    img = pil_image.convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img, dtype=np.uint8)
    img_array = enhance_image(img_array)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image_array):
    predictions = model.predict(image_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100
    all_probs = {CLASS_NAMES[i]: float(predictions[0][i]) * 100 for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence, all_probs


st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .result-box {
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1.2rem 0;
        text-align: center;
        font-size: 1.1rem;
    }
    .disclaimer {
        background: #f1f5f9;
        border-left: 4px solid #94a3b8;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Lung Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a chest X-ray image for AI-powered lung disease detection</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;">Developed by Faruk</div>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model = load_model()

st.success("Model loaded successfully!")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray in JPG or PNG format"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

with col2:
    st.subheader("Analysis Result")

    if uploaded_file:
        with st.spinner("Analyzing image..."):
            image_array = preprocess_image(image)
            predicted_class, confidence, all_probs = predict(model, image_array)

        info = CLASS_INFO[predicted_class]
        bg_color = info["color"] + "22"

        st.markdown(f"""
            <div class="result-box" style="background:{bg_color}; border: 2px solid {info['color']};">
                <div style="font-size:1.6rem; font-weight:800; color:{info['color']}; margin:0.4rem 0;">{predicted_class}</div>
                <div style="font-size:1rem; color:#475569;">{info['description']}</div>
            </div>
        """, unsafe_allow_html=True)

        st.metric(label="Confidence", value=f"{confidence:.1f}%")

        st.subheader("Class Probabilities")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            st.markdown(f"**{cls}**")
            st.progress(prob / 100)
            st.caption(f"{prob:.2f}%")

    else:
        st.info("Please upload a chest X-ray image on the left to begin analysis.")
        st.markdown("""
        **Detectable Conditions:**
        - **Normal** — Healthy lung X-ray with no abnormalities
        - **Lung Opacity** — Fluid or abnormal density present in the lungs
        - **Viral Pneumonia** — Signs consistent with viral lung infection
        """)

st.divider()

st.markdown("""
    <div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This tool is intended for educational and research purposes only.
    It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare
    provider for medical advice and interpretation of X-ray images.
    </div>
""", unsafe_allow_html=True)
