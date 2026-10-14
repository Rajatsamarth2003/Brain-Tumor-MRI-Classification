
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path


# ==========================================
# 1. Page configuration
# ==========================================

st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. Theme state
# ==========================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.session_state.dark_mode

if dark_mode:
    theme = {
        "background": "#0f172a",
        "card": "#1e293b",
        "text": "#f1f5f9",
        "muted": "#a8b5c7",
        "border": "#334155",
        "accent": "#2dd4bf",
        "header": "#172554",
        "upload": "#172033",
        "notice_bg": "#3b2f16",
        "notice_text": "#fde68a"
    }
else:
    theme = {
        "background": "#f4f7fb",
        "card": "#ffffff",
        "text": "#1e293b",
        "muted": "#64748b",
        "border": "#e1e8f0",
        "accent": "#0f766e",
        "header": "#eaf4ff",
        "upload": "#f8fbff",
        "notice_bg": "#fff8e8",
        "notice_text": "#704f12"
    }


# ==========================================
# 3. Custom CSS
# ==========================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {theme["background"]};
        color: {theme["text"]};
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, p, label {{
        color: {theme["text"]};
    }}

    .dashboard-header {{
        background: {theme["header"]};
        border: 1px solid {theme["border"]};
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 22px;
    }}

    .dashboard-title {{
        font-size: 30px;
        font-weight: 750;
        color: {theme["text"]};
        margin-bottom: 6px;
    }}

    .dashboard-subtitle {{
        color: {theme["muted"]};
        font-size: 15px;
        margin: 0;
    }}

    .section-card {{
        background: {theme["card"]};
        border: 1px solid {theme["border"]};
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(25, 55, 90, 0.04);
    }}

    .section-label {{
        color: {theme["muted"]};
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }}

    .prediction-title {{
        color: {theme["accent"]};
        font-size: 27px;
        font-weight: 750;
        margin: 5px 0 12px 0;
    }}

    .metric-card {{
        background: {theme["card"]};
        border: 1px solid {theme["border"]};
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }}

    .metric-label {{
        color: {theme["muted"]};
        font-size: 13px;
    }}

    .metric-value {{
        color: {theme["accent"]};
        font-size: 27px;
        font-weight: 750;
    }}

    .disclaimer {{
        background: {theme["notice_bg"]};
        border: 1px solid {theme["border"]};
        border-left: 5px solid #e5a323;
        border-radius: 10px;
        padding: 15px 18px;
        color: {theme["notice_text"]};
        font-size: 14px;
        margin: 18px 0;
    }}

    .footer-text {{
        text-align: center;
        color: {theme["muted"]};
        font-size: 12px;
        padding-top: 20px;
    }}

    div[data-testid="stFileUploader"] {{
        background: {theme["upload"]};
        border: 1px dashed {theme["border"]};
        border-radius: 12px;
        padding: 12px;
    }}

    div[data-testid="stFileUploader"] section {{
        background: transparent;
    }}

    div[data-testid="stFileUploader"] small {{
        color: {theme["muted"]};
    }}

    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
        border: 1px solid {theme["border"]};
    }}

    div[data-testid="stProgress"] {{
        margin-top: 6px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 4. Top bar and theme toggle
# ==========================================

top_title, top_theme = st.columns(
    [5, 1],
    vertical_alignment="center"
)

with top_theme:
    st.caption("DISPLAY THEME")

    if st.button(
        "☀️ Light Mode" if dark_mode else "🌙 Dark Mode",
        use_container_width=True
    ):
        st.session_state.dark_mode = not dark_mode
        st.rerun()


# ==========================================
# 5. Header
# ==========================================

st.markdown(
    f"""
    <div class="dashboard-header">
        <div class="dashboard-title">
            🧠 Brain MRI Classification
        </div>
        <p class="dashboard-subtitle">
            Deep Learning Research Dashboard |
            Experimental MRI Image Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 6. Medical disclaimer
# ==========================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important medical notice</strong><br>
        This application is an educational research project,
        not a medical diagnostic tool. Its predictions may be
        incorrect. Do not use them to make medical decisions.
        Consult a qualified medical professional for diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 7. Model configuration
# ==========================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

IMAGE_SIZE = (224, 224)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "efficientnetb0_fixed.keras"
)


# ==========================================
# 8. Load model
# ==========================================

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(str(model_path))


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please check "
        "models/efficientnetb0_fixed.keras."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# ==========================================
# 9. Upload and preview
# ==========================================

upload_col, preview_col = st.columns(
    [1, 1.2],
    gap="large"
)

with upload_col:
    st.markdown(
        '<div class="section-label">STEP 01 · UPLOAD IMAGE</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Select MRI image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="mri_upload"
    )

    st.caption(
        "Upload an MRI image for an experimental "
        "four-class model prediction."
    )

    st.markdown(
        f"""
        <div class="section-card">
            <strong>Model information</strong><br><br>
            Architecture: EfficientNetB0<br>
            Output classes: 4<br>
            Test accuracy: 87.40% on this project's test set
        </div>
        """,
        unsafe_allow_html=True
    )


with preview_col:
    st.markdown(
        '<div class="section-label">STEP 02 · IMAGE PREVIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown("### MRI image preview")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded MRI image",
                use_container_width=True
            )

        except Exception:
            st.error("Unable to read this image file.")
            st.stop()

    else:
        st.markdown(
            f"""
            <div class="section-card" style="
                min-height: 280px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                background: {theme["upload"]};
            ">
                <div style="font-size: 55px;">🧠</div>
                <h3>No image selected</h3>
                <p style="color: {theme["muted"]};">
                    Upload an MRI image to preview it here.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# 10. Action buttons
# ==========================================

st.divider()

button_col, reset_col, spacer_col = st.columns(
    [1.2, 1, 3]
)

with button_col:
    classify_clicked = st.button(
        "🔍 Classify MRI",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )

with reset_col:
    reset_clicked = st.button(
        "↻ Clear / Reset",
        use_container_width=True
    )


if reset_clicked:
    st.session_state.pop("mri_prediction", None)
    st.session_state.pop("mri_upload", None)
    st.rerun()


# ==========================================
# 11. Predict
# ==========================================

if classify_clicked and uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        with st.spinner("Processing MRI image..."):
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

        if len(predictions) != len(CLASS_NAMES):
            st.error("Unexpected model output.")
            st.stop()

        predicted_index = int(np.argmax(predictions))

        st.session_state["mri_prediction"] = {
            "predictions": predictions.tolist(),
            "predicted_index": predicted_index
        }

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)


# ==========================================
# 12. Display results
# ==========================================

result = st.session_state.get("mri_prediction")

if result is not None:
    predictions = np.array(
        result["predictions"],
        dtype=float
    )

    predicted_index = result["predicted_index"]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    st.markdown(
        '<div class="section-label">STEP 03 · MODEL OUTPUT</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Classification results")

    result_col, confidence_col = st.columns([1.5, 1])

    with result_col:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">PREDICTED CLASS</div>
                <div class="prediction-title">
                    {predicted_class.replace("_", " ").title()}
                </div>
                <p style="color: {theme["muted"]};">
                    This is the model's top predicted category,
                    not a confirmed medical diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with confidence_col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MODEL CONFIDENCE SCORE</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Class-wise prediction scores")

    st.caption(
        "Scores represent model outputs, not calibrated "
        "probabilities of a medical diagnosis."
    )

    for index, class_name in enumerate(CLASS_NAMES):
        score = float(predictions[index])

        score_col, bar_col = st.columns([1, 3])

        with score_col:
            st.write(class_name.replace("_", " ").title())

        with bar_col:
            st.progress(
                min(max(score, 0.0), 1.0),
                text=f"{score * 100:.2f}%"
            )


# ==========================================
# 13. Footer
# ==========================================

st.divider()

st.markdown(
    f"""
    <div class="footer-text">
        Brain MRI Classification · Deep Learning Research Project<br>
        Experimental results only — not for clinical use.
    </div>
    """,
    unsafe_allow_html=True
)
