import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from typing import Tuple, Any

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "notebooks/mlp_document_classifier_26-9-2025.h5"
VECTORIZER_PATH = "notebooks/tfidf_vectorizer_1-10-2025.pkl"
LABEL_ENCODER_PATH = "notebooks/label_encoder_1-10-2025.pkl"

# ------------------------------
# Load model and preprocessors
# ------------------------------
@st.cache_resource
def load_model_and_tools() -> Tuple[Any, Any, Any]:
    """Load and return the model, vectorizer and label encoder."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model or tools: {str(e)}")
        st.stop()

model, vectorizer, label_encoder = load_model_and_tools()

# ------------------------------
# Streamlit UI
# ------------------------------
def display_header():
    st.title("🧠 Sentiment / Text Classification App")
    st.write("Enter a text below to get the model's prediction.")
    st.markdown("---")

def get_user_input() -> str:
    """Get and validate user input text."""
    return st.text_area(
        "📝 Input text:",
        height=150,
        placeholder="Type a review, message, or sentence..."
    )

def display_prediction_results(pred_label: str, pred_proba: np.ndarray, label_encoder) -> None:
    """Display prediction results and confidence scores."""
    st.success(f"✅ **Prediction:** {pred_label}")

    # Show probability chart
    st.write("### 🔍 Prediction Confidence")
    for label, prob in zip(label_encoder.classes_, pred_proba[0]):
        st.progress(float(prob), text=f"{label}: {prob:.2%}")

    st.markdown("---")
    st.caption("Built with Streamlit, TensorFlow, and TF-IDF ✨")


# ------------------------------
# Main app flow
# ------------------------------
display_header()
user_input = get_user_input()

if st.button("Predict", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        X_input = vectorizer.transform([user_input])

        # Predict
        pred_proba = model.predict(X_input.toarray())
        pred_index = np.argmax(pred_proba, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        # Display results
        display_prediction_results(pred_label, pred_proba, label_encoder)
