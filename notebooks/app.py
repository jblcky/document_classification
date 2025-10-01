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
        st.error(f"⚠️ Error loading model or tools: {str(e)}")
        st.stop()

model, vectorizer, label_encoder = load_model_and_tools()

# ------------------------------
# UI Styling
# ------------------------------
st.set_page_config(page_title="Document Classifier", page_icon="🧠", layout="centered")

def display_header():
    st.title("🧠 Document Classification App")
    st.caption("Predicts the **category** of any text document using a trained Neural Network (TF-IDF + MLP).")
    st.markdown("---")

def get_user_input() -> str:
    """Get and validate user input text."""
    return st.text_area(
        "✍️ Enter your document or text:",
        height=180,
        placeholder="e.g. Artificial intelligence is transforming industries across the world..."
    )

def display_prediction_results(pred_label: str, pred_proba: np.ndarray, label_encoder) -> None:
    """Display prediction results and confidence scores."""
    st.markdown("## 🎯 Prediction Result")
    st.success(f"**Predicted Category:** {pred_label}")

    st.markdown("### 📊 Model Confidence")
    col1, col2 = st.columns(2)
    top_idx = np.argsort(pred_proba[0])[::-1][:2]

    # Confidence bars
    for i, (label, prob) in enumerate(zip(label_encoder.classes_, pred_proba[0])):
        st.progress(float(prob), text=f"{label}: {prob:.2%}")

    # Display top 2 with emoji for fun
    with col1:
        st.markdown("#### 🏆 Top Prediction")
        st.write(f"**{label_encoder.classes_[top_idx[0]]}** ({pred_proba[0][top_idx[0]]:.2%})")

    with col2:
        st.markdown("#### 🥈 2nd Best Guess")
        st.write(f"**{label_encoder.classes_[top_idx[1]]}** ({pred_proba[0][top_idx[1]]:.2%})")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, TensorFlow and TF-IDF.")


# ------------------------------
# Main app flow
# ------------------------------
display_header()
user_input = get_user_input()

if st.button("🚀 Predict", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing..."):
            # TF-IDF transform (don’t fit!)
            X_input = vectorizer.transform([user_input])

            # Model predict
            pred_proba = model.predict(X_input.toarray())
            pred_index = np.argmax(pred_proba, axis=1)[0]
            pred_label = label_encoder.inverse_transform([pred_index])[0]

        display_prediction_results(pred_label, pred_proba, label_encoder)
