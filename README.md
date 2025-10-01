# Document Classification using TF-IDF and Neural Network

## Overview
This project implements an end-to-end **text document classification system** using **TF-IDF feature extraction** with two models: **Naïve Bayes** (baseline) and a **TensorFlow Multilayer Perceptron (MLP)** neural network. The system predicts document categories, evaluates performance, and includes a **Streamlit web application** for real-time inference.

---

## Features
- **Baseline Model:** Naïve Bayes for initial benchmarking.
- **Neural Network:** TensorFlow MLP with 3 million parameters for enhanced performance.
- **Residual Analysis:** Identifies misclassified samples to improve model robustness.
- **Interactive App:** Streamlit interface for live predictions and confidence visualization.
- **Reproducible Workflow:** Saved model, TF-IDF vectorizer, and label encoder.

---

## Performance Metrics

| Model             | Accuracy | F1 Score | Loss / Log Loss |
|------------------|----------|----------|----------------|
| Naïve Bayes       | 82%      | 0.80     | 1.03           |
| TensorFlow MLP    | 84%      | 0.82     | 0.55           |

---

## Tech Stack
- Python 3.x
- TensorFlow
- Scikit-learn
- Streamlit
- NumPy, Pandas

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**
```bash
streamlit run app.py
```

---

## Project Structure

```
project/
│
├── notebooks/
│   ├── mlp_document_classifier_26-9-2025.h5
│   ├── tfidf_vectorizer_1-10-2025.pkl
│   └── label_encoder_1-10-2025.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## Usage
- Open the Streamlit app in your browser.
- Enter a document or text in the input box.
- Click **Predict** to see the predicted category and confidence scores.

---

## Key Highlights
- End-to-end pipeline: text preprocessing → model training → evaluation → deployment.
- Comparison between **baseline Naïve Bayes** and **MLP neural network**.
- Visualization of prediction confidence and residual analysis.
- Lightweight and CPU-friendly deployment via Streamlit.

---

## Author
**[Ling Chin Ung]** – Data Scientist / AI Enthusiast
Email: jasonling23@yahoo.com
LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
