import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("Bank Marketing Classification")
st.subheader("ML Assignment 2 – BITS")

# -------------------------------
# Model paths
# -------------------------------
MODEL_PATHS = {
    "Logistic Regression": "models/saved_models/logistic.pkl",
    "Decision Tree": "models/saved_models/dt.pkl",
    "Random Forest": "models/saved_models/rf.pkl",
    "KNN": "models/saved_models/knn.pkl",
    "Naive Bayes": "models/saved_models/nb.pkl",
    "XGBoost": "models/saved_models/xgb.pkl",
}

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# -------------------------------
# Model selection
# -------------------------------
model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))

# -------------------------------
# Main logic
# -------------------------------
if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data", data.head())

    # --------------------------------
    # Separate target if present
    # --------------------------------
    y_true = None
    if "target" in data.columns:
        y_true = data["target"]
        data = data.drop("target", axis=1)

    # Drop training-only label if exists
    if "y" in data.columns:
        data = data.drop("y", axis=1)

    # --------------------------------
    # Encode categorical features
    # --------------------------------
    data_encoded = pd.get_dummies(data, drop_first=True)

    # --------------------------------
    # Load selected model
    # --------------------------------
    with open(MODEL_PATHS[model_name], "rb") as f:
        model = pickle.load(f)

    # --------------------------------
    # Prediction
    # --------------------------------
    y_pred = model.predict(data_encoded)

    st.subheader("Predictions")
    st.write(pd.Series(y_pred).value_counts())

    # --------------------------------
    # Evaluation (only if target exists)
    # --------------------------------
    if y_true is not None:
        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))
