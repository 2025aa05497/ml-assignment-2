import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("Bank Marketing Classification")
st.write("ML Assignment 2 – BITS")

# Upload test data
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# Model selection
model_name = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

model_map = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "dt.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "nb.pkl",
    "Random Forest": "rf.pkl",
    "XGBoost": "xgb.pkl"
}

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Convert target
    if "y" in data.columns:
        data["target"] = data["y"].map({"yes": 1, "no": 0})

    data.drop("y", axis=1, inplace=True)

    # Encode categorical columns
    data_encoded = pd.get_dummies(data, drop_first=True)

    X = data_encoded.drop("target", axis=1)
    y = data_encoded["target"]

    # Load selected model
    model = joblib.load(f"models/saved_models/{model_map[model_name]}")

    y_pred = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
