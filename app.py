import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("Bank Marketing Classification")
st.caption("ML Assignment 2 – BITS")

# -------------------------------
# Cache TRAINING data + encoding
# -------------------------------
@st.cache_resource
def load_and_prepare_training_data():
    data = pd.read_csv("bank-full.csv", sep=";")
    data["target"] = data["y"].map({"yes": 1, "no": 0})
    data.drop("y", axis=1, inplace=True)

    encoded = pd.get_dummies(data, drop_first=True)
    X = encoded.drop("target", axis=1)
    y = encoded["target"]
    return X, y

X_train, y_train = load_and_prepare_training_data()

# -------------------------------
# Model selection
# -------------------------------
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "Naive Bayes"
    ]
)

@st.cache_resource
def train_model(model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    return model

model = train_model(model_name)

# -------------------------------
# Upload TEST data
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"],
    help="Upload test data (target column optional)"
)

if uploaded_file is not None:

    with st.spinner("Processing uploaded dataset..."):
        test_data = pd.read_csv(uploaded_file)

        # Save labels if present
        y_true = None
        if "target" in test_data.columns:
            y_true = test_data["target"]
            test_data.drop("target", axis=1, inplace=True)

        # Encode test data
        test_encoded = pd.get_dummies(test_data, drop_first=True)

        # Align with training columns
        test_encoded = test_encoded.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        # Predict
        y_pred = model.predict(test_encoded)

    st.success("Dataset uploaded and predictions generated successfully!")

    # -------------------------------
    # Prediction summary
    # -------------------------------
    st.subheader("Prediction Summary")
    pred_counts = pd.Series(y_pred).value_counts().rename(index={0: "No", 1: "Yes"})
    st.bar_chart(pred_counts)

    # -------------------------------
    # Evaluation (if labels exist)
    # -------------------------------
    if y_true is not None:

        # Classification report (colorful)
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("Classification Report")
        st.dataframe(report_df.style.background_gradient(cmap="Greens"))

        # Confusion matrix (colorful)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)
