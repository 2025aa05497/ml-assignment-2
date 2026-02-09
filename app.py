import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("Bank Marketing Classification")
st.subheader("ML Assignment 2 – BITS")

# -------------------------------
# Load training data
# -------------------------------
train_data = pd.read_csv("bank-full.csv")

# Convert target
train_data["target"] = train_data["y"].map({"yes": 1, "no": 0})
train_data.drop("y", axis=1, inplace=True)

# Encode categorical features
train_encoded = pd.get_dummies(train_data, drop_first=True)

X_train = train_encoded.drop("target", axis=1)
y_train = train_encoded["target"]

# -------------------------------
# Model selection
# -------------------------------
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "Naive Bayes"]
)

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

# Train model
model.fit(X_train, y_train)

# -------------------------------
# Upload test data
# -------------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data", test_data.head())

    # Separate target if present
    y_true = None
    if "target" in test_data.columns:
        y_true = test_data["target"]
        test_data.drop("target", axis=1, inplace=True)

    # Encode test data
    test_encoded = pd.get_dummies(test_data, drop_first=True)

    # Align columns with training data
    test_encoded = test_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Predict
    y_pred = model.predict(test_encoded)

    st.subheader("Predictions")
    st.write(pd.Series(y_pred).value_counts())

    if y_true is not None:
        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))
