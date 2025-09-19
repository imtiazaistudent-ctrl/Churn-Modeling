import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Churn Modeling App", layout="wide")

st.title("📊 Customer Churn Modeling")

# Upload dataset
uploaded_file = st.file_uploader("Upload Telco Customer Churn Dataset (CSV)", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Drop ID column
    df.drop("customerID", axis=1, inplace=True, errors="ignore")
    df.replace(" ", np.nan, inplace=True)

    # Handle TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "Churn":
            df[col] = LabelEncoder().fit_transform(df[col])

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Features & target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sidebar model selection
    st.sidebar.title("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Logistic Regression Results")
        st.write("Accuracy:", round(acc, 3))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Random Forest Results")
        st.write("Accuracy:", round(acc, 3))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ROC Curve
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='RF (AUC = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Random Forest")
        ax.legend(loc="lower right")
        st.pyplot(fig)
