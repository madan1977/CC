import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import os
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from pages.bilstm_model import AttentionLayer
from pages.train import train_classical_model, train_bilstm_model

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

def model_testing_app():
    st.title("Model Testing: Traditional vs Bi-LSTM")
    dataset_option = st.radio(
        "Select Dataset for Testing:",
        ("Right Fit Dataset", "Imbalanced Dataset"),
        index=0
    )
    if dataset_option == "Right Fit Dataset":
        file = os.path.join(current_dir, "backend_live_right.csv")
    else:
        file = os.path.join(current_dir, "backend_live_data.csv")
    file = os.path.join(current_dir, "backend_live_data.csv")
    df = pd.read_csv(file)
    st.dataframe(df.head())

    target_column = df.columns[-1]

    # === Train Classical Model ===
    clf_acc = train_classical_model(df, target_column)
    st.success(f"Classical Model Trained (Accuracy: {clf_acc:.2f})")

    # === Train Bi-LSTM Model ===
    bilstm_acc = train_bilstm_model(df, target_column)
    st.success(f"Bi-LSTM Model Trained (Accuracy: {bilstm_acc:.4f})")

    # === Prepare for Inference ===
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trad_model_path = os.path.join(current_dir, "classical_model.pkl")
    bilstm_model_path = os.path.join(current_dir, "bilstm_model.h5")
    scalar_bilstm_model_path = os.path.join(current_dir, "bilstm_scaler.pkl")
    labelencoders_bilstm_model_path = os.path.join(current_dir, "bilstm_labelencoders.pkl")
    labelencoder_target_path = os.path.join(current_dir, "bilstm_labelencoder.pkl")

    if os.path.exists(trad_model_path) and os.path.exists(bilstm_model_path):
        # === Traditional Inference ===
        with open(trad_model_path, "rb") as f:
            trad_model = pickle.load(f)

        expected_features = trad_model.feature_names_in_
        X_test_clf = X_test.reindex(columns=expected_features, fill_value=0)
        X_test_clf = X_test_clf.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_pred_trad = trad_model.predict(X_test_clf)

        st.subheader("🧠 Traditional Model Results")
        st.write("Accuracy:", accuracy_score(y_test, y_pred_trad))
        st.write("Precision:", precision_score(y_test, y_pred_trad, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred_trad, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred_trad, average='weighted'))
        st.text(classification_report(y_test, y_pred_trad))
        plot_confusion(y_test, y_pred_trad, labels=sorted(y.unique()))

        # === Bi-LSTM Inference ===
        with open(scalar_bilstm_model_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(labelencoders_bilstm_model_path, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(labelencoder_target_path, 'rb') as f:
            label_encoder = pickle.load(f)

        X_bilstm_pre = X_test.copy()
        for col in X_bilstm_pre.select_dtypes(include=['object', 'category']).columns:
            if col in label_encoders:
                le = label_encoders[col]
                X_bilstm_pre[col] = le.transform(X_bilstm_pre[col].astype(str))
            else:
                st.error(f"Missing encoder for column: {col}")
                return

        X_bilstm_pre = scaler.transform(X_bilstm_pre)
        y_test_enc = label_encoder.transform(y_test)
        X_bilstm_pre = X_bilstm_pre.reshape((X_bilstm_pre.shape[0], 1, X_bilstm_pre.shape[1]))

        bilstm_model = tf.keras.models.load_model(
            bilstm_model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )

        y_pred_prob = bilstm_model.predict(X_bilstm_pre)
        if y_pred_prob.shape[-1] > 1:
            y_pred = y_pred_prob.argmax(axis=-1)
        else:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        st.subheader("🧠 Bi-LSTM Model Results")
        st.write("Accuracy:", accuracy_score(y_test_enc, y_pred))
        st.write("Precision:", precision_score(y_test_enc, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test_enc, y_pred, average='weighted'))
        st.write("F1 Score (Weighted):", f1_score(y_test_enc, y_pred, average='weighted'))
        st.write("F1 Score (Macro):", f1_score(y_test_enc, y_pred, average='macro'))
        st.text(classification_report(y_test_enc, y_pred))
        plot_confusion(y_test_enc, y_pred, labels=label_encoder.classes_)

        # === Comparison ===
        acc_trad = accuracy_score(y_test, y_pred_trad)
        acc_bilstm = accuracy_score(y_test_enc, y_pred)
        better = "Traditional Model" if acc_trad > acc_bilstm else (
            "Bi-LSTM Model" if acc_bilstm > acc_trad else "Both models perform equally"
        )
        st.subheader("📊 Model Comparison")
        st.success(f"🧠 Better Performing Model: **{better}**")

        if acc_bilstm < 0.95:
            st.warning("⚠️ Bi-LSTM accuracy is below 95%. Consider using SMOTE, attention, or more features.")

    else:
        st.error("Model files not found. Ensure `.pkl` and `.h5` files exist.")

if __name__ == "__main__":
    model_testing_app()
