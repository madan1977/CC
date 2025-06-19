
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pages.bilstm_model import AttentionLayer # Assuming you have a function to build your Bi-LSTM model
from pages.train import train_classical_model, train_bilstm_model
# Ensure parent directory is in sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # pages/
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # StreamlitCreditCardFraud/
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



def model_testing_app():
    st.title("Model Testing: Traditional vs Bi-LSTM")
    current_dir = os.path.dirname(os.path.abspath(__file__))  # pages/
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # StreamlitCreditCardFraud/

        #st.set_page_config(page_title="Train Models", layout="wide")
        #st.title("🧪 Train Models on Uploaded Dataset")
      
    
    
    file = os.path.join(current_dir, "backend_live_data.csv")
    df = pd.read_csv(file)
            #st.dataframe(df.head())

            #st.markdown("### 🏷️ Target Inference")
    target_column = df.columns[-1]
            #st.success(f"Last column selected as target: **{target_column}**")

         
    clf_acc = train_classical_model(df, target_column, return_model=True)
    st.success(f"Classical Model Trained (Accuracy: {clf_acc:.2f})")


    lstm_acc = train_bilstm_model(df, target_column, return_model=True)
    st.success(f"Bi-LSTM Model Trained (Accuracy: {lstm_acc:.2f})")


    st.markdown("### 📤 Upload Test Dataset (Excel/xlsx)")
    test_file = st.file_uploader("Upload your test dataset (.xlsx)", type=["xlsx"])
    if test_file is not None:
        test_df = pd.read_excel(test_file)
        st.dataframe(test_df.head())

        # Ensure test data has the same columns as training data (except target)
        feature_columns = [col for col in df.columns if col != target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column] if target_column in test_df.columns else None

        # Load or re-train classical model
        
        classical_model = train_classical_model(df, target_column, return_model=True)
        y_pred_clf = classical_model.predict(X_test)
        clf_acc = accuracy_score(y_test, y_pred_clf)
        clf_prec = precision_score(y_test, y_pred_clf, zero_division=0)
        clf_rec = recall_score(y_test, y_pred_clf, zero_division=0)
        clf_f1 = f1_score(y_test, y_pred_clf, zero_division=0)

        # Load or re-train Bi-LSTM model
        bilstm_model, scaler, label_encoder = train_bilstm_model(df, target_column, return_model=True)
        X_test_scaled = scaler.transform(X_test)
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        y_pred_lstm = bilstm_model.predict(X_test_reshaped)
        y_pred_lstm = (y_pred_lstm > 0.5).astype(int).flatten()
        if label_encoder is not None:
            y_test_enc = label_encoder.transform(y_test)
        else:
            y_test_enc = y_test
        lstm_acc = accuracy_score(y_test_enc, y_pred_lstm)
        lstm_prec = precision_score(y_test_enc, y_pred_lstm, zero_division=0)
        lstm_rec = recall_score(y_test_enc, y_pred_lstm, zero_division=0)
        lstm_f1 = f1_score(y_test_enc, y_pred_lstm, zero_division=0)

        st.markdown("#### 📊 Model Performance on Test Data")
        st.write("**Classical Model:**")
        st.write(f"Accuracy: {clf_acc:.2f}")
        st.write(f"Precision: {clf_prec:.2f}")
        st.write(f"Recall: {clf_rec:.2f}")
        st.write(f"F1 Score: {clf_f1:.2f}")

        st.write("**Bi-LSTM Model:**")
        st.write(f"Accuracy: {lstm_acc:.2f}")
        st.write(f"Precision: {lstm_prec:.2f}")
        st.write(f"Recall: {lstm_rec:.2f}")
        st.write(f"F1 Score: {lstm_f1:.2f}")

        if lstm_acc > clf_acc:
            st.success("Bi-LSTM model performs better on the test data.")
        elif clf_acc > lstm_acc:
            st.success("Classical model performs better on the test data.")
        else:
            st.info("Both models perform equally well on the test data.")

            
if __name__ == "__main__":
    model_testing_app()
