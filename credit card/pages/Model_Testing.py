
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
    df1 = pd.read_csv(file)
    st.dataframe(df1.head())

            #st.markdown("### 🏷️ Target Inference")
    target_column = df1.columns[-1]
            #st.success(f"Last column selected as target: **{target_column}**")

         
    clf_acc = train_classical_model(df1, target_column)
    st.success(f"Classical Model Trained (Accuracy: {clf_acc:.2f})")


    lstm_acc = train_bilstm_model(df1, target_column)
    st.success(f"Bi-LSTM Model Trained (Accuracy: {lstm_acc:.2f})")
    st.markdown("### 📤 Upload a CSV file for testing")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    # Split data into features and target, then into train and test sets
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target_column])
        y = df[target_column]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        trad_model_path = "credit card/pages/classical_model.pkl"
        bilstm_model_path = "credit card/pages/bilstm_model.h5"

        if os.path.exists(trad_model_path) and os.path.exists(bilstm_model_path):
            with open(trad_model_path, "rb") as f:
                trad_model = pickle.load(f)

            expected_features = trad_model.feature_names_in_
            X_test = X_test.reindex(columns=expected_features, fill_value=0)
            X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

            y_pred_trad = trad_model.predict(X_test)

            st.subheader("🧠 Traditional Model Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_trad))
            st.write("Precision:", precision_score(y_test, y_pred_trad, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred_trad, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred_trad, average='weighted'))
            st.text(classification_report(y_test, y_pred_trad))

            # ==== Bi-LSTM Inference ====
            # Preprocessing
            X_bilstm_pre = X_test.copy()
            for col in X_bilstm_pre.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_bilstm_pre[col] = le.fit_transform(X_bilstm_pre[col].astype(str))

            scaler = StandardScaler()
            X_bilstm_pre = scaler.fit_transform(X_bilstm_pre)
            X_bilstm_pre = X_bilstm_pre.reshape((X_bilstm_pre.shape[0], 1, X_bilstm_pre.shape[1]))

            if y_test.dtype == 'object' or y_test.dtype.name == 'category':
                y_test_enc = LabelEncoder().fit_transform(y_test)
            else:
                y_test_enc = y_test

            bilstm_model = tf.keras.models.load_model(
                bilstm_model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )

            y_pred_bilstm_prob = bilstm_model.predict(X_bilstm_pre)
            if y_pred_bilstm_prob.shape[-1] > 1:
                y_pred_bilstm = y_pred_bilstm_prob.argmax(axis=-1)
            else:
                y_pred_bilstm = (y_pred_bilstm_prob > 0.5).astype(int).flatten()

            st.subheader("🧠 Bi-LSTM Model Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_bilstm))
            st.write("Precision:", precision_score(y_test, y_pred_bilstm, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred_bilstm, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred_bilstm, average='weighted'))
            st.text(classification_report(y_test, y_pred_bilstm))

            # ==== Comparison ====
            st.subheader("📊 Model Comparison")
            acc_trad = accuracy_score(y_test, y_pred_trad)
            acc_bilstm = accuracy_score(y_test, y_pred_bilstm)
            if acc_trad > acc_bilstm:
                better = "Traditional Model"
            elif acc_bilstm > acc_trad:
                better = "Bi-LSTM Model"
            else:
                better = "Both models perform equally"
            st.success(f"🧠 Better Performing Model: **{better}**")

            if acc_bilstm < 0.95:
                st.warning("⚠️ Bi-LSTM accuracy is below 95%. Consider tuning the model or improving the data.")

        else:
            st.error("Model files not found. Please make sure `.pkl` and `.h5` files are correctly placed.")


            
if __name__ == "__main__":
    model_testing_app()
