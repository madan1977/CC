
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

         
    clf_acc = train_classical_model(df, target_column)
    st.success(f"Classical Model Trained (Accuracy: {clf_acc:.2f})")


    lstm_acc = train_bilstm_model(df, target_column)
    st.success(f"Bi-LSTM Model Trained (Accuracy: {lstm_acc:.2f})")

    # Split data into features and target, then into train and test sets
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


    #trad_model_file = st.file_uploader("Upload Traditional Model (.pkl)", type=["pkl"])
        # Upload Bi-LSTM model
        #bilstm_model_file = st.file_uploader("Upload Bi-LSTM Model (.h5)", type=["h5"])
    trad_model_file = "credit card/pages/classical_model.pkl"
    bilstm_model_file_path = "credit card/pages/bilstm_model.h5"
    if trad_model_file and bilstm_model_file_path:
            # Load traditional model
            #trad_model_file.seek(0)
            with open(trad_model_file, "rb") as f:
                  trad_model = pickle.load(f)
            #trad_model = pickle.load(trad_model_file)
            # Align columns
            expected_features = trad_model.feature_names_in_
            X_test = X_test.reindex(columns=expected_features, fill_value=0)

            # Ensure all columns are numeric
            X_test = X_test.apply(pd.to_numeric, errors='coerce')
            #assert not X_test.isnull().any().any(), "X_test contains NaNs after conversion!"
            # After all preprocessing and before prediction:
            X_test = X_test.fillna(0)  # or use another strategy if 0 is not appropriate

            # Now the assertion should pass
            assert not X_test.isnull().any().any(), "X_test still contains NaNs!"

            # Predict
            y_pred_trad = trad_model.predict(X_test)
            st.subheader("Traditional Model Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_trad))
            st.write("Precision:", precision_score(y_test, y_pred_trad, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred_trad, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred_trad, average='weighted'))
            st.text(classification_report(y_test, y_pred_trad))
            # Load Bi-LSTM model
            #bilstm_model = tf.keras.models.load_model(bilstm_model_file)
            # Load Bi-LSTM model with custom AttentionLayer
            # Preprocessing for Bi-LSTM model
            # 1. Encode categorical variables if any

            X_bilstm_pre = X_test.copy()
            for col in X_bilstm_pre.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_bilstm_pre[col] = le.fit_transform(X_bilstm_pre[col].astype(str))

            # 2. Feature scaling (StandardScaler)
            scaler = StandardScaler()
            X_bilstm_pre = scaler.fit_transform(X_bilstm_pre)

            # 3. Reshape for LSTM input: (samples, timesteps, features)
            X_bilstm_pre = X_bilstm_pre.reshape((X_bilstm_pre.shape[0], 1, X_bilstm_pre.shape[1]))

            # 4. Encode y_test if categorical
            if y_test.dtype == 'object' or y_test.dtype.name == 'category':
                y_test_enc = LabelEncoder().fit_transform(y_test)
            else:
                y_test_enc = y_test

            # 5. Optionally, balance classes if highly imbalanced (not shown here, but consider SMOTE or similar)
            bilstm_model = tf.keras.models.load_model(
                bilstm_model_file_path, custom_objects={'AttentionLayer': AttentionLayer}
            )
            bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            #bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Prepare X_test for Bi-LSTM: (samples, timesteps, features)
            X_bilstm = X_test.values
            if X_bilstm.ndim == 2:
                X_bilstm = X_bilstm.reshape((X_bilstm.shape[0], 1, X_bilstm.shape[1]))

            # Optional: Feature scaling for neural network (improves accuracy)
            scaler = StandardScaler()
            X_bilstm_scaled = scaler.fit_transform(X_bilstm.reshape(X_bilstm.shape[0], -1))
            X_bilstm_scaled = X_bilstm_scaled.reshape(X_bilstm.shape)

            # Predict with Bi-LSTM model
            y_pred_bilstm_prob = bilstm_model.predict(X_bilstm_scaled)
            # Convert probabilities to class labels
            if y_pred_bilstm_prob.shape[-1] > 1:
                y_pred_bilstm = y_pred_bilstm_prob.argmax(axis=-1)
            else:
                y_pred_bilstm = (y_pred_bilstm_prob > 0.5).astype(int).flatten()

            # Align y_test type for metrics
            y_true = y_test.values if hasattr(y_test, "values") else y_test

            st.subheader("Bi-LSTM Model Results")
            st.write("Accuracy:", accuracy_score(y_true, y_pred_bilstm))
            st.write("Precision:", precision_score(y_true, y_pred_bilstm, average='weighted'))
            st.write("Recall:", recall_score(y_true, y_pred_bilstm, average='weighted'))
            st.write("F1 Score:", f1_score(y_true, y_pred_bilstm, average='weighted'))
            st.text(classification_report(y_true, y_pred_bilstm))

            # Compare models
            st.subheader("Model Comparison")
            trad_acc = accuracy_score(y_true, y_pred_trad)
            bilstm_acc = accuracy_score(y_true, y_pred_bilstm)
            if trad_acc > bilstm_acc:
                better = "Traditional Model"
            elif bilstm_acc > trad_acc:
                better = "Bi-LSTM Model"
            else:
                better = "Both models perform equally"
            st.write(f"Better Model: **{better}**")

            if bilstm_acc < 0.95:
                st.warning("Bi-LSTM accuracy is below 95%. Consider tuning your model or preprocessing steps for better results.")



            
if __name__ == "__main__":
    model_testing_app()
