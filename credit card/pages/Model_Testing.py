def model_testing_app():
    import streamlit as st
    import pandas as pd
    import pickle
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    from pages.bilstm_model import AttentionLayer
    import os
    from sklearn.preprocessing import StandardScaler

    st.title("Model Testing: Traditional vs Bi-LSTM")

    uploaded_file = st.file_uploader("Upload Excel file with test data", type=["xlsx"])
    if uploaded_file:
        test_df = pd.read_excel(uploaded_file)
        st.write("Test Data Preview:", test_df.head())

        target_col = st.selectbox("Select target column", test_df.columns)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        trad_model_file = st.file_uploader("Upload Traditional Model (.pkl)", type=["pkl"])
        bilstm_model_file_path = "credit card/pages/bilstm_model.h5"
        # If scaler or label encoder files do not exist, fit them on X_test here
        scaler_file = "credit card/pages/scaler_bilstm.pkl"
        label_encoder_file = "credit card/pages/label_encoder_bilstm.pkl"
        if not os.path.exists(scaler_file) or not os.path.exists(label_encoder_file):
            scaler = StandardScaler()
            X_bilstm = X_test.copy()
            # Encode categorical features
            for col in X_bilstm.select_dtypes(include=['object', 'category']).columns:
            X_bilstm[col] = X_bilstm[col].astype(str)
            le = LabelEncoder()
            X_bilstm[col] = le.fit_transform(X_bilstm[col])
            scaler.fit(X_bilstm)
            # Fit label encoder on y_test if needed
            if y_test.dtype == 'object' or y_test.dtype.name == 'category':
            label_encoder = LabelEncoder()
            label_encoder.fit(y_test.astype(str))
            else:
            label_encoder = None
            # Save for future use
            with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
            if label_encoder is not None:
            with open(label_encoder_file, "wb") as f:
                pickle.dump(label_encoder, f)
        else:
            with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
            with open(label_encoder_file, "rb") as f:
            label_encoder = pickle.load(f)
            st.warning("Scaler or Label Encoder files for Bi-LSTM not found. Bi-LSTM evaluation will be skipped.")
            return
        scaler_file = "credit card/pages/scaler_bilstm.pkl"
        label_encoder_file = "credit card/pages/label_encoder_bilstm.pkl"

        if trad_model_file and bilstm_model_file_path:
            trad_model_file.seek(0)
            trad_model = pickle.load(trad_model_file)
            expected_features = trad_model.feature_names_in_
            X_test_trad = X_test.reindex(columns=expected_features, fill_value=0)
            X_test_trad = X_test_trad.apply(pd.to_numeric, errors='coerce').fillna(0)
            assert not X_test_trad.isnull().any().any(), "X_test_trad still contains NaNs!"
            y_pred_trad = trad_model.predict(X_test_trad)

            st.subheader("Traditional Model Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_trad))
            st.write("Precision:", precision_score(y_test, y_pred_trad, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred_trad, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred_trad, average='weighted'))
            st.text(classification_report(y_test, y_pred_trad))

            # --- Bi-LSTM Section ---
            # Load scaler and label encoder used during training
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
            with open(label_encoder_file, "rb") as f:
                label_encoder = pickle.load(f)

            # Encode categorical features as during training
            X_bilstm = X_test.copy()
            for col in X_bilstm.select_dtypes(include=['object', 'category']).columns:
                X_bilstm[col] = X_bilstm[col].astype(str)
                if hasattr(label_encoder, 'classes_'):
                    # Use loaded label encoder if available
                    X_bilstm[col] = label_encoder.transform(X_bilstm[col])
                else:
                    # Otherwise, fit a new one (not recommended for production)
                    le = LabelEncoder()
                    X_bilstm[col] = le.fit_transform(X_bilstm[col])

            # Feature scaling using loaded scaler
            X_bilstm_scaled = scaler.transform(X_bilstm)

            # Reshape for LSTM input: (samples, timesteps, features)
            X_bilstm_scaled = X_bilstm_scaled.reshape((X_bilstm_scaled.shape[0], 1, X_bilstm_scaled.shape[1]))

            # Encode y_test using loaded label encoder
            if y_test.dtype == 'object' or y_test.dtype.name == 'category':
                y_true = label_encoder.transform(y_test.astype(str))
            else:
                y_true = y_test.values

            # Load Bi-LSTM model with custom AttentionLayer
            bilstm_model = tf.keras.models.load_model(
                bilstm_model_file_path, custom_objects={'AttentionLayer': AttentionLayer}
            )

            # Predict with Bi-LSTM model
            y_pred_bilstm_prob = bilstm_model.predict(X_bilstm_scaled)
            # For multi-class, use argmax; for binary, use threshold
            if y_pred_bilstm_prob.shape[-1] > 1:
                y_pred_bilstm = y_pred_bilstm_prob.argmax(axis=-1)
            else:
                # Try a lower threshold for higher recall (tune as needed)
                y_pred_bilstm = (y_pred_bilstm_prob > 0.4).astype(int).flatten()

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
                st.warning("Bi-LSTM accuracy is below 95%. Consider tuning your model, using class weights, or improving preprocessing for better results.")

if __name__ == "__main__":
    model_testing_app()
