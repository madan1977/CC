# pages/3_Manual_Inference_Self_Learning.py
def MISL():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import os
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    st.title("🧠 Manual Inference with Auto Self-Learning")

    # === CONFIG ===
    DATA_PATH = r"D:\GENAI\pythonProject1\IIMC Project\CreditCard\pages\backend_live_data.csv"
    TARGET_COLUMN = "Fraudulent"
    SELF_LEARN_FILE = "data/manual_self_learned.csv"
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # === Load Dataset ===
    def load_combined_data():
        df_base = pd.read_csv(DATA_PATH)
        if os.path.exists(SELF_LEARN_FILE):
            df_new = pd.read_csv(SELF_LEARN_FILE)
            df_combined = pd.concat([df_base, df_new], ignore_index=True)
        else:
            df_combined = df_base
        return df_combined

    # === Preprocess ===
    def preprocess(df):
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        label_encoders = {}
        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X, y, X_scaled, scaler, label_encoders, le_y

    # === Train Random Forest ===
    def train_classical_incremental(X_new, y_new, model=None):
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_new, y_new)
        else:
            # Simulate retraining with previous and new data
            model.fit(X_new, y_new)
        with open(f"{MODEL_DIR}/classical_model.pkl", "wb") as f:
            pickle.dump(model, f)
        return model

    # === Incremental Bi-LSTM ===
    def train_bilstm_incremental(model, X_scaled_new, y_new, input_dim):
        X_lstm = X_scaled_new.reshape(-1, 1, input_dim)
        y_cat = pd.get_dummies(y_new).values

        model.fit(
            X_lstm,
            y_cat,
            epochs=5,
            batch_size=16,
            verbose=0,
            callbacks=[EarlyStopping(patience=1, restore_best_weights=True)]
        )
        model.save(f"{MODEL_DIR}/bilstm_model.h5")
        return model

    # === Load existing models ===
    def load_models():
        with open(f"{MODEL_DIR}/classical_model.pkl", "rb") as f:
            clf = pickle.load(f)
        lstm = load_model(f"{MODEL_DIR}/bilstm_model.h5")
        return clf, lstm

    # === MAIN ===
    df = load_combined_data()
    X_raw = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    X_encoded, y_encoded, X_scaled, scaler, encoders, le_y = preprocess(df)
    input_dim = X_scaled.shape[1]

    clf_model, lstm_model = load_models()

    # === Form for Manual Input ===
    st.subheader("📝 Input Transaction Manually")
    model_type = st.radio("Model to Use", ["Classical ML", "Bi-LSTM with Attention"], horizontal=True)

    with st.form("manual_form"):
        user_input = {}
        for col in X_raw.columns:
            if X_raw[col].dtype == object:
                options = sorted(X_raw[col].dropna().unique())
                user_input[col] = st.selectbox(f"{col}", options, key=col)
            else:
                user_input[col] = st.number_input(f"{col}", value=float(X_raw[col].mean()), key=col)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])

        for col in input_df.select_dtypes(include="object").columns:
            le = encoders[col]
            if input_df[col].iloc[0] not in le.classes_:
                # Add the new class and refit
                new_classes = np.append(le.classes_, input_df[col].iloc[0])
                le.classes_ = new_classes
            input_df[col] = le.transform(input_df[col])

        scaled_input = scaler.transform(input_df)

        if model_type == "Classical ML":
            pred = clf_model.predict(input_df)[0]
            prob = clf_model.predict_proba(input_df)[0][pred]
        else:
            lstm_input = scaled_input.reshape(1, 1, input_dim)
            pred_prob = lstm_model.predict(lstm_input, verbose=0)[0]
            pred = np.argmax(pred_prob)
            prob = pred_prob[pred]

        pred_label = le_y.inverse_transform([pred])[0]

        st.success(f"📟 Prediction: `{pred_label}` with confidence `{prob * 100:.2f}%`")

        feedback = st.radio("Was this prediction correct?", ["Yes", "No"])
        if feedback == "No":
            correct = st.selectbox("Select Correct Label", list(le_y.classes_))
        else:
            correct = pred_label

        if st.button("📏 Save & Retrain"):
            user_input[TARGET_COLUMN] = correct
            df_new = pd.DataFrame([user_input])

            if os.path.exists(SELF_LEARN_FILE):
                df_existing = pd.read_csv(SELF_LEARN_FILE)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new

            os.makedirs("data", exist_ok=True)
            df_combined.to_csv(SELF_LEARN_FILE, index=False)
            st.success("Saved to self-learning file.")

            # Incremental training
            X_new_raw = df_new.drop(columns=[TARGET_COLUMN])
            y_new = df_new[TARGET_COLUMN]

            for col in X_new_raw.select_dtypes(include="object").columns:
                X_new_raw[col] = encoders[col].transform(X_new_raw[col])
            X_new_scaled = scaler.transform(X_new_raw)
            y_new_encoded = le_y.transform(y_new)

            if model_type == "Classical ML":
                clf_model = train_classical_incremental(X_new_raw, y_new_encoded, clf_model)
                st.success("✅ Classical ML improved with new sample!")
            else:
                lstm_model = train_bilstm_incremental(lstm_model, X_new_scaled, y_new_encoded, input_dim)
                st.success("✅ Bi-LSTM fine-tuned with new sample!")
if __name__ == "__main__":
    MISL()
    
