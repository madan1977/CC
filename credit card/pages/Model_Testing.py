import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# --- Preprocessing Function ---
def preprocess_tabular_sequence(df, target_col, seq_len=3):
    df = df.copy()
    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Scale features
    scaler = StandardScaler()
    features = df.drop(columns=[target_col])
    features_scaled = scaler.fit_transform(features)
    labels = df[target_col].values

    # Prepare sequences for LSTM
    X_seq, y_seq = [], []
    for i in range(len(features_scaled) - seq_len):
        X_seq.append(features_scaled[i:i+seq_len])
        y_seq.append(labels[i+seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    return X_seq, y_seq, scaler, label_encoders

# --- Build BiLSTM+Attention Model ---
def build_bilstm_attention(input_shape):
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(inp)
    x = AttentionLayer()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Streamlit App ---
def model_testing_app():
    st.title("BiLSTM with Attention: Credit Card Fraud Detection")

    # Load data
    df = pd.read_csv("credit card/pages/backend_live_data.csv")
    st.write("Sample Data", df.head())

    target_col = "Fraudulent"
    seq_len = 3  # You can tune this

    # Preprocess
    X, y, scaler, label_encoders = preprocess_tabular_sequence(df, target_col, seq_len=seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build model
    model = build_bilstm_attention(X_train.shape[1:])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Test Accuracy:** {acc:.4f}")
    st.text(classification_report(y_test, y_pred))

    if acc >= 0.98:
        st.success("🎉 BiLSTM+Attention model achieved ≥98% accuracy!")
    else:
        st.warning("Accuracy below 98%. Try tuning hyperparameters or using more data.")

if __name__ == "__main__":
    model_testing_app()
