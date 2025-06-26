import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import pickle
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # pages/
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # StreamlitCreditCardFraud/
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from pages.bilstm_model import build_bilstm_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE


def preprocess(df, target_column):
    df = df.copy()

    # Encode categorical columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

def train_classical_model(df, target_column):
    X, y = preprocess(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    with open(os.path.join(current_dir, "classical_model.pkl"), "wb") as f:
           pickle.dump(model, f)

    return acc


def train_bilstm_model(df, target_column):
    df = df.copy()
    import tensorflow as tf
    import streamlit as st
    import random

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        st.success(f"✅ GPU detected: {gpus}")
    else:
        st.warning("⚠️ GPU not detected, training on CPU.")

    st.write("This compares the time taken for a large matrix multiplication using CPU and GPU.")
    # Hide the Streamlit slider by setting matrix_size directly
    matrix_size = 5000  # Default value, adjust as needed
    #matrix_size = st.slider("Matrix Size", min_value=1000, max_value=10000, step=1000, value=5000)

    def run_benchmark(device_name):
        with tf.device(device_name):
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])
            start = time.time()
            c = tf.matmul(a, b)
            end = time.time()
            return end - start

    # Run on CPU
    st.subheader("🧠 CPU Benchmark")
    cpu_time = run_benchmark("/CPU:0")
    st.success(f"CPU Time: {cpu_time:.4f} seconds")

    # Check for GPU and run if available
    gpu_devices = tf.config.list_physical_devices("GPU")

    if gpu_devices:
        st.subheader("⚡ GPU Benchmark")
        gpu_time = run_benchmark("/GPU:0")
        st.success(f"GPU Time: {gpu_time:.4f} seconds")

        # Comparison
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        st.info(f"🚀 Speedup: GPU is approximately **{speedup:.2f}x** faster than CPU")
    else:
        st.warning("❌ No GPU detected. Using CPU only.")

    # Display detected devices
    st.sidebar.header("Device Info")
    st.sidebar.write("TensorFlow version:", tf.__version__)
    st.sidebar.write("Available devices:")
    for device in tf.config.list_logical_devices():
        st.sidebar.write(f"- {device.name}")

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Apply SMOTE to balance class distribution
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Reshape for LSTM input
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    #X_reshaped = X_reshaped.astype(np.float32)

    # Determine output dimensions and loss function
    num_classes = len(np.unique(y_resampled))
    if num_classes == 2:
        y_final = y_resampled
        loss_fn = 'binary_crossentropy'
        output_dim = 1
    else:
        y_final = to_categorical(y_resampled)
        loss_fn = 'categorical_crossentropy'
        output_dim = num_classes

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_final, test_size=0.2, random_state=42
    )

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_resampled),
        y=y_resampled
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Build Bi-LSTM model (with Attention Layer inside)
    from pages.bilstm_model import build_bilstm_model
    model = build_bilstm_model(
        input_shape=(1, X.shape[1]),
        output_dim=output_dim,
        loss_fn=loss_fn,
        lstm_units_1=256,
        lstm_units_2=128,
        dense_units=256,
        dropout_1=0.2,
        dropout_2=0.2,
        dropout_dense=0.2,
        learning_rate=0.0001,
        
    )

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)

    # Train
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stop, lr_scheduler],
        class_weight=class_weight_dict,
        verbose=1
    )
    acc = model.evaluate(X_test, y_test, verbose=1)[1]
    # Evaluate
    y_pred_prob = model.predict(X_test)
    if output_dim == 1:
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = y_test
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Save model and preprocessing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(current_dir, "bilstm_model.h5"))
    with open(os.path.join(current_dir, "bilstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(current_dir, "bilstm_labelencoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(current_dir, "bilstm_labelencoder.pkl"), "wb") as f:
        pickle.dump(le_target, f)

    return acc, macro_f1
