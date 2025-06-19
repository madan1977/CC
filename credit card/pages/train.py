import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

    with open("credit card/pages/classical_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return acc

def train_bilstm_model(df, target_column):
    X, y = preprocess(df, target_column)

    # Normalize & reshape for LSTM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.2, random_state=42)

    model = build_bilstm_model(input_shape=(1, X.shape[1]), output_dim=y_cat.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    acc = model.evaluate(X_test, y_test, verbose=1)[1]

    model.save("credit card/pages/bilstm_model.h5")
    return acc
