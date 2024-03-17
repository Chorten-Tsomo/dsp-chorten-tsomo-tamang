import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import joblib
from house_prices import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES


def fit_scaler(data: pd.DataFrame) -> None:
    scaler = StandardScaler()
    scaler.fit(data[CONTINUOUS_FEATURES])
    joblib.dump(scaler, "../models/scaler.joblib")


def fit_encoder(data: pd.DataFrame) -> None:
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(data[CATEGORICAL_FEATURES])
    joblib.dump(encoder, "../models/encoder.joblib")


def apply_scaler(data: pd.DataFrame, scaler_path: str) -> np.ndarray:
    scaler = joblib.load(scaler_path)
    X_continuous_scaled = scaler.transform(data[CONTINUOUS_FEATURES])
    return X_continuous_scaled


def apply_encoder(data: pd.DataFrame, encoder_path: str) -> np.ndarray:
    encoder = joblib.load(encoder_path)
    X_categorical_encoded = encoder.transform(
        data[CATEGORICAL_FEATURES]).toarray()
    return X_categorical_encoded


def merge_features(scaled_data: np.ndarray,
                   encoded_data: np.ndarray) -> np.ndarray:
    return np.concatenate((scaled_data, encoded_data), axis=1)


def fit_preprocessors(data: pd.DataFrame) -> None:
    fit_scaler(data)
    fit_encoder(data)


def apply_preprocessors(data: pd.DataFrame) -> np.ndarray:
    scaler_path = "../models/scaler.joblib"
    encoder_path = "../models/encoder.joblib"
    scaled_data = apply_scaler(data, scaler_path)
    encoded_data = apply_encoder(data, encoder_path)
    return merge_features(scaled_data, encoded_data)
