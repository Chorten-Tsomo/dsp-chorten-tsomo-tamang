import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import joblib
from house_prices import __init__.py


def fit_preprocessors(data: pd.DataFrame) -> None:
    CONTINUOUS_FEATURES
    CATEGORICAL_FEATURES

    # Initialize encoders and scalers
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")

    # Fit encoders and scalers to training data
    scaler.fit(data[CONTINUOUS_FEATURES])
    encoder.fit(data[CATEGORICAL_FEATURES])

    # Save the encoders and scalers
    joblib.dump(encoder, "../models/encoder.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")


def apply_preprocessors(data: pd.DataFrame) -> np.ndarray:
    # Load preprocessors
    scaler = joblib.load("../models/scaler.joblib")
    encoder = joblib.load("../models/encoder.joblib")
    CONTINUOUS_FEATURES
    CATEGORICAL_FEATURES

    # Transform continuous features
    X_continuous_scaled = scaler.transform(data[CONTINUOUS_FEATURES].copy())

    # One-hot encode categorical features
    X_categorical_encoded = encoder.transform(
        data[CATEGORICAL_FEATURES].copy()
    ).toarray()

    # Merge processed features
    X_processed = np.concatenate(
        (X_continuous_scaled, X_categorical_encoded), axis=1)

    return X_processed
