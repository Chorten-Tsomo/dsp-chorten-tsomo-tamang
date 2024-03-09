import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import joblib


def fit_preprocessors(data: pd.DataFrame) -> None:
    # Feature Selection
    continuous_features = ["LotArea", "YearBuilt"]
    categorical_features = ["Neighborhood", "ExterQual"]

    # Initialize encoders and scalers
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")

    # Fit encoders and scalers to training data
    scaler.fit(data[continuous_features])
    encoder.fit(data[categorical_features])

    # Save the encoders and scalers
    joblib.dump(encoder, "../models/encoder.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")


def apply_preprocessors(data: pd.DataFrame) -> np.ndarray:
    # Load preprocessors
    scaler = joblib.load("../models/scaler.joblib")
    encoder = joblib.load("../models/encoder.joblib")

    # Feature Selection
    continuous_features = ["LotArea", "YearBuilt"]
    categorical_features = ["Neighborhood", "ExterQual"]

    # Transform continuous features
    X_continuous_scaled = scaler.transform(data[continuous_features].copy())

    # One-hot encode categorical features
    X_categorical_encoded = encoder.transform(
        data[categorical_features].copy()
    ).toarray()

    # Merge processed features
    X_processed = np.concatenate(
        (X_continuous_scaled, X_categorical_encoded), axis=1)

    return X_processed
