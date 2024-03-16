import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import fit_preprocessors, apply_preprocessors
import numpy as np
import joblib


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    # Compute Root Mean Squared Log Error
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict:
    SELECTED_FEATURES

    # Separate data into features (X) and target (y)
    X = data[SELECTED_FEATURES]
    y = data["SalePrice"]

    # Splitting into train and test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit preprocessors
    fit_preprocessors(X_train)

    # Preprocess the data
    X_train_processed = apply_preprocessors(X_train)
    X_val_processed = apply_preprocessors(X_val)

    # Model
    model = LinearRegression()

    if X_train_processed is not None and y_train is not None:
        X_train_processed = np.atleast_2d(X_train_processed)
        if len(y_train.shape) == 1:
            y_train = np.atleast_2d(y_train).reshape(-1, 1)
        model.fit(X_train_processed, y_train)

        # Model performance evaluation
        predictions = model.predict(X_val_processed)

        # Persist the model
        joblib.dump(model, "../models/model.joblib")

        # Return model performances
        return {"rmsle": compute_rmsle(y_val, predictions)}
    else:
        return {"error": "Input data is None or not properly formatted"}
