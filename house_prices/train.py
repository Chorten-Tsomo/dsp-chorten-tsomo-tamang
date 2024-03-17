import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import fit_preprocessors, apply_preprocessors
import numpy as np
import joblib
from house_prices import SELECTED_FEATURES


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict:
    X = data[SELECTED_FEATURES]
    y = data["SalePrice"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    fit_preprocessors(X_train)
    X_train_processed = apply_preprocessors(X_train)
    X_val_processed = apply_preprocessors(X_val)
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_val_processed)
    rmsle = compute_rmsle(y_val, predictions)
    joblib.dump(model, "../models/model.joblib")
    return {"rmsle": rmsle}
