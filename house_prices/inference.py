import pandas as pd
import numpy as np
from house_prices.preprocess import apply_preprocessors
import joblib
from house_prices import SELECTED_FEATURES


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    X_processed = apply_preprocessors(input_data[SELECTED_FEATURES])
    model = joblib.load("../models/model.joblib")
    predictions = model.predict(X_processed)

    return predictions
