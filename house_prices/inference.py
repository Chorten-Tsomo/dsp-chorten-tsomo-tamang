import pandas as pd
import numpy as np
from house_prices.preprocess import apply_preprocessors
import joblib
from house_prices import __init__.py


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    SELECTED_FEATURES
    
    # Apply preprocessors to the input data
    X_processed = apply_preprocessors(input_data[SELECTED_FEATURES])

    # Load model
    model = joblib.load("../models/model.joblib")

    # Make predictions
    predictions = model.predict(X_processed)

    return predictions
