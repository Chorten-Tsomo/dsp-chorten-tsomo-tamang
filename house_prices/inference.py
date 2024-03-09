import pandas as pd
import numpy as np
from house_prices.preprocess import apply_preprocessors
import joblib


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    # Apply preprocessors to the input data
    X_processed = apply_preprocessors(input_data)

    # Load model
    model = joblib.load('../models/model.joblib')

    # Make predictions
    predictions = model.predict(X_processed)

    return predictions
    

  




    