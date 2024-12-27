import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def r2_in_production():
    # Step 1: Load the data (assuming it's in a CSV file)
    data = pd.read_csv("model_predictions.csv")

    # Step 2: Extract true and predicted targets
    true_targets = data['true_target']  # The actual values
    predicted_targets = data['predicted_target']  # The predicted values by the model

    # Step 4: Compute R-squared (R²)
    r2 = r2_score(true_targets, predicted_targets)
    
    return r2
