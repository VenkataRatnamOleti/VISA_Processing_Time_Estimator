import os
import joblib
import pandas as pd
import numpy as np

# Prevent pandas from showing downcasting warnings
pd.set_option('future.no_silent_downcasting', True)

# Define paths to your saved model artifacts (try several common names)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "..", "models")
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, 'processing_days_model.pkl'),
    os.path.join(MODELS_DIR, 'visa_model.pkl'),
    os.path.join(MODELS_DIR, 'visa_model_pipeline.pkl')
]
RMSE_CANDIDATES = [os.path.join(MODELS_DIR, 'model_rmse.pkl'), os.path.join(MODELS_DIR, 'processing_days_rmse.pkl')]
FEATURES_CANDIDATES = [os.path.join(MODELS_DIR, 'selected_features.pkl'), os.path.join(MODELS_DIR, 'model_features.pkl'), os.path.join(MODELS_DIR, 'model_features.pkl')]

def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


class VisaProcessingEstimator:
    def __init__(self):
        """Load the trained model and feature structure."""
        # attempt to find model, rmse, and features from common filenames
        model_path = find_first_existing(MODEL_CANDIDATES)
        rmse_path = find_first_existing(RMSE_CANDIDATES)
        feats_path = find_first_existing(FEATURES_CANDIDATES)
        try:
            if model_path is None:
                raise FileNotFoundError(f"No model file found in candidates: {MODEL_CANDIDATES}")
            self.predictor = joblib.load(model_path)
            if rmse_path and os.path.exists(rmse_path):
                try:
                    self.error_val = joblib.load(rmse_path)
                except Exception:
                    self.error_val = None
            else:
                self.error_val = None

            if feats_path and os.path.exists(feats_path):
                try:
                    self.model_columns = joblib.load(feats_path)
                except Exception:
                    self.model_columns = None
            else:
                self.model_columns = None

            self.active = True
            print(f"[+] Inference Engine successfully loaded from {model_path}.")
        except Exception as e:
            print(f"[-] Critical Error: Could not load model files. {e}")
            self.predictor = None
            self.error_val = None
            self.model_columns = None
            self.active = False

    def get_estimation(self, user_inputs):
        """
        Transforms user input to match training features and returns prediction.
        """
        if not self.active:
            return {"error": "Engine offline."}

        # If the model expects a specific feature list (pipeline created earlier), align to it
        # Create DataFrame from user inputs
        input_df = pd.DataFrame([user_inputs])
        # derive application_month if application_date present
        if 'application_date' in input_df.columns:
            try:
                input_df['application_date'] = pd.to_datetime(input_df['application_date'], errors='coerce')
                input_df['application_month'] = input_df['application_date'].dt.month
            except Exception:
                input_df['application_month'] = pd.NA

        if self.model_columns:
            # ensure all required columns exist
            for c in self.model_columns:
                if c not in input_df.columns:
                    input_df[c] = pd.NA
            proc = input_df[self.model_columns]
        else:
            # fallback: use all columns provided, with dummies
            proc = pd.get_dummies(input_df).fillna(0)

        # Predict using the loaded predictor
        try:
            preds = self.predictor.predict(proc)
            raw_pred = float(preds[0])
        except Exception as e:
            return {"error": f"prediction failed: {e}"}

        # uncertainty window
        if self.error_val is not None:
            deviation = max(self.error_val * 0.2, raw_pred * 0.1)
        else:
            deviation = max(1.0, raw_pred * 0.1)

        return {
            "estimated_days": round(raw_pred, 1),
            "window": f"{int(max(0, raw_pred - deviation))} to {int(raw_pred + deviation)} days"
        }

if __name__ == "__main__":
    # Test execution
    estimator = VisaProcessingEstimator()
    
    # Example input matching your dataset structure
    sample_request = {
        "no_of_employees": 150,
        "yr_of_estab": 2005,
        "prevailing_wage": 6500,
        "unit_of_wage": "Monthly",
        "continent": "Asia",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "region_of_employment": "West",
        "full_time_position": "Y",
        "visa_type": "Work",
        "processing_center": "Center B",
        "application_season": "Summer",
        "app_year": 2024,
        "app_month": 6,
        "app_quarter": 2,
        "is_peak_season": 0
    }

    if estimator.active:
        result = estimator.get_estimation(sample_request)
        print("\n" + "="*40)
        print(" VISA PROCESSING ESTIMATE ")
        print("="*40)
        print(f"Prediction: {result['estimated_days']} Days")
        print(f"Confidence Window: {result['window']}")
        print("="*40)