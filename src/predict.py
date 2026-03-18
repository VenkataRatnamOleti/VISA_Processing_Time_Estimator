import os
import joblib
import pandas as pd
import numpy as np

# Prevent pandas from showing downcasting warnings
pd.set_option('future.no_silent_downcasting', True)

# Define paths to your saved model artifacts
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_PATH, "..", "models", "visa_model.pkl")
RMSE_FILE = os.path.join(BASE_PATH, "..", "models", "model_rmse.pkl")
SCHEMA_FILE = os.path.join(BASE_PATH, "..", "models", "model_features.pkl")
ACCEPTANCE_MODEL_FILE = os.path.join(BASE_PATH, "..", "models", "visa_acceptance_model.pkl")
ACCEPTANCE_ENCODER_FILE = os.path.join(BASE_PATH, "..", "models", "visa_acceptance_encoder.pkl")
ACCEPTANCE_FEATURES_FILE = os.path.join(BASE_PATH, "..", "models", "acceptance_features.pkl")

class VisaProcessingEstimator:
    def __init__(self):
        """Load the trained model and feature structure."""
        try:
            self.predictor = joblib.load(MODEL_FILE)
            self.error_val = joblib.load(RMSE_FILE)
            self.model_columns = joblib.load(SCHEMA_FILE)
            self.acceptance_model = joblib.load(ACCEPTANCE_MODEL_FILE)
            self.acceptance_encoder = joblib.load(ACCEPTANCE_ENCODER_FILE)
            self.acceptance_features = joblib.load(ACCEPTANCE_FEATURES_FILE)
            self.active = True
            print("[+] Inference Engine successfully loaded.")
        except Exception as e:
            print(f"[-] Critical Error: Could not load model files. {e}")
            self.active = False

    def get_estimation(self, user_inputs):
        """
        Transforms user input to match training features and returns prediction.
        """
        if not self.active:
            return {"error": "Engine offline."}

        # 1. ENGINEER FEATURES (Must match train_model.py exactly)
        # Create company age
        user_inputs['company_age'] = 2024 - user_inputs.get('yr_of_estab', 2000)
        
        # Calculate annual wage based on unit
        wage_val = user_inputs.get('prevailing_wage', 0)
        unit_type = user_inputs.get('unit_of_wage', 'Yearly')
        rates = {'Monthly': 12, 'Weekly': 52, 'Hourly': 2080, 'Yearly': 1}
        user_inputs['prevailing_wage_annual'] = wage_val * rates.get(unit_type, 1)

        # 2. ALIGN WITH TRAINING SCHEMA
        # Convert dictionary to DataFrame and apply One-Hot Encoding (Dummies)
        input_df = pd.get_dummies(pd.DataFrame([user_inputs]))
        
        # Create a blank DataFrame with all training columns initialized to 0
        final_processed_df = pd.DataFrame(columns=self.model_columns)
        
        # Merge user input into the full feature set
        final_processed_df = pd.concat([final_processed_df, input_df], ignore_index=True).fillna(0)
        
        # CRITICAL: Reorder and filter columns to match training set exactly
        final_processed_df = final_processed_df[self.model_columns].astype(float)

        # 3. PREDICT
        # 
        raw_pred = self.predictor.predict(final_processed_df)[0]
        
        # Calculate uncertainty window
        deviation = max(self.error_val * 0.2, raw_pred * 0.1)
        
        return {
            "estimated_days": round(float(raw_pred), 1),
            "window": f"{int(raw_pred - deviation)} to {int(raw_pred + deviation)} days"
        }

    def get_acceptance(self, user_inputs):
        """
        Predicts if the visa application will be accepted or rejected.
        """
        if not self.active:
            return {"error": "Engine offline."}

        # Preprocess input
        input_df = pd.DataFrame([user_inputs])
        
        # Filter to expected features
        input_df = input_df[self.acceptance_features]
        
        # Apply ordinal encoding to education
        education_mapping = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
        if 'education_of_employee' in input_df.columns:
            input_df['education_of_employee'] = input_df['education_of_employee'].map(education_mapping)
        
        # Encode using the fitted encoder
        input_encoded = pd.DataFrame(self.acceptance_encoder.transform(input_df), columns=self.acceptance_encoder.get_feature_names_out())
        
        # Predict
        pred = self.acceptance_model.predict(input_encoded)[0]
        
        return {"acceptance": "Accepted" if pred == 1 else "Rejected"}

if __name__ == "__main__":
    # Test execution
    estimator = VisaProcessingEstimator()
    
    # Example input matching your dataset structure
    sample_request = {
        "no_of_employees": 10,
        "yr_of_estab": 1900,
        "prevailing_wage": 50000,
        "unit_of_wage": "Monthly",
        "continent": "Asia",
        "country_of_applicant": "Unknown",
        "education_of_employee": "Master's",
        "has_job_experience": "N",
        "requires_job_training": "N",
        "region_of_employment": "West",
        "full_time_position": "N",
        "visa_type": "Work",
        "processing_center": "Center B",
        "application_season": "Summer",
        "processing_time_days": 0,  # Dummy value for acceptance prediction
        "app_year": 2024,
        "app_month": 1,
        "app_quarter": 2,
        "is_peak_season": 0
    }

    if estimator.active:
        result = estimator.get_estimation(sample_request)
        print("\n" + "="*40)
        print(" VISA PROCESSING ESTIMATE ")
        print("="*40)
        print(f"Prediction: {result['estimated_days']} Days")
        
        acceptance_result = estimator.get_acceptance(sample_request)
        print(f"Acceptance: {acceptance_result['acceptance']}")