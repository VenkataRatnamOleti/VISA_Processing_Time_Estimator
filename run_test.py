import pandas as pd
import sys
sys.path.append('.')

from encoders import create_encoders
from visa_acceptance_model import train_visa_model
import joblib

# Load data
df = pd.read_csv('src/data/VisaFile.csv')

# Drop unnecessary columns
df = df.drop(['case_id', 'case_status'], axis=1, errors='ignore')

# Create target: accepted = 1 if Certified, 0 otherwise (but since we dropped case_status, wait no)
# Wait, need case_status for target
df['accepted'] = pd.read_csv('src/data/VisaFile.csv')['case_status'].apply(lambda x: 1 if x == 'Certified' else 0)

# Categorical columns
categorical_cols = ['continent', 'country_of_applicant', 'education_of_employee', 'has_job_experience', 'requires_job_training', 'region_of_employment', 'unit_of_wage', 'full_time_position', 'visa_type', 'application_season', 'processing_center']

# Train model
model, encoder = train_visa_model(df, target_col='accepted', categorical_cols=categorical_cols)

# Save model, encoder, and feature columns
joblib.dump(model, 'models/visa_acceptance_model.pkl')
joblib.dump(encoder, 'models/visa_acceptance_encoder.pkl')
joblib.dump(list(df.drop('accepted', axis=1).columns), 'models/acceptance_features.pkl')

print("Model trained and saved successfully.")