import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(r"a:\Infosys_SpringBoard\AI_Enabled_VISA_Status_Prediction_and_Processing_Time_Estimator\src\data\VisaFile.csv")
print("Loading:", csv_path)

df = pd.read_csv(csv_path, parse_dates=['application_date'])
print('\nShape:', df.shape)
print('\nColumns:')
print(list(df.columns))

print('\nDtypes:')
print(df.dtypes)

print('\nMissing values (count):')
print(df.isnull().sum())

# Quick unique counts for object columns (top 10)
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
print('\nSample unique counts (object cols):')
for c in obj_cols:
    print(f"- {c}: {df[c].nunique()} unique; top values: {df[c].value_counts().head(5).to_dict()}")

# Visa status distribution and drop Pending
print('\nvisa_status value counts:')
print(df['visa_status'].value_counts(dropna=False))

# Drop Pending for classifier as requested
df_bin = df[df['visa_status'].isin(['Approved','Rejected'])].copy()
print('\nAfter dropping Pending -> shape:', df_bin.shape)
print('visa_status counts (binary):')
print(df_bin['visa_status'].value_counts())

# Processing days target distribution
print('\nprocessing_days (describe):')
print(df['processing_days'].describe())
print('\nprocessing_time_days (describe):')
print(df['processing_time_days'].describe())

# Numeric correlations with processing_days
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'processing_days' in num_cols:
    corr = df[num_cols].corr()['processing_days'].sort_values(ascending=False)
    print('\nCorrelation of numeric features with processing_days:')
    print(corr)

# Derived feature: application month
df['application_month'] = df['application_date'].dt.month
print('\napplication_month distribution (top 12):')
print(df['application_month'].value_counts().sort_index())

# Show top categorical features that we'll consider
candidates = ['education_of_employee','has_job_experience','requires_job_training',
              'visa_type','processing_center','job_offer','documents_complete','case_status','destination_country','country_of_applicant']
print('\nCandidate categorical features top values:')
for c in candidates:
    if c in df.columns:
        print(f"\n{c}:\n", df[c].value_counts().head(8))

# Print a small sample
print('\nSample rows:')
print(df.head(5))

# Based on the above, print a recommended limited feature set (no leakage)
recommended_regression_features = [
    'visa_type', 'processing_center', 'education_of_employee',
    'has_job_experience', 'requires_job_training', 'job_offer', 'documents_complete',
    'years_experience', 'previous_visa_rejections', 'application_month'
]
recommended_classifier_features = [
    'visa_type', 'education_of_employee', 'has_job_experience',
    'requires_job_training', 'processing_center', 'job_offer', 'documents_complete',
    'years_experience', 'previous_visa_rejections', 'application_month'
]

print('\n\nRECOMMENDED FEATURES (limited list, no leakage):')
print('Regression (processing_days) features:')
print(recommended_regression_features)
print('\nClassifier (visa_status binary - Pending dropped) features:')
print(recommended_classifier_features)

print('\nNotes:')
print('- I DID NOT include `processing_time_days` or `processing_days` as features for the classifier (leakage).')
print('- I DID NOT include `visa_status` for regression.')
print('- `case_status` may be correlated with visa_status and could leak; include only if you want.')