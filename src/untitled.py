import os
import pandas as pd
import numpy as np

FILE_LOCATION = r"data\VisaFile.csv"

# STEP 1: Load Data
def read_dataset(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Visa dataset not found at: {file_path}")
    return pd.read_csv(file_path, encoding="latin1", low_memory=False)

# STEP 2: Clean Column Names
def normalize_headers(dataframe):
    dataframe.columns = [col.strip().lower().replace(" ", "_") for col in dataframe.columns]
    return dataframe

# STEP 3: Handle Missing Values & Duplicates
def treat_missing_data(dataframe):
    dataframe = dataframe.drop_duplicates().copy()
    
    # Drop columns with >60% missing values
    thresh = int(0.4 * len(dataframe))
    dataframe = dataframe.dropna(axis=1, thresh=thresh)

    for col in dataframe.columns:
        if dataframe[col].dtype == "O":
            dataframe[col] = dataframe[col].fillna("unknown")
        else:
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())
    return dataframe

# STEP 4: Date Processing
def generate_processing_time(dataframe):
    # Ensure date columns exist before processing
    date_cols = ["case_received_date", "decision_date"]
    if all(col in dataframe.columns for col in date_cols):
        dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime, errors="coerce")
        dataframe = dataframe.dropna(subset=date_cols)
        
        dataframe["processing_time_days"] = (
            dataframe["decision_date"] - dataframe["case_received_date"]
        ).dt.days
        
        # Keep only valid logical durations
        dataframe = dataframe[dataframe["processing_time_days"] >= 0]
    return dataframe

# STEP 5: Feature Engineering (Specific to your Columns)
def perform_feature_engineering(df):
    # 1. Normalize Wage to Annual (Yearly)
    if 'prevailing_wage' in df.columns and 'unit_of_wage' in df.columns:
        # Standard work conversion: 2080 hours/year, 52 weeks/year, 12 months/year
        multipliers = {'Hour': 2080, 'Week': 52, 'Month': 12, 'Year': 1}
        df['prevailing_wage_annual'] = df.apply(
            lambda x: x['prevailing_wage'] * multipliers.get(x['unit_of_wage'], 1), axis=1
        )

    # 2. Calculate Company Age
    if 'yr_of_estab' in df.columns:
        current_year = 2026 # Updated to current context
        df['company_age'] = current_year - df['yr_of_estab']
        df['company_age'] = df['company_age'].apply(lambda x: x if x >= 0 else 0)

    # 3. Binary Encoding for Experience/Training
    binary_map = {'Y': 1, 'N': 0}
    for col in ['has_job_experience', 'requires_job_training']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    return df

# MAIN EXECUTION
def run_preprocessing():
    print("🚀 Starting Preprocessing...")
    
    df = read_dataset(FILE_LOCATION)
    df = normalize_headers(df)
    df = treat_missing_data(df)
    df = generate_processing_time(df)
    df = perform_feature_engineering(df)

    print("\n✅ Preprocessing Completed")
    print(f"Final Dataset Shape: {df.shape}")
    print("\nColumns Available:\n", df.columns.tolist())
    
    # Save the cleaned data for model training
    # df.to_csv("cleaned_visa_data.csv", index=False)
    
    return df

if __name__ == "__main__":
    final_df = run_preprocessing()
    print(final_df.head())