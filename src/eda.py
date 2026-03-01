import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visual_insights(df):
    """
    Executes high-level EDA to visualize trends and bottlenecks.
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Target Variable Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['processing_time_days'], kde=True, color='teal')
    plt.title('Distribution of Visa Processing Times')
    plt.xlabel('Days')
    plt.show()

    # 2. Categorical Analysis: Visa Type vs. Time
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='visa_type', y='processing_time_days', data=df, palette='Set2')
    plt.title('Processing Time Variance by Visa Type')
    plt.xticks(rotation=45)
    plt.show()
    

    # 3. Correlation Matrix (Identifying Predictor Strength)
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()
    

def engineer_scalable_features(df):
    """
    Creates unique features to improve model accuracy and scalability.
    """
    # Feature 1: Center Workload Intensity
    # Captures how many applications are currently being handled by each processing center
    df['center_workload'] = df.groupby('processing_center')['case_id'].transform('count')
    
    # Feature 2: Historical Country Delay
    # Maps the average historical processing time for each country of origin
    df['country_avg_delay'] = df.groupby('country_of_applicant')['processing_time_days'].transform('mean')
    
    # Feature 3: Wage-to-Education Interaction
    # Creates a multiplier based on education level and wage (Assumes PhD/High Wage = Priority)
    edu_map = {'PhD': 4, "Master's": 3, "Bachelor's": 2, 'High School': 1}
    df['edu_rank'] = df['education_of_employee'].map(edu_map)
    df['priority_score'] = df['prevailing_wage_annual'] * df['edu_rank']
    
    # Feature 4: Seasonal Congestion Index
    # Calculates mean delay for each application season
    df['seasonal_index'] = df.groupby('application_season')['processing_time_days'].transform('mean')

    return df

def prepare_for_modeling(df):
    """
    Encodes categorical variables into a numerical form.
    """
    # High-impact categorical columns for One-Hot Encoding
    categorical_cols = ['continent', 'education_of_employee', 'processing_center', 'visa_type', 'application_season']
    
    # Generate dummy variables while avoiding the dummy variable trap (drop_first=True)
    df_ml = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_ml

if __name__ == "__main__":
    # Assume 'df' is our already preprocessed dataframe
    # df = pd.read_csv('data\\VisaFile.csv')

    # Execute EDA
    generate_visual_insights(df)

    # Execute Engineering
    df = engineer_scalable_features(df)

    # Prepare final dataset for the predictive model
    final_df = prepare_for_modeling(df)

  
    print(f"Total columns for training: {len(final_df.columns)}")
    print(final_df[['center_workload', 'country_avg_delay', 'priority_score', 'seasonal_index']].head())
