import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from encoders import create_encoders  # Import from the encoders file

def train_visa_model(df, target_col='accepted', categorical_cols=None):
    """
    Trains a RandomForest model to predict visa acceptance.
    Assumes df has features and target.
    """
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    if categorical_cols:
        encoder, X = create_encoders(X, categorical_cols)
    else:
        encoder = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    return model, encoder

def predict_acceptance(model, encoder, new_data_df, categorical_cols=None):
    """
    Predicts acceptance for new data.
    """
    if categorical_cols and encoder:
        # Apply same encoding
        education_mapping = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
        if 'education_of_employee' in new_data_df.columns:
            new_data_df['education_of_employee'] = new_data_df['education_of_employee'].map(education_mapping)
        new_data_df = pd.DataFrame(encoder.transform(new_data_df), columns=encoder.get_feature_names_out())
    return model.predict(new_data_df)

# Usage example:
# df = pd.read_csv('your_dataset.csv')
# categorical_cols = ['nationality', 'education_level', 'job_type']
# model = train_visa_model(df, target_col='accepted', categorical_cols=categorical_cols)
# new_applicant = pd.DataFrame({...})  # New data
# prediction = predict_acceptance(model, new_applicant, categorical_cols)