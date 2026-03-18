import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_encoders(df, categorical_cols):
    """
    Creates and fits encoders for categorical columns.
    - Ordinal encoding for education_of_employee with proper order.
    - OneHotEncoder for nominal categories.
    Returns fitted encoder and transformed DataFrame.
    """
    # Ordinal mapping for education_of_employee
    education_mapping = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    if 'education_of_employee' in df.columns:
        df['education_of_employee'] = df['education_of_employee'].map(education_mapping)
    
    # All other categorical as nominal
    nominal_cols = [col for col in categorical_cols if col != 'education_of_employee']
    
    # OneHotEncoder for nominal
    if nominal_cols:
        ct = ColumnTransformer([('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols)], remainder='passthrough')
        df_encoded = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())
        return ct, df_encoded
    else:
        return None, df

# Usage example:
# df = pd.read_csv('your_dataset.csv')
# categorical_cols = ['nationality', 'education_level', 'job_type']
# label_encoders, onehot_encoder, df_encoded = create_encoders(df, categorical_cols)