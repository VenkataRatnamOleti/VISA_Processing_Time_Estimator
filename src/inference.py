import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE.parent / 'models'

def load_artifacts():
    reg = joblib.load(MODELS_DIR / 'processing_days_model.pkl')
    clf = joblib.load(MODELS_DIR / 'visa_status_model.pkl')
    features = joblib.load(MODELS_DIR / 'selected_features.pkl')
    return reg, clf, features


def prepare_input(user_input, features):
    # user_input: dict
    df = pd.DataFrame([user_input])
    # derive application_month if application_date provided
    if 'application_date' in df.columns:
        df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
        df['application_month'] = df['application_date'].dt.month
    # ensure all features present and normalize types
    # Treat these as numeric features
    numeric_feats = set(['years_experience', 'previous_visa_rejections', 'application_month'])
    for f in features:
        if f not in df.columns:
            # set sensible defaults: np.nan for numerics, 'Missing' for categoricals
            if f in numeric_feats:
                df[f] = np.nan
            else:
                df[f] = 'Missing'
        else:
            # leave existing values; will coerce below
            pass

    # Coerce numeric features to numeric dtype (np.nan for non-convertible)
    for nf in numeric_feats:
        if nf in df.columns:
            df[nf] = pd.to_numeric(df[nf], errors='coerce')

    # Ensure categorical features are strings and fill missing
    for f in features:
        if f not in numeric_feats:
            df[f] = df[f].fillna('Missing').astype(str)

    # return columns in the expected order
    return df[features]


def predict_processing_days(user_input):
    reg, _, features = load_artifacts()
    X = prepare_input(user_input, features)
    pred = reg.predict(X)[0]
    return float(pred)


def predict_visa_status(user_input):
    _, clf, features = load_artifacts()
    X = prepare_input(user_input, features)
    prob = clf.predict_proba(X)[0,1] if hasattr(clf, 'predict_proba') else None
    pred = clf.predict(X)[0]
    label = 'Approved' if pred == 1 else 'Rejected'
    return {'label': label, 'probability_approved': float(prob) if prob is not None else None}


if __name__ == '__main__':
    # quick smoke test
    sample = {
        'visa_type': 'Work',
        'processing_center': 'Center B',
        'education_of_employee': "Master's",
        'has_job_experience': 'Y',
        'requires_job_training': 'N',
        'job_offer': 'Yes',
        'documents_complete': 'Yes',
        'years_experience': 5,
        'previous_visa_rejections': 0,
        'application_date': '2025-06-15'
    }
    print('Predict processing days: ', predict_processing_days(sample))
    print('Predict visa status: ', predict_visa_status(sample))
