import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib

try:
    from xgboost import XGBRegressor, XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False


DATA_PATH = Path(r"a:\Infosys_SpringBoard\AI_Enabled_VISA_Status_Prediction_and_Processing_Time_Estimator\src\data\VisaFile.csv")
MODELS_DIR = Path(r"a:\Infosys_SpringBoard\AI_Enabled_VISA_Status_Prediction_and_Processing_Time_Estimator\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare(df_path=DATA_PATH):
    df = pd.read_csv(df_path, parse_dates=['application_date'])
    # derived
    df['application_month'] = df['application_date'].dt.month
    return df


def build_preprocessor(cat_features, num_features):
    # numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ], remainder='drop')

    return preprocessor


def train_regression(df, features, target='processing_days'):
    X = df[features].copy()
    y = df[target].values

    num_features = [c for c in features if df[c].dtype.kind in 'fiu']
    cat_features = [c for c in features if c not in num_features]

    preprocessor = build_preprocessor(cat_features, num_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Candidate models
    candidates = {}
    # Random Forest
    rf = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    candidates['random_forest'] = rf

    if has_xgb:
        xgb = Pipeline([('pre', preprocessor), ('model', XGBRegressor(n_estimators=200, random_state=42, verbosity=0, n_jobs=1))])
        candidates['xgboost'] = xgb

    results = {}
    for name, pipe in candidates.items():
        print(f"Training regressor: {name}")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {'model': pipe, 'mae': mae, 'rmse': rmse}
        print(f" -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best = results[best_name]
    print(f"Best regressor: {best_name} (MAE={best['mae']:.4f})")

    # save
    joblib.dump(best['model'], MODELS_DIR / 'processing_days_model.pkl')
    print(f"Saved regression pipeline to {MODELS_DIR / 'processing_days_model.pkl'}")
    return best


def train_classifier(df, features, target='visa_status'):
    # Drop Pending rows as requested
    df_bin = df[df[target].isin(['Approved', 'Rejected'])].copy()
    # map to binary
    df_bin['visa_label'] = (df_bin[target] == 'Approved').astype(int)

    X = df_bin[features].copy()
    y = df_bin['visa_label'].values

    num_features = [c for c in features if df[c].dtype.kind in 'fiu']
    cat_features = [c for c in features if c not in num_features]

    preprocessor = build_preprocessor(cat_features, num_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {}
    # Logistic Regression baseline (with class weight)
    lr = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=400, class_weight='balanced'))])
    candidates['logistic_regression'] = lr

    # Random Forest
    rf = Pipeline([('pre', preprocessor), ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))])
    candidates['random_forest'] = rf

    if has_xgb:
        # For XGBoost, compute scale_pos_weight
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / max(1, pos)
        xgb = Pipeline([('pre', preprocessor), ('model', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, n_jobs=1, random_state=42))])
        candidates['xgboost'] = xgb

    results = {}
    for name, pipe in candidates.items():
        print(f"Training classifier: {name}")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        roc = roc_auc_score(y_test, probs) if probs is not None else None
        cm = confusion_matrix(y_test, preds)
        results[name] = {'model': pipe, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'roc_auc': roc, 'confusion_matrix': cm}
        print(f" -> Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, ROC-AUC: {roc}")

    # choose best by f1
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best = results[best_name]
    print(f"Best classifier: {best_name} (F1={best['f1']:.4f})")

    joblib.dump(best['model'], MODELS_DIR / 'visa_status_model.pkl')
    print(f"Saved classifier pipeline to {MODELS_DIR / 'visa_status_model.pkl'}")

    return best


def main():
    print("Loading data...")
    df = load_and_prepare()

    # feature lists (as recommended)
    features = ['visa_type', 'processing_center', 'education_of_employee',
                'has_job_experience', 'requires_job_training', 'job_offer', 'documents_complete',
                'years_experience', 'previous_visa_rejections', 'application_month']

    print("Training regression model...")
    best_reg = train_regression(df, features, target='processing_days')

    print("Training classifier model (Pending dropped)...")
    best_clf = train_classifier(df, features, target='visa_status')

    # save feature list
    joblib.dump(features, MODELS_DIR / 'selected_features.pkl')
    print(f"Saved selected feature list to {MODELS_DIR / 'selected_features.pkl'}")


if __name__ == '__main__':
    main()
