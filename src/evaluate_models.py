import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

MODELS_DIR = Path(r"a:\Infosys_SpringBoard\AI_Enabled_VISA_Status_Prediction_and_Processing_Time_Estimator\models")
DATA_PATH = Path(r"a:\Infosys_SpringBoard\AI_Enabled_VISA_Status_Prediction_and_Processing_Time_Estimator\src\data\VisaFile.csv")


def load_models():
    reg_file = MODELS_DIR / 'processing_days_model.pkl'
    clf_file = MODELS_DIR / 'visa_status_model.pkl'
    sel_file = MODELS_DIR / 'selected_features.pkl'
    reg = joblib.load(reg_file)
    clf = joblib.load(clf_file)
    features = joblib.load(sel_file)
    return reg, clf, features


def get_feature_names(pipeline, features):
    # Extract feature names after preprocessing
    pre = pipeline.named_steps['pre']
    try:
        names = pre.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback: try ColumnTransformer components
        names = []
        for name, trans, cols in pre.transformers_:
            if name == 'remainder':
                continue
            if hasattr(trans, 'named_steps') and 'ohe' in trans.named_steps:
                ohe = trans.named_steps['ohe']
                try:
                    ohe_names = ohe.get_feature_names_out(cols)
                except Exception:
                    ohe_names = [f"{c}_{i}" for c in cols for i in range(1)]
                names.extend(list(ohe_names))
            else:
                names.extend(list(cols))
        return names


def show_importances(pipeline, model_name):
    pre = pipeline.named_steps['pre']
    model = pipeline.named_steps['model']
    try:
        feat_names = get_feature_names(pipeline, None)
    except Exception:
        feat_names = None

    print(f"\nFeature importances for {model_name}:")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if feat_names is not None:
            imp_df = pd.Series(importances, index=feat_names).sort_values(ascending=False)
            print(imp_df.head(20))
        else:
            print(importances[:20])
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        # handle multiclass
        coef = coef.ravel() if coef.ndim > 1 else coef
        if feat_names is not None and len(feat_names) == len(coef):
            coef_df = pd.Series(coef, index=feat_names).sort_values(key=abs, ascending=False)
            print(coef_df.head(20))
        else:
            print(coef[:20])
    else:
        print("No feature importance or coefficients available for this model type.")


def cross_validate_pipeline(pipeline, X, y, task='reg'):
    print(f"\nRunning 5-fold cross-validation for {task}...")
    if task == 'reg':
        # MAE CV
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae_scores = -scores
        # RMSE via cross_val_predict
        preds = cross_val_predict(pipeline, X, y, cv=5, n_jobs=-1)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"MAE CV (5-fold) mean: {mae_scores.mean():.4f}, std: {mae_scores.std():.4f}")
        print(f"RMSE (CV preds): {rmse:.4f}")
        return mae_scores.mean(), rmse
    else:
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1', n_jobs=-1)
        print(f"F1 CV (5-fold) mean: {scores.mean():.4f}, std: {scores.std():.4f}")
        return scores.mean(), None


def retrain_without_cols(features, cols_to_remove):
    import train_models
    df = pd.read_csv(DATA_PATH, parse_dates=['application_date'])
    df['application_month'] = df['application_date'].dt.month

    new_features = [f for f in features if f not in cols_to_remove]
    print(f"Retraining without columns: {cols_to_remove}. New features: {new_features}")

    best_reg = train_models.train_regression(df, new_features, target='processing_days')
    joblib.dump(best_reg['model'], MODELS_DIR / 'processing_days_model_nofeatures.pkl')
    print("Saved regression (no features) model to processing_days_model_nofeatures.pkl")

    best_clf = train_models.train_classifier(df, new_features, target='visa_status')
    joblib.dump(best_clf['model'], MODELS_DIR / 'visa_status_model_nofeatures.pkl')
    print("Saved classifier (no features) model to visa_status_model_nofeatures.pkl")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=['application_date'])
    df['application_month'] = df['application_date'].dt.month

    reg, clf, features = load_models()

    # Show importances
    show_importances(reg, 'processing_days model')
    show_importances(clf, 'visa_status model')

    # CV evaluation
    X_reg = df[features]
    y_reg = df['processing_days'].values
    cross_validate_pipeline(reg, X_reg, y_reg, task='reg')

    df_bin = df[df['visa_status'].isin(['Approved', 'Rejected'])].copy()
    X_clf = df_bin[features]
    y_clf = (df_bin['visa_status'] == 'Approved').astype(int).values
    cross_validate_pipeline(clf, X_clf, y_clf, task='clf')

    # Retrain without problematic columns
    retrain_without_cols(features, ['job_offer', 'years_experience'])


if __name__ == '__main__':
    main()
