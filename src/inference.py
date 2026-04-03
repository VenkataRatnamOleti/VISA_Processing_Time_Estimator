import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE.parent / 'models'

def load_artifacts():
    def _load_compat(p):
        try:
            return joblib.load(p)
        except AttributeError as err:
            msg = str(err)
            if '_fill_dtype' in msg or 'SimpleImputer' in msg:
                try:
                    from sklearn.impute import SimpleImputer
                    if not hasattr(SimpleImputer, '_fill_dtype'):
                        SimpleImputer._fill_dtype = None
                except Exception:
                    pass
                obj = joblib.load(p)
            else:
                raise

        # best-effort: fix SimpleImputer instances so transform() won't error
        try:
            import numpy as np
            from sklearn.impute import SimpleImputer

            def _fix(si):
                try:
                    if getattr(si, '_fill_dtype', None) is None:
                        val = None
                        if hasattr(si, 'statistics_') and getattr(si, 'statistics_', None) is not None:
                            stats = si.statistics_
                            try:
                                if hasattr(stats, '__len__') and len(stats) > 0:
                                    val = stats[0]
                            except Exception:
                                val = None
                        if val is None and hasattr(si, 'fill_value'):
                            val = si.fill_value
                        if val is None:
                            val = 0
                        try:
                            si._fill_dtype = np.array([val]).dtype
                        except Exception:
                            si._fill_dtype = np.dtype('float64')
                except Exception:
                    pass

            def _walk(o, seen=None):
                if seen is None:
                    seen = set()
                oid = id(o)
                if oid in seen:
                    return
                seen.add(oid)
                if isinstance(o, SimpleImputer):
                    _fix(o)
                    return
                if isinstance(o, (list, tuple, set)):
                    for v in o:
                        _walk(v, seen)
                    return
                if isinstance(o, dict):
                    for v in o.values():
                        _walk(v, seen)
                    return
                try:
                    attrs = getattr(o, '__dict__', None)
                    if isinstance(attrs, dict):
                        for v in attrs.values():
                            _walk(v, seen)
                except Exception:
                    pass

            _walk(obj)
        except Exception:
            pass

        return obj

    reg = _load_compat(MODELS_DIR / 'processing_days_model.pkl')
    clf = _load_compat(MODELS_DIR / 'visa_status_model.pkl')
    features = _load_compat(MODELS_DIR / 'selected_features.pkl')
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
