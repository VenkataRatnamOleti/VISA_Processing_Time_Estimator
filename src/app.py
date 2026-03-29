from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback

# Ensure src/ is on import path so local modules can be imported when running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import VisaProcessingEstimator

# Frontend directory (served by Flask so frontend + API share origin)
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# Instantiate estimator once
estimator = VisaProcessingEstimator()

# Optional acceptance model
acceptance_model = None
acceptance_model_columns = None
try:
    import joblib
    acc_path = os.path.join(BASE_DIR, '..', 'models', 'acceptance_model.pkl')
    acc_cols_path = os.path.join(BASE_DIR, '..', 'models', 'acceptance_features.pkl')
    if os.path.exists(acc_path):
        acceptance_model = joblib.load(acc_path)
        if os.path.exists(acc_cols_path):
            acceptance_model_columns = joblib.load(acc_cols_path)
        print('[+] Acceptance model loaded.')
    else:
        print('[*] No acceptance model found; acceptance endpoint will use heuristic fallback.')
except Exception as e:
    print(f'[-] Could not load acceptance model: {e}')

try:
    visa_model_path = os.path.join(BASE_DIR, '..', 'models', 'visa_model.pkl')
    if os.path.exists(visa_model_path):
        estimator.load_model(visa_model_path)
        print('[+] Visa model loaded successfully.')
    else:
        print('[-] Visa model file not found. Ensure visa_model.pkl is in the models directory.')
except Exception as e:
    print(f'[-] Failed to load visa model: {e}')


@app.route('/api/estimate-days', methods=['POST'])
def estimate_days():
    """Return estimated processing days and confidence window using regression model."""
    try:
        payload = request.get_json(force=True)
        result = estimator.get_estimation(payload)
        if 'estimated_days' in result:
            return jsonify({'success': True, 'estimated_days': result['estimated_days'], 'window': result.get('window')}), 200
        return jsonify({'success': False, 'error': 'No prediction available'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/estimate-acceptance', methods=['POST'])
def estimate_acceptance():
    """Return acceptance probability and verdict.

    If an acceptance classifier is present it will be used. Otherwise use a
    heuristic mapping derived from estimated days.
    """
    try:
        payload = request.get_json(force=True)

        # Use classifier when available
        if acceptance_model is not None and acceptance_model_columns is not None:
            try:
                import pandas as pd
                input_df = pd.get_dummies(pd.DataFrame([payload]))
                final_df = pd.DataFrame(columns=acceptance_model_columns)
                final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)
                final_df = final_df[acceptance_model_columns].astype(float)
                proba = acceptance_model.predict_proba(final_df)[0]
                accept_score = float(proba[1])
                verdict = 'Accepted' if accept_score >= 0.5 else 'Rejected'
                return jsonify({'success': True, 'acceptance_score': round(accept_score, 3), 'verdict': verdict}), 200
            except Exception:
                traceback.print_exc()
                # fallback to heuristic

        # Fallback: compute estimated days and map to acceptance score
        est = estimator.get_estimation(payload)
        if 'estimated_days' in est:
            days = float(est['estimated_days'])
            score = max(0.0, min(1.0, 1.0 - (days / 200.0)))
            verdict = 'Accepted' if score >= 0.5 else 'Rejected'
            return jsonify({'success': True, 'acceptance_score': round(score, 3), 'verdict': verdict, 'based_on': 'heuristic'}), 200

        return jsonify({'success': False, 'error': 'Could not estimate acceptance'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        stats = {}
        stats['model_loaded'] = estimator.active
        stats['rmse'] = None
        stats['n_features'] = 0
        try:
            rmse_file = os.path.join(BASE_DIR, '..', 'models', 'model_rmse.pkl')
            import joblib
            stats['rmse'] = float(joblib.load(rmse_file))
        except Exception:
            stats['rmse'] = None

        try:
            stats['n_features'] = len(estimator.model_columns) if estimator.model_columns is not None else 0
        except Exception:
            stats['n_features'] = 0

        stats['synthetic_processing_days'] = [30, 45, 60, 90, 120, 15, 20, 200, 75, 40]
        stats['visa_type_breakdown'] = {'Work': 62, 'Student': 18, 'Visitor': 12, 'Other': 8}

        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        message = (data.get('message') or '').lower()
        reply = "I'm here to help — ask me about predictions, required inputs, or model performance."

        if 'estimate' in message or 'predict' in message:
            reply = 'To get an estimate, call the /api/estimate-days endpoint or visit the Predict page.'
        elif 'accepted' in message or 'reject' in message:
            reply = 'Acceptance predictions are heuristics in this demo. Shorter processing times usually increase likelihood.'
        elif 'rmse' in message or 'error' in message:
            reply = f"Model RMSE: {getattr(estimator, 'error_val', 'unknown')} (lower is better)."

        return jsonify({'success': True, 'reply': reply})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # Serve static frontend files; default to index.html
    if path != '' and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
