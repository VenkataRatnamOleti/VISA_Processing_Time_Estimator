from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback

# Ensure src/ is on import path so local modules can be imported when running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from inference import predict_processing_days, predict_visa_status
except Exception:
    predict_processing_days = None
    predict_visa_status = None
from predict import VisaProcessingEstimator

# Frontend directory (served by Flask so frontend + API share origin)
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)


def _save_analytics(updates: dict):
    """Merge `updates` into data/analytics.json (creating file if needed)."""
    try:
        data_dir = os.path.join(BASE_DIR, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        analytics_file = os.path.join(data_dir, 'analytics.json')
        import json
        if os.path.exists(analytics_file):
            try:
                with open(analytics_file, 'r', encoding='utf-8') as f:
                    a = json.load(f)
                    if not isinstance(a, dict):
                        a = {}
            except Exception:
                a = {}
        else:
            a = {}
        a.update(updates)
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(a, f, indent=2)
    except Exception:
        # Do not let analytics failures interrupt API behavior
        pass


# Removed Gemini server-side code - moved to frontend

    # Only use the explicitly configured model (param or environment). Avoid extra fallback models
    preferred = []
    if model:
        preferred.append(model)
    else:
        env_model = os.environ.get('GEMINI_MODEL')
        if env_model:
            preferred.append(env_model)
    # Keep REST timeout short for responsiveness (override with env LLM_REST_TIMEOUT if needed)
    timeout = int(os.environ.get('LLM_REST_TIMEOUT', '4'))

    # Try the modern google.genai client once (if available). Keep this to a single attempt for speed.
    try:
        try:
            from google import genai as google_genai
        except Exception:
            google_genai = None
        if google_genai is not None:
            try:
                contents = prompt if isinstance(prompt, (list, tuple)) else [prompt]
                # prefer client.models.generate_content when present
                ClientClass = getattr(google_genai, 'Client', None)
                if ClientClass is not None:
                    client = ClientClass()
                    if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
                        resp = client.models.generate_content(model=preferred[0] if preferred else None, contents=contents)
                        text = getattr(resp, 'text', None)
                        if not text:
                            try:
                                j = getattr(resp, 'to_dict', lambda: None)()
                            except Exception:
                                j = resp
                            if isinstance(j, dict):
                                if 'output' in j:
                                    text = j['output']
                                elif 'candidates' in j and isinstance(j['candidates'], list) and len(j['candidates'])>0:
                                    cand = j['candidates'][0]
                                    text = cand.get('content') or cand.get('text') or cand.get('output')
                        if text:
                            print('[+] _call_llm_server_side: used google.genai Client.models.generate_content')
                            return text
                # try top-level helper if present
                if hasattr(google_genai, 'generate_text'):
                    resp = google_genai.generate_text(model=preferred[0] if preferred else None, prompt=prompt)
                    text = getattr(resp, 'text', None) or (resp.get('candidates', [{}])[0].get('output') if isinstance(resp, dict) else None)
                    if text:
                        print('[+] _call_llm_server_side: used google.generativeai.generate_text')
                        return text
            except Exception:
                # single failure falls through to REST below
                pass
    except Exception:
        pass

    # Next try the older google.generativeai package (some environments provide this)
    try:
        import google.generativeai as genai
        try:
            # configure if available
            if hasattr(genai, 'configure'):
                try:
                    genai.configure(api_key=gemini_key)
                except Exception:
                    pass

            # prefer top-level helpers if present
            if hasattr(genai, 'generate_text'):
                try:
                    resp = genai.generate_text(model=model, prompt=prompt)
                    text = getattr(resp, 'text', None) or (resp.get('candidates', [{}])[0].get('output') if isinstance(resp, dict) else None)
                    if text:
                        print('[+] _call_llm_server_side: used google.generativeai.generate_text')
                        return text
                except Exception:
                    pass

            if hasattr(genai, 'generate'):
                try:
                    resp = genai.generate(model=model, prompt=prompt)
                    text = getattr(resp, 'text', None) or (resp.get('candidates', [{}])[0].get('output') if isinstance(resp, dict) else None)
                    if text:
                        print('[+] _call_llm_server_side: used google.generativeai.generate')
                        return text
                except Exception:
                    pass

            # some versions provide a Client class
            if hasattr(genai, 'Client'):
                try:
                    client = genai.Client()
                    if hasattr(client, 'generate_text'):
                        try:
                            resp = client.generate_text(model=model, prompt=prompt)
                            text = getattr(resp, 'text', None) or (resp.get('candidates', [{}])[0].get('output') if isinstance(resp, dict) else None)
                            if text:
                                print('[+] _call_llm_server_side: used google.generativeai.Client.generate_text')
                                return text
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # package not available — proceed to REST fallback
        pass

    # 3) REST fallback: call Google's Generative Language REST endpoint.
    try:
        import requests
        # If Application Default Credentials are available, prefer using an OAuth Bearer token
        auth_headers = None
        try:
            try:
                import google.auth
                from google.auth.transport.requests import Request as GARequest
                creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
                creds.refresh(GARequest())
                if getattr(creds, 'token', None):
                    auth_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
            except Exception:
                auth_headers = None
        except Exception:
            auth_headers = None

        # REST call: call the Generative Language endpoint once for the configured model.
        base = 'https://generativelanguage.googleapis.com/v1'
        gen_payload = { 'contents': [ { 'parts': [ { 'text': prompt } ] } ] }
        try:
            try_model = preferred[0] if preferred else None
            if not try_model:
                raise RuntimeError('No model specified for LLM call')
            url = f"{base}/models/{try_model}:generateContent"
            hdrs = auth_headers if auth_headers is not None else {'x-goog-api-key': gemini_key, 'Content-Type': 'application/json'}
            r = requests.post(url, json=gen_payload, headers=hdrs, timeout=timeout)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict):
                    if 'candidates' in j and isinstance(j['candidates'], list) and len(j['candidates'])>0:
                        cand = j['candidates'][0]
                        if 'content' in cand:
                            content = cand['content']
                            texts = []
                            if isinstance(content, list):
                                texts = [p.get('text') for p in content if isinstance(p, dict) and 'text' in p]
                            elif isinstance(content, dict) and 'parts' in content and isinstance(content['parts'], list):
                                texts = [p.get('text') for p in content['parts'] if isinstance(p, dict) and 'text' in p]
                            if texts:
                                return '\n'.join([t for t in texts if t])
                        if 'text' in cand:
                            return cand['text']
                        if 'output' in cand:
                            return cand['output']
                    if 'output' in j:
                        return j['output']
                    if 'text' in j:
                        return j['text']
                return str(j)
            else:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        except Exception as e:
            # propagate a clear error so callers can inform the frontend that Gemini failed
            raise RuntimeError(f"LLM REST call failed: {e}")
    except Exception as e:
        raise


# Instantiate estimator once
estimator = VisaProcessingEstimator()

# Optional acceptance model
acceptance_model = None
acceptance_model_columns = None
try:
    import joblib
    import glob
    models_dir = os.path.join(BASE_DIR, '..', 'models')
    # Prefer common filenames but also try to discover files matching visa/acceptance patterns
    candidate_names = [
        'acceptance_model.pkl',
        'visa_status_model.pkl',
        'visa_status.pkl',
        'acceptance.pkl'
    ]
    acc_path = None
    for nm in candidate_names:
        p = os.path.join(models_dir, nm)
        if os.path.exists(p):
            acc_path = p
            break
    # try wildcard matches if exact names not present
    if acc_path is None:
        matches = glob.glob(os.path.join(models_dir, '*visa*status*.pkl')) + glob.glob(os.path.join(models_dir, '*acceptance*.pkl'))
        if matches:
            acc_path = matches[0]

    if acc_path and os.path.exists(acc_path):
        try:
            acceptance_model = joblib.load(acc_path)
            # discover feature/columns file
            acc_cols_candidates = [
                'acceptance_features.pkl',
                'visa_status_features.pkl',
                'selected_features.pkl',
                'model_features.pkl',
                'model_features.pkl'
            ]
            acceptance_model_columns = None
            for nm in acc_cols_candidates:
                cpath = os.path.join(models_dir, nm)
                if os.path.exists(cpath):
                    try:
                        acceptance_model_columns = joblib.load(cpath)
                        break
                    except Exception:
                        continue
            print('[+] Acceptance model loaded from %s' % acc_path)
            if acceptance_model_columns is not None:
                print('[+] Acceptance model columns loaded.')
            else:
                print('[*] Acceptance model columns not found; classifier may require feature engineering at runtime.')
        except Exception as ie:
            print(f'[-] Failed to load acceptance model at {acc_path}: {ie}')
    else:
        print('[*] No acceptance model found; acceptance endpoint will use heuristic fallback.')
except Exception as e:
    print(f'[-] Could not load acceptance model: {e}')

try:
    visa_model_path = os.path.join(BASE_DIR, '..', 'models', 'visa_status_model.pkl')
    if os.path.exists(visa_model_path):
        # The estimator constructor attempts to load the model from the default path.
        # We avoid calling a non-existent `load_model` here; just report status.
        print('[+] Visa model file found. estimator.active=%s' % getattr(estimator, 'active', False))
    else:
        print('[-] Visa model file not found. Ensure visa_model.pkl is in the models directory.')
except Exception as e:
    print(f'[-] Failed to inspect visa model file: {e}')


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Accept feedback JSON and append to data/feedbacks.json"""
    try:
        data = request.get_json(force=True)
        feedback_dir = os.path.join(BASE_DIR, '..', 'data')
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_file = os.path.join(feedback_dir, 'feedbacks.json')
        import json, time

        entry = {
            'name': data.get('name'),
            'email': data.get('email'),
            'message': data.get('message'),
            'ts': time.time()
        }

        # load existing feedback list if present
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    arr = json.load(f)
                    if not isinstance(arr, list):
                        arr = []
            except Exception:
                arr = []
        else:
            arr = []

        arr.append(entry)
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(arr, f, indent=2)

        return jsonify({'success': True}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/estimate-days', methods=['POST'])
def estimate_days():
    """Return estimated processing days and confidence window using regression model."""
    try:
        payload = request.get_json(force=True)
        # Prefer the new inference helper if available
        if predict_processing_days is not None:
            days = float(predict_processing_days(payload))
            deviation = max(1.0, days * 0.10)
            window = f"{int(max(0, days - deviation))} to {int(days + deviation)} days"
            return jsonify({'success': True, 'estimated_days': round(days, 2), 'window': window}), 200

        # Fallback to legacy estimator
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

        # Prefer the new inference helper if available
        if predict_visa_status is not None:
            res = predict_visa_status(payload)
            label = res.get('label')
            prob = res.get('probability_approved') or res.get('probability_approved')
            verdict = 'Accepted' if label == 'Approved' else 'Rejected'
            try:
                import time
                _save_analytics({'latest_verdict': verdict, 'last_verdict_ts': time.time()})
            except Exception:
                pass
            return jsonify({'success': True, 'acceptance_score': round(prob, 3) if prob is not None else None, 'verdict': verdict}), 200

        # Use classifier when available
        if acceptance_model is not None:
            try:
                import pandas as pd, ast, re

                # First attempt: provide raw payload to the model pipeline (if it accepts DataFrame of raw cols)
                try:
                    input_df = pd.DataFrame([payload])
                    proba = acceptance_model.predict_proba(input_df)[0]
                except ValueError as ve:
                    # sklearn ColumnTransformer often raises a ValueError listing missing columns.
                    msg = str(ve)
                    missing = set()
                    m = re.search(r"columns are missing: (\{.*\})", msg)
                    if m:
                        try:
                            missing = set(ast.literal_eval(m.group(1)))
                        except Exception:
                            missing = set()

                    # Build a defaulted input dict using sensible defaults for expected fields
                    defaults = {
                        'country_of_applicant': 'Unknown',
                        'continent': 'Asia',
                        'visa_type': 'Work',
                        'requires_job_training': False,
                        'processing_time_days': float(payload.get('estimated_days') or payload.get('processing_time_days') or 0),
                        'unit_of_wage': 'Monthly',
                        'has_job_experience': False,
                        'education_of_employee': 'Bachelor',
                        'full_time_position': True,
                        'region_of_employment': 'Unknown',
                        'processing_center': 'Default',
                        'application_season': 'Other'
                    }

                    base = dict(payload) if isinstance(payload, dict) else {}
                    for col in missing:
                        if col not in base:
                            base[col] = defaults.get(col, 0)

                    input_df = pd.DataFrame([base])
                    proba = acceptance_model.predict_proba(input_df)[0]

                accept_score = float(proba[1])
                verdict = 'Accepted' if accept_score >= 0.5 else 'Rejected'
                try:
                    import time
                    _save_analytics({'latest_verdict': verdict, 'last_verdict_ts': time.time()})
                except Exception:
                    pass
                return jsonify({'success': True, 'acceptance_score': round(accept_score, 3), 'verdict': verdict}), 200
            except Exception as e:
                app.logger.warning('LLM call failed in chat(): %s', str(e))
                # fallback to heuristic

        # Fallback: compute estimated days and map to acceptance score
        est = estimator.get_estimation(payload)
        if 'estimated_days' in est:
            days = float(est['estimated_days'])
            score = max(0.0, min(1.0, 1.0 - (days / 200.0)))
            verdict = 'Accepted' if score >= 0.5 else 'Rejected'
            try:
                import time
                _save_analytics({'latest_verdict': verdict, 'last_verdict_ts': time.time()})
            except Exception:
                pass
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
        # Add simple monthly visits (synthetic) and feedback distribution derived from stored feedbacks
        try:
            # monthly visits (placeholder) - could be replaced with real analytics
            stats['visits'] = [120, 150, 180, 200, 250, 220, 210, 230, 240, 260, 270, 300]
        except Exception:
            stats['visits'] = [100,120,130,140,150]

        try:
            feedback_file = os.path.join(BASE_DIR, '..', 'data', 'feedbacks.json')
            pos = neu = neg = 0
            if os.path.exists(feedback_file):
                import json
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    arr = json.load(f)
                    for entry in arr:
                        msg = (entry.get('message') or '').lower()
                        if any(w in msg for w in ['good','great','thank','thanks','excellent','love','helpful']):
                            pos += 1
                        elif any(w in msg for w in ['bad','poor','terrible','hate','complaint','not']):
                            neg += 1
                        else:
                            neu += 1
            stats['feedback_distribution'] = {'positive': pos, 'neutral': neu, 'negative': neg}
            stats['feedback_count'] = pos + neu + neg
        except Exception:
            stats['feedback_distribution'] = {'positive': 0, 'neutral': 0, 'negative': 0}
            stats['feedback_count'] = 0

        # Load simple analytics persistence (visits, ratings)
        try:
            analytics_file = os.path.join(BASE_DIR, '..', 'data', 'analytics.json')
            import json
            if os.path.exists(analytics_file):
                with open(analytics_file, 'r', encoding='utf-8') as f:
                    a = json.load(f)
            else:
                a = {}
            stats['visit_count'] = int(a.get('total_visits', 0))
            stats['visits'] = a.get('monthly_visits', stats.get('visits', []))
            ratings = a.get('ratings', [])
            stats['rating_count'] = len(ratings)
            stats['avg_rating'] = round(sum(ratings)/len(ratings), 2) if ratings else None
            # include latest verdict if present
            stats['latest_verdict'] = a.get('latest_verdict', '—')
            stats['last_verdict_ts'] = a.get('last_verdict_ts', None)
        except Exception:
            stats['visit_count'] = 0
            stats['rating_count'] = 0
            stats['avg_rating'] = None

        # Provide a simple model accuracy estimate derived from RMSE (informational)
        try:
            if stats.get('rmse') is not None:
                # crude heuristic: map lower RMSE to higher percent score, cap between 0-100
                approx = max(0.0, min(100.0, 100.0 - float(stats['rmse'])))
                stats['model_accuracy_estimate'] = round(approx, 2)
                stats['model_accuracy_note'] = 'heuristic derived from RMSE; for classification use true holdout accuracy.'
            else:
                stats['model_accuracy_estimate'] = None
                stats['model_accuracy_note'] = None
        except Exception:
            stats['model_accuracy_estimate'] = None
            stats['model_accuracy_note'] = None

        # Load static model metrics
        try:
            metrics_file = os.path.join(BASE_DIR, '..', 'data', 'model_metrics.json')
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                stats.update(metrics)
            else:
                stats['regression_rmse_sample'] = None
                stats['classification_accuracy_sample'] = None
                stats['classification_roc_auc_sample'] = None
                stats['data_samples'] = None
        except Exception:
            pass

        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500



# Removed /api/chat - use frontend direct Gemini

    try:
        data = request.get_json(force=True)
        user_message = (data.get('message') or '').strip()
        # Chat API: use Gemini exclusively. If Gemini is not configured or a call fails, inform frontend.
        gemini_key = os.environ.get('GEMINI_API_KEY')
        app.logger.info('[*] chat(): GEMINI_API_KEY present=%s', bool(gemini_key))
        if not gemini_key:
            return jsonify({'success': False, 'error': 'Gemini not configured on server'}), 400

        try:
            model = os.environ.get('GEMINI_MODEL', 'gemini-3-small')
            prompt = f"User: {user_message}\nAssistant:" if user_message else 'Hello'
            resp_text = _call_llm_server_side(prompt, gemini_key, model=model)
            if resp_text:
                app.logger.info('[+] chat(): received LLM reply (len=%d)', len(resp_text))
                return jsonify({'success': True, 'reply': resp_text, 'source': model}), 200
            else:
                return jsonify({'success': False, 'error': 'Gemini is not working'}), 503
        except Exception as e:
            app.logger.warning('LLM call failed in chat(): %s', str(e))
            return jsonify({'success': False, 'error': 'Gemini is not working'}), 503
    except:
        pass
    # Generic fallback when unexpected error occurs
    return jsonify({'success': False, 'error': 'Chat service error'}), 500

@app.route('/api/track-visit', methods=['POST'])
def track_visit():
    """Increment a simple visit counter stored in data/analytics.json and return updated counts."""
    try:
        data_dir = os.path.join(BASE_DIR, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        analytics_file = os.path.join(data_dir, 'analytics.json')
        import json, time
        now = time.gmtime()
        month_idx = now.tm_mon - 1
        if os.path.exists(analytics_file):
            try:
                with open(analytics_file, 'r', encoding='utf-8') as f:
                    a = json.load(f)
            except Exception:
                a = {}
        else:
            a = {}

        a.setdefault('total_visits', 0)
        a['total_visits'] = int(a.get('total_visits', 0)) + 1
        mv = a.get('monthly_visits', [0]*12)
        if not isinstance(mv, list) or len(mv) != 12:
            mv = [0]*12
        mv[month_idx] = int(mv[month_idx]) + 1
        a['monthly_visits'] = mv

        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(a, f, indent=2)

        return jsonify({'success': True, 'total_visits': a['total_visits'], 'monthly_visits': a['monthly_visits']}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rate', methods=['POST'])
def rate():
    """Accept a numeric rating (1-5) and store it in analytics.json"""
    try:
        payload = request.get_json(force=True)
        rating = float(payload.get('rating') or 0)
        if rating <= 0 or rating > 5:
            return jsonify({'success': False, 'error': 'rating must be between 1 and 5'}), 400
        data_dir = os.path.join(BASE_DIR, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        analytics_file = os.path.join(data_dir, 'analytics.json')
        import json
        if os.path.exists(analytics_file):
            try:
                with open(analytics_file, 'r', encoding='utf-8') as f:
                    a = json.load(f)
            except Exception:
                a = {}
        else:
            a = {}

        ratings = a.get('ratings', [])
        ratings.append(rating)
        a['ratings'] = ratings
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(a, f, indent=2)

        return jsonify({'success': True, 'rating_count': len(ratings), 'avg_rating': round(sum(ratings)/len(ratings), 2)}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-insights', methods=['POST'])
def generate_insights():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()

        if not text:
            return jsonify({'success': False, 'error': 'Empty input'}), 400

        gemini_key = os.environ.get('GEMINI_API_KEY')

        if not gemini_key:
            return jsonify({'success': False, 'error': 'Gemini not configured on server'}), 400

        try:
            model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')

            prompt = f"""
You are an expert assistant.

Provide short, clear customer-service style suggestions based on this:

{text}

Keep it concise and helpful.
"""

            resp_text = _call_llm_server_side(prompt, gemini_key, model=model)

            if resp_text:
                return jsonify({
                    'success': True,
                    'suggestions': resp_text,
                    'source': model
                }), 200

            return jsonify({'success': False, 'error': 'Gemini failed'}), 503

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 503

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/llm-test', methods=['GET'])
def llm_test():
    """Diagnostic endpoint: reports whether GEMINI_API_KEY and `google.generativeai` are available
    and attempts a minimal model call so you can see the exact error returned by the client.
    """
    info = {}
    gemini_key = os.environ.get('GEMINI_API_KEY')
    info['gemini_key_present'] = bool(gemini_key)
    try:
        import google.generativeai as genai
        info['genai_import'] = True
        info['genai_has_generate_text'] = hasattr(genai, 'generate_text')
        info['genai_has_generate'] = hasattr(genai, 'generate')
    except Exception as e:
        info['genai_import'] = False
        info['genai_import_error'] = str(e)

    if not gemini_key:
        return jsonify({'success': False, 'info': info, 'error': 'GEMINI_API_KEY not set on server environment'}), 400

    # Try a minimal call through the robust helper and return the response or error to help debugging
    try:
        model = os.environ.get('GEMINI_MODEL', 'gemini-3-small')
        sample_prompt = 'Say hello in one short sentence.'
        reply = _call_llm_server_side(sample_prompt, gemini_key, model=model)
        return jsonify({'success': True, 'info': info, 'model': model, 'reply': reply}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'info': info, 'error': str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # Serve static frontend files; default to index.html
    if path != '' and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
