from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import urllib.parse
import re

app = Flask(__name__)

# Load the trained model, scaler, and feature columns
try:
    model = joblib.load('phishing_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('columns.pkl')
    print("Model, scaler, and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler, feature_columns = None, None, None

# Simplified feature extraction function
def extract_features(url):
    parsed_url = urllib.parse.urlparse(url)
    features = {
        'length_url': len(url),
        'length_hostname': len(parsed_url.netloc) if parsed_url.netloc else 0,
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_eq': url.count('='),
        'nb_slash': url.count('/'),
        'nb_www': 1 if 'www' in parsed_url.netloc.lower() else 0,
        'nb_com': 1 if '.com' in url.lower() else 0,
        'https_token': 1 if parsed_url.scheme == 'https' else 0,
        # Add placeholder zeros for remaining features to match the trained model's expected input
    }
    
    # Create a feature vector with zeros for all expected features
    feature_vector = np.zeros(len(feature_columns))
    for i, col in enumerate(feature_columns):
        if col in features:
            feature_vector[i] = features[col]
    
    return feature_vector

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded properly.'}), 500
    
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided.'}), 400
    
    try:
        # Extract features
        features = extract_features(url)
        # Scale features
        features_scaled = scaler.transform([features])
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        result = 'Phishing' if prediction == 1 else 'Legitimate'
        confidence = probability if prediction == 1 else 1 - probability
        
        return jsonify({
            'result': result,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)