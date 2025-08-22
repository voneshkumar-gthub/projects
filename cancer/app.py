from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# List of all 30 feature names (based on the dataset)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = []
    for feature in feature_names:
        value = float(request.form.get(feature, 0))
        features.append(value)
    
    # Convert to numpy array and scale
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    prediction_text = 'Malignant' if prediction == 1 else 'Benign'
    
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)