import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Preprocess the data
def preprocess_data(data):
    # Remove the 'url' column as it's not a feature for training
    if 'url' in data.columns:
        data = data.drop('url', axis=1)
    
    # Select numeric columns for filling missing values
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Fill missing values in numeric columns with their median
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Separate features and target
    X = data.drop('status', axis=1)
    y = data['status'].map({'legitimate': 0, 'phishing': 1})
    
    return X, y

# Train the model
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, X_train.columns

# Save the model and scaler
def save_model(model, scaler, columns, model_path='phishing_model.pkl', scaler_path='scaler.pkl', columns_path='columns.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(columns, columns_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Columns saved to {columns_path}")

def main():
    # Path to the dataset (update this path as needed)
    dataset_path = 'dataset_phishing.csv'
    
    # Load and preprocess data
    data = load_data(dataset_path)
    if data is None:
        return
    
    X, y = preprocess_data(data)
    
    # Train the model
    model, scaler, columns = train_model(X, y)
    
    # Save the model, scaler, and column names
    save_model(model, scaler, columns)

if __name__ == '__main__':
    main()