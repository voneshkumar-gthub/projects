# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# --------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --------------------- MODEL (SimpleCNN) ---------------------
MODEL_PATH = 'plant_disease_model.pth'
IMG_SIZE   = 224

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Class names – **must match the order in your training folder**
CLASSES = [
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)'
]

print("Loading SimpleCNN model...")
model = SimpleCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Inference transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# --------------------- PREDICTION ---------------------
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)               # (1, C, H, W)

    with torch.no_grad():
        output = model(img)
        probs  = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    label = CLASSES[pred.item()]
    confidence = conf.item() * 100
    return label, confidence

# --------------------- ROUTES ---------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            label, conf = predict_image(filepath)
            pretty = f"{label.replace('___', ' - ')} ({conf:.1f}% confidence)"
            return render_template('index.html',
                                   prediction=pretty,
                                   image=file.filename)
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(request.url)

    return redirect(request.url)

# --------------------- RUN ---------------------
if __name__ == '__main__':
    print(f"Server → http://127.0.0.1:5000")
    app.run(debug=True)