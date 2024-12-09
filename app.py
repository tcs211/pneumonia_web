from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
import os
import ssl
import sys
from PNModel import ChestXRayModel
from CXRAutoencoder import ChestXRayAutoencoder

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
autoencoder = ChestXRayAutoencoder().to(device)
classifier = ChestXRayModel().to(device)

# Load model weights
autoencoder.load_state_dict(torch.load('./models/best_autoencoder.pth', map_location=device,weights_only=False))
classifier.load_state_dict(torch.load('./models/epoch_4.pth', map_location=device,weights_only=False)['model_state_dict'])

autoencoder.eval()
classifier.eval()

# Image preprocessing
def preprocess_image(image, target_size=224):
    """Preprocess image for both autoencoder and classifier"""
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)
    
    # Transform for autoencoder
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])
    
    return transform(image).unsqueeze(0)

def add_prediction_text(image, is_xray, prediction=None, confidence=None):
    """Add prediction text to image"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    if not is_xray:
        result_text = "Not a chest X-ray image!"
        text_color = (255, 0, 0)
    else:
        result_text = f"{prediction} ({confidence:.1f}%)"
        text_color = (255, 0, 0) if prediction == "PNEUMONIA" else (0, 255, 0)
    
    draw.text((10, 10), result_text, fill=text_color, font=font)
    
    return np.array(image_pil)

def validate_xray(image_tensor, threshold=0.01):
    """Validate if image is an X-ray using autoencoder"""
    with torch.no_grad():
        reconstruction = autoencoder(image_tensor)
        error = nn.MSELoss()(reconstruction, image_tensor).item()
        return error < threshold, error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        # Read and process image
        file = request.files['file']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_image = image.copy()
        
        # Preprocess image
        processed_tensor = preprocess_image(image).to(device)
        
        # Validate if image is an X-ray
        is_xray, reconstruction_error = validate_xray(processed_tensor)
        
            # Prepare result image
        height, width = image.shape[:2]
        if not is_xray:
            # create a blank image with the same size as the original image
            blank_image = np.zeros((200, 600, 3), np.uint8)
            # set the blank image to white
            blank_image.fill(255)
            result_image = add_prediction_text(
                blank_image,
                is_xray=False
            )
        else:
            # Normalize for classifier
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            classifier_input = normalize(processed_tensor)
            
            # Get prediction
            with torch.no_grad():
                outputs = classifier(classifier_input)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = probabilities[0][1].item()
            
            # Determine class and confidence
            class_name = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
            confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100
            
            print(f"Raw prediction: {prediction}")
            print(f"Class: {class_name}")
            print(f"Confidence: {confidence}%")
            
            result_image = add_prediction_text(
                cv2.resize(original_image, (600, int(600 * height / width))),
                is_xray=True,
                prediction=class_name,
                confidence=confidence
            )
        
        # Convert to PNG
        _, buffer = cv2.imencode('.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        io_buf = io.BytesIO(buffer)
        
        return send_file(
            io_buf,
            mimetype='image/png',
            as_attachment=False
        )
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Check for SSL certificates
        if os.path.exists('/etc/letsencrypt/live/to-ai.net/fullchain.pem'):
            cert_path = '/etc/letsencrypt/live/to-ai.net/'
        elif os.path.exists('C:\\Certbot\\live\\to-ai.net-0001\\fullchain.pem'):
            cert_path = 'C:\\Certbot\\live\\to-ai.net-0001\\'
        else:
            raise FileNotFoundError("Certificate files not found")

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(
            certfile=os.path.join(cert_path, 'fullchain.pem'),
            keyfile=os.path.join(cert_path, 'privkey.pem')
        )
        app.run(host='0.0.0.0', port=443, ssl_context=context)
    except Exception as e:
        print(f"Error setting up HTTPS: {e}", file=sys.stderr)
        print("Falling back to HTTP...", file=sys.stderr)
        app.run(host='0.0.0.0', port=80)