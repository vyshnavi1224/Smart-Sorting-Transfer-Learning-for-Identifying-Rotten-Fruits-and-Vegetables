from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = tf.keras.models.load_model('healthy_vs_rotten.h5')

class_names = [
    'Apple_healthy', 'Apple_rotten', 'Banana_healthy', 'Banana_rotten',
    'Bell_pepper_healthy', 'Bell_pepper_rotten', 'Carrot_healthy', 'Carrot_rotten',
    'Cucumber_healthy', 'Cucumber_rotten']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            processed_image = preprocess_image(filepath)
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])
            
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            os.remove(filepath)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'image': img_base64
            })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)