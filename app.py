import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)

# --- Folder setup ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PREDICT_FOLDER = os.path.join('static', 'outputs', 'predict')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load YOLOv8 model ---
MODEL_PATH = os.path.join('yolov8_model', 'best.pt')
model = YOLO(MODEL_PATH)

# --- Home route ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', uploaded_image=None, result=None, detected_image=None)

# --- Upload route ---
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        uploaded_image = url_for('static', filename=f'uploads/{filename}')
        return render_template('index.html', uploaded_image=uploaded_image, result=None, detected_image=None)

# --- Detect route ---
@app.route('/detect', methods=['POST'])
def detect():
    image_path = request.form['image_path']
    full_path = image_path.replace('/static/', 'static/')

    # Run YOLOv8 detection
    results = model(full_path)

    # Save predicted image
    predicted_filename = os.path.basename(full_path)
    predicted_filepath = os.path.join(PREDICT_FOLDER, predicted_filename)
    results[0].save(filename=predicted_filepath)

    # Result message
    if results[0].boxes:
        result_text = "Garbage Detected!"
    else:
        result_text = "No Garbage Detected."

    predicted_image_path = url_for('static', filename=f'outputs/predict/{predicted_filename}')
    
    return render_template('index.html', uploaded_image=image_path, result=result_text, detected_image=predicted_image_path)

if __name__ == '__main__':
    app.run(debug=True)
