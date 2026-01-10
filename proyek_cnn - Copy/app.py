from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random

app = Flask(__name__)

# --- PERBAIKAN PATH AGAR TERHUBUNG DI HOSTING ---
# Mendapatkan lokasi folder tempat app.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. LOAD MODEL (Menggunakan jalur otomatis)
MODEL_PATH = os.path.join(BASE_DIR, 'model_uas_cnn.h5')
model = load_model(MODEL_PATH)

# 2. DAFTAR LABEL
labels = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'] 

# Pastikan folder upload ada di dalam folder static
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(img_path):
    # Menggunakan target_size sesuai model kamu
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ciri_ciri')
def ciri_ciri():
    return render_template('ciri_ciri.html')

@app.route('/manfaat')
def manfaat():
    return render_template('manfaat.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Gambar tidak ditemukan'}), 400
    
    file = request.files['file']
    if file:
        # Gunakan nama file yang aman dan path otomatis
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # PROSES PREDIKSI
        processed_img = prepare_image(filepath)
        prediction = model.predict(processed_img)
        
        result_index = np.argmax(prediction[0])
        hasil_prediksi = labels[result_index]
        
        confidence_score = float(np.max(prediction[0]))
        ai_percent = confidence_score * 100
        
        status = ""
        description = ""
        display_percent = 0 

        if 'rotten' in hasil_prediksi:
            status = "Kualitas Buruk / Busuk"
            description = "Buah sudah tidak layak konsumsi karena kerusakan jaringan yang parah, aroma busuk, dan tekstur hancur."
            display_percent = random.uniform(0.5, 10.0) 
        else:
            display_percent = ai_percent
            if 91 <= display_percent <= 100:
                status = "Sangat Segar (Optimal)"
                description = "Warna sangat cerah & merata, permukaan halus, tekstur sangat padat, aroma kuat."
            elif 71 <= display_percent <= 90:
                status = "Tinggi (Sangat Layak)"
                description = "Warna cerah alami, kulit kencang & mengkilap, tekstur padat."
            elif 41 <= display_percent <= 70:
                status = "Sedang (Cukup Segar)"
                description = "Warna cukup cerah, terdapat sedikit bercak ringan, tekstur agak padat."
            else:
                status = "Rendah (Segera Konsumsi)"
                description = "Warna mulai kusam, muncul banyak bercak cokelat, tekstur lembek."

        return jsonify({
            'prediction': hasil_prediksi,
            'confidence': confidence_score, 
            'display_percent': round(display_percent, 2),
            'status': status,
            'description': description,
            # Tambahkan path gambar agar bisa ditampilkan kembali di HTML
            'image_path': f"static/uploads/{file.filename}"
        })

if __name__ == '__main__':
    # Jangan gunakan debug=True saat sudah di hosting (Streamlit/PythonAnywhere)
    app.run()
