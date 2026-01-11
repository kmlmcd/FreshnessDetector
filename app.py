import streamlit as st
import os
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random
import threading

# --- 1. JEMBATAN STREAMLIT (Agar tampilan tidak kosong) ---
st.set_page_config(page_title="AI Vision Status", page_icon="🚀")
st.title("🤖 AI Vision Server")
st.success("Server deteksi kesegaran buah sedang berjalan!")
st.info("Catatan: Gunakan URL publik yang diberikan Streamlit untuk mengakses antarmuka Flask.")

# --- 2. KONFIGURASI FLASK ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. LOAD MODEL CNN
MODEL_PATH = os.path.join(BASE_DIR, 'model_uas_cnn.h5')
model = load_model(MODEL_PATH)

# 2. DAFTAR LABEL
labels = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'] 

# 3. FOLDER UPLOAD
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

# --- ROUTES ---
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
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

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
            'description': description
        })

# --- MENJALANKAN FLASK DI BACKGROUND ---
def run_flask():
    # Menggunakan port 8080 agar tidak bentrok dengan port default cloud
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

if __name__ == '__main__':
    t = threading.Thread(target=run_flask)
    t.start()
