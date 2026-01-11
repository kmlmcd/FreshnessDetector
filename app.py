import streamlit as st
import streamlit.components.v1 as components
import os
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random
import threading
import time

# --- 1. KONFIGURASI FLASK (MESIN BELAKANG) ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model AI
MODEL_PATH = os.path.join(BASE_DIR, 'model_uas_cnn.h5')
model = load_model(MODEL_PATH)

labels = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'] 

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

# Routes Flask untuk Tampilan HTML Kamu
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
        
        status, description = "", ""
        if 'rotten' in hasil_prediksi:
            status = "Kualitas Buruk / Busuk"
            description = "Buah sudah tidak layak konsumsi karena kerusakan jaringan yang parah."
            display_percent = random.uniform(0.5, 10.0) 
        else:
            display_percent = ai_percent
            if display_percent >= 91:
                status = "Sangat Segar"
                description = "Kondisi optimal untuk dikonsumsi."
            else:
                status = "Cukup Segar"
                description = "Masih layak, namun segera konsumsi."

        return jsonify({
            'prediction': hasil_prediksi,
            'display_percent': round(display_percent, 2),
            'status': status,
            'description': description
        })

# --- 2. MENJALANKAN FLASK DI THREAD TERPISAH ---
def run_flask():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# Cek apakah thread sudah jalan agar tidak double
if 'flask_started' not in st.session_state:
    thread = threading.Thread(target=run_flask)
    thread.daemon = True
    thread.start()
    st.session_state.flask_started = True

# --- 3. TAMPILAN FRONT-END STREAMLIT (JEMBATAN) ---
st.set_page_config(page_title="AI Freshness Detector", layout="wide")

# CSS untuk menyembunyikan header default Streamlit agar terlihat clean
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Tampilkan Web Flask asli kamu di dalam Iframe
st.info("Sedang memuat sistem AI Vision... Mohon tunggu sejenak.")
time.sleep(2) # Memberi waktu Flask untuk booting

# Alamat URL lokal di server Streamlit
components.iframe("http://localhost:8080", height=900, scrolling=True)
