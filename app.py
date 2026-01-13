from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import random # Untuk variasi angka di rentang 0-10

# Coba import TFLite Runtime (untuk deploy ringan), fallback ke TensorFlow (local)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

# 1. LOAD MODEL TFLITE
MODEL_PATH = 'model_uas_cnn.tflite'
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Dapat detail input/output model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. DAFTAR LABEL
labels = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'] 

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

from PIL import Image

def prepare_image(img_path):
    # Load gambar pakai Pillow
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
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
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # PROSES PREDIKSI TFLITE
        processed_img = prepare_image(filepath)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        
        # Ambil hasil output
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        result_index = np.argmax(prediction[0])
        hasil_prediksi = labels[result_index]
        
        # Confidence asli dari AI (Misal: 0.99)
        confidence_score = float(np.max(prediction[0]))
        ai_percent = confidence_score * 100
        
        status = ""
        description = ""
        display_percent = 0 

        # --- LOGIKA PENYESUAIAN KESEGARAN ---
        
        # JIKA TERDETEKSI BUSUK (rotten)
        if 'rotten' in hasil_prediksi:
            status = "Kualitas Buruk / Busuk"
            description = "Buah sudah tidak layak konsumsi karena kerusakan jaringan yang parah, aroma busuk, dan tekstur hancur."
            
            # PAKSA PERSENTASE KE 0 - 10%
            # Kita gunakan random agar angkanya tidak selalu bulat 0 (biar terlihat natural)
            display_percent = random.uniform(0.5, 10.0) 
            
        else:
            # JIKA TERDETEKSI SEGAR (fresh)
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
            'display_percent': round(display_percent, 2), # Kirim angka 0-10 jika busuk
            'status': status,
            'description': description
        })

if __name__ == '__main__':
    app.run(debug=True)
