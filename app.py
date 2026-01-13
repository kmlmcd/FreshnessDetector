from flask import Flask, render_template, request, jsonify
import os
import requests

# 1. INISIALISASI FLASK HARUS DI ATAS DULU
app = Flask(__name__)

# 2. KONFIGURASI API
HF_API_URL = "https://api-inference.huggingface.co/models/kmlmcd/model-freshness-detector"
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def query_huggingface(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(HF_API_URL, headers=headers, data=data)
    return response.json()

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 3. BARU ROUTING DI BAWAHNYA
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

        try:
            api_response = query_huggingface(filepath)
            
            if isinstance(api_response, list) and len(api_response) > 0:
                top_result = sorted(api_response, key=lambda x: x['score'], reverse=True)[0]
                hasil_prediksi = top_result['label']
                confidence_score = top_result['score']
                
                ai_percent = confidence_score * 100
                status = ""
                description = ""
                display_percent = 0
                
                # --- LOGIKA KESEGARAN ---
                if 'rotten' in hasil_prediksi:
                    status = "Kualitas Buruk / Busuk"
                    description = "Buah sudah tidak layak konsumsi."
                    import random
                    display_percent = random.uniform(0.5, 10.0) 
                else:
                    display_percent = ai_percent
                    if 91 <= display_percent <= 100:
                        status = "Sangat Segar (Optimal)"
                        description = "Kualitas terbaik, warna cerah, aroma kuat."
                    elif 71 <= display_percent <= 90:
                        status = "Tinggi (Sangat Layak)"
                        description = "Masih sangat segar dan layak konsumsi."
                    elif 41 <= display_percent <= 70:
                        status = "Sedang (Cukup Segar)"
                        description = "Cukup segar, segera konsumsi."
                    else:
                        status = "Rendah (Segera Konsumsi)"
                        description = "Mulai layu/kusam, konsumsi hari ini."
                
                return jsonify({
                    'prediction': hasil_prediksi,
                    'confidence': confidence_score, 
                    'display_percent': round(display_percent, 2), 
                    'status': status,
                    'description': description
                })
            else:
                 print("API Error:", api_response)
                 if isinstance(api_response, dict) and 'error' in api_response:
                     return jsonify({'error': f"AI sedang loading: {api_response['error']}. Coba lagi nanti."}), 503
                 return jsonify({'error': 'Gagal memproses gambar.'}), 500

        except Exception as e:
             print("Error:", e)
             return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
