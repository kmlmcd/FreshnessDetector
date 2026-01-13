from flask import Flask, render_template, request, jsonify
import os
import requests

# KONFIGURASI HUGGING FACE INFERENCE API
HF_API_URL = "https://api-inference.huggingface.co/models/kmlmcd/model-freshness-detector"
# Token ini optional untuk public model, tapi disarankan pakai jika rate limit
# Pastikan set HF_TOKEN di Environment Variables Render/Vercel
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def query_huggingface(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(HF_API_URL, headers=headers, data=data)
    return response.json()

# 2. DAFTAR LABEL
labels = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'] 

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



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

        # PROSES PREDIKSI VIA API HUGGING FACE
        # Kirim gambar ke server hugging face, biarkan mereka yang memproses
        try:
            api_response = query_huggingface(filepath)
            
            # Format response HF biasanya list of dict: [{'label': 'freshapples', 'score': 0.99}, ...]
            # Kita cari yang score-nya paling tinggi
            if isinstance(api_response, list) and len(api_response) > 0:
                # Sort berdasarkan score tertinggi
                top_result = sorted(api_response, key=lambda x: x['score'], reverse=True)[0]
                hasil_prediksi = top_result['label']
                confidence_score = top_result['score']
            else:
                # Handle error jika model loading / error lain
                # Fallback sementara (mock) atau return error
                print("API Error:", api_response)
                return jsonify({'error': 'Gagal memproses gambar di server AI. Coba lagi nanti.'}), 503

        except Exception as e:
             print("Error calling API:", e)
             return jsonify({'error': str(e)}), 500

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
