# Flask CNN Project

Projek ini telah dikonfigurasi untuk siap deploy di Vercel dan menggunakan model dari Hugging Face.

## Persiapan Deploy
1. Pastikan file `requirements.txt` sudah lengkap.
2. File `vercel.json` mengatur konfigurasi untuk Vercel.

## Konfigurasi Hugging Face
Agar aplikasi bisa mengunduh model dari Hugging Face otomatis jika file lokal tidak ada:
1. Upload file `model_uas_cnn.h5` ke repository Hugging Face Anda.
2. Saat deploy di Vercel, tambahkan Environment Variable:
   - `HF_REPO_ID`: `username_huggingface/nama_repo` (Contoh: `user123/cnn-fruit-model`)

## Jalankan Lokal
```bash
pip install -r requirements.txt
python app.py
```
