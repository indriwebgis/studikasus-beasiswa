# Sistem Seleksi Beasiswa (Streamlit)

Aplikasi ini membungkus notebook **Sistem Seleksi Beasiswa dengan Machine Learning** menjadi UI Streamlit dengan tahapan:

1. **Upload & Cleaning**
2. **EDA**
3. **Menjalankan Model**
4. **Membandingkan metrik (Accuracy, Precision, Recall, F1-score)**
5. **Memilih model terbaik (kuantitatif)**

> Jika pengguna tidak mengunggah data baru, aplikasi menyediakan opsi memakai **data contoh** `beasiswa.csv`.

## Skema data (6 kolom)
Aplikasi ini menggunakan **5 fitur + 1 target** (total 6 kolom):

- `IPK`
- `Pendapatan_Orang_Tua`
- `Prestasi_Akademik`
- `Prestasi_Non_Akademik`
- `Keikutsertaan_Organisasi`
- `Diterima_Beasiswa` (target, 0/1)

Data contoh `beasiswa.csv` di repo ini sudah mengikuti skema 6 kolom tersebut.

## Model
Model yang digunakan:

- Logistic Regression
- XGBoost

## Struktur
- `app.py` : aplikasi Streamlit
- `requirements.txt` : dependensi
- `beasiswa.csv` : data contoh (opsional, tapi disarankan agar opsi sample selalu tersedia)

## Cara jalan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Cloud
1) Upload folder ini ke GitHub (minimal `app.py` + `requirements.txt`)
2) Sertakan `beasiswa.csv` agar pengguna bisa memilih data contoh ketika tidak upload
3) Di Streamlit Cloud: pilih entry point **app.py**
