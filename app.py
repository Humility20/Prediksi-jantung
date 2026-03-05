import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. Menentukan lokasi file model.pkl
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts/model.pkl"

# 2. Memuat Model
model = joblib.load(MODEL_PATH)

# 3. Membuat Tampilan Web
st.title("❤️ Aplikasi Prediksi Serangan Jantung")
st.write("Masukkan data medis pasien di bawah ini untuk melihat hasil prediksi (Deployment by Gihon).")
st.markdown("---")

# 4. Form Input Fitur (Feature)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Umur (age)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Jenis Kelamin (sex)", [0, 1], help="0 = Wanita, 1 = Pria")
    cp = st.selectbox("Tipe Nyeri Dada (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Kolesterol (chol)", min_value=100, max_value=600, value=200)

with col2:
    fbs = st.selectbox("Gula Darah > 120 (fbs)", [0, 1])
    restecg = st.selectbox("Hasil ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Detak Jantung Maksimal (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Nyeri Olahraga (exang)", [0, 1])

with col3:
    oldpeak = st.number_input("Depresi ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Kemiringan ST (slope)", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Darah (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

st.markdown("---")

# 5. Tombol Eksekusi
if st.button("🔍 Prediksi Risiko"):
    # Menyusun data dari input web menjadi tabel (DataFrame)
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Memberikan data ke model untuk ditebak
    prediksi = model.predict(input_data)

    # Menampilkan hasil
    if prediksi[0] == 1:
        st.error("⚠️ HASIL: Pasien berisiko TINGGI terkena serangan jantung.")
    else:
        st.success("✅ HASIL: Pasien berisiko RENDAH terkena serangan jantung.")