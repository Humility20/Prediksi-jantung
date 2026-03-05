import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. LOAD MODEL (Hanya model.pkl saja)
model = joblib.load("artifacts/model.pkl")

# 2. BUAT SCALER OTOMATIS (Tanpa butuh file .pkl tambahan)
# Kita butuh data CSV sebentar untuk "mengajari" scaler tentang skala data kamu
df_train = pd.read_csv("Heart Attack Data Set.csv")
X_train = df_train.drop('target', axis=1) # Pastikan 'target' adalah nama kolom hasil di CSV-mu
scaler = StandardScaler()
scaler.fit(X_train) # Scaler sekarang sudah "pintar"

def main():
    st.title('Heart Attack Risk Prediction')
    st.write("Aplikasi ini menggunakan gaya pemrosesan manual sesuai standar kelas.")

    # Input User (Sesuaikan urutan dengan kolom di CSV kamu)
    # Urutan: age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall
    age = st.number_input('Age', value=50)
    sex = st.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    cp = st.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trtbps = st.number_input('Resting Blood Pressure', value=120)
    chol = st.number_input('Cholesterol', value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 (1=True, 0=False)', [0, 1])
    restecg = st.slider('Resting ECG (0-2)', 0, 2, 0)
    thalachh = st.number_input('Max Heart Rate', value=150)
    exng = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0, 1])
    oldpeak = st.number_input('Oldpeak', value=1.0, step=0.1)
    slp = st.slider('Slope (0-2)', 0, 2, 1)
    caa = st.slider('Number of vessels (0-4)', 0, 4, 0)
    thall = st.slider('Thallium Test (0-3)', 0, 3, 2)

    if st.button('Predict Risk'):
        # MASUKKAN KE LIST (Urutan harus SAMA dengan X_train)
        features = [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]
        result = make_prediction(features)
        
        if result == 1:
            st.error("Hasil: Berisiko Tinggi Serangan Jantung")
        else:
            st.success("Hasil: Risiko Rendah")

def make_prediction(features):
    # Logika yang dicari Dosen: Ada proses transform manual
    input_array = np.array(features).reshape(1, -1)
    
    # Proses scaling manual sebelum masuk ke model (Gaya Dosen)
    X_scaled = scaler.transform(input_array) 
    
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()