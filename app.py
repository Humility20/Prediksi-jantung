import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Menentukan lokasi model (karena model.pkl kamu ada di folder artifacts)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts/model.pkl"

# Load model (di dalam model.pkl ini sudah ada StandardScaler + Logistic Regression)
model = joblib.load(MODEL_PATH)

def main():
    st.title('Heart Attack Risk Prediction')
    st.write("Aplikasi prediksi risiko serangan jantung menggunakan Scikit-Learn Pipeline.")

    # Input User sesuai dengan urutan fitur di dataset jantung
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
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
        # Susun data ke dalam DataFrame agar Pipeline bisa membacanya dengan benar
        # Nama kolom harus sama dengan saat training agar tidak error
        columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 
                   'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
        
        input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, 
                                    thalachh, exng, oldpeak, slp, caa, thall]], 
                                   columns=columns)
        
        # Langsung prediksi (Scaling dilakukan otomatis oleh Pipeline di balik layar)
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("Hasil: Berisiko Tinggi Serangan Jantung")
        else:
            st.success("Hasil: Risiko Rendah")

if __name__ == '__main__':
    main()