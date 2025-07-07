import streamlit as st
import numpy as np
import pickle
model = pickle.load(open('model_rf.pkl', 'rb'))

st.set_page_config(page_title="Prediksi Kredit Macet", layout="centered")
st.title("🚨 Prediksi Peluang Kredit Macet (Default)")

st.markdown("Masukkan data nasabah di bawah ini:")
LIMIT_BAL = st.number_input("Limit Kartu Kredit (LIMIT_BAL)", min_value=0)
AGE = st.number_input("Umur (AGE)", min_value=18, max_value=100)
SEX = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
EDUCATION = st.selectbox("Pendidikan", ["SMA", "Diploma", "Sarjana", "Lainnya"])
MARRIAGE = st.selectbox("Status Pernikahan", ["Lajang", "Menikah", "Lainnya"])

PAY_0 = st.number_input("Status Pembayaran Bulan Ini (PAY_0)", min_value=-2, max_value=8)
BILL_AMT1 = st.number_input("Jumlah Tagihan Bulan Ini (BILL_AMT1)", min_value=0)
PAY_AMT1 = st.number_input("Jumlah Pembayaran Bulan Ini (PAY_AMT1)", min_value=0)
SEX = 1 if SEX == "Laki-laki" else 0
EDUCATION = {"SMA": 1, "Diploma": 2, "Sarjana": 3, "Lainnya": 0}[EDUCATION]
MARRIAGE = {"Lajang": 1, "Menikah": 2, "Lainnya": 0}[MARRIAGE]
if st.button("Prediksi"):
    features = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, BILL_AMT1, PAY_AMT1]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    if prediction == 1:
        st.error(f"❌ Nasabah **berisiko gagal bayar**. (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Nasabah **diprediksi lancar bayar**. (Probabilitas: {prob:.2f})")
