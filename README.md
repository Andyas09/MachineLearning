# Prediksi Risiko Kredit Kartu dengan Machine Learning

Proyek ini bertujuan untuk membangun sistem prediksi **risiko gagal bayar (default)** nasabah kartu kredit menggunakan beberapa algoritma klasifikasi, termasuk **Logistic Regression**, **Decision Tree**, dan **Random Forest**. Hasil prediksi disajikan melalui antarmuka web interaktif menggunakan **Streamlit**.

---

## 📁 A. Pendahuluan

### 🔹 Latar Belakang
Gagal bayar nasabah kartu kredit merupakan salah satu risiko utama bagi lembaga keuangan. Prediksi default secara akurat dapat membantu pengambilan keputusan kredit. Dataset yang digunakan diperoleh dari Kaggle:

🔗 **Dataset**: [Credit Card Defaulter Prediction](https://www.kaggle.com/datasets/gauravtopre/credit-card-defaulter-prediction)

### 🔹 Insight yang Diharapkan
- Mengidentifikasi fitur paling berpengaruh terhadap risiko gagal bayar.
- Membandingkan performa beberapa model klasifikasi.
- Menyediakan dashboard sederhana untuk testing model oleh pengguna non-teknis.

---

## 🎯 B. Tujuan

- Membangun model machine learning untuk memprediksi apakah nasabah akan default atau tidak.
- Menyajikan hasil prediksi melalui aplikasi web interaktif (Streamlit).
- Mengoptimalkan model terbaik menggunakan **GridSearchCV**.

---

## 📌 C. Batasan

### 🔹 Ruang Lingkup:
- Data hanya mencakup informasi demografis dan perilaku pembayaran nasabah kartu kredit.
- Tidak mencakup variabel eksternal (ekonomi makro, kredit lain, dll.)

### 🔹 Batasan Model:
- Fokus hanya pada task **klasifikasi biner**: `Default` (Yes/No)
- Model tidak mendukung prediksi multikategori atau regresi.

---

## ⚙️ D. Metode & Alur

### 🔹 Teknik yang Digunakan:
1. **Data Cleaning**: handling missing values, duplicates, dan outlier
2. **Encoding**: variabel kategorikal seperti `SEX`, `EDUCATION`, `MARRIAGE`
3. **Normalisasi**: MinMaxScaler untuk data numerik
4. **Train-Test Split**
5. **Modeling**: Logistic Regression, Decision Tree, Random Forest
6. **Evaluasi**: Confusion Matrix, Classification Report
7. **Hyperparameter Tuning**: GridSearchCV
8. **Deployment**: Streamlit Web App

### 🔄 Diagram Alur (PlantUML)
```plantuml
@startuml
start
:Load Dataset;
:Data Cleaning;
:Encoding Kategorikal;
:Normalisasi;
:Train-Test Split;
partition Model {
  :Logistic Regression;
  :Decision Tree;
  :Random Forest;
}
:Evaluasi Model;
:GridSearchCV;
if (Model Terbaik?) then (Ya)
  :Simpan model_rf.pkl;
  :Deploy dengan Streamlit;
else (Ulangi)
  -> Model
endif
stop
@enduml
