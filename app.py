from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
from werkzeug.utils import secure_filename

print(f"NumPy version used in Flask app: {np.__version__}")
# 🔹 Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Folder penyimpanan
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 🔹 Memuat Model XGBoost yang Sudah Dilatih
model_path = 'xgboost_model.model'  # Sesuaikan dengan path model Anda
xgb_model = XGBClassifier()
xgb_model.load_model(model_path)

# 🔹 Daftar kolom untuk One-Hot Encoding (sesuai dataset asli)
one_hot_columns = [
    "jenis_kelamin", "kluster", "univ", "Beasiswa", "Jabatan", "StatusNikah", "KampusDomisili"
]

# 🔹 Fungsi Preprocessing (Konversi ke One-Hot Encoding)
def preprocess_data(df):
    df = df.copy()

    # Lakukan One-Hot Encoding pada kolom kategori
    df = pd.get_dummies(df, columns=one_hot_columns)

    # Pastikan semua kolom yang dibutuhkan oleh model ada (tambahkan 0 jika tidak ada)
    for col in xgb_model.get_booster().feature_names:
        if col not in df.columns:
            df[col] = 0

    # Sesuaikan urutan kolom sesuai model
    df = df[xgb_model.get_booster().feature_names]

    return df

# 🔹 Halaman utama (web form)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 🔹 Endpoint Prediksi via Upload File
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        try:
            # Simpan file upload
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Membaca file (CSV/Excel)
            if file.filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Preprocessing
            df_processed = preprocess_data(df)

            # Lakukan prediksi menggunakan model XGBoost
            predictions = xgb_model.predict(df_processed)
            df['Predicted_Result'] = predictions  # Tambahkan hasil prediksi

            # Simpan file hasil prediksi
            result_filename = 'hasil_' + filename
            result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            df.to_csv(result_filepath, index=False)

            return jsonify({'message': 'Prediction successful', 'download_link': '/download/' + result_filename})

        except Exception as e:
            return jsonify({'error': f"Terjadi kesalahan dalam memproses file: {str(e)}"}), 500

    return jsonify({'error': 'File harus dalam format CSV atau Excel'}), 400

# 🔹 Endpoint Prediksi via Input Manual
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
         # Ambil data dari request JSON
        input_data = request.get_json()
        
        # Konversi ke DataFrame
        df_input = pd.DataFrame([input_data])

        # Konversi tipe data yang seharusnya numerik
        numeric_columns = ["Strata", "tahun_intake", "Umur_intake", "Lama_Bekerja"]
        for col in numeric_columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

        # Pastikan tidak ada nilai NaN setelah konversi
        if df_input[numeric_columns].isnull().any().any():
            return jsonify({'error': 'Beberapa input numerik tidak valid'}), 400
        
        # Preprocessing (konversi ke One-Hot Encoding)
        df_processed = preprocess_data(df_input)

        # Prediksi
        prediction = xgb_model.predict(df_processed)[0]

        return jsonify({'result': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

# 🔹 Endpoint Download Hasil
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

# 🔹 Menjalankan API Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
