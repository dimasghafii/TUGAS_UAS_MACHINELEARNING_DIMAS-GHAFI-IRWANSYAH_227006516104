# UAS Machine Learning — Dataset 4 (Intel Berkeley Lab Sensor Data)

Repositori ini menyiapkan skrip analisis untuk dataset **Intel Berkeley Lab Sensor Data** (temperature/kelembapan/CO₂/light, 54 sensor).

> Disusun untuk memenuhi instruksi UAS pada dokumen soal. Anda diminta menentukan tujuan & algoritma, membuat program Python, menghasilkan output, lalu mengunggah ke GitHub dan submit tautannya. (Lihat kutipan pada dokumen UAS).

## Tujuan
1. **Deteksi anomali** bacaan sensor (per sensor & global) untuk menemukan pembacaan tidak wajar/kerusakan sensor.
2. **Prediksi suhu jangka pendek** menggunakan variabel lingkungan & fitur waktu.

## Algoritma & Alasan
- **IsolationForest** (unsupervised) untuk deteksi anomali — efisien, tidak butuh label, cocok untuk mencari outlier.
- **RandomForestRegressor** untuk prediksi suhu — memodelkan hubungan non-linear, robust, dan menyediakan feature importance.

## Struktur
```
uas_dataset4_project/
├─ uas_dataset4_intel_lab_sensor.py   # Skrip utama
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ outputs/
   ├─ figs/       # grafik (otomatis dibuat)
   ├─ models/     # disiapkan jika ingin menyimpan model
   └─ tables/     # tabel hasil (json/csv)
```

## Cara Menjalankan
1. **Siapkan data** lokal berupa satu/lebih file CSV. Minimal kolom: `timestamp`, `sensor_id`, dan beberapa dari `temperature`, `humidity`, `co2`, `light`.  
   Nama kolom fleksibel, skrip akan mencoba mendeteksi otomatis (bisa override via argumen).
2. **Install dependency**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan skrip** (contoh):
   ```bash
   python uas_dataset4_intel_lab_sensor.py --data_path /path/ke/folder_csv --resample 5min --contamination 0.01 --test_size_days 7
   ```
   Opsi penting:
   - `--time_col` dan `--sensor_col` jika nama kolom tidak terdeteksi otomatis.
   - `--resample` untuk mengatur interval (mis. `1min`, `5min`, `10min`).
   - `--contamination` untuk persentase outlier pada IsolationForest.

4. **Output** akan muncul di folder `outputs/` berupa:
   - `tables/eda_summary.json` — ringkasan jumlah baris, rentang waktu, dll.
   - `tables/top_anomalies.csv` — 100 pembacaan paling anomali.
   - `tables/anomaly_rate_per_sensor.csv` — proporsi anomali per sensor.
   - `tables/regression_metrics.json` — MAE & RMSE prediksi suhu (jika kolom `temperature` tersedia memadai).
   - `tables/feature_importance.csv` — pentingnya fitur pada RandomForest.
   - `figs/sample_timeseries.png` — plot contoh deret waktu.
   - `figs/pred_vs_actual_temp.png` — perbandingan aktual vs prediksi (uji).

## Catatan Tambahan
- Skrip **tidak mengunduh** dataset otomatis; gunakan data lokal Anda.
- Untuk menyesuaikan target lain (mis. prediksi `co2`), ganti bagian fitur/target pada kode.
- Anda bisa menambahkan penyimpanan model (`joblib`) jika diperlukan.

Semoga membantu! Sukses UAS-nya.
