#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAS Machine Learning — Dataset 4 (Intel Berkeley Lab Sensor Data)
Author : (isi nama Anda)
NIM    : (isi NIM Anda)
Kelas  : (isi kelas Anda)

Tujuan & Alasan (singkat):
- Tujuan utama: Deteksi anomali bacaan sensor lingkungan (temp/kelembapan/CO₂/light) per sensor dan secara global.
  Alasan: Dataset sensor lingkungan cenderung mengandung noise, missing value, dan potensi kerusakan sensor;
          deteksi anomali membantu mengidentifikasi pembacaan tidak wajar maupun sensor bermasalah.
- Tujuan tambahan: Prediksi suhu jangka pendek (regresi) dari fitur lain + fitur waktu.
  Alasan: Suhu dipengaruhi oleh pola harian serta variabel lingkungan (kelembapan, CO₂, cahaya), bermanfaat untuk monitoring.

Algoritma yang digunakan & alasan:
- IsolationForest untuk deteksi anomali: efisien untuk data berdimensi menengah, tidak perlu label, andal untuk outlier.
- RandomForestRegressor untuk prediksi suhu: model non-linear, relatif robust terhadap skala/kolinearitas, dan memberi feature importance.

Cara pakai (singkat):
1) Siapkan data CSV (bisa satu file atau banyak file dalam satu folder). Minimal kolom: timestamp, sensor_id, dan beberapa dari: temperature, humidity, co2, light.
   Nama kolom fleksibel (mis. 'temp' atau 'temperature', 'hum' atau 'humidity', 'moteid'/'node_id' untuk sensor, dsb.).
2) Jalankan:
   python uas_dataset4_intel_lab_sensor.py --data_path /path/ke/folder_atau_file.csv --time_col <nama_kolom_waktu_opsional> --sensor_col <nama_kolom_sensor_opsional>
3) Hasil disimpan di folder outputs/: ringkasan, grafik, file anomali, dan metrik model regresi.

Catatan: Skrip ini tidak mengunduh data otomatis; jalankan pada lingkungan Anda yang memiliki data.
"""

import argparse
import os
import re
import glob
import math
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# ---------- Helpers ----------

CANDIDATE_TIME = ['timestamp','time','datetime','date','ts','epoch']
CANDIDATE_SENSOR = ['sensor','sensor_id','id','node','node_id','moteid','mote_id','mote']
CANDIDATE_TEMP = ['temp','temperature','temp_c','t']
CANDIDATE_HUM = ['hum','humidity','rel_hum','rh']
CANDIDATE_CO2 = ['co2','co₂','co2_ppm','ppm']
CANDIDATE_LIGHT = ['light','illum','lux']
CANDIDATE_VOLT = ['voltage','batt','battery']

def guess_col(cols, cands):
    cols_lower = [c.lower() for c in cols]
    for c in cands:
        if c in cols_lower:
            return cols[cols_lower.index(c)]
    # fuzzy: startswith/contains
    for c in cols:
        cl = c.lower()
        for k in cands:
            if cl.startswith(k) or k in cl:
                return c
    return None

def coerce_time(s: pd.Series) -> pd.Series:
    # Try numeric epoch, else parse datetime
    if np.issubdtype(s.dtype, np.number):
        # assume seconds since epoch if values look large
        try:
            s2 = pd.to_datetime(s, unit='s')
            return s2
        except Exception:
            pass
    # try parse strings
    return pd.to_datetime(s, errors='coerce', infer_datetime_format=True)

def load_any(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"Tidak ada file CSV di folder: {path}")
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                df['__source_file'] = os.path.basename(f)
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Gagal membaca {f}: {e}")
        data = pd.concat(dfs, ignore_index=True)
    else:
        data = pd.read_csv(path)
        data['__source_file'] = os.path.basename(path)
    return data

def standardize_columns(df: pd.DataFrame, time_col=None, sensor_col=None) -> Tuple[pd.DataFrame, Dict[str,str]]:
    cols = list(df.columns)
    # Identify columns
    time_c = time_col or guess_col(cols, CANDIDATE_TIME)
    sens_c = sensor_col or guess_col(cols, CANDIDATE_SENSOR)
    temp_c = guess_col(cols, CANDIDATE_TEMP)
    hum_c  = guess_col(cols, CANDIDATE_HUM)
    co2_c  = guess_col(cols, CANDIDATE_CO2)
    light_c= guess_col(cols, CANDIDATE_LIGHT)
    volt_c = guess_col(cols, CANDIDATE_VOLT)

    mapping = {}
    if time_c: mapping[time_c] = 'timestamp'
    if sens_c: mapping[sens_c] = 'sensor_id'
    if temp_c: mapping[temp_c] = 'temperature'
    if hum_c:  mapping[hum_c]  = 'humidity'
    if co2_c:  mapping[co2_c]  = 'co2'
    if light_c:mapping[light_c]= 'light'
    if volt_c: mapping[volt_c] = 'voltage'
    df = df.rename(columns=mapping)

    if 'timestamp' not in df.columns:
        raise ValueError("Kolom waktu tidak ditemukan. Gunakan --time_col untuk menentukan.")
    if 'sensor_id' not in df.columns:
        # jika tidak ada sensor_id, buat jadi 0
        df['sensor_id'] = 0

    df['timestamp'] = coerce_time(df['timestamp'])
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    return df, mapping

def resample_fill(group: pd.DataFrame, rule='5min') -> pd.DataFrame:
    group = group.set_index('timestamp').sort_index()
    # keep numeric columns only for resampling
    numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
    # we will also keep non-numeric by forward fill after merging
    res = group[numeric_cols].resample(rule).mean()
    res = res.interpolate(method='time', limit=3).ffill().bfill()
    res['sensor_id'] = group['sensor_id'].mode().iloc[0] if not group.empty else None
    return res.reset_index()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['timestamp'].dt.hour
    df['dow']  = df['timestamp'].dt.dayofweek
    df['month']= df['timestamp'].dt.month
    return df

def add_rolling_and_lags(df: pd.DataFrame, target='temperature', windows=(3,12), lags=(1,2,3)) -> pd.DataFrame:
    df = df.sort_values('timestamp')
    for w in windows:
        df[f'{target}_rollmean_{w}'] = df[target].rolling(w, min_periods=1).mean()
        df[f'{target}_rollstd_{w}']  = df[target].rolling(w, min_periods=1).std().fillna(0.0)
    for l in lags:
        df[f'{target}_lag_{l}'] = df[target].shift(l)
    return df

def train_isoforest(X: pd.DataFrame, contamination=0.01, random_state=42) -> IsolationForest:
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    model.fit(X)
    return model

def evaluate_regression(y_true, y_pred) -> Dict[str,float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {'MAE': float(mae), 'RMSE': float(rmse)}

def plot_sample(df: pd.DataFrame, outpath: str, cols: List[str]):
    plt.figure(figsize=(10,4))
    for c in cols:
        if c in df.columns:
            plt.plot(df['timestamp'], df[c], label=c)
    plt.legend()
    plt.title('Sampel Time Series')
    plt.xlabel('Waktu')
    plt.ylabel('Nilai')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def ensure_outputs(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for sub in ['figs','models','tables']:
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path ke file CSV atau folder berisi banyak CSV')
    parser.add_argument('--time_col', type=str, default=None, help='Nama kolom waktu (opsional)')
    parser.add_argument('--sensor_col', type=str, default=None, help='Nama kolom sensor_id (opsional)')
    parser.add_argument('--resample', type=str, default='5min', help='Frekuensi resampling (default 5min)')
    parser.add_argument('--contamination', type=float, default=0.01, help='Persentase outlier untuk IsolationForest')
    parser.add_argument('--test_size_days', type=int, default=7, help='Ukuran test set (hari terakhir) untuk regresi')
    parser.add_argument('--outdir', type=str, default='outputs', help='Folder output')
    args = parser.parse_args()

    ensure_outputs(args.outdir)

    print("== Memuat data ==")
    raw = load_any(args.data_path)
    std, mapping = standardize_columns(raw, time_col=args.time_col, sensor_col=args.sensor_col)
    print(f"Pemetaan kolom: {mapping}")

    core_vars = [c for c in ['temperature','humidity','co2','light','voltage'] if c in std.columns]
    print(f"Fitur tersedia: {core_vars}")

    # EDA ringkas
    eda_info = {
        'n_rows': int(len(std)),
        'n_sensors': int(std['sensor_id'].nunique()),
        'time_min': std['timestamp'].min().isoformat() if not std.empty else None,
        'time_max': std['timestamp'].max().isoformat() if not std.empty else None,
        'na_counts': std[ ['timestamp','sensor_id'] + core_vars ].isna().sum().to_dict()
    }
    with open(os.path.join(args.outdir, 'tables', 'eda_summary.json'), 'w') as f:
        json.dump(eda_info, f, indent=2, ensure_ascii=False)

    # Resample per sensor
    print("== Resampling & imputasi per sensor ==")
    resampled = (
        std
        .dropna(subset=['timestamp'])
        .sort_values(['sensor_id','timestamp'])
        .groupby('sensor_id', group_keys=False)
        .apply(lambda g: resample_fill(g, rule=args.resample))
        .reset_index(drop=True)
    )

    # Plot contoh
    plot_cols = [c for c in ['temperature','humidity','co2','light'] if c in resampled.columns][:3]
    if plot_cols:
        plot_sample(resampled.iloc[:2000], os.path.join(args.outdir,'figs','sample_timeseries.png'), ['temperature'] if 'temperature' in plot_cols else plot_cols)

    # ---------- Deteksi Anomali (IsolationForest) ----------
    print("== Deteksi anomali (IsolationForest) ==")
    feat_anom = [c for c in ['temperature','humidity','co2','light','voltage'] if c in resampled.columns]
    anom_results = []
    if feat_anom:
        X_all = resampled[feat_anom].copy()
        X_all = X_all.replace([np.inf,-np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        iso = train_isoforest(X_all, contamination=args.contamination)
        scores = iso.decision_function(X_all)
        preds = iso.predict(X_all)  # -1 outlier, 1 normal
        resampled['anomaly_score'] = scores
        resampled['is_anomaly'] = (preds == -1).astype(int)

        # Simpan top outlier
        top = resampled.sort_values('anomaly_score').head(100)
        top[['timestamp','sensor_id'] + feat_anom + ['anomaly_score','is_anomaly']].to_csv(
            os.path.join(args.outdir,'tables','top_anomalies.csv'), index=False
        )

        # ringkasan per sensor
        rate = resampled.groupby('sensor_id')['is_anomaly'].mean().reset_index().rename(columns={'is_anomaly':'anomaly_rate'})
        rate.to_csv(os.path.join(args.outdir,'tables','anomaly_rate_per_sensor.csv'), index=False)

    # ---------- Prediksi Suhu (RandomForest) ----------
    if 'temperature' in resampled.columns:
        print("== Prediksi suhu (RandomForestRegressor) ==")
        df_temp = resampled.dropna(subset=['temperature']).copy()
        df_temp = add_time_features(df_temp)
        df_temp = add_rolling_and_lags(df_temp, target='temperature', windows=(3,12), lags=(1,2,3))

        features = [c for c in [
            'humidity','co2','light','voltage',
            'hour','dow','month',
            'temperature_rollmean_3','temperature_rollstd_3',
            'temperature_rollmean_12','temperature_rollstd_12',
            'temperature_lag_1','temperature_lag_2','temperature_lag_3'
        ] if c in df_temp.columns]

        df_temp = df_temp.dropna(subset=features + ['temperature'])
        if not df_temp.empty:
            # split train/test by time (last N days as test)
            cutoff = df_temp['timestamp'].max() - pd.Timedelta(days=int(args.test_size_days))
            train = df_temp[df_temp['timestamp'] <= cutoff]
            test  = df_temp[df_temp['timestamp'] > cutoff]
            if len(train) > 100 and len(test) > 20:
                Xtr, ytr = train[features], train['temperature']
                Xte, yte = test[features], test['temperature']

                rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                rf.fit(Xtr, ytr)
                pred = rf.predict(Xte)
                metrics = evaluate_regression(yte, pred)

                with open(os.path.join(args.outdir,'tables','regression_metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)

                imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
                imp.to_csv(os.path.join(args.outdir,'tables','feature_importance.csv'))

                # Plot pred vs actual (sampel)
                sample_plot = test[['timestamp']].copy()
                sample_plot['y_true'] = yte.values
                sample_plot['y_pred'] = pred
                plt.figure(figsize=(10,4))
                plt.plot(sample_plot['timestamp'], sample_plot['y_true'], label='Actual')
                plt.plot(sample_plot['timestamp'], sample_plot['y_pred'], label='Pred')
                plt.legend()
                plt.title('Prediksi Suhu — Uji (sampel)')
                plt.xlabel('Waktu')
                plt.ylabel('Suhu')
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir,'figs','pred_vs_actual_temp.png'))
                plt.close()

    # ---------- Simpan ringkasan run ----------
    run_summary = {
        'mapping_used': mapping,
        'features_available': core_vars,
        'eda': eda_info,
        'resample_rule': args.resample,
        'isolationforest_contamination': args.contamination,
        'regression_test_days': args.test_size_days
    }
    with open(os.path.join(args.outdir,'run_summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print("Selesai. Cek folder outputs/ untuk hasil tabel & grafik.")

if __name__ == '__main__':
    main()
