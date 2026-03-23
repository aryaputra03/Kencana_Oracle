import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# ------------------------------------------------------------------------
# 1. SETUP PATH SYSTEM
# Memastikan direktori utama (ts-inference-engine) bisa terbaca oleh pytest
# ------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.preprocessor import create_xgboost_features, create_prophet_features
from src.predictor import predicted_next_days

# =======================================================================
# FIXTURE: Membuat Data Dummy
# Fixture ini akan otomatis dipanggil oleh fungsi test yang membutuhkannya
# =======================================================================
@pytest.fixture
def dummy_historical_data():
    """
    Membuat 300 baris data dummy (harga emas harian).
    XGBoost membutuhkan minimal 253 baris (karena ada lag 252 hari).
    """
    dates = pd.date_range(start="2023-01-01", periods=300, freq='D')
    # Harga simulasi: mulai dari 1.000.000, ditambah noise acak
    np.random.seed(42)
    prices = 1000000 + np.cumsum(np.random.normal(0, 5000, size=300))
    
    df = pd.DataFrame({
        "date": dates,
        "price": prices
    })
    return df

# =======================================================================
# TEST SUITE 1: PENGUJIAN PREPROCESSOR PROPHET
# =======================================================================
def test_create_prophet_features(dummy_historical_data):
    """
    Memastikan preprocessor Prophet berhasil mengubah kolom 'date' & 'price'
    menjadi format standar Prophet yaitu 'ds' dan 'y'.
    """
    df_prophet = create_prophet_features(dummy_historical_data)
    
    # Assert (Pastikan) kolom wajib ada
    assert 'ds' in df_prophet.columns, "Kolom 'ds' untuk Prophet tidak ditemukan!"
    assert 'y' in df_prophet.columns, "Kolom 'y' untuk Prophet tidak ditemukan!"
    
    # Assert tipe data ds harus datetime
    assert pd.api.types.is_datetime64_any_dtype(df_prophet['ds']), "Kolom 'ds' harus bertipe datetime"
    
    # Assert jumlah baris tidak berubah
    assert len(df_prophet) == len(dummy_historical_data), "Jumlah baris data Prophet tidak boleh berkurang"

# =======================================================================
# TEST SUITE 2: PENGUJIAN FEATURE ENGINEERING XGBOOST
# =======================================================================
def test_create_xgboost_features_completeness(dummy_historical_data):
    """
    Memastikan preprocessor XGBoost menghasilkan seluruh fitur yang dibutuhkan:
    Fitur waktu, fitur Lag (selisih), dan fitur Rolling.
    """
    df_xgb = create_xgboost_features(dummy_historical_data)
    
    # List kolom fitur yang wajib dihasilkan oleh XGBoost preprocessor
    expected_features = [
        'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
        'lag1', 'lag2', 'lag3', 'lag5', 'lag21', 'lag63', 'lag252',
        'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7', 'rolling_std_30', 
        'rolling_max_7', 'rolling_min_7'
    ]
    
    for feature in expected_features:
        assert feature in df_xgb.columns, f"Fitur '{feature}' gagal di-generate oleh preprocessor!"
        
    # Memastikan tidak ada nilai NaN/Kosong setelah proses shift() dan rolling()
    assert df_xgb.isnull().sum().sum() == 0, "Ditemukan nilai NaN pada data yang sudah di-preprocess. Pastikan ada fungsi dropna()."

def test_xgboost_lag_logic(dummy_historical_data):
    """
    Menguji kebenaran matematis dari perhitungan selisih (diff) dan Lag.
    Ini untuk mencegah 'Data Leakage' atau perhitungan yang salah arah.
    """
    # Ambil data sebelum di preprocess
    raw_df = dummy_historical_data.copy()
    raw_df['price_diff'] = raw_df['price'].diff()
    
    # Proses data
    df_xgb = create_xgboost_features(dummy_historical_data)
    
    # Karena ada lag252 dan dropna(), baris pertama di df_xgb sebenarnya adalah 
    # baris ke-253 dari raw_df. Mari kita cek apakah lag1-nya cocok.
    
    # Index pertama dari df_xgb yang bersih
    first_clean_index = df_xgb.index[0]
    
    # Di dalam XGBoost preprocessor Anda (berdasarkan best practice):
    # 'lag1' adalah nilai 'price_diff' di H-1.
    # Kita ambil sampel baris terakhir untuk dicocokkan.
    baris_terakhir_raw = raw_df.iloc[-2]['price_diff'] # H-1 dari baris terakhir
    baris_terakhir_xgb = df_xgb.iloc[-1]['lag1']
    
    # Pengecekan matematis (menggunakan isclose untuk antisipasi selisih koma float)
    assert np.isclose(baris_terakhir_raw, baris_terakhir_xgb), "Logika perhitungan Lag 1 salah! Nilai diff tidak cocok."

# =======================================================================
# TEST SUITE 3: PENGUJIAN ALUR INFERENCE (MOCKING MODEL)
# =======================================================================
@patch('src.predictor.loader')
def test_xgboost_inference_flow(mock_loader, dummy_historical_data):
    """
    Menguji alur router di predictor.py untuk XGBoost.
    Kita memalsukan (Mock) model XGBoost agar mengembalikan nilai prediksi dummy (misal: naik Rp 5000).
    Ini memastikan loop prediksi multi-step (autoregressive) tidak error (IndexError/KeyError).
    """
    # 1. Menyiapkan Model Palsu (Mock Model)
    mock_xgb_model = MagicMock()
    # Setting model palsu agar selalu menebak selisih harga (diff) sebesar +5000 setiap harinya
    mock_xgb_model.predict.return_value = np.array([5000.0]) 
    
    # Mengelabui fungsi loader.get_model() agar me-return model palsu kita
    mock_loader.get_model.return_value = mock_xgb_model
    
    # 2. Menjalankan fungsi prediksi utama (Minta tebakan 5 hari ke depan)
    steps_to_predict = 5
    result = predicted_next_days(
        steps=steps_to_predict, 
        model_type="xgboost", 
        historical_df=dummy_historical_data
    )
    
    # 3. Assertions (Pengecekan)
    assert result['status'] == 'success', f"Inference gagal: {result.get('message', '')}"
    assert len(result['data']) == steps_to_predict, f"Jumlah hari prediksi harus {steps_to_predict}"
    
    # Pastikan model.predict dipanggil tepat sebanyak jumlah steps 
    # (karena XGBoost memprediksi hari demi hari / iteratif)
    assert mock_xgb_model.predict.call_count == steps_to_predict, "Model XGBoost tidak dipanggil secara iteratif sesuai jumlah steps!"
    
    # Cek format data kembalian
    first_pred = result['data'][0]
    assert 'date' in first_pred, "Key 'date' tidak ada di respons prediksi"
    assert 'price' in first_pred, "Key 'price' tidak ada di respons prediksi"