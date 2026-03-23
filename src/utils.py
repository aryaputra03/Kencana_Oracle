import pandas as pd
import numpy as np
from datetime import datetime

def generate_future_dates(start_date: str = None, steps: int = 30, freq: str = 'B') -> list:
    """
    Membuat daftar tanggal masa depan untuk sumbu X pada grafik/JSON.
    
    Args:
        start_date (str): Tanggal awal (format 'YYYY-MM-DD'). Jika None, pakai hari ini.
        steps (int): Jumlah hari ke depan.
        freq (str): Frekuensi ('D' = Daily, 'B' = Business Day/Senin-Jumat). 
                    Default 'B' karena pasar emas libur weekend.
    
    Returns:
        list: List berisi string tanggal (contoh: ['2024-01-01', '2024-01-02'])
    """
    if start_date is None:
        start_obj = datetime.now()
    else:
        try:
            start_obj = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            start_obj = datetime.now()
    
    dates = pd.date_range(start=start_obj, periods=steps + 1, freq=freq)

    return [d.strftime("%Y-%m-%d") for d in dates[1:]]

def format_currency_idr(value: float) -> str:
    """
    Mengubah angka float menjadi format mata uang Rupiah yang rapi.
    Contoh: 1500000.50 -> "Rp 1.500.000"
    """
    try:
        formatted = "{:,.0f}".format(value)
        formatted_indo = formatted.replace(",", ".")
        return f"Rp {formatted_indo}"
    except:
        return str(value)

def sanitize_numpy_output(data):
    """
    Fungsi SANGAT PENTING untuk API.
    Mengubah tipe data Numpy (int64, float32, ndarray) menjadi tipe data Python native (int, float, list).
    
    Kenapa? Karena library JSON di FastAPI tidak bisa membaca format Numpy.
    Tanpa ini, API akan error: "Object of type float32 is not JSON serializable".
    """
    if isinstance(data, dict):
        return {k: sanitize_numpy_output(v) for k, v in data.items()}

    elif isinstance(data, list):
        return [sanitize_numpy_output(v) for v in data]
    
    elif isinstance(data, np.ndarray):
        return sanitize_numpy_output(data.tolist())
    
    elif isinstance(data, (np.float32, np.float64, np.float_)):
        return float(data)
    
    elif isinstance(data, (np.int32, np.int64, np.int_)):
        return int(data)
    
    else:
        return data

# ==========================================
# TEST AREA (Jalankan file ini untuk cek)
# ==========================================

if __name__ == "__main__":
    print("--- Test 1: Generate Dates ---")
    dates = generate_future_dates(steps=5)
    print("5 Hari Kerja ke depan:", dates)

    print("\n--- Test 2: Format Rupiah ---")
    uang = 1250500.99
    print(f"Asli: {uang} -> Format: {format_currency_idr(uang)}")

    print("\n--- Test 3: Sanitize Numpy ---")
    numpy_data = {
        "prediksi" : np.array([100.5, 200.1], dtype=np.float32),
        "error" : np.int64(5)
    }
    clean_data = sanitize_numpy_output(numpy_data)
    print("Data Bersih (siap JSON):", clean_data)
    print("Tipe data item pertama:", type(clean_data['prediksi'][0]))