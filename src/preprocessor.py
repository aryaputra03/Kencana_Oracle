import pandas as pd
import numpy as np

# ==============================================================================
# 1. PREPROCESSOR UNTUK XGBOOST
# ==============================================================================
def create_xgboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mereplikasi persis Feature Engineering dari notebook 'xgb.ipynb'.
    
    PENTING:
    - Dataframe HARUS memiliki minimal 253 baris data historis (karena ada lag252).
    - Jika data kurang dari 252, hasil prediksi akan mengandung NaN dan error.
    """
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date').reset_index(drop=True)

    df['price_diff'] = df['price'].diff()

    # ==========================================
    # A. TIME FEATURES (Sesuai fungsi create_features)
    # Di notebook Anda pakai df.index, di API kita pakai kolom df['date'].dt
    # ==========================================
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # ==========================================
    # B. LAG FEATURES (Sesuai fungsi add_lags)
    # Penamaan wajib sama persis: lag1, lag2, lag3, lag5, lag21, lag63, lag252
    # ==========================================
    df['lag1'] = df["price_diff"].shift(1)
    df['lag2'] = df["price_diff"].shift(2)
    df['lag3'] = df["price_diff"].shift(3)
    df['lag5'] = df["price_diff"].shift(5)
    df['lag21'] = df["price_diff"].shift(21)
    df['lag63'] = df["price_diff"].shift(63)
    df['lag252'] = df["price_diff"].shift(252)

    # ==========================================
    # C. ROLLING WINDOWS (Sesuai fungsi add_rolling_windows)
    # ==========================================
    df['rolling_mean_7'] = df["price_diff"].shift(1).rolling(window=7).mean()
    df['rolling_mean_30'] = df["price_diff"].shift(1).rolling(window=30).mean()

    df['rolling_std_7'] = df["price_diff"].shift(1).rolling(window=7).std()
    df['rolling_std_30'] = df["price_diff"].shift(1).rolling(window=30).std()

    df['rolling_max_7'] = df["price_diff"].shift(1).rolling(window=7).max()
    df['rolling_min_7'] = df["price_diff"].shift(1).rolling(window=7).min()

    df = df.dropna().reset_index(drop=True)

    return df

# ==============================================================================
# 2. PREPROCESSOR UNTUK PROPHET
# ==============================================================================
def create_prophet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menyiapkan DataFrame untuk model Prophet.
    Wajib menggunakan kolom 'ds' dan 'y'.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df['date']

    if 'price' in df.columns:
        prophet_df['y'] = df['price']

    return prophet_df 

# ==============================================================================
# TEST AREA
# ==============================================================================
if __name__ == '__main__':
    print("--- Testing Preprocessor XGBoost & Prophet ---")

    dates = pd.date_range(start="2023-01-01", periods=260)
    prices = np.random.randint(1000000, 1500000, size=260)
    dummy_df = pd.DataFrame({
        "date" : dates,
        "price" : prices
    })

    print("\n[1] Memproses Fitur XGBoost...")
    xgb_df = create_xgboost_features(dummy_df)

    print("Daftar Kolom XGBoost:")
    print(xgb_df.columns.to_list())
    print(f"Total Kolom: {len(xgb_df.columns)}")

    print("\n2 Baris terakhir (Pastikan tidak ada NaN di baris terbawah):")
    kolom_cek = ['date', 'price', 'lag1', 'lag252', 'rolling_mean_30']
    print(xgb_df[kolom_cek].tail(2))

    