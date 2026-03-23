import sys
import numpy as  np
from pathlib import Path
import warnings
import pandas as pd

# ------------------------------------------------------------------------
# 1. SETUP PATH (Agar bisa dijalankan langsung untuk test)
# ------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config.setting import DEFAULT_FORECAST_STEPS, CONFIDENCE_LEVEL, ALPHA
from src.loader import loader
from src.utils import generate_future_dates, sanitize_numpy_output
from src.preprocessor import create_xgboost_features

warnings.filterwarnings("ignore")

def predicted_next_days(steps: int = DEFAULT_FORECAST_STEPS, model_type: str = "sarima", historical_df: pd.DataFrame = None):
    """
    Fungsi Utama Forecasting (Router untuk semua model).
    """
    try:
        model = loader.get_model(model_type)
        

        future_dates = generate_future_dates(steps=steps, freq='B')
        prediction_data = []

        # ====================================================================
        # LOGIKA 1: ARIMA & SARIMA [TETAP]
        # (Menggunakan skala Logaritma lalu di-Inverse dengan np.exp)
        # ====================================================================
        if model_type in ["arima", "sarima"]:
            forecast_result = model.get_forecast(steps=steps)
            predicted_log = forecast_result.predicted_mean
            conf_int_log = forecast_result.conf_int(alpha=ALPHA)

            forecast_values = np.exp(predicted_log)
            lower_bound = np.exp(conf_int_log.iloc[:, 0])
            upper_bound = np.exp(conf_int_log.iloc[:, 1])


            vals_list = forecast_values.tolist()
            low_list = lower_bound.tolist()
            up_list = upper_bound.tolist()

            for i in range(len(future_dates)):
                prediction_data.append({
                    "date" : future_dates[i],
                    "price" : vals_list[i],
                    "lower_bound" : low_list[i],
                    "upper_bound" : up_list[i]
                })

        # ====================================================================
        # LOGIKA 2: PROPHET [BARU]
        # ====================================================================
        elif model_type == "prophet":
            future_df = pd.DataFrame({
                'ds' : future_dates
            })
            forecast = model.predict(future_df)

            for i in range(len(future_dates)):
                prediction_data.append({
                    "date" : future_dates[i],
                    "price" : float(forecast['yhat'].iloc[i]),
                    "lower_bound" : float(forecast['yhat_lower'].iloc[i]),
                    "upper_bound" : float(forecast['yhat_upper'].iloc[i])
                })
        
        # ====================================================================
        # LOGIKA 3: XGBOOST [BARU]
        # (Auto-Regressive Iterative Forecasting)
        # ====================================================================
        elif model_type == "xgboost":
            if historical_df is None or len(historical_df) < 253:
                raise ValueError("XGBoost membutuhkan parameter 'historical_df' yang berisi minimal 253 baris data harga emas terakhir!")
            
            current_df = historical_df[['date', 'price']].copy()
            current_df['date'] = pd.to_datetime(current_df['date'])

            for i in range(steps):
                next_date = pd.to_datetime(future_dates[i])

                new_row = pd.DataFrame({'date': [next_date],
                                        'price': [np.nan]})
                current_df = pd.concat([current_df, new_row], ignore_index=True)

                features_df = create_xgboost_features(current_df)
                last_row = features_df.iloc[[-1]].copy()

                x_pred = last_row.drop(columns=['date', 'price', 'price_diff'], errors='ignore')

                pred_diff = float(model.predict(x_pred)[0])

                previous_price = current_df.iloc[-2]['price']
                pred_price = previous_price + pred_diff

                current_df.iloc[-1, current_df.columns.get_loc('price')] = pred_price

                margin = pred_price * 0.015

                prediction_data.append({
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price' : pred_price,
                    'lower_bound' : pred_price - margin,
                    'upper_bound' : pred_price + margin
                })

        # ====================================================================
        # BUNGKUS HASIL (RESPONSE) [TETAP]
        # ====================================================================


        response = {
            "status": "success",
            "model_used": model_type,
            "total_steps": steps,
            "confidence_level": CONFIDENCE_LEVEL,
            "data": prediction_data
        }

        return sanitize_numpy_output(response)
    
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Gagal melakukan prediksi dengan model {model_type}. Error: {str(e)}"
        }

    

# ========================================================================
# AREA TEST (Jalankan file ini langsung untuk cek apakah prediksi keluar)
# ========================================================================

if __name__ == "__main__":
    print("--- MULAI TEST PREDIKSI ---")
    
    dates = pd.date_range(start="2023-01-01", periods=260)
    prices = np.random.randint(1000000, 1500000, size=260)
    dummy_history = pd.DataFrame({
        "date" : dates,
        "price" : prices
    })

    models_to_test = ["sarima", "prophet", "xgboost"]

    for m_type in models_to_test:
        print(f"\nTesting Model: {m_type.upper()}")

        if m_type == "xgboost":
            hasil = predicted_next_days(steps=10, model_type=m_type, historical_df=dummy_history)
        else:
            hasil = predicted_next_days(steps=10, model_type=m_type)

        if hasil['status'] == "success":
            print("-" * 60)
            print(f"{'Tanggal':<15} | {'Prediksi (Rp)':<20} | {'Range (Lower - Upper)'}")
            print("-" * 60)
            for row in hasil['data']:
                tgl = str(row['date'])[:10]
                harga = f"Rp {row['price']:,.0f}"
                range_harga = f"Rp {row['lower_bound']:,.0f} - {row['upper_bound']:,.0f}"
                print(f"{tgl:<15} | {harga:<20} | {range_harga}")
            print("-" * 60)
        else:
            print("Test Gagal:", hasil['message'])