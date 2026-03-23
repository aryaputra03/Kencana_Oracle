import pickle
import sys
import os
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from prophet.serialize import model_from_json

# ------------------------------------------------------------------------
# 1. SETUP PATH SYSTEM
# Trik ini memastikan Python bisa menemukan folder 'config' 
# meskipun script dijalankan dari dalam folder 'src'
# ------------------------------------------------------------------------

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from config.setting import MODEL_PATHS
except ImportError:
    raise ImportError("Gagal mengimport settings. Pastikan file config/settings.py ada dan struktur folder benar.")

# ------------------------------------------------------------------------
# 2. CLASS MODEL LOADER
# ------------------------------------------------------------------------
class ModelLoader:
    """
    Class ini bertugas untuk memuat file binary model (.pkl) dari disk ke RAM.
    Menggunakan pola Singleton agar model hanya diload satu kali.
    """

    def __init__(self):
        self._loaded_models = {
            "arima" : None,
            "sarima" : None,
            "prophet" : None,
            "xgboost" : None
        }

    def get_model(self, model_type: str):
        """
        Fungsi utama untuk mengambil model.
        
        Args:
            model_type (str): Jenis model ('arima', 'sarima', 'prophet', 'xgboost')
            
        Returns:
            Object model statsmodels yang sudah dilatih.
        """
        model_type = model_type.lower()
        if model_type not in MODEL_PATHS:
            raise ValueError(f"Tipe model '{model_type}' tidak dikenal. Pilihan: {list(MODEL_PATHS.keys())}")
    
        if self._loaded_models[model_type] is not None:
            return self._loaded_models[model_type]
        
        return self._load_from_disk(model_type)
    
    def _load_from_disk(self, model_type: str):
        """
        Internal function untuk membaca file fisik model (.pkl atau .json)
        """
        path = MODEL_PATHS[model_type]
        print(f"System: Sedang memuat model {model_type.upper()} dari {path}...")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CRITICAL ERROR: File model tidak ditemukan di {path}.\n"
                f"Pastikan Anda sudah menjalankan Notebook dan menyimpan model ke folder artifacts/."
            )

        try:
            if model_type in ['arima', 'sarima']:
                with open(path, 'rb') as f:
                    model = pickle.load(f)

            elif model_type == 'prophet':
                with open(path, 'r') as f:
                    model = model_from_json(json.load(f))

            elif model_type == "xgboost":
                    model = xgb.XGBRegressor()
                    model.load_model(path)    
        
            self._loaded_models[model_type] = model
            print(f"System: Model {model_type.upper()} berhasil dimuat ke memori!")
            return model
        
        except Exception as e:
            raise RuntimeError(f"Gagal me-load file model {model_type}. Error: {str(e)}")

    def get_model_metrics(self, model_type: str):
        """
        Mengambil metrik ARIMA/SARIMA atau mengekstrak 
        Feature Importance (XGBoost) / Components (Prophet).
        """
        model = self.get_model(model_type)
        try:
            # ==========================================
            # 1. LOGIKA UNTUK XGBOOST (Feature Importance)
            # ==========================================
            if model_type == "xgboost":
                booster = model.get_booster()
                importance = booster.get_score(importance_type='weight')

                sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
                top_3_features = list(sorted_importance.keys())[:3] if importance else ['None']
                
                return {
                    "model_type": model_type,
                    "aic": 0.0, "bic": 0.0, "hqic": 0.0, "n_observations": 0, "mape": 0.0,
                    "interpretation": f"XGBoost Aktif. Top 3 Fitur Utama: {', '.join(top_3_features)}",
                    "extra_info" : sorted_importance,
                    "status" : "Active"
                }
            
            elif model_type == "prophet":
                seasonalities = list(model.seasonalities.keys())
                has_holiday = (model.holidays is not None) or (model.country_holidays is not None)

                components_status = {
                    "trend" : True,
                    "yearly" : "yearly" in seasonalities,
                    "weekly" : "weekly" in seasonalities,
                    "daily" : "daily" in seasonalities,
                    "holiday" : has_holiday
                }

                komponen_aktif = [k for k, v in components_status.items() if v]

                return {
                    "model_type" : model_type,
                    "aic": 0.0, "bic": 0.0, "hqic": 0.0, "n_observations": 0, "mape": 0.0,
                    "interpretation": f"Prophet Aktif. Komponen Aktif: {', '.join(komponen_aktif)}",
                    "extra_info" : components_status,
                    "status" : "Active"
                }
            

            aic_val = model.aic
            bic_val = model.bic
            residuals = model.resid
            actuals = model.data.endog

            with np.errstate(divide='ignore', invalid='ignore'):
                mape_val = np.mean(np.abs(residuals/actuals)) * 100

            if np.isnan(mape_val):
                mape_val = 0.0

            if mape_val < 10:
                interpret = "⭐⭐⭐ Sangat Bagus (Akurat)"
            elif mape_val < 20:
                interpret = "⭐⭐ Bagus (Layak)"
            elif mape_val < 50:
                interpret = "⭐ Cukup (Perlu Hati-hati)"
            else:
                interpret = "❌ Buruk (Tidak Akurat)"


            return {
                "model_type": model_type,
                "aic": aic_val,
                "bic": bic_val,
                "hqic": model.hqic,
                "n_observations": int(model.nobs),
                "mape": round(mape_val, 2),
                "interpretation": interpret,
                "extra_info" : None, 
                "status": "Active"
            }
        except AttributeError:
            return {
                "model_type": model_type,
                "aic": 0.0, "bic": 0.0, "hqic": 0.0, "n_observations": 0,
                "mape": 0.0,
                "interpretation": f"Error calculating metrics/info: {str(e)}",
                "extra_info": None,
                "status": "Error"
            }

# ------------------------------------------------------------------------
# 3. INSTANCE GLOBAL
# Variabel 'loader' inilah yang akan di-import oleh file lain (predictor.py)
# ------------------------------------------------------------------------
loader = ModelLoader()

if __name__ == "__main__":
    print("Testing ModelLoader...")
    try:
        model = loader.get_model("prophet")
        print("Test Sukses: Object model ditemukan:", type(model))
    except Exception as e:
        print("Test Gagal:", e)