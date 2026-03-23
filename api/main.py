from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import time
import datetime
import pandas as pd

# =================================================================
# 1. SETUP IMPORTS
# =================================================================
from src.database import engine, get_db
from src import models, crud
from api import schemas
from src.predictor import predicted_next_days
from src.loader import loader
from src import cache_manager

models.Base.metadata.create_all(bind=engine)

# =================================================================
# 2. INISIALISASI APP
# =================================================================
app = FastAPI(
    title="Gold Price Forecasting API V2",
    description='API untuk memantau harga emas historis dan melakukan prediksi masa depan menggunakan AI (ARIMA/SARIMA/Prophet/XGBoost) dengan Redis Cache.',
    version='2.1.0',
    docs_url="/docs",
    redoc_url="/redoc"
)

# =================================================================
# 3. API ENDPOINTS (ROUTES)
# =================================================================
@app.get("/", tags=["General"])
def read_root():
    """
    Cek status server (Health Check).
    """
    return {
        "status" : "active",
        "message" : "Welcome to Gold Price API V2. Visit /docs for documentation."
    }

# --- BAGIAN DATA HISTORIS (CRUD BASIC) ---

@app.get("/prices", response_model=List[schemas.GoldPriceResponse], tags=["Gold Prices"])
def read_price(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Mendapatkan daftar harga emas historis.
    
    - **skip**: Jumlah data yang dilewati (untuk pagination).
    - **limit**: Jumlah maksimal data yang diambil (default 100).
    """
    prices = crud.get_gold_price(db, skip=skip, limit=limit)
    return prices

@app.get("/prices/latest", response_model=schemas.GoldPriceResponse, tags=['Gold Prices'])
def read_latest_price(db: Session = Depends(get_db)):
    """
    Mendapatkan 1 data harga emas paling aktual (terbaru).
    """
    prices = crud.get_prices(db, skip=0, limit=1, sort_order='dsc')

    if not prices:
        raise HTTPException(status_code=404, detail="Belum ada data harga emas di database.")
    
    return prices[0]

@app.get("/prices/{target_date}", response_model=schemas.GoldPriceResponse, tags=["Gold Prices"])
def read_price_by_date(
    target_date: datetime.date,
    db: Session = Depends(get_db)
):
    """
    Mencari data harga emas berdasarkan tanggal spesifik.
    Format tanggal: YYYY-MM-DD (Contoh: 2023-08-17)
    """
    price_item = crud.get_price_by_date(db, date_val=target_date)

    if not price_item:
        raise HTTPException(
            status_code=404,
            detail=f"Data harga untuk tanggal {target_date} tidak ditemukan."
        )
    return price_item

@app.post("/prices", response_model=schemas.GoldPriceResponse, tags=["Gold Prices"])
def create_price(
    price_data: schemas.GoldPriceCreate,
    db: Session = Depends(get_db)
):
    """
    Menambahkan data harga emas baru secara manual.
    """
    existing_price = crud.get_price_by_date(db, date_val=price_data.date)
    if existing_price:
        raise HTTPException(status_code=400, detail="Data untuk tanggal ini sudah ada.")
    
    return crud.create_price_entry(db, price_data.date, price_data.price)

# --- BAGIAN AI FORECASTING (V2 UPDATE) ---

@app.get("/model-info/{model_type}", response_model=schemas.ModelMetricsResponse, tags=["AI Forecasting"])
def get_model_info(model_type: str):
    """
    Melihat statistik kualitas model (AIC, BIC, MAPE).
    
    - **model_type**: Pilih 'arima', 'sarima', 'xgboost', 'prophet'.
    """
    valid_models = ['arima', 'sarima', 'prophet', 'xgboost']
    if model_type not in valid_models:
        raise HTTPException(status_code=400, detail=f"Model harus salah satu dari: {', '.join(valid_models)}")
    
    try:
        metrics = loader.get_model_metrics(model_type)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat info model: {str(e)}")

@app.post("/predict", response_model=schemas.PredictionResponse, tags=["AI Forecasting"])
def predict_future_prices(
    request: schemas.PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Melakukan prediksi harga emas ke depan menggunakan model AI.
    Mendukung ARIMA, SARIMA, Prophet, dan XGBoost. Terintegrasi dengan Redis Cache.
    """
    cached_result = cache_manager.get_cached_prediction(request.model_type, request.steps)
    if cached_result:
        return cached_result

    start_time = time.time()
    historical_df = None

    if request.model_type == "xgboost":
        db_history = crud.get_gold_price(db, skip=0, limit=260)
        if len(db_history) < 253:
            raise HTTPException(status_code=400, detail="Data sejarah di database kurang! XGBoost butuh minimal 253 baris (kurang lebih setahun hari kerja).")
        historical_df = pd.DataFrame([{"date": item.date, "price": item.price} for item in db_history])
    
    result = predicted_next_days(
        steps=request.steps,
        model_type=request.model_type,
        historical_df=historical_df
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    
    cache_manager.set_cached_prediction(request.model_type, request.steps, result)
    
    exec_time = (time.time() - start_time)*1000
    forecast_data = result.get("data", [])

    if forecast_data:
        start_raw = forecast_data[0]['date']
        end_raw = forecast_data[-1]['date']

        start_date_pred = datetime.datetime.strptime(str(start_raw), "%Y-%m-%d").date() if isinstance(start_raw, str) else start_raw
        end_date_pred = datetime.datetime.strptime(str(end_raw), "%Y-%m-%d").date() if isinstance(end_raw, str) else end_raw
    else:
        start_date_pred = datetime.datetime.now().date()
        end_date_pred = datetime.datetime.now().date()
    
    try:
        crud.create_prediction_log(
            db=db,
            model_used=request.model_type,
            steps=request.steps,
            start_date=start_date_pred,
            end_date=end_date_pred,
            execution_time=exec_time
        )
    except Exception as e:
        print(f"[WARNING] Gagal menyimpan log prediksi: {e}")
    
    return result

# --- ENDPOINT BARU (V2) ---
@app.post("/predict/compare", tags=['AI Forecasting'])
def compare_predictions(request: schemas.CompareRequest, db: Session = Depends(get_db)):
    """
    Mengadu prediksi dari beberapa model sekaligus (misal: XGBoost vs ARIMA vs Prophet).
    Sangat berguna untuk ditampilkan di multi-line chart pada Dashboard Streamlit.
    """
    comparison_result = {}

    for model_name in request.models:
        try:
            single_request = schemas.PredictionRequest(model_type=model_name, steps=request.steps)
            result = predict_future_prices(request=single_request, db=db)
            comparison_result[model_name] = result["data"]
        except Exception as e:
            comparison_result[model_name] = {"error": str(e)}
    
    return {
        "status": "success",
        "steps": request.steps,
        "comparison": comparison_result
    }

@app.get("/predictions/history", tags=["Monitoring"])
def get_prediction_history(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Mengambil riwayat log prediksi yang pernah dilakukan oleh sistem."""
    return crud.get_prediction_log(db, skip=skip, limit=limit)
