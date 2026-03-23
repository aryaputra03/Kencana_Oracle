from pydantic import BaseModel, Field, ConfigDict
from datetime import date
from typing import Optional, List, Literal, Dict, Any

# =================================================================
# 1. SCHEMAS HARGA EMAS (GOLD PRICE)
# Digunakan untuk validasi data dari Database ke API (Response)
# =================================================================
class GoldPriceBase(BaseModel):
    """
    Schema dasar. Field ini wajib ada saat Input maupun Output.
    """
    date: date
    price: float = Field(..., gt=0, description="Harga emas (harus lebih besar dari 0)")

class GoldPriceCreate(GoldPriceBase):
    """
    Schema khusus saat user ingin INPUT data baru (POST).
    Mewarisi semua field dari GoldPriceBase.
    """
    pass 

class GoldPriceResponse(GoldPriceBase):
    """
    Schema khusus saat API memberikan OUTPUT (GET).
    Kita tambahkan 'id' karena database otomatis generate ID.
    """
    id: int

    model_config = ConfigDict(from_attributes=True)

# =================================================================
# 2. SCHEMAS PREDIKSI (FORECASTING)
# Digunakan nanti saat kita membuat endpoint prediksi AI
# =================================================================
class PredictionRequest(BaseModel):
    """
    Apa yang dikirim user saat minta prediksi?
    Contoh JSON Request:
    {
        "steps": 30,
        "model_type": "arima"
    }
    """
    steps: int = Field(
        30,
        ge=1,
        le=365,
        description="Jumlah hari ke depan yang ingin diprediksi (1 s/d 365 hari)"
    )
    model_type: Literal['arima', 'sarima', 'prophet', 'xgboost'] = Field(
        default="xgboost",
        description="Pilih jenis model AI: 'arima', 'sarima', 'prophet', atau 'xgboost'"
    )

class PredictionItem(BaseModel):
    """
    Schema untuk setiap butir data harian.
    Sekarang menampung lower_bound & upper_bound.
    """
    date: date
    price: float
    lower_bound: float
    upper_bound: float

class PredictionResponse(BaseModel):
    """
    Format jawaban API setelah selesai memprediksi.
    """
    status: str
    model_used: str
    total_steps: int
    confidence_level: float
    data: List[PredictionItem]

    model_config = ConfigDict(from_attributes=True)

# =================================================================
# 3. SCHEMAS INFO MODEL (METRICS)
# =================================================================
class ModelMetricsResponse(BaseModel):
    """
    Schema untuk menampilkan kualitas statistik model.
    """
    model_type: str
    aic: Optional[float] = Field(None, description="Akaike Information Criterion (Semakin kecil semakin bagus)")
    bic: Optional[float] = Field(None, description="Bayesian Information Criterion (Semakin kecil semakin bagus)")
    hqic: Optional[float] = Field(None,description="Hannan-Quinn Information Criterion")

    mape: float = Field(...,description="Persentase Error Rata-rata (0-100%). Semakin kecil semakin bagus.")
    interpretation: str = Field(...,description="Penjelasan manusiawi (Sangat Bagus/Buruk)")
    n_observations: int = Field(...,description="Jumlah data historis yang digunakan untuk melatih model")
    status: str = "Loaded"

    extra_info: Optional[Dict[str, Any]] = Field(None, description="Informasi tambahan spesifik model")

# =================================================================
# 4. SCHEMAS KOMPARASI (COMPARE PREDICTIONS) [BARU]
# Digunakan untuk endpoint adu mekanik antar model AI
# =================================================================
class CompareRequest(BaseModel):
    """
    Apa yang dikirim user saat minta komparasi prediksi?
    Contoh JSON Request:
    {
        "steps": 30,
        "models": ["xgboost", "prophet", "arima"]
    }
    """
    steps: int = Field(
        30,
        ge=1,
        le=365,
        description="Jumlah hari ke depan yang ingin diprediksi (1 s/d 365 hari)"
    )
    models: List[Literal['arima', 'sarima', 'prophet', 'xgboost']] = Field(
        default=["xgboost", "prophet"],
        min_length=2,
        description="Daftar model AI yang ingin dibandingkan (minimal 2 model)"
    )

class CompareResponse(BaseModel):
    """
    Format jawaban API setelah selesai mengadu model.
    Mengembalikan dictionary di mana key-nya adalah nama model, 
    dan value-nya adalah list hasil prediksi dari model tersebut (atau pesan error).
    """
    status: str
    steps: int
    comparison: Dict[str, Any]