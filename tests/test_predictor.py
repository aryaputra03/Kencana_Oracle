import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from src.database import get_db
from src.predictor import predicted_next_days

# ==========================================
# 1. SETUP & MOCKING DATABASE (FastAPI)
# ==========================================
def override_get_db():
    db_mock = MagicMock()
    yield db_mock

app.dependency_overrides[get_db]  = override_get_db

client = TestClient(app)

# ==========================================
# 2. FIXTURE UNTUK MOCK MODEL ARIMA/SARIMA
# ==========================================
@pytest.fixture
def mock_model_loader():
    """
    Fixture ini meniru output dari statsmodels (get_forecast).
    Karena di predictor.py data di-log (np.exp), kita harus memberikan nilai log.
    Misal log(1.000.000) adalah sekitar 13.8155.
    """
    with patch("src.predictor.loader.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_forecast = MagicMock()

        mock_forecast.predicted_mean = pd.Series([13.8155, 13.8200])
        mock_forecast.conf_int.return_value = pd.DataFrame([
            [13.8000, 13.8300],
            [13.8100, 13.8400]
        ])

        mock_model.get_forecast.return_value = mock_forecast
        mock_get_model.return_value = mock_model

        yield mock_get_model

# ==========================================
# 3. UNIT TEST: CORE LOGIC (src/predictor.py)
# ==========================================
def test_predicted_next_days_success(mock_model_loader):
    """
    Skenario: Memastikan fungsi `predicted_next_days` memproses perhitungan logaritmik,
    inverse transform (np.exp), dan format output dictionary dengan benar.
    """
    result = predicted_next_days(steps=2, model_type='arima')

    assert result["status"] == "success"
    assert result["model_used"] == "arima"
    assert result["total_steps"] == 2
    assert 'data' in result

    data = result['data']
    assert len(data) == 2

    assert data[0]['price'] > 900000
    assert data[0]['lower_bound'] < data[0]["price"]
    assert data[0]['upper_bound'] > data[0]["price"]

def test_predicted_next_days_error():
    """
    Skenario: Mengetes mekanisme try-except di fungsi predictor.
    Jika terjadi error (misal model tidak ditemukan), ia harus mengembalikan dictionary error.
    """
    with patch("src.predictor.loader.get_model", side_effect=Exception("Model rusak")):
        result = predicted_next_days(steps=2, model_type="arima")

        assert result["status"] == "error"
        assert "Model rusak" in result["message"]

# ==========================================
# 4. INTEGRATION TEST: API ENDPOINT (/predict)
# ==========================================
def test_api_predict_success(mock_model_loader):
    """
    Skenario: User menembak endpoint POST /predict.
    Ekspektasi: Skenario sukses, respons 200, format sesuai schemas.PredictionResponse.
    """
    payload = {
        "steps": 2,
        "model_type": "sarima"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["model_used"] == "sarima"
    assert data["total_steps"] == 2
    assert "confidence_level" in data

    assert isinstance(data["data"], list)
    assert len(data['data']) == 2
    assert "date" in data["data"][0]
    assert "price" in data["data"][0]
    assert "lower_bound" in data["data"][0]
    assert "upper_bound" in data["data"][0]

def test_api_predict_invalid_model():
    """
    Skenario: Request menggunakan model_type yang tidak diizinkan oleh Literal/Pydantic di schemas.py.
    Ekspektasi: Pydantic menolak, mengembalikan error 422 Unprocessable Entity.
    """
    payload = {
        "steps": 5,
        "model_type": "lstm"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_api_predict_internal_error():
    """
    Skenario: Jika `predicted_next_days` gagal, API harus mengembalikan status 500.
    """
    with patch("api.main.predicted_next_days") as mock_core:
        mock_core.return_value = {
            "status" : "error",
            "message": "Gagal total"
        }

        payload = {
            "steps" : 5,
            "model_type": "arima"
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 500
        assert response.json()["detail"] == "Gagal total"