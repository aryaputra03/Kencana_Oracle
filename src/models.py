import sys
from pathlib import Path

# ------------------------------------------------------------------------
# 1. SETUP PATH
# ------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.append(str(project_dir))

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, func
from src.database import Base, engine

# =========================================================
# TABEL 1: DATA HISTORIS EMAS
# Menyimpan data harga emas harian (jika Anda ingin menyimpan history di DB)
# =========================================================

class GoldPrice(Base):
    __tablename__ = "gold_price"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    price = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<GoldPrice(date={self.date}, price={self.price})>"
    
# =========================================================
# TABEL 2: LOG PREDIKSI (HISTORY FORECAST)
# Menyimpan hasil prediksi yang pernah diminta user.
# Berguna untuk analisis: "Seberapa sering user minta prediksi?"
# =========================================================

class PredictionLog(Base):
    __tablename__ = "prediction_gold_logs"

    id = Column(Integer,primary_key=True, index=True)
    request_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_used = Column(String, index=True)

    start_date = Column(Date)
    end_date = Column(Date)
    steps = Column(Integer)

    execution_time_ms = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, model={self.model_used}, steps={self.steps})>"
    
# =========================================================
# SCRIPT PEMBUAT TABEL (MIGRATION)
# =========================================================
# Bagian ini akan membuat tabel otomatis di Supabase
# Jika file ini dijalankan langsung.

if __name__ == "__main__":
    print("--- SEDANG MEMBUAT TABEL DI DATABASE... ---")
    try:
        Base.metadata.create_all(bind=engine)
        print("SUKSES! Tabel 'gold_prices' dan 'prediction_logs' berhasil dibuat.")
        print("Silakan cek dashboard Supabase Anda (Table Editor).")
    except Exception as e:
        print("GAGAL MEMBUAT TABEL.")
        print(f"Error: {e}")