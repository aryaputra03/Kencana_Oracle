import sys
from pathlib import Path
from typing import Optional, List
from datetime import date as date_type

# ------------------------------------------------------------------------
# 1. SETUP PATH SYSTEM
# ------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.models import GoldPrice, PredictionLog

# ========================================================================
# BAGIAN 1: OPERASI TABEL GOLD_PRICES (DATA HISTORIS)
# ========================================================================
def get_gold_price(db: Session, skip: int = 0, limit: int = 100, chronological: bool = True) -> List[GoldPrice]:
    """
    [MODIFIKASI] Fungsi ini menggabungkan get_prices dan get_recent_price milik Anda sebelumnya.
    Sangat krusial untuk XGBoost di api/main.py.
    
    Cara kerja:
    1. Mengambil data TERBARU (descending) sebanyak 'limit'. Jika butuh 260 data untuk lag XGBoost,
       kita pastikan itu adalah 260 hari terakhir, bukan 260 hari dari tahun 2010.
    2. Jika chronological=True, hasil yang asalnya terbalik itu diputar balik (reverse) 
       agar menjadi LAMA ke BARU (Chronological). Time series ML akan error jika data tidak urut maju!
    """
    items = db.query(GoldPrice).order_by(desc(GoldPrice.date)).offset(skip).limit(limit).all()

    if chronological:
        items.reverse()

    return items

def create_price_entry(db: Session, date_val: date_type, price: float) -> GoldPrice:
    """
    Menyimpan satu data harga emas baru.
    Fungsi ini otomatis mengecek apakah tanggal tersebut sudah ada datanya.
    Jika sudah ada, dia TIDAK akan duplikat (Skip).
    """
    existing_price = db.query(GoldPrice).filter(GoldPrice.date == date_val).first()

    if existing_price:
        return existing_price
    
    new_entry = GoldPrice(date=date_val, price=price)
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry

def get_price_by_date(db: Session, date_val: date_type) -> Optional[GoldPrice]:
    """
    Mencari harga emas pada tanggal spesifik.
    """
    return db.query(GoldPrice).filter(GoldPrice.date == date_val).first()

# ========================================================================
# BAGIAN 2: OPERASI TABEL PREDICTION_LOGS (AUDIT / RIWAYAT)
# ========================================================================

def create_prediction_log(
        db: Session,
        model_used: str,
        steps: int,
        start_date: date_type,
        end_date: date_type,
        execution_time: float = 0.0
) -> PredictionLog:
    """
    Mencatat log bahwa ada user yang melakukan prediksi.
    Dipanggil otomatis oleh API setiap kali endpoint /predict diakses.
    """
    log_entry = PredictionLog(
        model_used=model_used,
        steps=steps,
        start_date=start_date,
        end_date=end_date,
        execution_time_ms=execution_time
    )

    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry

def get_prediction_log(db: Session, skip: int = 0, limit: int = 10) -> List[PredictionLog]:
    """
    [MODIFIKASI] Menambahkan parameter 'skip' untuk fitur Pagination.
    Mengambil daftar riwayat prediksi terakhir.
    Berguna untuk halaman Admin atau Monitoring di Streamlit nanti.
    """
    return db.query(PredictionLog).order_by(desc(PredictionLog.request_timestamp)).offset(skip).limit(limit).all()

# ========================================================================
# TEST AREA (Jalankan file ini langsung untuk cek fungsi)
# ========================================================================
if __name__  == "__main__":
    from src.database import SessionLocal, engine
    from src.models import Base
    from datetime import date

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    print("--- TESTING CRUD OPERATIONS ---")

    try:
        print("1. Menambah data dummy harga emas...")
        dummy_date = date(2023, 1, 1)
        item = create_price_entry(db, dummy_date, 1000000.0)
        print(f"    Sukses: ID={item.id}, Tgl={item.date}, Harga={item.price}")

        print("2. Mengambil data harga...")
        prices = get_gold_price(db, limit=5)
        print(f"   Ditemukan {len(prices)} data.")
        if len(prices) > 0:
            print(f"   Data pertama: {prices[0].date}, Data terakhir: {prices[-1].date}")

        print("3. Mencatat log prediksi...")
        log = create_prediction_log(
            db,
            model_used="xgboost",
            steps=7,
            start_date=date(2023, 1, 2), 
            end_date=date(2023, 1, 9),
            execution_time=24.5
        )
        print(f"   Sukses Log ID={log.id} disimpan pada {log.request_timestamp}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

