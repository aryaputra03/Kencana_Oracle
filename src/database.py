import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ------------------------------------------------------------------------
# SETUP PATH (Agar bisa import config)
# ------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config.setting import DATABASE_URL

if not DATABASE_URL:
    raise ValueError("DATABASE_URL belum diset di file .env!")

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Membuka koneksi database, menyerahkannya ke request, 
    dan WAJIB menutupnya setelah request selesai (walaupun error).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================================================================
# AREA TEST KONEKSI (Jalankan file ini langsung)
# ========================================================================
if __name__ == "__main__":
    print("--- TESTING KONEKSI KE SUPABASE ---")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("Koneksi BERHASIL! Database merespon.")

            version = connection.execute(text("SELECT version()")).fetchone()
            print(f"Versi DB: {version[0]}")

    except Exception as e:
        print("Koneksi GAGAL!")
        print(f"Error: {e}")
        print("\nTips: Cek password di .env dan pastikan IP Anda tidak diblokir Supabase.")