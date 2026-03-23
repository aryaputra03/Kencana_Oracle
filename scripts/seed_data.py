import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# ========================================================================
# 1. SETUP PATH SYSTEM
# ========================================================================
curent_file = Path(__file__).resolve().parent
project_root = curent_file.parent
sys.path.append(str(project_root))

from src.database import SessionLocal, engine
from src.models import Base
from src import crud

def seed_gold_price(csv_path: str):
    """
    Membaca file CSV dan memasukkan data harga emas ke database.
    Menggunakan crud.create_price_entry agar aman dari duplikasi.
    """
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        file_path = Path(csv_path)
        if not file_path.exists():
            print(f"  Error: File tidak ditemukan di: {file_path}")
            print("   Pastikan Anda sudah menaruh file 'gold_price.csv' di folder 'data/'")
            return
        
        print(f"Membaca file: {file_path.name}...")
        df = pd.read_csv(file_path)

        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in df.columns if 'date' in c or'tgl' in c), None)
        price_col = next((c for c in df.columns if 'price' in c or 'close' in c), None)

        if not date_col or  not price_col:
            print(f"   Error: Tidak bisa menemukan kolom Tanggal atau Harga.")
            print(f"   Kolom yang ditemukan: {list(df.columns)}")
            return
        
        print(f"   Ditemukan {len(df)} baris data.")
        print(f"   • Kolom Tanggal: '{date_col}'")
        print(f"   • Kolom Harga  : '{price_col}'")
        print("-" * 50)

        success_count = 0
        skip_count = 0
        error_count = 0

        for index, row in df.iterrows():
            try:
                raw_date = row[date_col]
                date_val = pd.to_datetime(raw_date).date()

                raw_price = str(row[price_col])
                clean_price = raw_price.replace(',', '').replace('Rp', '').strip()

                # if clean_price.count('.') > 1:
                #     clean_price = clean_price.replace('.', '') karena saya ada titik untuk desimal

                price_val = float(clean_price)

                entry = crud.create_price_entry(db, date_val, price_val)

                success_count += 1

                if (index + 1)% 50 == 0:
                    sys.stdout.write(f"\r Memproses... {index + 1}/{len(df)}")
                    sys.stdout.flush()

            except ValueError:
                error_count += 1
            
            except Exception as e:
                print(f"\nError di baris {index}: {e}")
                error_count += 1
        
        print(f"\n\n{'='*30}")
        print(" PROSES SELESAI")
        print(f"{'='*30}")
        print(f" Total Data   : {len(df)}")
        print(f" Sukses Masuk : {success_count}")
        print(f" Gagal/Error  : {error_count}")
        print(f"{'='*30}")

    except Exception as e:
        print(f"Terjadi kesalahan fatal: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    csv_file = r"D:\AI AND DATA SCIENCE\Time Series\ts_inference_engine\data\data_RP.csv"

    seed_gold_price(str(csv_file))