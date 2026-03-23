import hashlib
from typing import Optional, Any, Dict
import json
import redis

from config.setting import REDIS_URL, CACHE_TTL


# =================================================================
# SETUP UPSTASH REDIS CACHE
# =================================================================
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("[CACHE] Berhasil terhubung ke Upstash Redis!")
    else:
        redis_client = None
        print("[CACHE] REDIS_URL kosong. Fitur caching dinonaktifkan.")
except Exception as e:
    redis_client = None
    print(f"[CACHE] Gagal terhubung ke Upstash Redis: {e}. Fitur caching dinonaktifkan.")

def generate_cache_key(model_type: str, steps: int) -> str:
    """
    Membuat 'KTP' (Kunci unik) untuk setiap request.
    Contoh: Jika user minta XGBoost 30 hari, key-nya gabungan kata tersebut.
    Kita gunakan Hash MD5 agar seragam dan rapi.
    """
    raw_key = f"{model_type}_{steps}"
    hashed_key = hashlib.md5(raw_key.encode()).hexdigest()
    return f"gold_predict:{hashed_key}"

def get_cached_prediction(model_type: str, steps: int) -> Optional[Dict[str, Any]]:
    """
    Mengambil data dari Upstash Redis.
    """
    if not redis_client:
        return None
    
    key = generate_cache_key(model_type, steps)

    try:
        cached_data = redis_client.get(key)
        if cached_data:
            print(f"[CACHE HIT] Mengambil prediksi {model_type} ({steps} hari) dari Upstash Redis!") 
            return json.loads(cached_data)
    except Exception as e:
        print(f"[CACHE ERROR] Gagal membaca dari Redis: {e}")
    
    print(f"[CACHE MISS] Belum ada cache untuk {model_type} ({steps} hari). Harus hitung ML...")
    return None

def set_cached_prediction(model_type: str, steps: int, prediction_data: Dict[str, Any]) -> None:
    """
    Menyimpan hasil ke Upstash Redis dengan batas waktu (TTL) dari setting.py.
    """

    if not redis_client:
        return
    key = generate_cache_key(model_type, steps)
    
    try:
        json_data = json.dumps(prediction_data)
        redis_client.set(key, json_data, ex=CACHE_TTL)
        print(f"[CACHE SAVED] Hasil prediksi {model_type} ({steps} hari) disimpan ke Redis selama {CACHE_TTL} detik.")
    except Exception as e:
        print(f"[CACHE ERROR] Gagal menyimpan ke Redis: {e}")

def clear_prediction_cache() -> None:
    """
    Menghapus semua ingatan cache (Berguna jika kita baru saja melatih ulang/update model).
    """
    if not redis_client:
        return
    
    try:
        keys_to_delete = redis_client.keys("gold_predict:*")
        if keys_to_delete:
            redis_client.delete(*keys_to_delete)
        print("[CACHE CLEARED] Semua memori prediksi di Redis telah dihapus.")
    except Exception as e:
        print(f"[CACHE ERROR] Gagal menghapus cache di Redis: {e}")

