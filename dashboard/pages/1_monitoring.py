import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Monitoring Prediksi",
    page_icon="🛡️",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

# ==========================================
# 2. FUNGSI HELPER (Ambil Data Log)
# ==========================================
# Menggunakan cache dengan batas waktu (TTL) 10 detik agar tidak membebani server
# Tapi data tetap cukup real-time
@st.cache_data(ttl=10)
def fetch_prediction_logs(limit: int = 100):
    """Mengambil riwayat log prediksi dari API Backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/history?limit={limit}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: Gagal mengambil data log.")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Gagal terhubung ke API. Pastikan server FastAPI (main.py) sudah berjalan!") 
        return None

# ==========================================
# 3. HEADER & KONTROL UI
# ==========================================
st.title("Dashboard Monitoring & Log Prediksi")   
st.write("Halaman administrator untuk memantau aktivitas sistem, riwayat penggunaan model AI, dan performa komputasi.")

col_limit, col_btn, _ = st.columns([2,2,6])
with col_limit:
    limit_data = st.selectbox("Tampilkan jumlah data terakhir:", [50, 100, 200, 500], index=1)
with col_btn:
    st.write("")
    st.write("")
    if st.button("Refresh Data Manual"):
        fetch_prediction_logs.clear()

st.markdown("---")

# ==========================================
# 4. PENGAMBILAN & PEMROSESAN DATA
# ==========================================
with st.spinner("Memuat data log dari database..."):
    raw_logs = fetch_prediction_logs(limit=limit_data)

if raw_logs:
    df = pd.DataFrame(raw_logs)

    if not df.empty:
        df['request_timestamp'] = pd.to_datetime(df['request_timestamp'])
        df['Waktu Request'] = df['request_timestamp'].dt.strftime("%d %b %Y %H:%M:%S")
        df['Model'] = df['model_used'].str.upper()
        df['Jml Hari'] = df['steps']
        df['Tgl Mulai'] = pd.to_datetime(df['start_date']).dt.strftime("%d-%m-%Y")
        df['Tgl Akhir'] = pd.to_datetime(df['end_date']).dt.strftime("%d-%m-%Y")

        df['execution_time_ms'] = df['execution_time_ms'].fillna(0)
        df['Waktu Eksekusi (ms)'] = df['execution_time_ms'].apply(lambda x: f"{x:.2f}")

        # ==========================================
        # 5. METRIK KPI (Key Performance Indicators)
        # ==========================================
        total_requests = len(df)
        avg_execution_time = df['execution_time_ms'].mean()
        most_used_model = df['Model'].mode()[0] if not df['Model'].empty else "N/A"

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Request (Sesuai Limit)", total_requests)
        m2.metric("Rata-rata Waktu Eksekusi AI", f"{avg_execution_time:.2f} ms")
        m3.metric("Model Paling Sering Dipakai", most_used_model)

        st.markdown("---")

        # ==========================================
        # 6. VISUALISASI GRAFIK
        # ==========================================
        st.subheader("Distribusi Penggunaan Model")

        _, col_pie, _ = st.columns([1, 2, 1])
        
        with col_pie:
            model_count = df['Model'].value_counts().reset_index()
            model_count.columns = ['Model', 'Jumlah']

            fig_pie = px.pie(
                model_count,
                names='Model',
                values='Jumlah',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")

        st.subheader("Performa Waktu Eksekusi (ms)")
        df_sorted = df.sort_values('request_timestamp').reset_index(drop=True)

        fig_line = px.line(
                df_sorted,
                x='request_timestamp',
                y='execution_time_ms',
                color='Model',
                markers=True,
                labels={'request_timestamp': 'Waktu Request', 'execution_time_ms': 'Waktu (ms)'}
            )
        
        fig_line.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # ==========================================
        # 7. TABEL DATA DETAIL
        # ==========================================
        st.subheader("Tabel Detail Riwayat Request")

        display_cols = [
            'id', 'Waktu Request', 'Model', 'Jml Hari', 
            'Tgl Mulai', 'Tgl Akhir', 'Waktu Eksekusi (ms)'
        ]
        
        df_display = df[display_cols].rename(columns={'id' : 'ID'})
        df_display = df_display.sort_values(by='ID', ascending=False).reset_index(drop=True)
        st.dataframe(df_display, use_container_width=True)
    
    else:
        st.info("Koneksi ke database berhasil, tetapi belum ada riwayat prediksi yang tercatat.")
else:
    pass