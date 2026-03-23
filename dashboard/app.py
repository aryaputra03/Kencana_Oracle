import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. KONFIGURASI UMUM
# ==========================================
st.set_page_config(
    page_title="Gold Price Forecasting V2",
    page_icon="🪙",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 2. FUNGSI HELPER (KOMUNIKASI KE API)
# ==========================================
def fetch_api(endpoint: str):
    """Fungsi untuk melakukan HTTP GET ke API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Gagal terhubung ke server API. Pastikan FastAPI (main.py) sudah berjalan!")
        return None
    
def post_api(endpoint: str, payload: dict):
    """Fungsi untuk melakukan HTTP POST ke API."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Gagal terhubung ke server API. Pastikan FastAPI (main.py) sudah berjalan!")
        return None
    
# ==========================================
# 3. KUMPULAN FUNGSI HALAMAN (HALAMAN LOKAL)
# ==========================================
# st.sidebar.title("Navigation")
# menu = st.sidebar.radio(
#     "Pilih Menu:",
#     ["Data Historis Emas", 
#      "Cari & Kelola Data",
#      "Prediksi Harga AI", 
#      "Cek Kualitas Model",
#      ]
# )

# ==========================================
# 4. HALAMAN: DATA HISTORIS
# ==========================================
def halaman_historis():
    st.title("Data Historis Harga Emas")
    st.write("Melihat data harga emas yang tersimpan di dalam database.")

    data = fetch_api("/prices?limit=100000")
    
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        min_date_db = df['date'].min().date()
        max_date_db = df['date'].max().date()

        st.markdown("---")
        st.subheader("Filter Rentang Waktu")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Pilih Tanggal Awal:",
                value=min_date_db,
                min_value=min_date_db,
                max_value=max_date_db
            )

        with col2:
            end_date = st.date_input(
                "Pilih Tanggal Akhir:",
                value=max_date_db,
                min_value=min_date_db,
                max_value=max_date_db
            )

        if start_date > end_date:
            st.error("Tanggal Awal tidak boleh lebih besar dari Tanggal Akhir. Silakan perbaiki pilihan Anda.")

        else:
            mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
            df_filtered = df.loc[mask]

            if st.button("Tampilkan Grafik", type="primary"):
                if not df_filtered.empty:
                    with st.spinner("Merangkai visualisasi..."):
                        latest_price = df_filtered.iloc[-1]
                        st.metric(
                            label=f"Harga Terakhir (Per {latest_price['date'].strftime('%d %b %Y')})",
                            value=f"Rp {latest_price['price']:,.0f}"
                        )

                        st.subheader(f"Pergerakan Harga ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')})")
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df_filtered['date'], 
                            y=df_filtered['price'], 
                            mode='lines', 
                            name='Harga Aktual', 
                            line=dict(color='goldenrod')))
                        
                        fig.update_layout(xaxis_title="Tanggal", yaxis_title="Harga (Rp)", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Lihat Tabel"):
                            st.dataframe(df_filtered.sort_values('date', ascending=False).reset_index(drop=True))
                else:
                    st.warning("Tidak ada data emas yang ditemukan pada rentang tanggal tersebut.")
    else:
        st.error("Gagal terhubung ke database. Pastikan FastAPI sudah berjalan!")


# ==========================================
# 5. HALAMAN: KELOLA DATA (Pencarian & Input)
# ==========================================
def halaman_kelola_data():
    st.title("Pencarian & Kelola Data Emas")
    st.write("Cari harga emas di masa lalu atau tambahkan data harga terbaru secara manual.")

    all_data = fetch_api("/prices?limit=100000")

    min_date_db = None
    max_date_db = None

    if all_data:
        df_all = pd.DataFrame(all_data)
        if not df_all.empty:
            df_all['date'] = pd.to_datetime(df_all['date'])
            min_date_db = df_all['date'].min().date()
            max_date_db = df_all['date'].max().date()

    tab1, tab2 = st.tabs(["Cari Berdasarkan Tanggal", "Tambah Data Manual"])

    with tab1:
        st.subheader("Cari Harga Historis")

        if min_date_db and max_date_db:
            st.info(f"**Catatan:** Rentang data yang tersedia di database adalah dari **{min_date_db.strftime('%d-%m-%Y')}** sampai **{max_date_db.strftime('%d-%m-%Y')}**.")

            search_date = st.date_input("Pilih Tanggal yang ingin dicari:",
                                        value=max_date_db,
                                        min_value=min_date_db,
                                        max_value=max_date_db)

        if st.button("Cari Harga", key='btn_cari'):
            with st.spinner("Mencari data ke database..."):
                result = fetch_api(f"/prices/{search_date.strftime('%Y-%m-%d')}")
                if result:
                    st.success("Data berhasil ditemukan!")
                    st.metric(label=f"Harga Emas pada{result['date']}", value=f"Rp {result['price']:,.0f}")
                else:
                    st.warning(f"Data tidak ditemukan. Mungkin tanggal tersebut adalah hari libur (pasar tutup).")
        else:
            st.warning("Silahkan tekan button 'Cari Harga' di atas untuk menampilkan data harga emas berdasarkan waktu")
    
    with tab2:
        st.subheader("Input Data Harga Baru")
        st.info("Fitur ini akan menyimpan data langsung ke dalam Database (Supabase) secara permanen.")

        new_date = st.date_input("Tanggal:", key="input_date")
        new_price = st.number_input("Harga Emas (Rp):", min_value=0.0, step=1000.0, format="%.2f")

        if st.button("Simpan ke Database", type="primary"):
            with st.spinner("Menyimpan data..."):
                payload = {
                    "date": new_date.strftime("%Y-%m-%d"),
                    "price": float(new_price)
                }

                save_result = post_api("/prices", payload)

                if save_result:
                    st.success(f"Berhasil! Data tanggal {save_result['date']} dengan harga Rp {save_result['price']:,.0f} telah ditambahkan ke database.")


# ==========================================
# 6. HALAMAN: PREDIKSI HARGA AI
# ==========================================
def halaman_prediksi_ai():
    st.title("Mesin Peramal Harga Emas")
    st.write("Gunakan kecerdasan buatan untuk melihat estimasi harga di masa depan.")

    predict_mode = st.radio("Pilih Mode:", ["Single Model (Detail)", "Compare Models (Adu Cepat)"], horizontal=True)
    st.markdown("---")

    model_option = ["arima", "sarima", "prophet", "xgboost"]

    col1, col2 = st.columns(2)
    with col1:
        if predict_mode == "Single Model (Detail)":
            selected_model = st.selectbox("Pilih Model AI:", model_option, index=0)
        else:
            selected_model = st.multiselect("Pilih Model untuk Diadu:", model_option, default=["arima", "xgboost"])

    with col2:
        steps_ahead = st.slider("Prediksi untuk beberapa hari ke depan?", min_value=1, max_value=365, value=30)
    
    if st.button("Jalankan Prediksi", type="primary"):
        with st.spinner("AI sedang berpikir (Mengecek Redis Cache & Menghitung)..."):
            if predict_mode == "Single Model (Detail)":
                payload = {"steps": steps_ahead, "model_type": selected_model}
                result = post_api("/predict", payload)
                
                if result and result.get("status") == "success":
                    st.success(f"Prediksi berhasil! Diambil dari {'Cache' if result.get('cached') else 'Perhitungan Model'}.")

                    df_pred = pd.DataFrame(result["data"])
                    df_pred['date'] = pd.to_datetime(df_pred['date'])

                    y_col = 'price'

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_pred['date'], y=df_pred[y_col],
                        mode='lines+markers', name=f'Prediksi {selected_model.upper()}', line=dict(color='blue')
                ))

                if 'upper_bound' in df_pred.columns and 'lower_bound' in df_pred.columns:
                    fig.add_trace(go.Scatter(
                        x=df_pred['date'].tolist() + df_pred['date'].tolist()[::-1],
                        y=df_pred['upper_bound'].tolist() + df_pred['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0, 0, 255, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Rentang Toleransi (Batas Atas & Bawah)'
                ))
                    
                fig.update_layout(title=f"Hasil Forecasting ({selected_model.upper()})", xaxis_title="Tanggal", yaxis_title="Harga (Rp)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                st.write("**Detail Angka Prediksi:**")
                st.dataframe(df_pred)

            else:
                if len (selected_model) < 2:
                    st.warning("Pilih minimal 2 model untuk dibandingkan!")
                else:
                    payload = {"steps" : steps_ahead, "models": selected_model}
                    result = post_api("/predict/compare", payload)

                    if result and result.get("status") == "success":
                        st.success("Pertarungan Model Selesai!")
                        compare_data = result["comparison"]

                        fig_compare = go.Figure()
                        colors = ['red', 'green', 'orange', 'purple', 'blue', 'cyan']

                        st.markdown("###  Hasil Prediksi Individu per Model")
                        valid_models_count = 0

                        for idx, (m_name, m_data) in enumerate(compare_data.items()):
                            if isinstance(m_data, dict) and "error" in m_data:
                                st.error(f"Error pada {m_name.upper()}: {m_data['error']}")
                                continue
                            
                            valid_models_count += 1
                            df_pred = pd.DataFrame(m_data)
                            df_pred['date'] = pd.to_datetime(df_pred['date'])
                            # y_col = 'price'

                            fig_ind = go.Figure()
                            
                            fig_ind.add_trace(
                                go.Scatter(
                                    x=df_pred['date'], y=df_pred['price'],
                                    mode='lines+markers', name=f'Prediksi {m_name.upper()}',
                                    line=dict(color=colors[idx % len(colors)])
                                )
                            )

                            if 'upper_bound' in df_pred.columns and 'lower_bound' in df_pred.columns:
                                fig_ind.add_trace(
                                    go.Scatter(
                                        x=df_pred['date'].tolist() + df_pred['date'].tolist()[::-1],
                                        y=df_pred['upper_bound'].tolist() + df_pred['lower_bound'].tolist()[::-1],
                                        fill='toself',
                                        fillcolor='rgba(128, 128, 128, 0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        name='Rentang Toleransi',
                                        showlegend=False
                                    )
                                )

                            fig_ind.update_layout(
                                title=f"Prediksi Spesifik: {m_name.upper()}",
                                xaxis_title="Tanggal",
                                yaxis_title="Harga (Rp)",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_ind, use_container_width=True)

                            fig_compare.add_trace(go.Scatter(
                                x=df_pred['date'], y=df_pred['price'],
                                mode='lines', name=f'{m_name.upper()}',
                                line=dict(color=colors[idx % len(colors)], dash='dash')
                            ))

                            last_date = df_pred['date'].iloc[-1]
                            last_price = float(df_pred['price'].iloc[-1])
                        
                            offset_list = [
                                {"ax": 60, "ay": -50},  
                                {"ax": 60, "ay": 50},   
                                {"ax": 130, "ay": -20}, 
                                {"ax": 130, "ay": 20},  
                                {"ax": 200, "ay": -40}, 
                                {"ax": 200, "ay": 40}   
                            ]
                            
                            current_offset = offset_list[idx % len(offset_list)]

                            fig_compare.add_annotation(
                                x=last_date,
                                y=last_price,
                                text=f"<b>{m_name.upper()}</b><br>Rp {last_price:,.0f}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowcolor=colors[idx % len(colors)],
                                ax=current_offset["ax"], 
                                ay=current_offset["ay"],
                                font=dict(size=11, color=colors[idx % len(colors)]),
                                bgcolor="rgba(255, 255, 255, 0.95)", 
                                bordercolor=colors[idx % len(colors)],
                                borderwidth=1,
                                borderpad=4
                            )

                        if valid_models_count > 0:
                            st.markdown("---")
                            st.markdown("###  Grafik Kombinasi (Adu Cepat)")
                            fig_compare.update_layout(
                                    title="Perbandingan Tren Semua Model Terpilih",
                                    xaxis_title="Tanggal", 
                                    yaxis_title="Harga (Rp)", 
                                    template="plotly_white",
                                    margin=dict(r=250)
                                )

                            st.plotly_chart(fig_compare, use_container_width=True)

# ==========================================
# 7. HALAMAN: CEK KUALITAS MODEL
# ==========================================
def halaman_kualitas_model():
    st.title("Spesifikasi & Kualitas Model AI")
    st.write("Di sini Anda bisa melihat metrik evaluasi model (Nilai Rapor AI) untuk memastikan kelayakannya.")

    info_model = st.selectbox("Pilih Model untuk dicek:", ["arima", "sarima", "prophet", "xgboost"])

    if st.button("Lihat Rapor Model", type="primary"):
        with st.spinner("Menarik data evaluasi..."):
            # ==============================================================
            # 1. HARDCODE KHUSUS XGBOOST
            # ==============================================================
            if info_model == "xgboost":
                st.markdown("### Model: XGBOOST")
                st.info("**Catatan:** Metrik ini dipublish pada tanggal **20 Maret 2026**.")

                st.metric(
                    label='Tingkat Error Rata-rata (MAPE)',
                    value="0.48%",
                    delta="-Semakin Kecil Semakin Bagus",
                    delta_color="inverse"
                )
                st.success("Status Model: **⭐⭐⭐ Sangat Bagus (Sangat Akurat)**")

                with st.expander("Lihat Detail Rapor & Fitur (Untuk Data Scientist)"):
                    st.write("Metrik Tambahan (Semakin kecil semakin baik):")
                    c1, c2 = st.columns(2)

                    c1.metric("MAE (Mean Absolute Error)", "Rp 158.75")
                    c2.metric("RMSE (Root Mean Square Error)", "Rp 217.95")

                    st.markdown("---")
                    st.write("**XGBoost ditest menggunakan kombinasi fitur berikut (Differencing):**")
                    st.code("dayofweek, quarter, month, year, dayofyear, dayofmonth, weekofyear, lag1, lag2, lag3, lag5, lag21, lag63, lag252, rolling_mean_7, rolling_mean_30, rolling_std_7, rolling_std_30, rolling_max_7, rolling_min_7", language="text")

            # ==============================================================
            # 2. HARDCODE KHUSUS PROPHET
            # ==============================================================
            elif info_model == "prophet":
                st.markdown("### Model: PROPHET")
                st.info("**Catatan:** Metrik ini dipublish pada tanggal **20 Maret 2026**.")

                st.metric(
                    label="Tingkat Error Rata-rata (MAPE)",
                    value="3.96%",
                    delta="-Semakin Kecil Semakin Bagus",
                    delta_color="inverse"
                )
                st.success("Status Model: **⭐⭐ Bagus (Cukup Akurat)**")

                with st.expander("Lihat Detail Rapor (Untuk Data Scientist)"):
                    st.write("Metrik Tambahan (Semakin kecil semakin baik):")
                    c1, c2 = st.columns(2)
                    c1.metric("MAE (Mean Absolute Error)", "Rp 1,201.87")
                    c2.metric("RMSE (Root Mean Square Error)", "Rp 1,498.93")

            else:
                info = fetch_api(f"/model-info/{info_model}")

                st.markdown(f"### Model: {info['model_type'].upper()}")
                st.metric(
                    label="Tingkat Error Rata-rata (MAPE)",
                    value=f"{info['mape']}%",
                    delta="-Semakin Kecil Semakin Bagus",
                    delta_color="inverse"
                )
                st.success(f"Status Model: **{info['interpretation']}**")

                with st.expander("Lihat Detail Statistik (Untuk Data Scientist)"):
                    st.write("Angka di bawah ini digunakan untuk perbandingan. Cari nilai yang paling kecil (paling negatif).")
                    c1, c2, c3 = st.columns(3)
                    
                    aic_val = f"{info['aic']:.2f}" if info.get('aic') is not None else "N/A"
                    bic_val = f"{info['bic']:.2f}" if info.get('bic') is not None else "N/A"
                    hqic_val = f"{info['hqic']:.2f}" if info.get('hqic') is not None else "N/A"

                    c1.metric("AIC (Akaike)", aic_val)
                    c2.metric("BIC (Bayesian)", bic_val)
                    c3.metric("HQIC", hqic_val)

                    st.caption(f"Jumlah baris data yang dipelajari: {info.get('n_observations', 0)} baris")

# ==========================================
# 8. KONFIGURASI NATIVE MULTIPAGE & MENU
# ==========================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

page_historis = st.Page(halaman_historis, title="Data Historis Emas", icon="📊", default=True)
page_kelola = st.Page(halaman_kelola_data, title="Cari & Kelola Data", icon="🔍")
page_prediksi = st.Page(halaman_prediksi_ai, title="Prediksi Harga AI", icon="🤖")
page_kualitas = st.Page(halaman_kualitas_model, title="Cek Kualitas Model", icon="🎯")

page_monitoring = st.Page("pages/1_monitoring.py", title="Monitoring API", icon="🛡️")

pg = st.navigation({
    "Fitur Utama" : [page_historis, page_kelola, page_prediksi, page_kualitas],
    "Administrator" : [page_monitoring]
})

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini menggunakan model Machine Learning (ARIMA, SARIMA, Prophet, XGBoost) dengan Redis Cache untuk meramal harga emas.")

if st.session_state.admin_logged_in:
    if st.sidebar.button("Logout Admin"):
        st.session_state.admin_logged_in = False
        st.rerun()

if pg.title == "Monitoring API" and not st.session_state.admin_logged_in:
    st.title("Akses Terbatas")
    st.warning("Halaman ini khusus Administrator. Silakan masukkan password untuk melanjutkan.")

    col_pwd, _ = st.columns([1,2])
    with col_pwd:
        pwd_input = st.text_input("Password Admin", type="password")
        if st.button("Masuk", type="primary"):
            if pwd_input == "admin123":
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Password salah! Silakan coba lagi.")

    st.stop()

pg.run()