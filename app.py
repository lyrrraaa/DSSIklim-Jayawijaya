# app_streamlit_dss_iklim_jayawijaya_full.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---------- CONFIG ----------
st.set_page_config(page_title="DSS Iklim - Jayawijaya", layout="wide")
st.title("🌦️ Decision Support System Iklim — Jayawijaya")
st.markdown(
    "Dashboard prediksi & analisis iklim Jayawijaya. "
    "Data otomatis dimuat dari `Jayawijaya.xlsx` atau dibuat contoh jika file tidak ada."
)

LOCAL_EXCEL_PATH = "Jayawijaya.xlsx"

# ---------- DSS HELPER FUNCTIONS ----------
def klasifikasi_cuaca(ch, matahari):
    if ch > 20:
        return "Hujan"
    elif ch > 5:
        return "Berawan"
    elif matahari > 4:
        return "Cerah"
    else:
        return "Berawan"

def risiko_kekeringan_score(ch, matahari):
    ch_clamped = np.clip(ch, 0, 200)
    matahari_clamped = np.clip(matahari, 0, 16)
    score = (1 - (ch_clamped / 200)) * 0.7 + (matahari_clamped / 16) * 0.3
    return float(np.clip(score, 0, 1))

def risiko_kekeringan_label(score, thresholds=(0.6, 0.3)):
    high, med = thresholds
    if score >= high:
        return "Risiko Tinggi"
    elif score >= med:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"

def hujan_ekstrem_flag(ch, threshold=50):
    return int(ch > threshold)

def compute_weather_index(df):
    eps = 1e-6
    # Curah hujan
    r = df['curah_hujan'].fillna(0).astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)
    # Tavg
    t = df['Tavg'].fillna((df['Tn'] + df['Tx']) / 2).astype(float).values
    comfy_low, comfy_high = 24, 28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)
    # Kelembaban
    h = df['kelembaban'].fillna(0).astype(float).values
    hum_dist = np.maximum(0, np.maximum(40 - h, h - 70))
    h_norm = (hum_dist - hum_dist.min()) / (hum_dist.max() - hum_dist.min() + eps)
    # Angin
    w = df['kecepatan_angin'].fillna(0).astype(float).values
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)
    composite = 0.35 * r_norm + 0.25 * t_norm + 0.2 * h_norm + 0.2 * w_norm
    return np.clip(composite, 0, 1)

# ---------- DATA LOADING ----------
@st.cache_data(show_spinner=False)
def load_data(local_path=LOCAL_EXCEL_PATH):
    try:
        df = pd.read_excel(local_path, parse_dates=['Tanggal'])
        st.sidebar.success(f"Loaded local Excel: {local_path}")
    except Exception:
        st.sidebar.info("Local Excel tidak ditemukan — membuat data contoh 2 tahun.")
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=730)
        rng = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Tanggal': rng,
            'curah_hujan': np.random.gamma(1.5, 8, len(rng)).round(1),
            'Tn': np.random.normal(22, 2, len(rng)).round(1),
            'Tx': np.random.normal(31, 2.5, len(rng)).round(1),
            'kelembaban': np.random.randint(50, 95, len(rng)),
            'matahari': np.clip(np.random.normal(5, 2, len(rng)), 0, 12).round(1),
            'kecepatan_angin': np.random.uniform(0, 20, len(rng)).round(1)
        })
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    for col in ['curah_hujan', 'Tn', 'Tx', 'kelembaban', 'matahari', 'kecepatan_angin']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Tavg'] = (df['Tn'] + df['Tx']) / 2
    return df.sort_values('Tanggal').reset_index(drop=True)

# ---------- FORECAST ALL ----------
def forecast_all(df, years=50):
    future_dfs = []
    future_days = years*365
    last_date = df['Tanggal'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    
    numeric_cols = ['curah_hujan','Tn','Tx','kelembaban','matahari','kecepatan_angin']
    forecast_values = {}
    
    for col in numeric_cols:
        X = np.arange(len(df)).reshape(-1,1)
        y = df[col].values
        model = LinearRegression().fit(X,y)
        X_future = np.arange(len(df), len(df)+future_days).reshape(-1,1)
        pred = model.predict(X_future)
        
        df['month'] = df['Tanggal'].dt.month
        seasonal = df.groupby('month')[col].mean()
        future_month = future_dates.month
        seasonal_adj = future_month.map(seasonal)
        pred = 0.5*pred + 0.5*seasonal_adj
        forecast_values[col] = pred
        
    df_future = pd.DataFrame({
        'Tanggal': future_dates,
        'curah_hujan': forecast_values['curah_hujan'],
        'Tn': forecast_values['Tn'],
        'Tx': forecast_values['Tx'],
        'Tavg': (forecast_values['Tn'] + forecast_values['Tx'])/2,
        'kelembaban': forecast_values['kelembaban'],
        'matahari': forecast_values['matahari'],
        'kecepatan_angin': forecast_values['kecepatan_angin'],
        'Sumber':'Prediksi'
    })
    df_hist = df.copy()
    df_hist['Sumber'] = 'Historis'
    return pd.concat([df_hist, df_future]).reset_index(drop=True)

# ---------- LOAD + FORECAST ----------
data = load_data()
data_forecast = forecast_all(data)

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm/hari)", value=50, min_value=1)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi", 0.0, 1.0, 0.6, 0.01)
risk_med = st.sidebar.slider("Ambang Risiko Sedang", 0.0, 1.0, 0.3, 0.01)

# ---------- FILTER ----------
st.sidebar.header("📅 Filter Tanggal")
min_date = data_forecast['Tanggal'].min().date()
max_date = data_forecast['Tanggal'].max().date()
date_range = st.sidebar.date_input("Rentang Tanggal", value=(min_date,max_date), min_value=min_date, max_value=max_date)
start_date, end_date = map(pd.to_datetime, date_range)
df = data_forecast[(data_forecast['Tanggal'] >= start_date) & (data_forecast['Tanggal'] <= end_date)].copy()

# ---------- DERIVED ----------
df['Prediksi Cuaca'] = df.apply(lambda r: klasifikasi_cuaca(r['curah_hujan'], r.get('matahari',5)), axis=1)
df['Hujan Ekstrem'] = df['curah_hujan'].apply(lambda x: "Ya" if x>extreme_threshold else "Tidak")
df['RiskScore'] = df.apply(lambda r: risiko_kekeringan_score(r['curah_hujan'], r.get('matahari',5)), axis=1)
df['RiskLabel'] = df['RiskScore'].apply(lambda s: risiko_kekeringan_label(s, (risk_high,risk_med)))
df['WeatherIndex'] = compute_weather_index(df)

# ---------- 1. Prediksi Curah Hujan ----------
st.markdown("---")
st.header("1. Prediksi Curah Hujan (Rainfall Forecast)")
st.markdown("Visualisasi ini membantu memantau tren curah hujan historis dan prediksi jangka panjang, termasuk distribusi bulanan.")

# Rainfall Line
fig_line = px.line(df, x='Tanggal', y='curah_hujan', color='Sumber', title="Rainfall Over Time")
st.plotly_chart(fig_line, use_container_width=True)

# Rainfall Area
fig_area = px.area(df, x='Tanggal', y='curah_hujan', color='Sumber', title="Rainfall Area Chart")
st.plotly_chart(fig_area, use_container_width=True)

# Monthly Rainfall Sum
monthly_sum = df.groupby(df['Tanggal'].dt.to_period('M'))['curah_hujan'].sum().reset_index()
monthly_sum['Tanggal'] = monthly_sum['Tanggal'].dt.to_timestamp()
fig_monthly = px.bar(monthly_sum, x='Tanggal', y='curah_hujan', title="Monthly Rainfall Sum")
st.plotly_chart(fig_monthly, use_container_width=True)

# ---------- 2. Prediksi Temperatur ----------
st.markdown("---")
st.header("2. Prediksi Hari Panas / Temperatur (Temperature Forecast)")
st.markdown("Menampilkan tren harian suhu minimum (Tn), rata-rata (Tavg), dan maksimum (Tx) beserta distribusi panas bulanan.")

fig_temp = px.line(df, x='Tanggal', y=['Tn','Tavg','Tx'], labels={'value':'Temperature (°C)','variable':'Type'}, title="Temperature Trends")
st.plotly_chart(fig_temp, use_container_width=True)

# Heatmap Month x Day
df['day'] = df['Tanggal'].dt.day
df['month'] = df['Tanggal'].dt.month
heatmap_data = df.pivot_table(index='month', columns='day', values='Tavg', aggfunc='mean')
fig_heat = px.imshow(heatmap_data, labels={'x':'Day','y':'Month','color':'Tavg'}, title="Temperature Heatmap")
st.plotly_chart(fig_heat, use_container_width=True)

# ---------- 3. Prediksi Risiko Kekeringan ----------
st.markdown("---")
st.header("3. Prediksi Risiko Kekeringan")
st.markdown("Menunjukkan risiko kekeringan harian dan rata-rata risiko wilayah Jayawijaya.")

avg_risk = df['RiskScore'].mean()
st.metric("Average RiskScore wilayah Jayawijaya", f"{avg_risk:.2f}")

fig_risk = px.line(df, x='Tanggal', y='RiskScore', title="Risk Score Over Time")
st.plotly_chart(fig_risk, use_container_width=True)

# ---------- 4. Prediksi Hujan Ekstrem ----------
st.markdown("---")
st.header("4. Prediksi Hujan Ekstrem")
st.markdown("Memvisualisasikan frekuensi hujan ekstrem, curah hujan vs waktu, dan probabilitas hujan ekstrem rolling 30 hari.")

# Frekuensi Hujan Ekstrem per Bulan
extreme_count = df[df['Hujan Ekstrem']=='Ya'].groupby(df['Tanggal'].dt.to_period('M'))['Hujan Ekstrem'].count().reset_index()
extreme_count['Tanggal'] = extreme_count['Tanggal'].dt.to_timestamp()
fig_extreme_bar = px.bar(extreme_count, x='Tanggal', y='Hujan Ekstrem', title="Frekuensi Hujan Ekstrem Per Bulan")
st.plotly_chart(fig_extreme_bar, use_container_width=True)

# Scatter Curah Hujan vs Waktu
fig_scatter = px.scatter(df, x='Tanggal', y='curah_hujan', color='Hujan Ekstrem', title="Curah Hujan vs Waktu")
st.plotly_chart(fig_scatter, use_container_width=True)

# Rolling 30-day probability
df['ExtremeFlag'] = df['Hujan Ekstrem'].map({'Ya':1,'Tidak':0})
df['Rolling30'] = df['ExtremeFlag'].rolling(30,min_periods=1).mean()
fig_rolling = px.line(df, x='Tanggal', y='Rolling30', title="Rolling 30-day Probability of Extreme Rain")
st.plotly_chart(fig_rolling, use_container_width=True)

# ---------- 5. Weather Index ----------
st.markdown("---")
st.header("5. Prediksi Indeks Cuaca Gabungan (Weather Index Prediction)")
st.markdown("Membantu memonitor kenyamanan kondisi cuaca dengan menggabungkan curah hujan, suhu, kelembaban, dan angin.")

# Radar
latest = df.iloc[-1]
categories = ['Rain','Temperature','Humidity','Wind']
values = [latest['curah_hujan'], latest['Tavg'], latest['kelembaban'], latest['kecepatan_angin']]
fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Weather Index Components (Latest Day)")
st.plotly_chart(fig_radar, use_container_width=True)

# Composite over time
fig_weather = px.line(df, x='Tanggal', y='WeatherIndex', title="Composite Weather Index Over Time")
st.plotly_chart(fig_weather, use_container_width=True)

# ---------- 6. Tren Kualitas Iklim Bulanan/Tahunan ----------
st.markdown("---")
st.header("6. Tren Kualitas Iklim Bulanan/Tahunan")
st.markdown("Memantau rata-rata curah hujan bulanan dan moving average harian.")

# Monthly Average Rainfall by Year
df['year'] = df['Tanggal'].dt.year
monthly_avg = df.groupby(['year', 'month'])['curah_hujan'].mean().reset_index()
fig_monthly_avg = px.line(monthly_avg, x='month', y='curah_hujan', color='year', title="Monthly Average Rainfall by Year")
st.plotly_chart(fig_monthly_avg, use_container_width=True)

# Moving average
df['Rain_MA7'] = df['curah_hujan'].rolling(7).mean()
fig_ma = px.line(df, x='Tanggal', y='Rain_MA7', title="Moving Average Rainfall (7-day)")
st.plotly_chart(fig_ma, use_container_width=True)

# ---------- 7. Prediksi Anomali Iklim ----------
st.markdown("---")
st.header("7. Prediksi Anomali Iklim")
st.markdown("Menunjukkan anomali suhu dibanding baseline historis.")

baseline = df.groupby(df['Tanggal'].dt.month)['Tavg'].mean()
df['TempAnomaly'] = df.apply(lambda r: r['Tavg'] - baseline[r['Tanggal'].month], axis=1)
anomaly_pivot = df.pivot_table(index=df['Tanggal'].dt.year, columns=df['Tanggal'].dt.month, values='TempAnomaly')
fig_anomaly = px.imshow(anomaly_pivot, labels={'x':'Month','y':'Year','color':'Temperature Anomaly'}, title="Temperature Anomaly (Year x Month)")
st.plotly_chart(fig_anomaly, use_container_width=True)

# Tabel & Download
st.markdown("---")
with st.expander("📁 Lihat dan Unduh Data Lengkap"):
    st.dataframe(df)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Hasil_DSS", index=False)
    buffer.seek(0)
    st.download_button(
        "Unduh Excel",
        data=buffer.getvalue(),
        file_name="hasil_dss_iklim_jayawijaya.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
