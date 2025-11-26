# app_streamlit_dss_iklim_jayawijaya.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="DSS Iklim - Jayawijaya", layout="wide")
st.title("🌦️ Decision Support System Iklim — Jayawijaya")
st.markdown(
    "Dashboard prediksi & analisis iklim. Data akan otomatis dimuat dari file lokal "
    "`Jayawijaya.xlsx` jika tersedia; jika tidak aplikasi membuat data contoh."
)

LOCAL_EXCEL_PATH = "Jayawijaya.xlsx"

# ----------------------------------------------------
# DSS FUNCTIONS
# ----------------------------------------------------
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


# ----------------------------------------------------
# WEATHER INDEX (menghitung Tavg otomatis)
# ----------------------------------------------------
def compute_weather_index(df):
    eps = 1e-6

    # Curah hujan
    r = df['curah_hujan'].fillna(0).astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)

    # Temperatur (Tavg = (Tn+Tx)/2)
    tn = df.get('Tn', pd.Series([0]*len(df))).astype(float)
    tx = df.get('Tx', pd.Series([0]*len(df))).astype(float)
    t = ((tn + tx) / 2).values

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


# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
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

    required = ['curah_hujan','Tn','Tx','kelembaban','matahari','kecepatan_angin']
    for col in required:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Hitung Tavg otomatis
    df['Tavg'] = (df['Tn'] + df['Tx']) / 2

    return df.sort_values('Tanggal').reset_index(drop=True)


# ----------------------------------------------------
# FORECAST 50 TAHUN (hanya curah hujan)
# ----------------------------------------------------
def forecast_50_years(df, target_col="curah_hujan", years=50):
    df = df.copy().dropna(subset=[target_col])
    df['time_index'] = np.arange(len(df))

    X = df[['time_index']]
    y = df[target_col]

    model = LinearRegression()
    model.fit(X, y)

    last_date = df['Tanggal'].max()
    future_days = years * 365
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    future_index = np.arange(len(df), len(df) + future_days).reshape(-1, 1)

    future_pred = model.predict(future_index)

    df['month'] = df['Tanggal'].dt.month
    seasonal = df.groupby('month')[target_col].mean()
    future_month = future_dates.month
    seasonal_adjustment = future_month.map(seasonal)

    final_pred = future_pred * 0.5 + seasonal_adjustment * 0.5

    future_df = pd.DataFrame({
        "Tanggal": future_dates,
        target_col: final_pred,
        "Sumber": "Prediksi"
    })

    df_hist = df[['Tanggal', target_col]].copy()
    df_hist["Sumber"] = "Historis"

    combined = pd.concat([df_hist, future_df]).reset_index(drop=True)
    return combined


# ----------------------------------------------------
# LOAD + FORECAST
# ----------------------------------------------------
data = load_data()
data_forecast = forecast_50_years(data)

# Lengkapi kolom yang hilang agar Tavg, WeatherIndex, RiskScore tidak error
extra_cols = ['Tn', 'Tx', 'kelembaban', 'matahari', 'kecepatan_angin']

for col in extra_cols:
    if col not in data_forecast.columns:
        last_val = data[col].iloc[-1] if col in data.columns else 0
        data_forecast[col] = last_val

data_forecast["Tavg"] = (data_forecast["Tn"] + data_forecast["Tx"]) / 2


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm/hari)", value=50, min_value=1)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi", 0.0, 1.0, 0.6, 0.01)
risk_med = st.sidebar.slider("Ambang Risiko Sedang", 0.0, 1.0, 0.3, 0.01)

st.sidebar.header("📅 Filter Tanggal")
min_date = data_forecast['Tanggal'].min().date()
max_date = data_forecast['Tanggal'].max().date()

date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)

start_date, end_date = map(pd.to_datetime, date_range)
df = data_forecast[(data_forecast['Tanggal'] >= start_date) & (data_forecast['Tanggal'] <= end_date)].copy()


# ----------------------------------------------------
# DERIVED FIELDS
# ----------------------------------------------------
df['Prediksi Cuaca'] = df.apply(lambda r: klasifikasi_cuaca(r['curah_hujan'], r.get('matahari', 5)), axis=1)
df['Hujan Ekstrem'] = df['curah_hujan'].apply(lambda x: "Ya" if x > extreme_threshold else "Tidak")
df['RiskScore'] = df.apply(lambda r: risiko_kekeringan_score(r['curah_hujan'], r.get('matahari', 5)), axis=1)
df['RiskLabel'] = df['RiskScore'].apply(lambda s: risiko_kekeringan_label(s, (risk_high, risk_med)))
df['WeatherIndex'] = compute_weather_index(df)


# ----------------------------------------------------
# TOP METRICS
# ----------------------------------------------------
st.markdown("---")
st.subheader("Ringkasan Cepat")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Periode", f"{df['Tanggal'].min().date()} — {df['Tanggal'].max().date()}")
c2.metric("Avg Rain (mm)", f"{df['curah_hujan'].mean():.2f}")
c3.metric("Avg Temp (°C)", f"{df['Tavg'].mean():.2f}")
c4.metric("Avg RiskScore", f"{df['RiskScore'].mean():.2f}")


# ----------------------------------------------------
# GRAFIK
# ----------------------------------------------------
st.markdown("---")
st.header("1. Prediksi Curah Hujan")

fig_rain_line = px.line(
    df,
    x="Tanggal",
    y="curah_hujan",
    color="Sumber",
    title="Historis & Prediksi 50 Tahun ke Depan",
    labels={"curah_hujan": "Curah Hujan (mm)"}
)

st.plotly_chart(fig_rain_line, use_container_width=True)


# ----------------------------------------------------
# DATA VIEW + DOWNLOAD
# ----------------------------------------------------
st.markdown("---")
with st.expander("📁 Lihat dan Unduh Data"):
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

st.caption(
    "Gunakan file `Jayawijaya.xlsx` dengan kolom minimal: Tanggal, curah_hujan, Tn, Tx, kelembaban, matahari, kecepatan_angin."
)
