# app_streamlit_dss_iklim_jayawijaya.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------- CONFIG ----------
st.set_page_config(page_title="DSS Iklim - Jayawijaya", layout="wide")
st.title("🌦️ Decision Support System Iklim — Jayawijaya")
st.markdown(
    "Dashboard prediksi & analisis iklim untuk wilayah Jayawijaya. "
    "Data dapat dimuat dari `Jayawijaya.xlsx` atau dibuat otomatis."
)

LOCAL_EXCEL_PATH = "Jayawijaya.xlsx"

# ---------- DSS helper functions ----------
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

def risiko_kekeringan_label(score, thresholds=(0.6,0.3)):
    high, med = thresholds
    if score >= high:
        return "Risiko Tinggi"
    elif score >= med:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"

def compute_weather_index(df):
    eps = 1e-6
    r = df['curah_hujan'].fillna(0).astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)
    tn = df.get('Tn', pd.Series([0]*len(df))).astype(float)
    tx = df.get('Tx', pd.Series([0]*len(df))).astype(float)
    t = ((tn + tx)/2).values
    comfy_low, comfy_high = 24,28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)
    h = df['kelembaban'].fillna(0).astype(float).values
    hum_dist = np.maximum(0, np.maximum(40-h,h-70))
    h_norm = (hum_dist - hum_dist.min())/(hum_dist.max()-hum_dist.min()+eps)
    w = df['kecepatan_angin'].fillna(0).astype(float).values
    w_norm = (w-w.min())/(w.max()-w.min()+eps)
    composite = 0.35*r_norm + 0.25*t_norm + 0.2*h_norm + 0.2*w_norm
    return np.clip(composite,0,1)

# ---------- Load Data ----------
@st.cache_data
def load_data(path=LOCAL_EXCEL_PATH):
    try:
        df = pd.read_excel(path, parse_dates=['Tanggal'])
    except:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=730)
        rng = pd.date_range(start,end,freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Tanggal': rng,
            'curah_hujan': np.random.gamma(1.5,8,len(rng)).round(1),
            'Tn': np.random.normal(22,2,len(rng)).round(1),
            'Tx': np.random.normal(31,2.5,len(rng)).round(1),
            'kelembaban': np.random.randint(50,95,len(rng)),
            'matahari': np.clip(np.random.normal(5,2,len(rng)),0,12).round(1),
            'kecepatan_angin': np.random.uniform(0,20,len(rng)).round(1)
        })
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    for col in ['curah_hujan','Tn','Tx','kelembaban','matahari','kecepatan_angin']:
        if col not in df.columns: df[col]=0
        df[col] = pd.to_numeric(df[col],errors='coerce').fillna(0)
    df['Tavg'] = (df['Tn']+df['Tx'])/2
    return df.sort_values('Tanggal').reset_index(drop=True)

# ---------- Forecast ----------
def forecast_50_years(df,target='curah_hujan',years=50):
    df = df.copy().dropna(subset=[target])
    df['time_index'] = np.arange(len(df))
    X = df[['time_index']]; y = df[target]
    model = LinearRegression(); model.fit(X,y)
    last_date = df['Tanggal'].max()
    future_days = years*365
    future_dates = pd.date_range(last_date+pd.Timedelta(days=1),periods=future_days)
    future_index = np.arange(len(df),len(df)+future_days).reshape(-1,1)
    future_pred = model.predict(future_index)
    df['month'] = df['Tanggal'].dt.month
    seasonal = df.groupby('month')[target].mean()
    future_month = future_dates.month
    seasonal_adjustment = future_month.map(seasonal)
    future_final = future_pred*0.5 + seasonal_adjustment*0.5
    future_df = pd.DataFrame({
        'Tanggal':future_dates,
        target:future_final,
        'Sumber':'Prediksi'
    })
    df_hist = df[['Tanggal',target]].copy(); df_hist['Sumber']='Historis'
    return pd.concat([df_hist,future_df]).reset_index(drop=True)

# ---------- Load & Forecast ----------
data = load_data()
df_forecast = forecast_50_years(data)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm/hari)",value=50)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi",0.0,1.0,0.6)
risk_med = st.sidebar.slider("Ambang Risiko Sedang",0.0,1.0,0.3)
st.sidebar.header("📅 Filter Tanggal")
min_date,max_date = df_forecast['Tanggal'].min().date(),df_forecast['Tanggal'].max().date()
date_range = st.sidebar.date_input("Rentang Tanggal",(min_date,max_date),min_value=min_date,max_value=max_date)
start_date,end_date = map(pd.to_datetime,date_range)
df = df_forecast[(df_forecast['Tanggal']>=start_date)&(df_forecast['Tanggal']<=end_date)].copy()

# ---------- Derived Fields ----------
df['PrediksiCuaca'] = df.apply(lambda r: klasifikasi_cuaca(r['curah_hujan'],r.get('matahari',5)),axis=1)
df['HujanEkstrem'] = df['curah_hujan'].apply(lambda x:"Ya" if x>extreme_threshold else "Tidak")
df['RiskScore'] = df.apply(lambda r: risiko_kekeringan_score(r['curah_hujan'],r.get('matahari',5)),axis=1)
df['RiskLabel'] = df['RiskScore'].apply(lambda s: risiko_kekeringan_label(s,(risk_high,risk_med)))
df['WeatherIndex'] = compute_weather_index(df)

# ---------- Top Metrics ----------
st.markdown("---"); st.subheader("Ringkasan Cepat")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Periode",f"{df['Tanggal'].min().date()} — {df['Tanggal'].max().date()}")
c2.metric("Avg Rain (mm)",f"{df['curah_hujan'].mean():.2f}")
c3.metric("Avg Temp (°C)",f"{df['Tavg'].mean():.2f}")
c4.metric("Avg RiskScore",f"{df['RiskScore'].mean():.2f}")

# ---------- 1. Prediksi Curah Hujan ----------
st.markdown("---"); st.header("1. Prediksi Curah Hujan (Rainfall Forecast)")
st.markdown("**Cara membaca:** Garis menunjukkan curah hujan harian historis & prediksi. Area chart memudahkan visualisasi total curah hujan. Sum bulanan menunjukkan akumulasi hujan tiap bulan.")

fig_line = px.line(df,x='Tanggal',y='curah_hujan',color='Sumber',labels={'curah_hujan':'Curah Hujan (mm)'})
st.plotly_chart(fig_line,use_container_width=True)
fig_area = px.area(df,x='Tanggal',y='curah_hujan',color='Sumber')
st.plotly_chart(fig_area,use_container_width=True)
monthly_sum = df.groupby(df['Tanggal'].dt.to_period('M'))['curah_hujan'].sum().reset_index(); monthly_sum['Tanggal']=monthly_sum['Tanggal'].dt.to_timestamp()
fig_month_sum = px.bar(monthly_sum,x='Tanggal',y='curah_hujan',title="Monthly Rainfall Sum")
st.plotly_chart(fig_month_sum,use_container_width=True)

# ---------- 2. Prediksi Hari Panas / Temperatur ----------
st.markdown("---"); st.header("2. Prediksi Hari Panas / Temperatur (Temperature Forecast)")
st.markdown("**Cara membaca:** Tn=Minimum, Tx=Maximum, Tavg=Rata-rata harian. Heatmap mempermudah melihat tren bulanan.")

fig_temp_trend = px.line(df,x='Tanggal',y=['Tn','Tavg','Tx'],labels={'value':'°C','variable':'Temperature'})
st.plotly_chart(fig_temp_trend,use_container_width=True)

# Heatmap Month x Day
df['Month'],df['Day']=df['Tanggal'].dt.month,df['Tanggal'].dt.day
temp_matrix = df.pivot_table(index='Day',columns='Month',values='Tavg',aggfunc='mean')
fig_temp_heatmap = px.imshow(temp_matrix,color_continuous_scale='RdYlBu_r',labels=dict(x='Month',y='Day',color='Tavg'))
st.plotly_chart(fig_temp_heatmap,use_container_width=True)

# ---------- 3. Prediksi Risiko Kekeringan ----------
st.markdown("---"); st.header("3. Prediksi Risiko Kekeringan")
st.markdown("**Cara membaca:** Angka RiskScore 0–1. Semakin tinggi, semakin besar risiko kekeringan di wilayah Jayawijaya.")

fig_risk = px.line(df,x='Tanggal',y='RiskScore',color='RiskLabel',labels={'RiskScore':'Risk Score'})
st.plotly_chart(fig_risk,use_container_width=True)

# ---------- 4. Prediksi Hujan Ekstrem ----------
st.markdown("---"); st.header("4. Prediksi Hujan Ekstrem")
st.markdown("**Cara membaca:** Frekuensi bulanan menunjukkan bulan-bulan dengan potensi hujan ekstrem. Scatter plot menunjukkan curah hujan vs waktu. Rolling probability memberi perkiraan 30 hari.")

extreme_count = df[df['HujanEkstrem']=='Ya'].groupby(df['Tanggal'].dt.to_period('M'))['HujanEkstrem'].count().reset_index()
extreme_count['Tanggal']=extreme_count['Tanggal'].dt.to_timestamp()
fig_extreme_bar = px.bar(extreme_count,x='Tanggal',y='HujanEkstrem',title="Frekuensi Hujan Ekstrem Per Bulan")
st.plotly_chart(fig_extreme_bar,use_container_width=True)

fig_extreme_scatter = px.scatter(df,x='Tanggal',y='curah_hujan',color='HujanEkstrem')
st.plotly_chart(fig_extreme_scatter,use_container_width=True)

df['HujanEkstrem_flag'] = (df['curah_hujan']>extreme_threshold).astype(int)
df['HujanEkstrem_roll30'] = df['HujanEkstrem_flag'].rolling(30,min_periods=1).mean()
fig_extreme_roll = px.line(df,x='Tanggal',y='HujanEkstrem_roll30',labels={'HujanEkstrem_roll30':'Probabilitas 30-hari'})
st.plotly_chart(fig_extreme_roll,use_container_width=True)

# ---------- 5. Prediksi Indeks Cuaca ----------
st.markdown("---"); st.header("5. Prediksi Indeks Cuaca Gabungan (Weather Index Prediction)")
st.markdown("**Cara membaca:** Radar menampilkan komponen indeks cuaca. Garis composite menampilkan tren indeks gabungan dari 0–1.")

fig_weather_line = px.line(df,x='Tanggal',y='WeatherIndex',labels={'WeatherIndex':'Composite Weather Index'})
st.plotly_chart(fig_weather_line,use_container_width=True)

# ---------- 6. Tren Kualitas Iklim Bulanan/Tahunan ----------
st.markdown("---"); st.header("6. Tren Kualitas Iklim Bulanan/Tahunan")
st.markdown("**Cara membaca:** Rainfall rata-rata bulanan per tahun dan moving average membantu melihat tren curah hujan jangka panjang.")

monthly_avg = df.groupby([df['Tanggal'].dt.year,df['Tanggal'].dt.month])['curah_hujan'].mean().reset_index()
monthly_avg.rename(columns={'Tanggal':'Year','curah_hujan':'AvgRain'},inplace=True)
fig_monthly_avg = px.line(monthly_avg,x='Tanggal',y='AvgRain',title="Monthly Avg Rainfall")
st.plotly_chart(fig_monthly_avg,use_container_width=True)

df['Rain_MA7'] = df['curah_hujan'].rolling(7,min_periods=1).mean()
fig_rain_ma7 = px.line(df,x='Tanggal',y='Rain_MA7',labels={'Rain_MA7':'7-day Moving Average Rainfall'})
st.plotly_chart(fig_rain_ma7,use_container_width=True)

# ---------- 7. Prediksi Anomali Iklim ----------
st.markdown("---"); st.header("7. Prediksi Anomali Iklim")
st.markdown("**Cara membaca:** Heatmap menunjukkan deviasi temperatur dari baseline per bulan/tahun. Garis menunjukkan perbandingan dengan baseline.")

baseline_temp = df.groupby(df['Tanggal'].dt.dayofyear)['Tavg'].mean()
df['TempAnomaly'] = df['Tavg'] - df['Tanggal'].dt.dayofyear.map(baseline_temp)
anomaly_matrix = df.pivot_table(index=df['Tanggal'].dt.year,columns=df['Tanggal'].dt.month,values='TempAnomaly')
fig_anomaly = px.imshow(anomaly_matrix,color_continuous_scale='RdBu',labels=dict(x='Month',y='Year',color='TempAnomaly'))
st.plotly_chart(fig_anomaly,use_container_width=True)

fig_temp_baseline = px.line(df,x='Tanggal',y=['Tavg'],labels={'value':'°C','variable':'Temperature'})
st.plotly_chart(fig_temp_baseline,use_container_width=True)

# ---------- Data Viewer & Download ----------
st.markdown("---")
with st.expander("📁 Lihat dan Unduh Data"):
    st.dataframe(df)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer,engine='xlsxwriter') as writer:
        df.to_excel(writer,sheet_name='Hasil_DSS',index=False)
    buffer.seek(0)
    st.download_button("Unduh Excel",data=buffer.getvalue(),file_name="hasil_dss_iklim_jayawijaya.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Gunakan file `Jayawijaya.xlsx` dengan kolom minimal: Tanggal, curah_hujan, Tn, Tx, kelembaban, matahari, kecepatan_angin.")
