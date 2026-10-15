import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import plotly.graph_objects as go
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Regresi Fiskal - Surabaya",
    page_icon="📊",
    layout="wide"
)

# Header Utama
st.title("📊 Dashboard Analisis Regresi Lineer Berganda")
st.caption("Studi Kasus: Regresi Kemiskinan terhadap BM Ekonomi & DBH Pajak (Kota Surabaya)")

# 1. READ & FILTER DATASET
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/main/dataset_fiskal_97_hipoFull.csv"
    df_all = pd.read_csv(url)
    df_surabaya = df_all[df_all['Kota'] == 'Kota Surabaya'].copy()
    return df_surabaya

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Informasi Dataset")
st.sidebar.metric("Total Data Surabaya", f"{len(df)} baris")
st.sidebar.markdown("---")
st.sidebar.markdown("**Variabel Regresi:**")
st.sidebar.write("• **Dependent (Y):** KEMISKINAN")
st.sidebar.write("• **Independent (X1):** BM Ekonomi (`bm_ekonomi`)")
st.sidebar.write("• **Independent (X2):** DBH Pajak (`DBH_PAJAK`)")

# 2. MODEL REGRESI
X = df[['bm_ekonomi', 'DBH_PAJAK']]
Y = df['KEMISKINAN']
X_sm = sm.add_constant(X)
model = sm.OLS(Y, X_sm).fit()

# Ringkasan Performa Model (Metric Cards)
col1, col2, col3, col4 = st.columns(4)
col1.metric("R-Squared (R²)", f"{model.rsquared:.4f}")
col2.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
col3.metric("F-Statistic", f"{model.fvalue:.2f}")
col4.metric("Prob (F-Statistic)", f"{model.f_pvalue:.4f}")

st.markdown("---")

# TABS UNTUK INTERFACE
tab1, tab2, tab3, tab4 = st.tabs(["📉 Visualisasi 3D", "📋 Uji Asumsi Klasik", "📑 Ringkasan OLS", "🔍 Data Frame"])

# TAB 1: VISUALISASI 3D INTERAKTIF
with tab1:
    st.subheader("Plot Bidang Regresi 3D Interaktif")
    
    # Grid Bidang Regresi
    x_range = np.linspace(df['bm_ekonomi'].min(), df['bm_ekonomi'].max(), 20)
    y_range = np.linspace(df['DBH_PAJAK'].min(), df['DBH_PAJAK'].max(), 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = model.params['const'] + (model.params['bm_ekonomi'] * x_grid) + (model.params['DBH_PAJAK'] * y_grid)

    fig = go.Figure()

    # Permukaan Bidang
    fig.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        name='Bidang Regresi',
        colorscale='Viridis',
        opacity=0.8,
        showscale=False
    ))

    # Titik Data Aktual Asli
    fig.add_trace(go.Scatter3d(
        x=df['bm_ekonomi'],
        y=df['DBH_PAJAK'],
        z=df['KEMISKINAN'],
        mode='markers',
        name='Data Aktual',
        marker=dict(size=6, color='red', line=dict(color='white', width=1))
    ))

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='BM Ekonomi',
            yaxis_title='DBH PAJAK',
            zaxis_title='KEMISKINAN'
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

# TAB 2: UJI ASUMSI KLASIK
with tab2:
    st.subheader("Pemeriksaan Uji Asumsi Klasik")
    
    # Kalkulasi Uji Klasik
    shapiro_stat, shapiro_p = stats.shapiro(model.resid)
    
    vif_data = pd.DataFrame()
    vif_data["Variabel"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
    dw_value = sm.stats.stattools.durbin_watson(model.resid)

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 1. Normalitas (Shapiro-Wilk)")
        st.write(f"**P-Value:** `{shapiro_p:.4f}`")
        if shapiro_p > 0.05:
            st.success("✅ **Lolos** (Residul berdistribusi normal, p-value > 0.05)")
        else:
            st.error("❌ **Tidak Lolos** (Residul tidak berdistribusi normal)")

        st.markdown("### 2. Heteroskedastisitas (Breusch-Pagan)")
        st.write(f"**P-Value:** `{bp_test[1]:.4f}`")
        if bp_test[1] > 0.05:
            st.success("✅ **Lolos** (Bebas Heteroskedastisitas, p-value > 0.05)")
        else:
            st.error("❌ **Terjadi Heteroskedastisitas**")

    with col_b:
        st.markdown("### 3. Multikolinearitas (VIF)")
        st.dataframe(vif_data, use_container_width=True, hide_index=True)
        if vif_data['VIF'].max() < 10:
            st.success("✅ **Lolos** (Tidak ada multikolinearitas, VIF < 10)")
        else:
            st.error("❌ **Terjadi Multikolinearitas**")

        st.markdown("### 4. Autokorelasi (Durbin-Watson)")
        st.write(f"**Nilai DW:** `{dw_value:.4f}`")
        if 1.5 <= dw_value <= 2.5:
            st.success("✅ **Lolos** (Mendekati nilai 2, bebas autokorelasi)")
        else:
            st.warning("⚠️ **Indikasi Autokorelasi**")

# TAB 3: SUMMARY OLS
with tab3:
    st.subheader("Persamaan Regresi (Data Asli)")
    const_val = model.params['const']
    bm_val = model.params['bm_ekonomi']
    dbh_val = model.params['DBH_PAJAK']
    
    st.code(
        f"KEMISKINAN = {const_val:.4f} + ({bm_val:.4f} * BM_Ekonomi) - ({abs(dbh_val):.4f} * DBH_PAJAK)",
        language="text"
    )

    st.subheader("Ringkasan Statistik Model OLS")
    st.text(model.summary().as_text())

# TAB 4: DATASET PREVIEW
with tab4:
    st.subheader("Tabel Data Kota Surabaya")
    st.dataframe(df, use_container_width=True)