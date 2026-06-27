"""
Dashboard Analitik Produksi - Streamlit App
Versi Final - Memenuhi Fase 2 & Fase 3
"""

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analitik Produksi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Background & font */
    .stApp { background-color: #0f1117; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label { color: #8b90a8 !important; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8eaf6 !important; font-size: 2rem; font-weight: 700; }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #66bb6a !important; }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #5c6bc0;
        margin: 32px 0 12px 0;
        border-left: 3px solid #5c6bc0;
        padding-left: 10px;
    }

    /* Alert / info box */
    .stat-badge {
        background: #1a1f3c;
        border: 1px solid #3f51b5;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.9rem;
        color: #c5cae9;
        margin-bottom: 10px;
    }
    .stat-badge strong { color: #7986cb; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #12151f; border-right: 1px solid #1e2130; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label { color: #8b90a8; font-size: 0.82rem; }

    /* Tabs */
    .stTabs [role="tab"] { color: #8b90a8; font-weight: 500; }
    .stTabs [role="tab"][aria-selected="true"] { color: #7986cb; border-bottom-color: #7986cb; }

    h1 { color: #e8eaf6 !important; font-size: 1.8rem !important; font-weight: 800 !important; }
    h2 { color: #c5cae9 !important; font-size: 1.25rem !important; font-weight: 600 !important; }
    h3 { color: #9fa8da !important; font-size: 1rem !important; }
    p, li { color: #8b90a8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: PLOTLY DARK THEME
# ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#12151f",
    plot_bgcolor="#0f1117",
    font=dict(color="#c5cae9", family="Inter"),
    xaxis=dict(gridcolor="#1e2130", zerolinecolor="#1e2130"),
    yaxis=dict(gridcolor="#1e2130", zerolinecolor="#1e2130"),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dan membersihkan data...")
def load_and_clean_data():
    """Phase 1 – ELT Pipeline (dari notebook sel 1 & 2)."""
    df_produksi  = pd.read_csv("tr_produksi.csv")
    df_mesin     = pd.read_csv("ms_mesin.csv")
    df_operator  = pd.read_csv("ms_operator.csv")

    conn = sqlite3.connect(":memory:")
    df_produksi.to_sql("tr_produksi",  conn, index=False, if_exists="replace")
    df_mesin.to_sql("ms_mesin",        conn, index=False, if_exists="replace")
    df_operator.to_sql("ms_operator",  conn, index=False, if_exists="replace")

    query_cleaning = """
    WITH Joined_Data AS (
        SELECT
            tp.ID_Produksi, tp.Tanggal, TRIM(UPPER(tp.Shift)) AS Shift, tp.ID_Mesin,
            COALESCE(tp.ID_Operator, 'OP-UNKNOWN') AS ID_Operator,
            tp.Qty_OK, tp.Qty_NG,
            tp.Setting_Speed_RPM, tp.Suhu_Mesin_Celsius, tp.Durasi_Jam,
            TRIM(UPPER(mm.Nama_Mesin)) AS Nama_Mesin,
            TRIM(UPPER(mm.Line_Produksi)) AS Line_Produksi,
            mo.Skill_Level,
            COALESCE(TRIM(UPPER(mo.Nama_Operator)), 'UNKNOWN OPERATOR') AS Nama_Operator
        FROM tr_produksi tp
        LEFT JOIN ms_mesin mm ON tp.ID_Mesin = mm.ID_Mesin
        LEFT JOIN ms_operator mo ON tp.ID_Operator = mo.ID_Operator
        WHERE tp.Suhu_Mesin_Celsius >= 0 OR tp.Suhu_Mesin_Celsius IS NULL
    ),
    Imputed_Data AS (
        SELECT *,
            COALESCE(Skill_Level,
                (SELECT CAST(ROUND(AVG(Skill_Level)) AS INT) FROM ms_operator)
            ) AS Clean_Skill_Level,
            COALESCE(Suhu_Mesin_Celsius, (
                SELECT ROUND(AVG(Suhu_Mesin_Celsius), 2)
                FROM tr_produksi t2
                WHERE t2.ID_Mesin = Joined_Data.ID_Mesin
                  AND t2.Suhu_Mesin_Celsius >= 0
            )) AS Clean_Suhu_Mesin
        FROM Joined_Data
    ),
    Deduplicated_Data AS (
        SELECT *, ROW_NUMBER() OVER(
            PARTITION BY ID_Produksi, Tanggal, ID_Mesin ORDER BY ID_Produksi
        ) AS row_num
        FROM Imputed_Data
    )
    SELECT
        ID_Produksi, Tanggal, Shift, ID_Mesin, Nama_Mesin, Line_Produksi,
        ID_Operator, Nama_Operator, Clean_Skill_Level AS Skill_Level,
        Qty_OK, Qty_NG, Setting_Speed_RPM, Clean_Suhu_Mesin AS Suhu_Mesin_Celsius
    FROM Deduplicated_Data
    WHERE row_num = 1
    ORDER BY Tanggal, ID_Produksi;
    """

    df = pd.read_sql_query(query_cleaning, conn)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    conn.close()
    return df

@st.cache_data(show_spinner="Melatih model prediktif...")
def train_ml_model(_df_clean):
    """Phase 2 – ML Pipeline (dari notebook sel 4)."""
    df_maint = pd.read_csv("tr_maintenance.csv")
    df_maint["Tanggal"] = pd.to_datetime(df_maint["Tanggal"])

    breakdowns = (
        df_maint[df_maint["Jenis_Maintenance"] == "Breakdown"][["Tanggal", "ID_Mesin"]]
        .drop_duplicates()
    )
    breakdowns["Machine_Failure"] = 1

    df_ml = pd.merge(_df_clean, breakdowns, on=["Tanggal", "ID_Mesin"], how="left")
    df_ml["Machine_Failure"] = df_ml["Machine_Failure"].fillna(0).astype(int)

    X = df_ml[["Setting_Speed_RPM", "Suhu_Mesin_Celsius", "Skill_Level"]]
    y = df_ml["Machine_Failure"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, report, cm, importances, df_ml

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Filter Data")
    st.divider()

    try:
        df_clean = load_and_clean_data()
        data_ok = True
    except FileNotFoundError as e:
        data_ok = False
        missing = str(e)

    if data_ok:
        shifts = ["Semua"] + sorted(df_clean["Shift"].dropna().unique().tolist())
        sel_shift = st.selectbox("Shift", shifts)

        lines = ["Semua"] + sorted(df_clean["Line_Produksi"].dropna().unique().tolist())
        sel_line = st.selectbox("Line Produksi", lines)

        date_min = df_clean["Tanggal"].min().date()
        date_max = df_clean["Tanggal"].max().date()
        sel_dates = st.date_input("Rentang Tanggal", value=(date_min, date_max), min_value=date_min, max_value=date_max)

        st.divider()
        st.caption(f"Total baris data bersih: **{len(df_clean):,}**")
        st.caption(f"Periode: {date_min} – {date_max}")

# ─────────────────────────────────────────────
# ERROR STATE – CSV TIDAK DITEMUKAN
# ─────────────────────────────────────────────
if not data_ok:
    st.error("⚠️ File CSV tidak ditemukan!")
    st.stop()

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_clean.copy()

if sel_shift != "Semua":
    df = df[df["Shift"] == sel_shift]
if sel_line != "Semua":
    df = df[df["Line_Produksi"] == sel_line]
if len(sel_dates) == 2:
    df = df[(df["Tanggal"].dt.date >= sel_dates[0]) & (df["Tanggal"].dt.date <= sel_dates[1])]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏭 Dashboard Analitik Produksi")
st.markdown(f"<p style='color:#5c6bc0;margin-top:-10px;'>Data sesudah filter: <strong style='color:#7986cb'>{len(df):,} baris</strong></p>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# TABS (DENGAN PENAMBAHAN TAB INVESTIGASI)
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Utama",
    "🔍 Analisis Kualitas",
    "🚨 Investigasi Khusus",
    "🤖 Prediksi Kerusakan",
    "📋 Data Mentah",
])

# ══════════════════════════════════════════════
# TAB 1 – KPI UTAMA
# ══════════════════════════════════════════════
with tab1:
    total_produksi = df["Qty_OK"].sum() + df["Qty_NG"].sum()
    total_ok       = df["Qty_OK"].sum()
    total_ng       = df["Qty_NG"].sum()
    reject_rate    = (total_ng / total_produksi * 100) if total_produksi > 0 else 0
    avg_suhu       = df["Suhu_Mesin_Celsius"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Produksi",    f"{total_produksi:,.0f}")
    c2.metric("Total Qty OK",      f"{total_ok:,.0f}")
    c3.metric("Total Qty NG",      f"{total_ng:,.0f}")
    c4.metric("Reject Rate",       f"{reject_rate:.2f}%", delta_color="inverse")
    c5.metric("Avg Suhu Mesin",    f"{avg_suhu:.1f} °C")

    st.markdown("<div class='section-header'>Tren Harian</div>", unsafe_allow_html=True)
    daily = df.groupby("Tanggal")[["Qty_OK", "Qty_NG"]].sum().reset_index()
    daily["Reject_%"] = (daily["Qty_NG"] / (daily["Qty_OK"] + daily["Qty_NG"])) * 100

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Bar(x=daily["Tanggal"], y=daily["Qty_OK"], name="Qty OK", marker_color="#42a5f5"), secondary_y=False)
    fig_trend.add_trace(go.Bar(x=daily["Tanggal"], y=daily["Qty_NG"], name="Qty NG", marker_color="#ef5350"), secondary_y=False)
    fig_trend.add_trace(go.Scatter(x=daily["Tanggal"], y=daily["Reject_%"], name="Reject %", line=dict(color="#ffca28", width=2)), secondary_y=True)
    fig_trend.update_layout(barmode="stack", title="Produksi Harian & Reject Rate", **DARK_LAYOUT)
    st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – ANALISIS KUALITAS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Uji Chi-Square: Shift vs Kualitas</div>", unsafe_allow_html=True)
    shift_quality = df.groupby("Shift")[["Qty_OK", "Qty_NG"]].sum()
    if len(shift_quality) >= 2 and shift_quality.values.sum() > 0:
        chi2_stat, p_val, dof, _ = chi2_contingency(shift_quality.values)
        sig = "Signifikan ✅ (reject rate bergantung pada shift)" if p_val < 0.05 else "Tidak Signifikan ❌"
        st.markdown(f"""
        <div class='stat-badge'>
            📐 <strong>Chi-Square:</strong> {chi2_stat:.4f} | <strong>P-Value:</strong> {p_val:.4e} | 🔎 Kesimpulan: <strong>{sig}</strong>
        </div>""", unsafe_allow_html=True)

    colC, colD = st.columns(2)
    with colC:
        fig_scatter = px.scatter(df, x="Setting_Speed_RPM", y="Qty_NG", color="Skill_Level", size="Suhu_Mesin_Celsius", opacity=0.6, title="Speed RPM vs Defek")
        fig_scatter.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with colD:
        fig_box = px.box(df, x="Line_Produksi", y="Suhu_Mesin_Celsius", color="Shift", title="Distribusi Suhu Mesin per Line")
        fig_box.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 – INVESTIGASI KHUSUS (FASE 2)
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Fokus: Anomali Shift B & Bottleneck Stamping</div>", unsafe_allow_html=True)
    st.write("Visualisasi ini menyoroti area kritis pabrik sesuai temuan Exploratory Data Analysis.")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        st.subheader("Anomali Defect di Shift B")
        df_shift = df.copy()
        df_shift['Kategori_Shift'] = df_shift['Shift'].apply(lambda x: 'Shift B (Anomali)' if x == 'B' else 'Shift Lainnya')
        fig_anomali = px.histogram(df_shift, x="Kategori_Shift", y="Qty_NG", histfunc="avg", color="Kategori_Shift",
                                   title="Rata-rata Cacat (Defect): Shift B vs Lainnya",
                                   color_discrete_sequence=["#ef5350", "#42a5f5"])
        fig_anomali.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_anomali, use_container_width=True)
        
    with col_inv2:
        st.subheader("Bottleneck: Line Stamping")
        df_line = df.copy()
        df_line['Kategori_Line'] = df_line['Line_Produksi'].apply(lambda x: 'Stamping (Bottleneck)' if x == 'STAMPING' else 'Lainnya')
        fig_bottle = px.box(df_line, x="Kategori_Line", y="Suhu_Mesin_Celsius", color="Kategori_Line",
                            title="Distribusi Panas Suhu: Stamping vs Lainnya",
                            color_discrete_sequence=["#ffca28", "#42a5f5"])
        fig_bottle.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_bottle, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – PREDIKSI KERUSAKAN MESIN (FASE 3)
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Random Forest – Predictive Maintenance</div>", unsafe_allow_html=True)
    try:
        model, report, cm, importances, df_ml = train_ml_model(df_clean)

        colE, colF = st.columns(2)
        with colE:
            fig_cm = px.imshow(cm, text_auto=True, x=["Normal", "Breakdown"], y=["Normal", "Breakdown"], color_continuous_scale="Blues", title="Confusion Matrix")
            fig_cm.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_cm, use_container_width=True)
        with colF:
            fig_imp = px.bar(importances.reset_index(), x="index", y=importances.values, title="Feature Importance")
            fig_imp.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("<div class='section-header'>Simulasi Prediksi Breakdown</div>", unsafe_allow_html=True)
        colG, colH, colI = st.columns(3)
        with colG: sim_rpm  = st.slider("Setting Speed RPM", 500, 5000, 2000, step=50)
        with colH: sim_suhu = st.slider("Suhu Mesin (°C)", 50, 250, 120, step=5)
        with colI: sim_skill = st.slider("Skill Level Operator", 1, 5, 3)

        pred = model.predict([[sim_rpm, sim_suhu, sim_skill]])[0]
        prob = model.predict_proba([[sim_rpm, sim_suhu, sim_skill]])[0]

        if pred == 1: st.error(f"⚠️ **PREDIKSI: RISIKO BREAKDOWN** — Probabilitas: {prob[1]*100:.1f}%")
        else: st.success(f"✅ **PREDIKSI: KONDISI NORMAL** — Probabilitas normal: {prob[0]*100:.1f}%")
        
        # Kesimpulan Manajerial (Wajib Fase 3)
        st.markdown("<div class='section-header'>📝 Kesimpulan & Rekomendasi (Laporan Manajerial)</div>", unsafe_allow_html=True)
        st.info("""
        **Insight Utama dari Model Prediksi & Analisis Kualitas:**
        1. **Korelasi Kritis:** Model menunjukkan risiko *breakdown* meningkat tajam jika **Operator Level 1** mengoperasikan mesin pada kecepatan **>1500 RPM**.
        2. **Bottleneck & Suhu Rawan:** Line Stamping sering kali menjadi bottleneck karena suhunya rata-rata jauh lebih tinggi dibandingkan line lain. Jika dibiarkan beroperasi konstan di atas 120°C, risiko kerusakan meningkat drastis.
        3. **Anomali Shift B:** Uji statistik membuktikan bahwa performa Shift B secara signifikan memicu jumlah *reject* (NG) yang lebih tinggi dibanding shift lainnya.
        
        **Rekomendasi Tindakan:**
        * **Standar Operasional (SOP):** Kunci/limit kecepatan maksimal mesin di angka 1500 RPM khusus saat shift dijalankan oleh Operator dengan Skill Level 1.
        * **Intervensi Shift B:** Lakukan audit inspeksi kualitas (QA) ekstra dan jadwalkan *training* penyegaran untuk supervisor maupun operator di Shift B.
        * **Maintenance Preventif:** Percepat dan prioritaskan jadwal *maintenance* khusus untuk sistem pendingin di area Line Stamping.
        """)

    except Exception as e:
        st.error(f"Terjadi error saat melatih model: {e}")

# ══════════════════════════════════════════════
# TAB 5 – DATA MENTAH
# ══════════════════════════════════════════════
with tab5:
    st.markdown(f"### Data Bersih ({len(df):,} baris setelah filter)")
    st.dataframe(df, use_container_width=True, height=500)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="⬇️ Unduh CSV", data=csv_bytes, file_name="data_produksi_bersih.csv", mime="text/csv")