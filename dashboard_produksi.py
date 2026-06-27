"""
Dashboard Analitik Produksi - Streamlit App
Konversi dari notebook UAS_IAD_1.ipynb

Cara menjalankan:
    streamlit run dashboard_produksi.py

Pastikan file CSV berada di folder yang sama:
    - tr_produksi.csv
    - ms_mesin.csv
    - ms_operator.csv
    - tr_maintenance.csv
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
    st.markdown(f"""
    **Detail error:** `{missing}`

    Pastikan file-file berikut berada di folder yang **sama** dengan `dashboard_produksi.py`:
    - `tr_produksi.csv`
    - `ms_mesin.csv`
    - `ms_operator.csv`
    - `tr_maintenance.csv`

    Lalu jalankan ulang: `streamlit run dashboard_produksi.py`
    """)
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
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 KPI Utama",
    "🔍 Analisis Kualitas",
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
    avg_rpm        = df["Setting_Speed_RPM"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Produksi",    f"{total_produksi:,.0f}")
    c2.metric("Total Qty OK",      f"{total_ok:,.0f}",  f"+{total_ok/total_produksi*100:.1f}%" if total_produksi else "–")
    c3.metric("Total Qty NG",      f"{total_ng:,.0f}")
    c4.metric("Reject Rate",       f"{reject_rate:.2f}%", delta_color="inverse")
    c5.metric("Avg Suhu Mesin",    f"{avg_suhu:.1f} °C")

    st.markdown("<div class='section-header'>Tren Harian</div>", unsafe_allow_html=True)

    daily = df.groupby("Tanggal")[["Qty_OK", "Qty_NG"]].sum().reset_index()
    daily["Reject_%"] = (daily["Qty_NG"] / (daily["Qty_OK"] + daily["Qty_NG"])) * 100

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Bar(x=daily["Tanggal"], y=daily["Qty_OK"], name="Qty OK",
                                marker_color="#42a5f5", opacity=0.8), secondary_y=False)
    fig_trend.add_trace(go.Bar(x=daily["Tanggal"], y=daily["Qty_NG"], name="Qty NG",
                                marker_color="#ef5350", opacity=0.8), secondary_y=False)
    fig_trend.add_trace(go.Scatter(x=daily["Tanggal"], y=daily["Reject_%"], name="Reject %",
                                    line=dict(color="#ffca28", width=2), mode="lines+markers"), secondary_y=True)
    fig_trend.update_layout(barmode="stack", title="Produksi Harian & Reject Rate",
                             yaxis2=dict(title="Reject %", gridcolor="#1e2130", ticksuffix="%"),
                             **DARK_LAYOUT)
    fig_trend.update_yaxes(title_text="Jumlah Unit", secondary_y=False)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<div class='section-header'>Performa per Line & Shift</div>", unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        line_g = df.groupby("Line_Produksi")[["Qty_OK", "Qty_NG"]].sum().reset_index()
        line_g["Reject_%"] = line_g["Qty_NG"] / (line_g["Qty_OK"] + line_g["Qty_NG"]) * 100
        fig_line = px.bar(line_g, x="Line_Produksi", y="Reject_%",
                           color="Reject_%", color_continuous_scale="RdYlGn_r",
                           title="Reject Rate per Line Produksi", labels={"Reject_%": "Reject %"})
        fig_line.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_line, use_container_width=True)

    with colB:
        shift_g = df.groupby("Shift")[["Qty_OK", "Qty_NG"]].sum().reset_index()
        shift_g["Reject_%"] = shift_g["Qty_NG"] / (shift_g["Qty_OK"] + shift_g["Qty_NG"]) * 100
        fig_shift = px.pie(shift_g, names="Shift", values="Qty_NG",
                            title="Distribusi NG per Shift", hole=0.45,
                            color_discrete_sequence=px.colors.sequential.Magma)
        fig_shift.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_shift, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 – ANALISIS KUALITAS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Uji Chi-Square: Shift vs Kualitas</div>", unsafe_allow_html=True)

    shift_quality = df.groupby("Shift")[["Qty_OK", "Qty_NG"]].sum()
    if len(shift_quality) >= 2 and shift_quality.values.sum() > 0:
        try:
            chi2_stat, p_val, dof, _ = chi2_contingency(shift_quality.values)
            sig = "Signifikan ✅ (reject rate bergantung pada shift)" if p_val < 0.05 else "Tidak Signifikan ❌ (tidak ada perbedaan nyata antar shift)"
            st.markdown(f"""
            <div class='stat-badge'>
                📐 <strong>Chi-Square Statistic:</strong> {chi2_stat:.4f} &nbsp;|&nbsp;
                <strong>P-Value:</strong> {p_val:.4e} &nbsp;|&nbsp;
                <strong>Derajat Kebebasan:</strong> {dof}<br>
                🔎 Kesimpulan: <strong>{sig}</strong>
            </div>""", unsafe_allow_html=True)
        except Exception:
            st.info("Data tidak cukup untuk uji Chi-Square dengan filter saat ini.")
    else:
        st.info("Minimal 2 shift diperlukan untuk uji Chi-Square.")

    shift_quality_reset = shift_quality.reset_index()
    shift_quality_reset["Reject_%"] = (
        shift_quality_reset["Qty_NG"] /
        (shift_quality_reset["Qty_OK"] + shift_quality_reset["Qty_NG"]) * 100
    )
    avg_reject = shift_quality_reset["Reject_%"].mean()

    colC, colD = st.columns(2)
    with colC:
        fig_bar = px.bar(shift_quality_reset, x="Shift", y="Reject_%",
                          color="Shift", title="Defect Rate (%) per Shift",
                          color_discrete_sequence=px.colors.sequential.Magma)
        fig_bar.add_hline(y=avg_reject, line_dash="dash", line_color="#ef5350",
                           annotation_text=f"Rata-rata Pabrik: {avg_reject:.2f}%")
        fig_bar.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_bar, use_container_width=True)

    with colD:
        fig_scatter = px.scatter(
            df, x="Setting_Speed_RPM", y="Qty_NG",
            color="Skill_Level", size="Suhu_Mesin_Celsius",
            opacity=0.6, color_continuous_scale="viridis",
            title="Kecepatan Mesin vs Defek (per Skill & Suhu)",
            labels={"Setting_Speed_RPM": "Speed RPM", "Qty_NG": "Qty NG", "Skill_Level": "Skill Level"},
        )
        fig_scatter.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<div class='section-header'>Distribusi Suhu Mesin per Line</div>", unsafe_allow_html=True)
    fig_box = px.box(df, x="Line_Produksi", y="Suhu_Mesin_Celsius", color="Shift",
                      title="Distribusi Suhu Mesin", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 – PREDIKSI KERUSAKAN MESIN
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Random Forest – Predictive Maintenance</div>", unsafe_allow_html=True)

    try:
        model, report, cm, importances, df_ml = train_ml_model(df_clean)

        # Confusion Matrix
        colE, colF = st.columns(2)
        with colE:
            fig_cm = px.imshow(cm, text_auto=True,
                                x=["Normal", "Breakdown"], y=["Normal", "Breakdown"],
                                color_continuous_scale="Blues",
                                title="Confusion Matrix",
                                labels={"x": "Prediksi", "y": "Aktual"})
            fig_cm.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_cm, use_container_width=True)

        with colF:
            fig_imp = px.bar(importances.reset_index(), x="index", y=importances.values,
                              title="Feature Importance – Penyebab Breakdown",
                              labels={"index": "Fitur", "y": "Importance"},
                              color=importances.values, color_continuous_scale="Viridis")
            fig_imp.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Model report
        st.markdown("<div class='section-header'>Laporan Evaluasi Model</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='stat-badge'>
            🎯 <strong>Accuracy:</strong> {report['accuracy']:.4f} &nbsp;|&nbsp;
            <strong>Precision (Breakdown):</strong> {report.get('1', {}).get('precision', 0):.4f} &nbsp;|&nbsp;
            <strong>Recall (Breakdown):</strong> {report.get('1', {}).get('recall', 0):.4f} &nbsp;|&nbsp;
            <strong>F1-Score:</strong> {report.get('1', {}).get('f1-score', 0):.4f}
        </div>""", unsafe_allow_html=True)

        # Simulasi prediksi
        st.markdown("<div class='section-header'>Simulasi Prediksi Breakdown</div>", unsafe_allow_html=True)
        colG, colH, colI = st.columns(3)
        with colG:
            sim_rpm  = st.slider("Setting Speed RPM", 500, 5000, 2000, step=50)
        with colH:
            sim_suhu = st.slider("Suhu Mesin (°C)", 50, 250, 120, step=5)
        with colI:
            sim_skill = st.slider("Skill Level Operator", 1, 5, 3)

        pred = model.predict([[sim_rpm, sim_suhu, sim_skill]])[0]
        prob = model.predict_proba([[sim_rpm, sim_suhu, sim_skill]])[0]

        if pred == 1:
            st.error(f"⚠️ **PREDIKSI: RISIKO BREAKDOWN** — Probabilitas: {prob[1]*100:.1f}%")
        else:
            st.success(f"✅ **PREDIKSI: KONDISI NORMAL** — Probabilitas normal: {prob[0]*100:.1f}%")

    except FileNotFoundError:
        st.warning("File `tr_maintenance.csv` tidak ditemukan. Tab ini memerlukan file tersebut.")
    except Exception as e:
        st.error(f"Terjadi error saat melatih model: {e}")


# ══════════════════════════════════════════════
# TAB 4 – DATA MENTAH
# ══════════════════════════════════════════════
with tab4:
    st.markdown(f"### Data Bersih ({len(df):,} baris setelah filter)")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh CSV",
        data=csv_bytes,
        file_name="data_produksi_bersih.csv",
        mime="text/csv",
    )

    with st.expander("📊 Statistik Deskriptif"):
        st.dataframe(df.describe(), use_container_width=True)
