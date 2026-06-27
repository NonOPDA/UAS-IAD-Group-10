"""
===========================================================
DASHBOARD ANALITIK PRODUKSI
Project UAS - Intelligent Analytics Dashboard

Fase 1 : Data Acquisition & Data Cleaning
Fase 2 : Statistical Analysis & Executive Dashboard
Fase 3 : Machine Learning Prediction

Author : Nama Anda
===========================================================
"""

# =========================================================
# BAGIAN 1 - PERSIAPAN I
# IMPORT LIBRARY
# =========================================================

# Dashboard
import streamlit as st

# Data Processing
import pandas as pd
import numpy as np
import sqlite3

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical Analysis
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    zscore
)

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Model Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# Data Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# BAGIAN 2 - PERSIAPAN II
# KONFIGURASI DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Dashboard Analitik Produksi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#003366;
}

h2{
    color:#0b5394;
}

div[data-testid="metric-container"]{
    background:white;
    border:1px solid #d9d9d9;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Konstanta
# ---------------------------------------------------------

TEMPERATURE_MIN = 20
TEMPERATURE_MAX = 200

RPM_MIN = 500
RPM_MAX = 3000

OUTLIER_METHOD = "IQR"

# ---------------------------------------------------------
# Helper Function
# ---------------------------------------------------------

def calculate_reject_rate(ok, ng):
    total = ok + ng
    if total == 0:
        return 0
    return (ng / total) * 100


def calculate_quality(ok, ng):
    total = ok + ng
    if total == 0:
        return 0
    return ok / total


def calculate_availability(runtime, planned):
    if planned == 0:
        return 0
    return runtime / planned


def calculate_performance(actual_output, ideal_output):
    if ideal_output == 0:
        return 0
    return actual_output / ideal_output


def calculate_oee(availability, performance, quality):
    return availability * performance * quality
# =========================================================
# BAGIAN 3 - FASE 1
# DATA ACQUISITION
# =========================================================

@st.cache_data(show_spinner="Membaca dataset...")
def load_raw_data():
    """
    Membaca seluruh dataset mentah.
    """

    produksi = pd.read_csv("tr_produksi.csv")
    mesin = pd.read_csv("ms_mesin.csv")
    operator = pd.read_csv("ms_operator.csv")
    material = pd.read_csv("ms_material.csv")
    maintenance = pd.read_csv("tr_maintenance.csv")

    return (
        produksi,
        mesin,
        operator,
        material,
        maintenance
    )


# ---------------------------------------------------------
# SQL JOIN
# ---------------------------------------------------------

@st.cache_data(show_spinner="Menggabungkan data menggunakan SQL JOIN...")
def sql_join_data(
    produksi,
    mesin,
    operator,
    material
):
    """
    Menggabungkan data produksi dengan seluruh master data
    menggunakan SQL JOIN.
    """

    conn = sqlite3.connect(":memory:")

    produksi.to_sql(
        "tr_produksi",
        conn,
        index=False,
        if_exists="replace"
    )

    mesin.to_sql(
        "ms_mesin",
        conn,
        index=False,
        if_exists="replace"
    )

    operator.to_sql(
        "ms_operator",
        conn,
        index=False,
        if_exists="replace"
    )

    material.to_sql(
        "ms_material",
        conn,
        index=False,
        if_exists="replace"
    )

    query = """
    SELECT

        tp.*,

        mm.Nama_Mesin,
        mm.Line_Produksi,

        op.Nama_Operator,
        op.Skill_Level,

        mt.Nama_Material,
        mt.Jenis AS Jenis_Material

    FROM tr_produksi tp

    LEFT JOIN ms_mesin mm
        ON tp.ID_Mesin = mm.ID_Mesin

    LEFT JOIN ms_operator op
        ON tp.ID_Operator = op.ID_Operator

    LEFT JOIN ms_material mt
        ON tp.ID_Material = mt.ID_Material

    ORDER BY
        tp.Tanggal,
        tp.ID_Produksi

    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df
# =========================================================
# BAGIAN 4 - FASE 1
# DATA CLEANING
# =========================================================

@st.cache_data(show_spinner="Membersihkan data...")
def clean_data(df):
    """
    Membersihkan data produksi hasil SQL JOIN.
    """

    report = {}

    # --------------------------------------------
    # Data sebelum cleaning
    # --------------------------------------------

    report["Jumlah Data Awal"] = len(df)

    # --------------------------------------------
    # Menghapus duplikasi
    # --------------------------------------------

    duplicate = df.duplicated().sum()

    report["Duplicate"] = duplicate

    df = df.drop_duplicates()

    # --------------------------------------------
    # Missing Value
    # --------------------------------------------

    report["Missing Value"] = df.isnull().sum().sum()

    # Kolom numerik

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    # Kolom kategori

    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:
        df[col] = df[col].fillna("UNKNOWN")

    # --------------------------------------------
    # Konversi Tipe Data
    # --------------------------------------------

    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(
    df["Tanggal"],
    errors="coerce"
)

    numeric_convert = [

        "Qty_OK",

        "Qty_NG",

        "Setting_Speed_RPM",

        "Suhu_Mesin_Celsius",

        "Skill_Level"

    ]

    for col in numeric_convert:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------
    # Validasi Suhu Mesin
    # --------------------------------------------

    invalid_temperature = (

        (df["Suhu_Mesin_Celsius"] < TEMPERATURE_MIN)

        |

        (df["Suhu_Mesin_Celsius"] > TEMPERATURE_MAX)

    ).sum()

    report["Invalid Temperature"] = invalid_temperature

    df.loc[
        df["Suhu_Mesin_Celsius"] < TEMPERATURE_MIN,
        "Suhu_Mesin_Celsius"
    ] = np.nan

    df.loc[
        df["Suhu_Mesin_Celsius"] > TEMPERATURE_MAX,
        "Suhu_Mesin_Celsius"
    ] = np.nan

    df["Suhu_Mesin_Celsius"] = df[
        "Suhu_Mesin_Celsius"
    ].fillna(
        df["Suhu_Mesin_Celsius"].median()
    )

    # --------------------------------------------
    # Validasi RPM
    # --------------------------------------------

    invalid_rpm = (

        (df["Setting_Speed_RPM"] < RPM_MIN)

        |

        (df["Setting_Speed_RPM"] > RPM_MAX)

    ).sum()

    report["Invalid RPM"] = invalid_rpm

    df.loc[
        df["Setting_Speed_RPM"] < RPM_MIN,
        "Setting_Speed_RPM"
    ] = np.nan

    df.loc[
        df["Setting_Speed_RPM"] > RPM_MAX,
        "Setting_Speed_RPM"
    ] = np.nan

    df["Setting_Speed_RPM"] = df[
        "Setting_Speed_RPM"
    ].fillna(
        df["Setting_Speed_RPM"].median()
    )

    # --------------------------------------------
    # Outlier (IQR)
    # --------------------------------------------

    outlier_total = 0

    outlier_columns = [

        "Qty_OK",

        "Qty_NG",

        "Setting_Speed_RPM",

        "Suhu_Mesin_Celsius"

    ]

    for col in outlier_columns:

        if col in df.columns:

            q1 = df[col].quantile(0.25)

            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outlier = (

                (df[col] < lower)

                |

                (df[col] > upper)

            )

            outlier_total += outlier.sum()

            df.loc[outlier, col] = df[col].median()

    report["Outlier"] = outlier_total

    # --------------------------------------------
    # Data sesudah cleaning
    # --------------------------------------------

    report["Jumlah Data Akhir"] = len(df)

    return df, report
# =========================================================
# BAGIAN 5 - FASE 1
# DATA QUALITY REPORT & ETL PIPELINE
# =========================================================

def data_quality_report(report):
    """
    Mengubah hasil cleaning menjadi DataFrame agar
    mudah ditampilkan pada dashboard.
    """

    report_df = pd.DataFrame({

        "Parameter": list(report.keys()),

        "Hasil": list(report.values())

    })

    return report_df


# ---------------------------------------------------------
# Menyimpan hasil cleaning
# ---------------------------------------------------------

def save_clean_data(df):
    """
    Menyimpan data hasil cleaning.
    """

    df.to_csv(
        "pt_andalas_cleaned.csv",
        index=False
    )


# ---------------------------------------------------------
# Pipeline ETL
# ---------------------------------------------------------

@st.cache_data(show_spinner="Menjalankan ETL...")
def load_and_clean_data():

    (
        produksi,
        mesin,
        operator,
        material,
        maintenance

    ) = load_raw_data()

    df = sql_join_data(

        produksi,

        mesin,

        operator,

        material

    )

    df_clean, report = clean_data(df)

    save_clean_data(df_clean)

    report_df = data_quality_report(report)

    return (

        df_clean,

        maintenance,

        report_df

    )
# =========================================================
# BAGIAN 6 - FASE 2
# SIDEBAR & FILTER DATA
# =========================================================

# Menjalankan proses ETL

try:

    df_clean, maintenance, quality_report = load_and_clean_data()

    data_ok = True

except Exception as e:

    data_ok = False

    error_message = str(e)


# ---------------------------------------------------------
# Error Handling
# ---------------------------------------------------------

if not data_ok:

    st.error("Dataset tidak dapat dimuat.")

    st.code(error_message)

    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("🏭 Dashboard Produksi")

    st.markdown("---")

    st.subheader("Filter Data")


    # ----------------------------------------
    # Filter Shift
    # ----------------------------------------

    daftar_shift = ["Semua"] + sorted(

        df_clean["Shift"].dropna().unique()

    )

    shift = st.selectbox(

        "Shift",

        daftar_shift

    )


    # ----------------------------------------
    # Filter Line Produksi
    # ----------------------------------------

    daftar_line = ["Semua"] + sorted(

        df_clean["Line_Produksi"].dropna().unique()

    )

    line = st.selectbox(

        "Line Produksi",

        daftar_line

    )


    # ----------------------------------------
    # Filter Mesin
    # ----------------------------------------

    daftar_mesin = ["Semua"] + sorted(

        df_clean["Nama_Mesin"].dropna().unique()

    )

    mesin = st.selectbox(

        "Mesin",

        daftar_mesin

    )


    # ----------------------------------------
    # Filter Operator
    # ----------------------------------------

    daftar_operator = ["Semua"] + sorted(

        df_clean["Nama_Operator"].dropna().unique()

    )

    operator = st.selectbox(

        "Operator",

        daftar_operator

    )


    # ----------------------------------------
    # Filter Tanggal
    # ----------------------------------------

    tanggal_awal = df_clean["Tanggal"].min()

    tanggal_akhir = df_clean["Tanggal"].max()

    tanggal = st.date_input(

        "Rentang Tanggal",

        value=(

            tanggal_awal,

            tanggal_akhir

        )

    )

    st.markdown("---")

    st.subheader("Data Quality")

    st.dataframe(

        quality_report,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")

    st.caption(

        f"Jumlah Data : {len(df_clean):,}"

    )
# =========================================================
# FILTER DATAFRAME
# =========================================================

df = df_clean.copy()


if shift != "Semua":

    df = df[

        df["Shift"] == shift

    ]


if line != "Semua":

    df = df[

        df["Line_Produksi"] == line

    ]


if mesin != "Semua":

    df = df[

        df["Nama_Mesin"] == mesin

    ]


if operator != "Semua":

    df = df[

        df["Nama_Operator"] == operator

    ]


if len(tanggal) == 2:

    df = df[

        (df["Tanggal"] >= pd.to_datetime(tanggal[0]))

        &

        (df["Tanggal"] <= pd.to_datetime(tanggal[1]))

    ]
# =========================================================
# BAGIAN 7 - FASE 2
# EXECUTIVE KPI DASHBOARD
# =========================================================

st.title("🏭 Dashboard Analitik Produksi")

st.markdown(
    """
Dashboard untuk monitoring produksi, kualitas, dan prediksi
kerusakan mesin.
"""
)

st.markdown("---")


# ---------------------------------------------------------
# KPI
# ---------------------------------------------------------

total_ok = df["Qty_OK"].sum()

total_ng = df["Qty_NG"].sum()

total_produksi = total_ok + total_ng


reject_rate = calculate_reject_rate(

    total_ok,

    total_ng

)


quality = calculate_quality(

    total_ok,

    total_ng

)


# Jika belum tersedia pada dataset,
# gunakan nilai default

availability = 0.95

performance = 0.90


oee = calculate_oee(

    availability,

    performance,

    quality

)


avg_rpm = df["Setting_Speed_RPM"].mean()

avg_temp = df["Suhu_Mesin_Celsius"].mean()


jumlah_mesin = df["ID_Mesin"].nunique()

jumlah_operator = df["ID_Operator"].nunique()


# ---------------------------------------------------------
# KPI ROW 1
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(

    "Total Produksi",

    f"{total_produksi:,.0f}"

)

col2.metric(

    "Qty OK",

    f"{total_ok:,.0f}"

)

col3.metric(

    "Qty NG",

    f"{total_ng:,.0f}"

)

col4.metric(

    "Reject Rate",

    f"{reject_rate:.2f}%"

)


# ---------------------------------------------------------
# KPI ROW 2
# ---------------------------------------------------------

col5, col6, col7, col8 = st.columns(4)

col5.metric(

    "OEE",

    f"{oee*100:.2f}%"

)

col6.metric(

    "Average RPM",

    f"{avg_rpm:.0f}"

)

col7.metric(

    "Average Temperature",

    f"{avg_temp:.1f} °C"

)

col8.metric(

    "Machine",

    jumlah_mesin

)


# ---------------------------------------------------------
# KPI ROW 3
# ---------------------------------------------------------

col9, col10 = st.columns(2)

col9.metric(

    "Operator",

    jumlah_operator

)

col10.metric(

    "Quality",

    f"{quality*100:.2f}%"

)

st.markdown("---")
# =========================================================
# BAGIAN 8 - FASE 2
# EXECUTIVE DASHBOARD
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📊 Executive Dashboard",

    "⚙️ Bottleneck Stamping",

    "🏭 Shift B Investigation",

    "🤖 Machine Learning",

    "📄 Raw Data"

])


# =========================================================
# TAB 1
# EXECUTIVE DASHBOARD
# =========================================================

with tab1:

    st.subheader("Executive Dashboard")

    # -----------------------------------------------------
    # Trend Produksi
    # -----------------------------------------------------

    produksi_harian = (

        df.groupby("Tanggal")[

            ["Qty_OK", "Qty_NG"]

        ]

        .sum()

        .reset_index()

    )

    fig = px.line(

        produksi_harian,

        x="Tanggal",

        y=["Qty_OK", "Qty_NG"],

        markers=True,

        title="Trend Produksi Harian"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # -----------------------------------------------------
    # Reject Rate per Shift
    # -----------------------------------------------------

    shift_summary = (

        df.groupby("Shift")[

            ["Qty_OK", "Qty_NG"]

        ]

        .sum()

        .reset_index()

    )

    shift_summary["Reject Rate"] = (

        shift_summary["Qty_NG"]

        /

        (

            shift_summary["Qty_OK"]

            +

            shift_summary["Qty_NG"]

        )

    ) * 100

    fig = px.bar(

        shift_summary,

        x="Shift",

        y="Reject Rate",

        text="Reject Rate",

        color="Reject Rate",

        title="Reject Rate per Shift"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # -----------------------------------------------------
    # Produksi per Mesin
    # -----------------------------------------------------

    mesin_summary = (

        df.groupby("Nama_Mesin")[

            ["Qty_OK", "Qty_NG"]

        ]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        mesin_summary,

        x="Nama_Mesin",

        y=["Qty_OK", "Qty_NG"],

        barmode="group",

        title="Output Produksi per Mesin"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # -----------------------------------------------------
    # Distribusi RPM
    # -----------------------------------------------------

    fig = px.histogram(

        df,

        x="Setting_Speed_RPM",

        nbins=30,

        title="Distribusi Kecepatan Mesin (RPM)"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # -----------------------------------------------------
    # Distribusi Suhu
    # -----------------------------------------------------

    fig = px.box(

        df,

        y="Suhu_Mesin_Celsius",

        color="Shift",

        title="Distribusi Suhu Mesin"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # -----------------------------------------------------
    # Executive Insight
    # -----------------------------------------------------

    st.subheader("Executive Insight")

    mesin_ng = (

        df.groupby("Nama_Mesin")["Qty_NG"]

        .sum()

        .idxmax()

    )

    shift_ng = (

        df.groupby("Shift")["Qty_NG"]

        .sum()

        .idxmax()

    )

    operator_ng = (

        df.groupby("Nama_Operator")["Qty_NG"]

        .sum()

        .idxmax()

    )

    st.info(f"""

• Mesin dengan reject tertinggi : **{mesin_ng}**

• Shift dengan reject tertinggi : **{shift_ng}**

• Operator dengan reject tertinggi : **{operator_ng}**

• Reject Rate saat ini : **{reject_rate:.2f}%**

• Quality Rate : **{quality*100:.2f}%**

• OEE : **{oee*100:.2f}%**

""")
# =========================================================
# BAGIAN 9 - FASE 2
# BOTTLENECK ANALYSIS (STAMPING)
# =========================================================

with tab2:

    st.subheader("Analisis Bottleneck Stasiun Stamping")

    # -----------------------------------------------------
    # Filter Stamping
    # -----------------------------------------------------

    if "Line_Produksi" in df.columns:

        df_stamping = df[
            df["Line_Produksi"].str.upper() == "STAMPING"
        ].copy()

    else:

        df_stamping = df.copy()

        st.warning(
            "Kolom Line_Produksi tidak ditemukan. Seluruh data digunakan."
        )


    if df_stamping.empty:

        st.warning("Data Stamping tidak tersedia.")

    else:

        # -------------------------------------------------
        # KPI
        # -------------------------------------------------

        total_ok_stamp = df_stamping["Qty_OK"].sum()

        total_ng_stamp = df_stamping["Qty_NG"].sum()

        reject_stamp = calculate_reject_rate(
            total_ok_stamp,
            total_ng_stamp
        )

        mesin_stamp = df_stamping["Nama_Mesin"].nunique()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Produksi Stamping",
            f"{total_ok_stamp + total_ng_stamp:,.0f}"
        )

        c2.metric(
            "Reject Rate",
            f"{reject_stamp:.2f}%"
        )

        c3.metric(
            "Jumlah Mesin",
            mesin_stamp
        )


        # -------------------------------------------------
        # Output Mesin
        # -------------------------------------------------

        mesin_summary = (

            df_stamping

            .groupby("Nama_Mesin")[

                ["Qty_OK", "Qty_NG"]

            ]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            mesin_summary,

            x="Nama_Mesin",

            y=["Qty_OK", "Qty_NG"],

            barmode="group",

            title="Output Produksi Mesin Stamping"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------------------------------
        # Pareto Defect
        # -------------------------------------------------

        pareto = (

            df_stamping

            .groupby("Nama_Mesin")["Qty_NG"]

            .sum()

            .sort_values(ascending=False)

            .reset_index()

        )

        pareto["Persen"] = (

            pareto["Qty_NG"]

            /

            pareto["Qty_NG"].sum()

        ) * 100

        pareto["Kumulatif"] = pareto["Persen"].cumsum()

        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        fig.add_bar(

            x=pareto["Nama_Mesin"],

            y=pareto["Qty_NG"],

            name="Reject"

        )

        fig.add_scatter(

            x=pareto["Nama_Mesin"],

            y=pareto["Kumulatif"],

            mode="lines+markers",

            name="Kumulatif",

            secondary_y=True

        )

        fig.update_layout(
            title="Pareto Defect Mesin Stamping"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------------------------------
        # Heatmap Mesin vs Shift
        # -------------------------------------------------

        pivot = pd.pivot_table(

            df_stamping,

            values="Qty_NG",

            index="Nama_Mesin",

            columns="Shift",

            aggfunc="sum",

            fill_value=0

        )

        fig = px.imshow(

            pivot,

            text_auto=True,

            aspect="auto",

            title="Heatmap Reject Mesin vs Shift"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------------------------------
        # Downtime
        # -------------------------------------------------

        if not maintenance.empty:

            downtime = (

                maintenance

                .groupby("ID_Mesin")["Durasi_Jam"]

                .sum()

                .reset_index()

            )

            downtime = downtime.merge(

                df_stamping[
                    ["ID_Mesin", "Nama_Mesin"]
                ].drop_duplicates(),

                on="ID_Mesin",

                how="left"

            )

            fig = px.bar(

                downtime,

                x="Nama_Mesin",

                y="Durasi_Jam",

                color="Durasi_Jam",

                title="Total Downtime Mesin"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # -------------------------------------------------
        # Root Cause
        # -------------------------------------------------

        mesin_terburuk = (

            pareto.iloc[0]["Nama_Mesin"]

        )

        reject_mesin = (

            pareto.iloc[0]["Qty_NG"]

        )

        st.subheader("Root Cause Analysis")

        st.success(f"""

Mesin dengan reject tertinggi adalah **{mesin_terburuk}**.

Jumlah produk cacat yang dihasilkan sebanyak **{reject_mesin:,.0f} unit**.

Mesin ini menjadi kandidat utama bottleneck karena memiliki kontribusi cacat terbesar pada proses Stamping.

Rekomendasi awal:

• Prioritaskan preventive maintenance.

• Evaluasi parameter RPM dan suhu operasi.

• Periksa kondisi tooling dan dies.

• Evaluasi kompetensi operator pada mesin tersebut.

""")
# =========================================================
# BAGIAN 10 - FASE 2
# SHIFT B INVESTIGATION
# =========================================================

with tab3:

    st.subheader("Investigasi Anomali Produk Cacat - Shift B")

    # -----------------------------------------------------
    # Filter Shift B
    # -----------------------------------------------------

    df_shift_b = df[
        df["Shift"].astype(str).str.upper() == "B"
    ].copy()

    if df_shift_b.empty:

        st.warning("Data Shift B tidak tersedia.")

    else:

        # -------------------------------------------------
        # KPI Shift B
        # -------------------------------------------------

        total_ok_b = df_shift_b["Qty_OK"].sum()

        total_ng_b = df_shift_b["Qty_NG"].sum()

        reject_b = calculate_reject_rate(
            total_ok_b,
            total_ng_b
        )

        avg_rpm_b = df_shift_b["Setting_Speed_RPM"].mean()

        avg_temp_b = df_shift_b["Suhu_Mesin_Celsius"].mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Produksi",
            f"{total_ok_b + total_ng_b:,.0f}"
        )

        c2.metric(
            "Reject Rate",
            f"{reject_b:.2f}%"
        )

        c3.metric(
            "Average RPM",
            f"{avg_rpm_b:.0f}"
        )

        c4.metric(
            "Average Temperature",
            f"{avg_temp_b:.1f} °C"
        )

        # -------------------------------------------------
        # Trend Produksi Shift B
        # -------------------------------------------------

        trend = (

            df_shift_b

            .groupby("Tanggal")[

                ["Qty_OK", "Qty_NG"]

            ]

            .sum()

            .reset_index()

        )

        fig = px.line(

            trend,

            x="Tanggal",

            y=["Qty_OK", "Qty_NG"],

            markers=True,

            title="Trend Produksi Shift B"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Scatter RPM vs Reject
        # -------------------------------------------------

        fig = px.scatter(

            df_shift_b,

            x="Setting_Speed_RPM",

            y="Qty_NG",

            color="Nama_Mesin",

            hover_data=["Nama_Operator"],

            title="RPM vs Produk Cacat"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Scatter Temperature vs Reject
        # -------------------------------------------------

        fig = px.scatter(

            df_shift_b,

            x="Suhu_Mesin_Celsius",

            y="Qty_NG",

            color="Nama_Mesin",

            hover_data=["Nama_Operator"],

            title="Temperatur vs Produk Cacat"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Boxplot RPM
        # -------------------------------------------------

        fig = px.box(

            df_shift_b,

            x="Nama_Mesin",

            y="Setting_Speed_RPM",

            color="Nama_Mesin",

            title="Distribusi RPM per Mesin"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Boxplot Temperatur
        # -------------------------------------------------

        fig = px.box(

            df_shift_b,

            x="Nama_Mesin",

            y="Suhu_Mesin_Celsius",

            color="Nama_Mesin",

            title="Distribusi Temperatur per Mesin"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Korelasi
        # -------------------------------------------------

        st.subheader("Correlation Matrix")

        corr = df_shift_b[

            [

                "Qty_OK",

                "Qty_NG",

                "Setting_Speed_RPM",

                "Suhu_Mesin_Celsius"

            ]

        ].corr()

        fig = px.imshow(

            corr,

            text_auto=True,

            color_continuous_scale="RdBu_r",

            title="Korelasi Variabel"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Operator dengan Reject Terbesar
        # -------------------------------------------------

        operator_summary = (

            df_shift_b

            .groupby("Nama_Operator")["Qty_NG"]

            .sum()

            .sort_values(ascending=False)

            .reset_index()

        )

        fig = px.bar(

            operator_summary,

            x="Nama_Operator",

            y="Qty_NG",

            color="Qty_NG",

            title="Reject per Operator"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Mesin dengan Reject Terbesar
        # -------------------------------------------------

        mesin_summary = (

            df_shift_b

            .groupby("Nama_Mesin")["Qty_NG"]

            .sum()

            .sort_values(ascending=False)

            .reset_index()

        )

        fig = px.bar(

            mesin_summary,

            x="Nama_Mesin",

            y="Qty_NG",

            color="Qty_NG",

            title="Reject per Mesin"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # Executive Insight
        # -------------------------------------------------

        operator_terburuk = operator_summary.iloc[0]["Nama_Operator"]

        mesin_terburuk = mesin_summary.iloc[0]["Nama_Mesin"]

        st.subheader("Executive Insight")

        st.info(f"""

Shift B menunjukkan reject rate sebesar **{reject_b:.2f}%**.

Operator dengan reject tertinggi adalah **{operator_terburuk}**.

Mesin dengan reject tertinggi adalah **{mesin_terburuk}**.

Berdasarkan visualisasi RPM, temperatur, dan distribusi reject, investigasi lanjutan perlu difokuskan pada parameter operasi mesin serta kompetensi operator pada Shift B.

""")
# =========================================================
# BAGIAN 11 - FASE 2
# ANALISIS STATISTIK
# =========================================================

with tab3:

    st.markdown("---")

    st.subheader("Analisis Statistik")

    # -----------------------------------------------------
    # CHI-SQUARE
    # Shift vs Skill Level
    # -----------------------------------------------------

    st.markdown("### 1. Uji Chi-Square")

    if "Skill_Level" in df_shift_b.columns:

        contingency = pd.crosstab(

            df_shift_b["Shift"],

            df_shift_b["Skill_Level"]

        )

        chi2, p_value, dof, expected = chi2_contingency(
            contingency
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Chi-Square",
            f"{chi2:.3f}"
        )

        c2.metric(
            "p-value",
            f"{p_value:.4f}"
        )

        if p_value < 0.05:

            st.success(
                "Terdapat hubungan yang signifikan (p < 0.05)."
            )

        else:

            st.info(
                "Tidak terdapat hubungan yang signifikan (p ≥ 0.05)."
            )

    else:

        st.warning(
            "Kolom Skill_Level tidak ditemukan."
        )


    # -----------------------------------------------------
    # ANOVA
    # RPM berdasarkan Mesin
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("### 2. Uji ANOVA")

    kelompok = []

    for nama_mesin in df_shift_b["Nama_Mesin"].unique():

        kelompok.append(

            df_shift_b[
                df_shift_b["Nama_Mesin"] == nama_mesin
            ]["Setting_Speed_RPM"]

        )

    if len(kelompok) >= 2:

        f_stat, p_value = f_oneway(*kelompok)

        c1, c2 = st.columns(2)

        c1.metric(
            "F Statistic",
            f"{f_stat:.3f}"
        )

        c2.metric(
            "p-value",
            f"{p_value:.4f}"
        )

        if p_value < 0.05:

            st.success(
                "Terdapat perbedaan RPM yang signifikan antar mesin."
            )

        else:

            st.info(
                "Tidak terdapat perbedaan RPM yang signifikan."
            )


    # -----------------------------------------------------
    # KRUSKAL WALLIS
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("### 3. Uji Kruskal-Wallis")

    if len(kelompok) >= 2:

        h_stat, p_value = kruskal(*kelompok)

        c1, c2 = st.columns(2)

        c1.metric(
            "H Statistic",
            f"{h_stat:.3f}"
        )

        c2.metric(
            "p-value",
            f"{p_value:.4f}"
        )

        if p_value < 0.05:

            st.success(
                "Distribusi RPM berbeda secara signifikan."
            )

        else:

            st.info(
                "Distribusi RPM tidak berbeda secara signifikan."
            )


    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("### 4. Korelasi")

    correlation = df_shift_b[
        [
            "Qty_OK",
            "Qty_NG",
            "Setting_Speed_RPM",
            "Suhu_Mesin_Celsius"
        ]
    ].corr()

    st.dataframe(
        correlation.round(3),
        use_container_width=True
    )


    # -----------------------------------------------------
    # RINGKASAN
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("Kesimpulan Statistik")

    st.info("""

Analisis statistik digunakan untuk memvalidasi apakah hubungan atau
perbedaan yang terlihat pada visualisasi memang signifikan secara statistik.

Interpretasi dilakukan berdasarkan nilai p-value:

• p-value < 0.05 → signifikan

• p-value ≥ 0.05 → tidak signifikan

Hasil analisis ini menjadi dasar dalam menyusun rekomendasi pada fase
Machine Learning dan Executive Summary.

""")
