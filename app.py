"""
================================================================================
FINAL PROJECT: SISTEM PENDUKUNG KEPUTUSAN (DECISION SUPPORT SYSTEM)

JUDUL APLIKASI: DSS Loyalty: Churn Prediction & Win-Back Strategy Optimizer

Deskripsi Proyek:
Sistem Pendukung Keputusan (DSS) berbasis Data Mining untuk memprediksi loyalitas 
pelanggan (churn risk) pada platform E-Commerce. Hasil prediksi digunakan untuk 
membuat segmentasi strategis (Value vs. Risk) dan memberikan rekomendasi tindakan 
(prescriptive action) yang dioptimalkan dengan simulasi Return on Investment (ROI).

Disusun Oleh:
Adelia Felisha – 140810230003
Nikita Putri Prabowo – 140810230010

Program Studi: S1 Teknik Informatika
Universitas: Universitas Padjadjaran

Tanggal Pengumpulan: 9 November 2025
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from io import BytesIO
from scipy import stats 

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder, 
    RobustScaler 
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib


# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------

st.set_page_config(
    page_title="DSS Loyalitas - Marketing Intelligence", 
    page_icon="📊", 
    layout="wide"
)

# CSS Executive + Hero + Tabs
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #e0ecff 0, #f5f7fa 35%, #e8eef5 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* --------- HEADERS --------- */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }

    h1 {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4em !important;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 0.5rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(37,99,235,0.08);
        border-radius: 999px;
        padding: 6px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #1d4ed8;
        border: 1px solid rgba(59,130,246,0.4);
        margin-bottom: 0.4rem;
    }

    .hero-badge span {
        font-size: 1.0rem;
    }

    h2 {
        color: #111827;
        border-left: 5px solid #4f46e5;
        padding-left: 12px;
        background: #ffffff;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(15,23,42,0.06);
        font-weight: 700;
        margin-top: 1.2rem;
    }

    /* --------- HERO SECTION --------- */
    .hero-container {
        background: linear-gradient(135deg, #111827 0%, #1f2937 40%, #4b5563 100%);
        border-radius: 20px;
        padding: 22px 26px;
        color: #e5e7eb;
        box-shadow: 0 18px 45px rgba(15,23,42,0.45);
        position: relative;
        overflow: hidden;
        margin-bottom: 18px;
    }

    .hero-left-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .hero-left-sub {
        font-size: 0.95rem;
        color: #d1d5db;
        max-width: 420px;
    }

    .hero-chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 6px;
    }

    .hero-chip {
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(209,213,219,0.5);
        color: #e5e7eb;
        background: rgba(31,41,55,0.7);
        backdrop-filter: blur(8px);
    }

    .hero-right-card {
        background: rgba(17,24,39,0.85);
        border-radius: 18px;
        padding: 14px 16px;
        border: 1px solid rgba(75,85,99,0.8);
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }

    .hero-right-title {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 6px;
    }

    .hero-step {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 4px;
        font-size: 0.8rem;
        color: #e5e7eb;
    }

    .hero-step-badge {
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(135deg,#4f46e5,#6366f1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
        margin-top: 2px;
    }

    /* --------- METRICS --------- */
    [data-testid="stMetric"] {
        background: white;
        padding: 22px 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        transition: all 0.18s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 40px rgba(15,23,42,0.12);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.3em !important;
        font-weight: 800;
        color: #4f46e5;
    }

    /* --------- BUTTONS --------- */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
        color: white;
        border-radius: 999px;
        padding: 0.55rem 1.6rem;
        font-weight: 700;
        border: none;
        box-shadow: 0 10px 25px rgba(129,140,248,0.55);
        transition: all 0.18s ease-out;
        font-size: 0.9rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 14px 35px rgba(129,140,248,0.7);
    }

    /* --------- TABS --------- */
    .stTabs [role="tablist"] {
        gap: 6px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 4px;
        margin-bottom: 0.4rem;
    }

    .stTabs [role="tab"] {
        border-radius: 999px;
        padding: 8px 16px !important;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,#4f46e5,#6366f1);
        color: white !important;
    }

    /* --------- GENERIC CARDS --------- */
    .section-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    .section-title-inline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 6px;
    }

    .section-pill {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .section-caption {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 4px;
    }

    /* --------- BADGE PRIORITY --------- */
    .priority-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #ef4444;
        font-size: 0.8rem;
    }
    
    .priority-high {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #f59e0b;
        font-size: 0.8rem;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #3b82f6;
        font-size: 0.8rem;
    }
    
    .priority-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #10b981;
        font-size: 0.8rem;
    }

    /* --------- FOOTER --------- */
    .footer-note {
        color:#9ca3af;
        font-size: 0.75rem;
        text-align:center;
        margin-top: 25px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def format_currency(value):
    """Format nilai mata uang menjadi Rupiah."""
    return f"Rp {value:,.0f}"

def format_percentage(value):
    """Format nilai menjadi persentase."""
    return f"{value:.1f}%"

def get_priority_badge(segment):
    """Memberikan tag visual berdasarkan segmen prioritas."""
    if "High Value - High Risk" in segment:
        return '<span class="priority-critical">🔴 CRITICAL - WIN BACK NOW</span>'
    elif "Low Value - High Risk" in segment:
        return '<span class="priority-high">🟠 HIGH - Quick Action</span>'
    elif "High Value - Low Risk" in segment:
        return '<span class="priority-medium">🔵 MEDIUM - Retain & Grow</span>'
    else:
        return '<span class="priority-low">🟢 LOW - Maintain</span>'

def label_clusters_by_centers(kmeans):
    """Melabeli klaster K-Means berdasarkan rata-rata Recency, Frequency, Monetary."""
    centers = kmeans.cluster_centers_
    labels = []
    # Menggunakan median dari pusat klaster untuk penentuan batas
    median_r = np.median(centers[:, 0]) if len(centers) > 0 else 0
    median_f = np.median(centers[:, 1]) if len(centers) > 0 else 0
    median_m = np.median(centers[:, 2]) if len(centers) > 0 else 0

    for c in centers:
        r, f, m = c
        tag_r = "Fresh" if r < median_r else "Stale"
        tag_f = "Frequent" if f >= median_f else "Occasional"
        tag_m = "HighValue" if m >= median_m else "LowValue"
        labels.append("-".join([tag_r, tag_f, tag_m]))
    return {i: labels[i] for i in range(len(labels))}

def recommend_action(strategy_segment: str, risk_score: float) -> str:
    """Memberikan rekomendasi tindakan spesifik berdasarkan segmen dan skor risiko."""
    if strategy_segment == "High Value - High Risk":
        return "Personal CS Call + 40% Voucher + Free shipping (Budget: High)"
    elif strategy_segment == "Low Value - High Risk":
        return "Automated Email + 20% Discount Code (Budget: Medium)" if risk_score >= 0.70 else "📱 Push Notification + 10% Off (Budget: Low)"
    elif strategy_segment == "High Value - Low Risk":
        return "VIP Program + Early Access + Loyalty Points 2x (Retention/Upsell)"
    else:
        return "Newsletter + Product Recommendations (Automated/Low Cost)"

def loyal_probability(pipe, X):
    """Mendapatkan probabilitas kelas Loyal (1) dari model."""
    if hasattr(pipe[-1], "predict_proba"):
        proba = pipe.predict_proba(X)
        classes_ = getattr(pipe[-1], "classes_", np.array([0, 1]))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            try:
                # Cari indeks untuk kelas Loyal (1)
                idx1 = int(np.where(classes_ == 1)[0][0])
            except:
                idx1 = 1
            return proba[:, idx1]
    return np.zeros(len(X))


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("##  Control Panel")
    st.markdown("---")
    
    tx_file = st.file_uploader("📁 Upload Data CSV", type=["csv"])
    
    st.markdown("###  Parameter Bisnis")
    
    # Parameter Kunci untuk DSS
    risk_threshold = st.slider(" Ambang Batas Risiko (Risk Threshold)", 0.3, 0.8, 0.6, 0.05, help="Di atas nilai ini dianggap 'High Risk' (Potensi Churn). Mempengaruhi Cost Efficiency vs Risk Coverage.")
    value_percentile = st.slider(" Persentil Nilai Tinggi (High Value)", 50, 90, 70, 5, help="Persentil total pengeluaran 90 hari terakhir. Pelanggan di atas persentil ini dianggap 'High Value'.")
    
    # --- MODIFIKASI: K_CLUSTERS DIBUAT STATIS (K=4) ---
    k_clusters = 4
    st.markdown(f"**👥 Jumlah Segmen RFM (K):** **{k_clusters}** (Ditetapkan Statis)")
    # --- AKHIR MODIFIKASI ---
    
    st.markdown("---")
    st.markdown("###  ROI Calculator")
    
    avg_clv = st.number_input("Avg CLV/Pelanggan (Rp)", 100000, 50000000, 5000000, 100000, help="Rata-rata Customer Lifetime Value, digunakan menghitung Potensi Kerugian.")
    winback_cost = st.number_input("Biaya Win-back/Pelanggan (Rp)", 10000, 1000000, 150000, 10000, help="Anggaran rata-rata untuk intervensi win-back (misal: diskon, panggilan CS, hadiah).")
    success_rate = st.slider("Tingkat Keberhasilan Win-back (%)", 10, 80, 30, 5, help="Estimasi persentase pelanggan berisiko yang berhasil dipertahankan melalui intervensi.")
    
    st.markdown("---")
    
    with st.expander(" Advanced ML Settings"):
        model_name = st.selectbox("Algoritma Klasifikasi", ["RandomForest", "DecisionTree"])
        test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)
        
        if model_name == "RandomForest":
            n_estimators = st.slider("Jumlah Pohon (Trees)", 100, 600, 300, 50)
            max_depth = st.slider("Kedalaman Maksimum", 2, 20, 8)
        else:
            max_depth = st.slider("Kedalaman Maksimum", 2, 20, 8)
            n_estimators = 0
    
    st.markdown("---")
    want_export = st.checkbox("📥 Enable Downloads")

# ------------------------------------------------------------
# MAIN - DATA LOADING & PREPROCESSING
# ------------------------------------------------------------

if tx_file is None:
    st.markdown("#  Marketing Intelligence Dashboard")
    st.info(" Upload data CSV di sidebar untuk memulai analisis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "---")
    col2.metric("Loyalty Rate", "---")
    col3.metric("At-Risk Customers", "---")
    col4.metric("Potential Loss", "---")
    
    st.stop()

# Preprocessing
try:
    df_raw = pd.read_csv(tx_file)
    
    # Validasi Kolom Wajib
    required_cols = ["Churn"] 
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"❌ File harus memiliki kolom wajib: {', '.join(required_cols)}")
        st.stop()
        
    df_raw["customer_id"] = np.arange(len(df_raw)) + 1
    df_raw["loyal_label"] = 1 - df_raw["Churn"].astype(int)
    
    # Standardisasi Nama Kolom (penting untuk konsistensi model)
    column_mapping = {
        "DaySinceLastOrder": "recency_days",
        "NumberOfDeviceRegistered": "frequency_90d",
        "CashbackAmount": "total_spent_90d",
    }
    tx = df_raw.rename(columns=column_mapping, errors="ignore")
    tx["transaction_date"] = pd.to_datetime(dt.date.today())
    
    numeric_cols_clean = ['Tenure', 'WarehouseToHome', 'frequency_90d', 'SatisfactionScore', 
                          'NumberOfAddress', 'Complain', 'recency_days', 'total_spent_90d']
    
    # Konversi & Imputasi
    for col in numeric_cols_clean:
        tx[col] = pd.to_numeric(tx[col], errors='coerce')
    
    tx.drop_duplicates(inplace=True)
    
    for col in numeric_cols_clean:
        tx[col].fillna(tx[col].median(), inplace=True)
    
    tx.dropna(subset=['loyal_label'], inplace=True)
    
    # Penghapusan outlier (Robust Cleaning)
    z_scores = np.abs(stats.zscore(tx[numeric_cols_clean].fillna(0)))
    tx = tx[~((z_scores > 3).any(axis=1))]
    
    total_customers = len(tx)
    
    # Encoding Kategori
    le = LabelEncoder()
    for col in ['PreferedOrderCat', 'MaritalStatus']:
        if col in tx.columns:
            tx[col] = tx[col].fillna('Missing').astype(str)
            tx[col] = le.fit_transform(tx[col])

except Exception as e:
    st.error(f"❌ Error saat memuat atau membersihkan data. Pastikan format kolom sesuai. Detail: {e}")
    st.stop()

# ------------------------------------------------------------
# MODEL TRAINING & SCORING
# ------------------------------------------------------------

df_base = tx.copy()
df_base["avg_basket_value"] = np.where(
    df_base["frequency_90d"] > 0,
    df_base["total_spent_90d"] / df_base["frequency_90d"],
    0.0
)
df_base = df_base.drop(columns=["Churn"], errors="ignore")

y = df_base["loyal_label"].astype(int)
X = df_base.drop(columns=["loyal_label", "customer_id", "transaction_date", 
                          "DaySinceLastOrder", "NumberOfDeviceRegistered", "CashbackAmount"], errors="ignore")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
], remainder="drop")

# Penambahan class_weight="balanced" untuk menangani dataset yang tidak seimbang (imbalance)
clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42, 
        class_weight="balanced"
    ) if model_name == "RandomForest" else DecisionTreeClassifier(
        max_depth=max_depth, 
        random_state=42, 
        class_weight="balanced"
    )

pipe = Pipeline([("prep", preprocessor), ("model", clf)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
pipe.fit(X_train, y_train)

y_prob = loyal_probability(pipe, X_test)
full_prob = loyal_probability(pipe, X)

scored = df_base.copy()
scored["loyal_prob"] = full_prob
scored["risk_score"] = 1.0 - scored["loyal_prob"]
scored["predicted_loyal"] = (scored["risk_score"] < risk_threshold).astype(int)

# Feature Importance
feature_importances = pd.Series(dtype=float)
if model_name == "RandomForest":
    try:
        feature_importances = pd.Series(
            pipe["model"].feature_importances_,
            index=pipe['prep'].get_feature_names_out()
        ).sort_values(ascending=False).head(10)
    except:
        pass

# ------------------------------------------------------------
# SEGMENTATION & PRIORITIZATION (RFM-R)
# ------------------------------------------------------------

rfm_for_cluster = scored[["customer_id","recency_days","frequency_90d","total_spent_90d"]].copy()
scaler_robust = RobustScaler()
rfm_scaled = scaler_robust.fit_transform(rfm_for_cluster[["recency_days","frequency_90d","total_spent_90d"]])

# K-MEANS menggunakan nilai K statis = 4
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
rfm_for_cluster["cluster"] = kmeans.fit_predict(rfm_scaled)
rfm_for_cluster["segment_name"] = rfm_for_cluster["cluster"].map(label_clusters_by_centers(kmeans))

joined = scored.merge(rfm_for_cluster[["customer_id","cluster","segment_name"]], on="customer_id", how="left")

# Penentuan Segmen Prioritas Bisnis (Value vs Risk)
value_thresh = np.percentile(joined["total_spent_90d"], value_percentile)
joined["value_level"] = np.where(joined["total_spent_90d"] >= value_thresh, "High Value", "Low Value")
joined["risk_level"] = np.where(joined["risk_score"] >= risk_threshold, "High Risk", "Low Risk")
joined["strategy_segment"] = joined["value_level"] + " - " + joined["risk_level"]
joined["action"] = joined.apply(lambda r: recommend_action(r["strategy_segment"], r["risk_score"]), axis=1)
joined["priority_badge"] = joined["strategy_segment"].apply(get_priority_badge)

# ------------------------------------------------------------
# 1. KPI & HERO SECTION
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <div style="display:flex; gap:22px; align-items:stretch; flex-wrap:wrap;">
            <div style="flex: 1 1 260px; min-width:260px;">
                <div class="hero-badge">
                    <span>📊</span> Decision Support – Customer Loyalty & Churn Win-Back
                </div>
                <div class="hero-left-title">Marketing Intelligence Dashboard</div>
                <div class="hero-left-sub">
                    Dashboard ini membantu tim marketing memutuskan 
                    <b>siapa yang harus diselamatkan dulu</b>, 
                    berapa <b>potensi kerugian</b>, dan 
                    skenario <b>ROI campaign win-back</b> terbaik.
                </div>
                <div class="hero-chip-row">
                    <div class="hero-chip">Predictive Churn</div>
                    <div class="hero-chip">Segmentation RFM</div>
                    <div class="hero-chip">ROI Simulation</div>
                    <div class="hero-chip">Actionable Playbook</div>
                </div>
            </div>
            <div style="flex:0 0 280px; max-width:320px;">
                <div class="hero-right-card">
                    <div class="hero-right-title">Cara Pakai Dashboard</div>
                    <div class="hero-step">
                        <div class="hero-step-badge">1</div>
                        <div>Upload data CSV dan atur parameter di <b>sidebar</b>.</div>
                    </div>
                    <div class="hero-step">
                        <div class="hero-step-badge">2</div>
                        <div>Lihat tab <b>Overview</b> untuk status loyalitas dan risiko.</div>
                    </div>
                    <div class="hero-step">
                        <div class="hero-step-badge">3</div>
                        <div>Buka tab <b>ROI</b> & <b>Segments</b> untuk simulasi dan prioritas budget.</div>
                    </div>
                    <div class="hero-step">
                        <div class="hero-step-badge">4</div>
                        <div>Gunakan tab <b>Action Plan</b> untuk daftar pelanggan yang harus dihubungi.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# KPI utama
loyal_pct = 100.0 * scored["loyal_label"].mean()
churn_pct = 100.0 - loyal_pct
high_risk_count = len(joined[joined["risk_score"] >= risk_threshold])
hvhr_count = len(joined[joined["strategy_segment"] == "High Value - High Risk"])

# ROI Calculations
potential_loss = high_risk_count * avg_clv
winback_investment = high_risk_count * winback_cost
expected_recovery = potential_loss * (success_rate / 100.0)
net_roi = expected_recovery - winback_investment
roi_percentage = (net_roi / winback_investment * 100) if winback_investment > 0 else 0

# Data untuk Priority Matrix
priority_matrix = joined.groupby("strategy_segment").agg(
    Jumlah=('customer_id', 'size'),
    Avg_Risk=('risk_score', 'mean'),
    Avg_Value=('total_spent_90d', 'mean'),
    Total_Value=('total_spent_90d', 'sum')
).reset_index().sort_values('Total_Value', ascending=False)

priority_matrix['Persentase'] = (priority_matrix['Jumlah'] / total_customers * 100).apply(lambda x: f"{x:.1f}%")
priority_matrix['Priority'] = priority_matrix['strategy_segment'].apply(get_priority_badge)

# ------------------------------------------------------------
# 2. MAIN TABS (VERSI LEBIH BISNIS)
# ------------------------------------------------------------

tab_overview, tab_roi, tab_segments, tab_action, tab_model, tab_summary = st.tabs(
    ["📌 Overview", "💰 ROI & Finance", "🎯 Segments & Matrix", "📋 Action Plan", "⚙️ Model & Trade-Off", "📝 Executive Summary"]
)

# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    st.markdown("### 📌 Ringkasan Status Loyalitas & Risiko")
    st.markdown(
        """
        <div class="section-caption">
        Ringkasan ini cocok untuk slide awal presentasi: kondisi loyalitas, seberapa besar risiko churn,
        dan nilai yang sedang “dipertaruhkan” jika tidak ada intervensi.
        </div>
        """, unsafe_allow_html=True
    )

    # KPI Utama
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{total_customers:,}", delta="Data bersih", delta_color="off")
    kpi2.metric("Loyalty Rate", f"{loyal_pct:.1f}%", delta=f"{int(total_customers * loyal_pct/100):,} loyal")
    kpi3.metric("At-Risk Customers", f"{high_risk_count:,}", delta=f"{hvhr_count} critical", delta_color="inverse")
    kpi4.metric("Potential Revenue Loss", format_currency(potential_loss), delta="Tanpa intervensi", delta_color="inverse")

    # Health Index sederhana (gabung loyalitas & share HVHR)
    hvhr_share = hvhr_count / total_customers if total_customers > 0 else 0
    if loyal_pct >= 75 and hvhr_share < 0.05:
        health_label = "HEALTHY"
        health_color = "#10b981"
        health_desc = "Mayoritas pelanggan loyal dan hanya sedikit high-value yang berisiko. Fokus di retention & upsell."
    elif loyal_pct >= 60 and hvhr_share < 0.10:
        health_label = "WATCHLIST"
        health_color = "#f59e0b"
        health_desc = "Secara umum masih aman, tetapi mulai ada kantong pelanggan high-value yang berisiko."
    else:
        health_label = "CRITICAL"
        health_color = "#ef4444"
        health_desc = "Risiko churn cukup tinggi terutama di segmen bernilai besar. Perlu campaign win-back terarah."

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title-inline">
                <div><b>Loyalty Health Status</b></div>
                <div class="section-pill" style="background:{health_color}1A; color:{health_color}; border:1px solid {health_color};">
                    Status: {health_label}
                </div>
            </div>
            <div style="font-size:0.9rem; color:#374151; margin-bottom:6px;">{health_desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown("#### 🔍 Distribusi Risiko Churn")
        fig_risk_dist = px.histogram(
            joined,
            x="risk_score",
            color="strategy_segment",
            nbins=30,
            title="Risk Score Distribution per Segment",
        )
        fig_risk_dist.update_layout(height=380, showlegend=True)
        st.plotly_chart(fig_risk_dist, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🧩 Komposisi Segmen Strategis")
        fig_seg = px.pie(
            joined,
            names="strategy_segment",
            title="Share Pelanggan per Strategy Segment",
            color_discrete_map={
                "High Value - High Risk": "#ef4444",
                "Low Value - High Risk": "#f59e0b",
                "High Value - Low Risk": "#3b82f6",
                "Low Value - Low Risk": "#10b981"
            }
        )
        fig_seg.update_layout(height=380)
        st.plotly_chart(fig_seg, use_container_width=True)

    # Insight naratif singkat untuk manajemen
    hvhr_value = joined.loc[joined["strategy_segment"] == "High Value - High Risk", "total_spent_90d"].sum()
    lvhr_count = len(joined[joined["strategy_segment"] == "Low Value - High Risk"])
    hvlr_count = len(joined[joined["strategy_segment"] == "High Value - Low Risk"])

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title-inline">
                <div><b>Insight Cepat untuk Rapat Manajemen</b></div>
                <div class="section-pill">Storytelling</div>
            </div>
            <ul style="font-size:0.85rem; color:#4b5563; margin-bottom:0;">
                <li>Saat ini sekitar <b>{loyal_pct:.1f}%</b> pelanggan dikategorikan loyal, dengan <b>{churn_pct:.1f}%</b> berada pada zona risiko.</li>
                <li>Ada <b>{hvhr_count}</b> pelanggan di segmen <b>High Value - High Risk</b> dengan total nilai transaksi sekitar <b>{format_currency(hvhr_value)}</b>.</li>
                <li>Segmen <b>High Value - Low Risk</b> berjumlah <b>{hvlr_count}</b> pelanggan yang layak dijadikan fokus <i>upsell</i> & program loyalitas.</li>
                <li>Segmen <b>Low Value - High Risk</b> (± {lvhr_count} pelanggan) dapat ditangani dengan program massal berbiaya rendah (email, push notification).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===================== TAB 2: ROI & FINANCE =====================
with tab_roi:
    st.markdown("### 💰 ROI Win-Back Campaign Optimization")

    st.markdown(
        """
        <div class="section-caption">
        Di tab ini, kamu bisa bermain dengan skenario: pilih segmen mana yang mau diikutkan campaign,
        dan lihat dampaknya ke ROI.
        </div>
        """, unsafe_allow_html=True
    )

    # --- Scenario builder: pilih segmen & budget ---
    st.markdown("#### 🎛️ Pengaturan Skenario Kampanye")

    available_segments = priority_matrix["strategy_segment"].unique().tolist()
    default_segments = [s for s in ["High Value - High Risk", "Low Value - High Risk"] if s in available_segments]

    selected_segments = st.multiselect(
        "Pilih segmen yang akan diikutkan dalam campaign:",
        options=available_segments,
        default=default_segments
    )

    # Basis: hanya pelanggan high risk
    campaign_base = joined[joined["risk_level"] == "High Risk"].copy()
    if selected_segments:
        campaign_base = campaign_base[campaign_base["strategy_segment"].isin(selected_segments)]

    total_candidates = len(campaign_base)
    st.caption(f"📌 Dengan kriteria di atas, terdapat <b>{total_candidates}</b> pelanggan kandidat campaign.</b>", unsafe_allow_html=True)

    # Batas budget
    limit_by_budget = st.checkbox("Batasi berdasarkan total budget campaign", value=False)
    if limit_by_budget:
        max_budget = st.number_input(
            "Total Budget Maksimal (Rp)",
            min_value=winback_cost,
            value=int(max(winback_investment, winback_cost)),
            step=winback_cost
        )
        max_customers_affordable = int(max_budget // winback_cost)
        max_customers_affordable = max(1, max_customers_affordable)
        selected_customers = campaign_base.sort_values(
            ["strategy_segment", "total_spent_90d", "risk_score"],
            ascending=[True, False, False]
        ).head(max_customers_affordable)
    else:
        selected_customers = campaign_base
        max_budget = len(campaign_base) * winback_cost

    targeted_n = len(selected_customers)

    # ROI skenario
    scenario_potential_loss = targeted_n * avg_clv
    scenario_investment = targeted_n * winback_cost
    scenario_expected_recovery = scenario_potential_loss * (success_rate / 100.0)
    scenario_net_roi = scenario_expected_recovery - scenario_investment
    scenario_roi_pct = (scenario_net_roi / scenario_investment * 100) if scenario_investment > 0 else 0

    # --- Visualisasi ROI skenario ---
    roi_col1, roi_col2 = st.columns([3, 2])

    with roi_col1:
        fig_roi = go.Figure()
        
        categories = ['Potential Loss', 'Investment', 'Recovery', 'Net ROI']
        values = [scenario_potential_loss, -scenario_investment, scenario_expected_recovery, scenario_net_roi]
        
        fig_roi.add_trace(go.Waterfall(
            name="ROI Scenario",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=categories,
            y=values,
            text=[format_currency(abs(v)) for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#4f46e5"}}
        ))
        
        fig_roi.update_layout(
            title="Financial Impact – Skenario Campaign Terpilih",
            height=420,
            showlegend=False
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)

    with roi_col2:
        st.markdown("#### 📈 Ringkasan Skenario")

        base_net_str = format_currency(net_roi)
        base_roi_str = f"{roi_percentage:.1f}%"

        st.markdown(f"""
        **Target Customers (Skenario Ini)** <b>{targeted_n}</b> pelanggan
        
        **Investment (Skenario Ini)** {format_currency(scenario_investment)}
        
        **Expected Recovery (Skenario Ini)** {format_currency(scenario_expected_recovery)}
        
        **Net ROI (Skenario Ini)** <span style="font-size: 1.5em; color: {'#10b981' if scenario_net_roi > 0 else '#ef4444'}; font-weight: 800;">
        {format_currency(scenario_net_roi)}
        </span>
        
        **ROI % (Skenario Ini)** <span style="font-size: 1.2em; color: {'#10b981' if scenario_roi_pct > 0 else '#ef4444'}; font-weight: 700;">
        {scenario_roi_pct:.1f}%
        </span>
        
        <hr/>
        <span style="font-size:0.8rem; color:#6b7280;">
        📎 Sebagai pembanding, baseline semua pelanggan high-risk:<br/>
        Net ROI: <b>{base_net_str}</b> · ROI%: <b>{base_roi_str}</b>
        </span>
        """, unsafe_allow_html=True)
        
        if scenario_net_roi > 0:
            st.success("✅ Skenario ini menghasilkan ROI positif. Cocok dijadikan usulan awal campaign.")
        else:
            st.warning("⚠️ ROI skenario ini negatif. Kurangi segmen yang disasar atau sesuaikan biaya per pelanggan.")

    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title-inline">
                <div><b>Guideline singkat untuk Finance & CMO:</b></div>
                <div class="section-pill">Finance & Budgeting</div>
            </div>
            <ul style="font-size:0.85rem; color:#4b5563; margin-bottom:0;">
                <li>Gunakan <b>multiselect segmen</b> di atas untuk melihat trade-off antara cakupan pelanggan dan keuntungan finansial.</li>
                <li>Jika budget terbatas, aktifkan opsi <b>batasi berdasarkan budget</b> dan lihat berapa pelanggan yang paling “worth it” dijangkau.</li>
                <li>Angka-angka di sini dapat langsung dibawa ke <i>business case</i> untuk justifikasi campaign marketing.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

# ===================== TAB 3: SEGMENTS & MATRIX =====================
with tab_segments:
    st.markdown("### 🎯 Priority Matrix - Nilai Pelanggan vs Risiko Churn")

    matrix_col1, matrix_col2 = st.columns([2, 3])

    with matrix_col1:
        st.markdown("#### 📊 Distribusi Segmen (Tabel Konsultan)")
        
        display_matrix = priority_matrix[['strategy_segment', 'Jumlah', 'Persentase', 'Avg_Value', 'Total_Value']].copy()
        display_matrix['Avg_Value'] = display_matrix['Avg_Value'].apply(format_currency)
        display_matrix['Total_Value'] = display_matrix['Total_Value'].apply(format_currency)
        
        st.dataframe(
            display_matrix,
            use_container_width=True,
            column_config={
                "strategy_segment": "Segmen Prioritas",
                "Jumlah": st.column_config.NumberColumn("Customers", format="%d"),
                "Persentase": "% dari Total",
                "Avg_Value": "Avg Spending",
                "Total_Value": "Total Value"
            },
            hide_index=True
        )

    with matrix_col2:
        st.markdown("#### 🗺️ Visualisasi Kuadran Keputusan")
        
        fig_matrix = px.scatter(
            joined,
            x="total_spent_90d",
            y="risk_score",
            color="strategy_segment",
            size="frequency_90d",
            hover_data=["customer_id"],
            color_discrete_map={
                "High Value - High Risk": "#ef4444",
                "Low Value - High Risk": "#f59e0b",
                "High Value - Low Risk": "#3b82f6",
                "Low Value - Low Risk": "#10b981"
            },
            labels={
                "total_spent_90d": "Customer Value (Rp)",
                "risk_score": "Churn Risk Score"
            }
        )
        
        fig_matrix.add_hline(y=risk_threshold, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
        fig_matrix.add_vline(x=value_thresh, line_dash="dash", line_color="blue", annotation_text="Value Threshold")
        
        fig_matrix.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig_matrix, use_container_width=True)

    # Detail per segmen / persona
    st.markdown("#### 👤 Detail Segmen & Persona Bisnis")

    seg_selected = st.selectbox(
        "Pilih segmen untuk melihat detail:",
        options=priority_matrix["strategy_segment"].tolist()
    )

    seg_df = joined[joined["strategy_segment"] == seg_selected]
    seg_n = len(seg_df)
    seg_avg_val = seg_df["total_spent_90d"].mean() if seg_n > 0 else 0
    seg_avg_risk = seg_df["risk_score"].mean() if seg_n > 0 else 0

    persona_desc = {
        "High Value - High Risk": "Pelanggan bernilai besar tetapi mulai menjauh. Butuh hubungan personal + insentif kuat.",
        "Low Value - High Risk": "Pelanggan sensitif harga dan mudah pindah. Cocok untuk promo massal berbiaya rendah.",
        "High Value - Low Risk": "Pelanggan setia bernilai tinggi. Fokus pada loyalty program & cross-sell/upsell.",
        "Low Value - Low Risk": "Pelanggan stabil namun nilai kecil. Cukup dipertahankan dengan komunikasi ringan."
    }.get(seg_selected, "Segmen campuran berdasarkan nilai dan risiko. Sesuaikan strategi channel & pesan komunikasi.")

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title-inline">
                <div><b>{seg_selected}</b></div>
                <div class="section-pill">Segment Insight</div>
            </div>
            <p style="font-size:0.9rem; color:#374151; margin-bottom:4px;">{persona_desc}</p>
            <ul style="font-size:0.85rem; color:#4b5563; margin-bottom:8px;">
                <li>Jumlah pelanggan: <b>{seg_n}</b></li>
                <li>Rata-rata pengeluaran (90 hari): <b>{format_currency(seg_avg_val)}</b></li>
                <li>Rata-rata risk score: <b>{seg_avg_risk:.2f}</b></li>
            </ul>
            <p style="font-size:0.8rem; color:#6b7280; margin-bottom:0;">
                💡 Gunakan insight ini untuk merancang <i>message</i> dan channel yang paling relevan untuk segmen ini.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===================== TAB 4: ACTION PLAN =====================
with tab_action:
    st.markdown("### 📋 Prescriptive Action Plan - Target Customers")
    st.markdown(
        """
        <div class="section-caption">
        Tab ini berisi daftar pelanggan prioritas yang dapat langsung dieksekusi oleh tim CRM / sales.
        Kamu juga bisa mensimulasikan berapa banyak pelanggan yang bisa dijangkau sesuai budget.
        </div>
        """, unsafe_allow_html=True
    )

    subtab1, subtab2, subtab3 = st.tabs(["🔴 Critical (HVHR)", "🟠 All High Risk", "📊 Ringkasan Segmen"])

    with subtab1:
        st.markdown("#### Pelanggan Critical - Immediate Action Required")
        
        critical_customers = joined[joined["strategy_segment"] == "High Value - High Risk"].sort_values("total_spent_90d", ascending=False)
        
        if len(critical_customers) > 0:
            st.error(f"⚠️ {len(critical_customers)} pelanggan high-value berisiko churn! Potensi kerugian: {format_currency(len(critical_customers) * avg_clv)}")
            
            display_critical = critical_customers[["customer_id", "recency_days", "frequency_90d", "total_spent_90d", "risk_score", "action"]].head(50)
            
            st.dataframe(
                display_critical.style.format({
                    "recency_days": "{:.0f}",
                    "frequency_90d": "{:.0f}",
                    "total_spent_90d": lambda x: format_currency(x),
                    "risk_score": "{:.2f}"
                }).background_gradient(cmap='Reds', subset=['risk_score']),
                use_container_width=True,
                column_config={
                    "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
                    "recency_days": "Last Order (Days)",
                    "frequency_90d": "Frequency",
                    "total_spent_90d": "Total Spent",
                    "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=1),
                    "action": "Recommended Action"
                },
                hide_index=True
            )
        else:
            st.success("✅ Tidak ada pelanggan di kategori critical!")

    with subtab2:
        st.markdown("#### High Risk Customers - Quick Win Opportunities")
        
        high_risk = joined[joined["risk_level"] == "High Risk"].sort_values(
            ["total_spent_90d", "risk_score"], ascending=[False, False]
        )
        
        st.info(f"📊 Total {len(high_risk)} pelanggan berisiko tinggi")

        # Budget-based targeting
        st.markdown("##### 🎯 Prioritas Berdasarkan Budget")
        budget_for_highrisk = st.number_input(
            "Total Budget untuk Campaign High-Risk (Rp)",
            min_value=winback_cost,
            value=int(min(len(high_risk) * winback_cost, max(winback_investment, winback_cost))),
            step=winback_cost
        )
        max_targets = int(budget_for_highrisk // winback_cost)
        max_targets = min(max_targets, len(high_risk))
        st.caption(f"Dengan budget ini, maksimal sekitar <b>{max_targets}</b> pelanggan bisa dihubungi.", unsafe_allow_html=True)

        ranked_targets = high_risk.sort_values(
            ["strategy_segment", "total_spent_90d", "risk_score"],
            ascending=[True, False, False]
        ).head(max_targets)

        st.dataframe(
            ranked_targets[["customer_id", "strategy_segment", "total_spent_90d", "risk_score", "action"]].style.format({
                "total_spent_90d": lambda x: format_currency(x),
                "risk_score": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Estimasi dampak finansial untuk budget ini
        est_loss_protected = max_targets * avg_clv
        est_recovery = est_loss_protected * (success_rate / 100.0)
        est_net = est_recovery - budget_for_highrisk

        st.markdown(
            f"""
            <div class="section-card" style="margin-top:10px;">
                <b>Estimasi Dampak untuk Budget Ini:</b>
                <ul style="font-size:0.85rem; color:#4b5563; margin-bottom:0;">
                    <li>Pelanggan yang dijangkau: <b>{max_targets}</b></li>
                    <li>Potensi kerugian yang “diproteksi”: <b>{format_currency(est_loss_protected)}</b></li>
                    <li>Perkiraan recovery (dengan success rate {success_rate}%): <b>{format_currency(est_recovery)}</b></li>
                    <li>Perkiraan Net ROI campaign ini: <b>{format_currency(est_net)}</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with subtab3:
        st.markdown("#### Ringkasan Agregat per Strategy Segment")
        
        segment_summary = joined.groupby("strategy_segment").agg({
            "customer_id": "count",
            "total_spent_90d": ["mean", "sum"],
            "risk_score": "mean"
        }).round(2)
        
        st.dataframe(segment_summary, use_container_width=True)

# ===================== TAB 5: MODEL & TRADE-OFF =====================
with tab_model:
    # 5. ROOT CAUSE ANALYSIS (DRIVERS)
    if not feature_importances.empty:
        st.markdown("### 💡 Key Insights - Top Drivers Churn")
        
        insight_col1, insight_col2 = st.columns([3, 2])
        
        with insight_col1:
            fig_fi = px.bar(
                feature_importances,
                orientation='h',
                title="Top 10 Faktor Bisnis yang Mendorong Pelanggan Churn",
                labels={'value': 'Kekuatan Pengaruh (Relative Importance)', 'index': 'Faktor Bisnis'},
                color_discrete_sequence=['#4f46e5']
            )
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)
        
        with insight_col2:
            top_factor = feature_importances.index[0]
            
            st.markdown("#### 🎯 Rekomendasi Strategis Berdasarkan Akar Masalah")
            st.markdown(f"""
            **Faktor Churn Utama:** `{top_factor}`  
            <br>
            **Action Items (Structural):**
            1. 🔍 <b>Deep Dive:</b> Lakukan analisis mendalam pada {top_factor} (contoh: mengapa nilai/keluhan di faktor ini tinggi?).  
            2. 🛠️ <b>Implementasi:</b> Rancang intervensi perbaikan proses, bukan hanya promosi diskon jangka pendek.  
            3. 📊 <b>Monitor:</b> Pantau perubahan metrik setelah perbaikan (NPS, Satisfaction Score, Complaint Rate, dll).  
            <br>
            **Quick Wins:**
            - Alokasikan budget terbesar untuk segmen <b>High Value - High Risk</b>.  
            - Distribusi awal yang direkomendasikan: 60% HVHR, 30% LVHR, 10% Retention & Upsell.  
            - Target ROI: minimal <b>{roi_percentage:.1f}%</b> atau di atas standar perusahaan.
            """, unsafe_allow_html=True)
    else:
        top_factor = None
        st.info("Model tidak mengembalikan feature importance (mungkin karena Decision Tree sederhana).")

    st.markdown("---")
    st.markdown("### 📊 Kualitas Model dalam Bahasa Bisnis")

    # Confusion matrix versi bisnis: churn vs flagged as high-risk
    # True label: churner (1) vs loyal (0)
    churn_test = (1 - y_test).astype(int)
    risk_test = (1.0 - y_prob >= risk_threshold).astype(int)  # 1 = di-flag high risk

    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(churn_test, risk_test)  # baris: true, kolom: pred
        if cm.shape == (2, 2):
            # tp = True Positive (churner terdeteksi high-risk)
            # fn = False Negative (churner lolos, tidak terdeteksi high-risk)
            # fp = False Positive (loyal terdeteksi high-risk, biaya terbuang)
            # tn = True Negative (loyal terdeteksi low-risk, aman)
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
    except Exception:
        tn = fp = fn = tp = 0

    col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
    col_cm1.metric("📩 Intervensi Tepat (TP)", tp, help="Pelanggan berisiko yang berhasil di-flag dan bisa diselamatkan.")
    col_cm2.metric("💸 Uang Terbuang (FP)", fp, help="Pelanggan loyal yang ikut dikirimi promo (biaya campaign tanpa risiko).")
    col_cm3.metric("⚠️ Risiko Lolos (FN)", fn, help="Pelanggan berisiko yang tidak terdeteksi model.")
    col_cm4.metric("✅ Aman (TN)", tn, help="Pelanggan loyal yang tidak perlu intervensi khusus.")

    st.markdown(
        """
        <div class="section-card">
            <b>Interpretasi:</b>
            <ul style="font-size:0.85rem; color:#4b5563; margin-bottom:0;">
                <li><b>FP tinggi</b> ⇒ banyak pelanggan loyal yang ikut kena promo → biaya marketing boros.</li>
                <li><b>FN tinggi</b> ⇒ banyak churner yang tidak terdeteksi → kerugian revenue jangka panjang.</li>
                <li>Goal-nya adalah menekan <b>FP</b> dan <b>FN</b> sambil menjaga ROI tetap positif.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚖️ Optimalisasi Biaya vs. Cakupan Risiko (Model Trade-Off)")
    st.info("Lihat bagaimana pengaturan Ambang Batas Risiko (slider di sidebar) mengubah keseimbangan antara **efisiensi biaya** dan **cakupan risiko**.")

    # Recalculate Precision and Recall across a range of thresholds
    thresholds = np.linspace(0, 1, 101)
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        y_pred_t = (1.0 - y_prob < t).astype(int) 
        prec = precision_score(y_test, y_pred_t, zero_division=0)
        rec = recall_score(y_test, y_pred_t, zero_division=0)
        precision_scores.append(prec)
        recall_scores.append(rec)

    tradeoff_df = pd.DataFrame({
        'Threshold': thresholds,
        'Precision (Loyal)': precision_scores,
        'Recall (Loyal)': recall_scores,
    })

    fig_tradeoff = make_subplots(specs=[[{"secondary_y": True}]])
    fig_tradeoff.add_trace(
        go.Scatter(x=tradeoff_df['Threshold'], y=tradeoff_df['Precision (Loyal)'], 
                   name='Cost Efficiency (Precision)', line=dict(color='#3b82f6', width=3)), 
        secondary_y=False
    )
    fig_tradeoff.add_trace(
        go.Scatter(x=tradeoff_df['Threshold'], y=tradeoff_df['Recall (Loyal)'], 
                   name='Risk Coverage (Recall)', line=dict(color='#ef4444', width=3)), 
        secondary_y=True
    )
    fig_tradeoff.add_vline(
        x=risk_threshold, line_dash="dash", line_color="#7c3aed", 
        annotation_text=f"Ambang Saat Ini ({risk_threshold})", 
        annotation_position="top left"
    )
    fig_tradeoff.update_layout(
        title_text="Keseimbangan Biaya vs Risiko Berdasarkan Ambang Batas Klasifikasi",
        height=480
    )
    fig_tradeoff.update_xaxes(title_text="Ambang Batas Risiko (0.0 = Agresif Win-Back, 1.0 = Selektif)")
    fig_tradeoff.update_yaxes(title_text="Cost Efficiency", secondary_y=False, range=[0, 1.05])
    fig_tradeoff.update_yaxes(title_text="Risk Coverage", secondary_y=True, range=[0, 1.05])

    st.plotly_chart(fig_tradeoff, use_container_width=True)

# ===================== TAB 6: EXECUTIVE SUMMARY & DOWNLOADS =====================
with tab_summary:
    st.markdown("### 📝 Executive Summary & Final Recommendations")

    summary_data = []
    summary_data.append([
        "Status Pasar Loyalitas",
        f"{format_percentage(loyal_pct)} Loyal / {format_percentage(churn_pct)} At-Risk",
        "Jaga program loyalitas yang sudah ada dan gunakan dashboard ini untuk fokus pada pelanggan berisiko."
    ])

    if hvhr_count > 0:
        summary_data.append([
            "Prioritas Kritis (HVHR)",
            f"{hvhr_count} pelanggan High-Value berisiko hilang ({format_currency(hvhr_count * avg_clv)} potensi kerugian).",
            "Intervensi segera dan personal: kontak langsung (telepon/WA) + penawaran premium (voucher besar / benefit eksklusif)."
        ])
    else:
        summary_data.append([
            "Prioritas Kritis (HVHR)",
            "Tidak ada pelanggan di segmen High Value - High Risk.",
            "Fokuskan anggaran retention ke segmen High Value - Low Risk dan program loyalitas jangka panjang."
        ])

    if not feature_importances.empty and 'top_factor' in locals() and top_factor is not None:
        summary_data.append([
            "Akar Masalah (Root Cause)",
            f"Pendorong churn utama adalah **{top_factor}**.",
            "Lakukan root-cause analysis & perbaikan proses struktural pada faktor ini (SOP, kualitas layanan, SLA, dsb)."
        ])

    summary_data.append([
        "Dampak Finansial Campaign (Baseline High-Risk)",
        f"Investasi: {format_currency(winback_investment)} | Target ROI Bersih: {roi_percentage:.1f}%",
        "Lanjutkan campaign dengan konfigurasi saat ini" if net_roi > 0 else "Revisi strategi: fokus ke segmen tertentu atau negosiasikan ulang biaya campaign."
    ])

    st.table(pd.DataFrame(summary_data, columns=["Aspek", "Temuan Kunci (DSS)", "Strategi Manajemen"]))

    st.markdown("#### 📆 Recommended Action Plan (30 Hari ke Depan)")
    st.markdown(
        f"""
        <div class="section-card">
            <ul style="font-size:0.9rem; color:#374151; margin-bottom:0;">
                <li><b>Minggu 1:</b> Finalisasi parameter campaign dan daftar pelanggan target (terutama {hvhr_count} HVHR & segmen high-risk prioritas).</li>
                <li><b>Minggu 2:</b> Launch campaign tahap 1 (HVHR) + test A/B untuk segmen Low Value - High Risk.</li>
                <li><b>Minggu 3:</b> Evaluasi respon awal, hitung ulang ROI aktual, sesuaikan benefit & channel yang paling efektif.</li>
                <li><b>Minggu 4:</b> Scale up campaign di segmen yang ROI-nya paling tinggi dan presentasikan hasil ke manajemen.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### ✅ Checklist Eksekusi")
    st.markdown(
        """
        <ul style="font-size:0.85rem; color:#4b5563;">
            <li>[ ] Menyetujui segmen prioritas yang akan diaktifkan (CMO / Head of Marketing).</li>
            <li>[ ] Sinkronisasi daftar pelanggan target ke CRM / sistem campaign.</li>
            <li>[ ] Menetapkan target KPI (ROI %, churn rate, response rate).</li>
            <li>[ ] Menyiapkan laporan mingguan berbasis dashboard ini untuk monitoring.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    if want_export:
        st.markdown("#### ⬇️ Unduh Data Aksi")

        risky_customers_df = joined[joined["risk_level"] == "High Risk"].sort_values(
            ["strategy_segment", "total_spent_90d"], ascending=[True, False]
        )
        
        risky_export = risky_customers_df[[
            "customer_id", "recency_days", "total_spent_90d", "risk_score", 
            "strategy_segment", "action"
        ]].to_csv(index=False).encode("utf-8")
        
        model_bytes = BytesIO()
        joblib.dump(pipe, model_bytes)
        model_bytes.seek(0)
        
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button("Download Model Klasifikasi (.joblib)", data=model_bytes, file_name="loyalty_model.joblib")
        col_dl2.download_button("Download Pelanggan Berisiko (CSV)", data=risky_export, file_name="target_customers_high_risk.csv")

    st.markdown(
        "<div class='footer-note'>DSS Loyalty Engine - Advanced Version · Powered by Python & Streamlit</div>",
        unsafe_allow_html=True
    )