import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====================== CONFIG ======================
st.set_page_config(
    page_title="🏭 Industrial Maintenance Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk styling yang lebih baik
st.markdown("""
    <style>
    .main-header {
        font-size: 48px !important;

        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: black 3px solid;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏭 Industrial Machine Maintenance Recommender</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>Analisis prediktif kondisi mesin industri dengan "
    "<strong>XGBoost</strong> dan interpretasi <strong>Feature Importance</strong></p>",
    unsafe_allow_html=True
)

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/ai4i2020.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ File 'data/ai4i2020.csv' tidak ditemukan. Pastikan file ada di direktori yang benar.")
        st.stop()

df = load_data()

# ====================== TRAIN MODEL ======================
@st.cache_resource
def train_model(df):
    X = df.drop(["UDI", "Product ID", "Machine failure"], axis=1)
    y = df["Machine failure"]
    
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = ["Type"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(int)
    
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / pos if pos != 0 else 1.0
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=150,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, num_cols, cat_cols

with st.spinner('🔄 Melatih model XGBoost...'):
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, num_cols, cat_cols = train_model(df)

# Predictions
df["Failure_Prob"] = model.predict_proba(df.drop(["UDI", "Product ID", "Machine failure"], axis=1))[:, 1]

def get_recommendation(prob):
    if prob > 0.6:
        return "🚨 Perawatan Berat"
    elif prob > 0.3:
        return "⚙️ Perawatan Sedang"
    else:
        return "✅ Aman"

df["Recommendation"] = df["Failure_Prob"].apply(get_recommendation)

# Feature Importance
xgb_model = model.named_steps["clf"]
try:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
except:
    feature_names = num_cols

importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Fitur': feature_names[:len(importances)],
    'Importance': importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

def clean_feature_name(raw_name):
    raw_name = raw_name.replace("num__", "").replace("cat__", "").replace("Type_", "Type: ")
    pretty_map = {
        "Air temperature [K]": "Air Temperature",
        "Process temperature [K]": "Process Temperature",
        "Rotational speed [rpm]": "Rotational Speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool Wear",
    }
    return pretty_map.get(raw_name, raw_name)

feature_importance_df["Fitur"] = feature_importance_df["Fitur"].apply(clean_feature_name)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/maintenance.png", width=80)
    st.markdown("## ⚙️ Filter & Navigasi")
    
    page = st.radio("📑 **Menu Utama**", [
        "🏠 Dashboard",
        "📋 Detail Data",
        "📊 Model Performance",
        "🔍 Feature Analysis",
        "🤖 Prediction Simulator"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🔍 Filter Data")
    
    type_filter = st.multiselect(
        "**Tipe Mesin**",
        options=sorted(df["Type"].unique()),
        default=df["Type"].unique()
    )
    
    temp_range = st.slider(
        "**Suhu Proses (K)**",
        float(df["Process temperature [K]"].min()),
        float(df["Process temperature [K]"].max()),
        (float(df["Process temperature [K]"].min()), float(df["Process temperature [K]"].max()))
    )
    
    risk_filter = st.selectbox(
        "**Filter Rekomendasi**",
        ["Semua", "✅ Aman", "⚙️ Perawatan Sedang", "🚨 Perawatan Berat"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Model Info")
    st.info(f"""
    **Akurasi**: {classification_report(y_test, y_pred, output_dict=True)['accuracy']:.2%}
    
    **ROC-AUC**: {roc_auc_score(y_test, y_pred_proba):.3f}
    
    **Total Data**: {len(df):,}
    """)

# Apply filters (tanpa risk filter untuk detail page)
filtered_df = df[
    (df["Type"].isin(type_filter)) &
    (df["Process temperature [K]"].between(temp_range[0], temp_range[1]))
]

# Apply risk filter hanya untuk page selain Detail Data
if risk_filter != "Semua" and page != "📋 Detail Data":
    filtered_df = filtered_df[filtered_df["Recommendation"] == risk_filter]

# ====================== PAGES ======================
if page == "🏠 Dashboard":
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="margin:0;">🏭</h2>
            <h1 style="
            margin:0.5rem 0;
            color: white;
            text-shadow:
                -2px -2px 0 #000,
                2px -2px 0 #000,
                -2px 2px 0 #000,
                2px 2px 0 #000;
        ">{:,}</h1>
            <p style="
            margin:0;
            color: white;
            font-size: 1rem;
            text-shadow:
                -1px -1px 0 #000,
                1px -1px 0 #000,
                -1px 1px 0 #000,
                1px 1px 0 #000;
        ">Total Mesin</p>
        </div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        berat = len(df[df["Recommendation"] == "🚨 Perawatan Berat"])
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2 style="margin:0;">🚨</h2>
            <h1 style="
            margin:0.5rem 0;
            color: white;
            text-shadow:
                -2px -2px 0 #000,
                2px -2px 0 #000,
                -2px 2px 0 #000,
                2px 2px 0 #000;
        ">{:,}</h1>
            <p style="
            margin:0;
            color: white;
            font-size: 1rem;
            text-shadow:
                -1px -1px 0 #000,
                1px -1px 0 #000,
                -1px 1px 0 #000,
                1px 1px 0 #000;
        ">Risiko Berat</p>
        </div>
        """.format(berat), unsafe_allow_html=True)
    
    with col3:
        sedang = len(df[df["Recommendation"] == "⚙️ Perawatan Sedang"])
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffa751 0%, #ffe259 100%);">
            <h2 style="margin:0;">⚙️</h2>
            <h1 style="
            margin:0.5rem 0;
            color: white;
            text-shadow:
                -2px -2px 0 #000,
                2px -2px 0 #000,
                -2px 2px 0 #000,
                2px 2px 0 #000;
        ">{:,}</h1>
            <p style="
            margin:0;
            color: white;
            font-size: 1rem;
            text-shadow:
                -1px -1px 0 #000,
                1px -1px 0 #000,
                -1px 1px 0 #000,
                1px 1px 0 #000;
        ">Risiko Sedang</p>
        </div>
        </div>
        """.format(sedang), unsafe_allow_html=True)
    
    with col4:
        aman = len(df[df["Recommendation"] == "✅ Aman"])
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #0ba360 0%, #3cba92 100%);">
            <h2 style="margin:0;">✅</h2>
            <h1 style="
            margin:0.5rem 0;
            color: white;
            text-shadow:
                -2px -2px 0 #000,
                2px -2px 0 #000,
                -2px 2px 0 #000,
                2px 2px 0 #000;
        ">{:,}</h1>
            <p style="
            margin:0;
            color: white;
            font-size: 1rem;
            text-shadow:
                -1px -1px 0 #000,
                1px -1px 0 #000,
                -1px 1px 0 #000,
                1px 1px 0 #000;
        ">Aman</p>
        </div>
        </div>
        """.format(aman), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribusi Rekomendasi")
        rec_counts = df["Recommendation"].value_counts()
        fig_pie = px.pie(
            values=rec_counts.values,
            names=rec_counts.index,
            color_discrete_sequence=['#0ba360', '#ffa751', '#f5576c'],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Probabilitas Kegagalan")
        fig_hist = px.histogram(
            df,
            x="Failure_Prob",
            nbins=50,
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            xaxis_title="Probabilitas Kegagalan",
            yaxis_title="Jumlah Mesin",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Risk by Machine Type
    st.markdown("### 🏭 Analisis Risiko per Tipe Mesin")
    risk_by_type = df.groupby("Type").agg({
        "Failure_Prob": "mean",
        "Machine failure": "sum"
    }).reset_index()
    risk_by_type.columns = ["Type", "Avg_Risk", "Total_Failures"]
    
    fig_bar = px.bar(
        risk_by_type,
        x="Type",
        y="Avg_Risk",
        color="Total_Failures",
        color_continuous_scale="Reds",
        text="Total_Failures"
    )
    fig_bar.update_layout(
        xaxis_title="Tipe Mesin",
        yaxis_title="Rata-rata Probabilitas Kegagalan",
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "📋 Detail Data":
    st.markdown("## 📋 Detail Data Mesin")
    
    # Filter controls di bagian atas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detail_risk_filter = st.selectbox(
            "Filter Status",
            ["Semua", "✅ Aman", "⚙️ Perawatan Sedang", "🚨 Perawatan Berat"],
            key="detail_risk"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Urutkan berdasarkan", 
            ["Failure_Prob", "Tool wear [min]", "Torque [Nm]"],
            key="sort_by"
        )
    
    with col3:
        sort_order = st.radio(
            "Urutan", 
            ["Descending", "Ascending"], 
            horizontal=True,
            key="sort_order"
        )
    
    # Apply filter khusus untuk detail data
    detail_filtered_df = filtered_df.copy()
    if detail_risk_filter != "Semua":
        detail_filtered_df = detail_filtered_df[detail_filtered_df["Recommendation"] == detail_risk_filter]
    
    # Metric dan download button
    col_metric, col_download = st.columns([1, 2])
    with col_metric:
        st.metric("Data Terfilter", f"{len(detail_filtered_df):,} dari {len(df):,}")
    
    with col_download:
        csv = detail_filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name='maintenance_data.csv',
            mime='text/csv',
        )
    
    # Display data dengan sorting
    display_df = detail_filtered_df[[
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Failure_Prob", "Recommendation"
    ]].sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
    
    # Color coding function
    def color_recommendation(val):
        if val == "🚨 Perawatan Berat":
            return 'background-color: #ffcccc'
        elif val == "⚙️ Perawatan Sedang":
            return 'background-color: #fff4cc'
        else:
            return 'background-color: #ccffcc'
    
    # Apply styling and display
    styled_df = display_df.style.applymap(color_recommendation, subset=['Recommendation'])
    st.dataframe(styled_df, use_container_width=True, height=600)

elif page == "📊 Model Performance":
    st.markdown("## 📊 Performa Model XGBoost")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report['accuracy']:.2%}")
    col2.metric("Precision (Fail)", f"{report.get('1', {}).get('precision', 0):.2%}")
    col3.metric("Recall (Fail)", f"{report.get('1', {}).get('recall', 0):.2%}")
    col4.metric("F1-Score (Fail)", f"{report.get('1', {}).get('f1-score', 0):.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Failure', 'Failure'],
            y=['No Failure', 'Failure'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']), use_container_width=True)

elif page == "🔍 Feature Analysis":
    st.markdown("## 🔍 Analisis Fitur")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Importance", "🔗 Korelasi", "📈 Distribusi", "🎯 Top Machines"])
    
    with tab1:
        st.markdown("### 🏆 Feature Importance")
        fig_imp = px.bar(
            feature_importance_df.head(10),
            x='Importance',
            y='Fitur',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        
        if len(feature_importance_df) >= 3:
            st.success(f"""
            🧠 **Interpretasi Otomatis:**
            
            Fitur paling berpengaruh: **{feature_importance_df.iloc[0]['Fitur']}**
            
            Diikuti oleh: **{feature_importance_df.iloc[1]['Fitur']}** dan **{feature_importance_df.iloc[2]['Fitur']}**
            
            Fokuskan monitoring pada fitur-fitur ini untuk deteksi dini kegagalan mesin.
            """)
    
    with tab2:
        st.markdown("### 🔗 Correlation Heatmap")
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Distribusi Fitur")
        selected_feature = st.selectbox("Pilih fitur:", num_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_dist = px.histogram(
                df,
                x=selected_feature,
                nbins=50,
                marginal='box',
                color_discrete_sequence=['#667eea']
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                df,
                x='Recommendation',
                y=selected_feature,
                color='Recommendation',
                color_discrete_map={
                    '✅ Aman': '#0ba360',
                    '⚙️ Perawatan Sedang': '#ffa751',
                    '🚨 Perawatan Berat': '#f5576c'
                }
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔥 Top 10 Mesin Berisiko Tinggi")
        top10 = df.nlargest(10, "Failure_Prob")[[
            "Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "Failure_Prob", "Recommendation"
        ]]
        st.dataframe(top10, use_container_width=True)

elif page == "🤖 Prediction Simulator":
    st.markdown("## 🤖 Simulator Prediksi Kegagalan")
    st.info("Masukkan parameter mesin untuk memprediksi probabilitas kegagalan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌡️ Parameter Suhu & Mesin")
        machine_type = st.selectbox("Tipe Mesin", df["Type"].unique())
        air_temp = st.slider("Air Temperature (K)", 
                            float(df["Air temperature [K]"].min()),
                            float(df["Air temperature [K]"].max()),
                            float(df["Air temperature [K]"].mean()))
        process_temp = st.slider("Process Temperature (K)",
                                float(df["Process temperature [K]"].min()),
                                float(df["Process temperature [K]"].max()),
                                float(df["Process temperature [K]"].mean()))
    
    with col2:
        st.markdown("#### ⚙️ Parameter Operasional")
        rotation_speed = st.slider("Rotational Speed (rpm)",
                                   int(df["Rotational speed [rpm]"].min()),
                                   int(df["Rotational speed [rpm]"].max()),
                                   int(df["Rotational speed [rpm]"].mean()))
        torque = st.slider("Torque (Nm)",
                          float(df["Torque [Nm]"].min()),
                          float(df["Torque [Nm]"].max()),
                          float(df["Torque [Nm]"].mean()))
        tool_wear = st.slider("Tool Wear (min)",
                             int(df["Tool wear [min]"].min()),
                             int(df["Tool wear [min]"].max()),
                             int(df["Tool wear [min]"].mean()))
    
    # Failure mode checkboxes (optional - set to 0 by default)
    with st.expander("🔧 Mode Kegagalan (Opsional - Default: Tidak Ada)"):
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            twf = st.checkbox("TWF (Tool Wear Failure)", value=False)
        with col_b:
            hdf = st.checkbox("HDF (Heat Dissipation Failure)", value=False)
        with col_c:
            pwf = st.checkbox("PWF (Power Failure)", value=False)
        with col_d:
            osf = st.checkbox("OSF (Overstrain Failure)", value=False)
        with col_e:
            rnf = st.checkbox("RNF (Random Failure)", value=False)
    
    if st.button("🔮 Prediksi", type="primary", use_container_width=True):
        # Buat input data dengan semua kolom yang diperlukan
        input_data = pd.DataFrame({
            "Type": [machine_type],
            "Air temperature [K]": [air_temp],
            "Process temperature [K]": [process_temp],
            "Rotational speed [rpm]": [rotation_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
            "TWF": [1 if twf else 0],
            "HDF": [1 if hdf else 0],
            "PWF": [1 if pwf else 0],
            "OSF": [1 if osf else 0],
            "RNF": [1 if rnf else 0]
        })
        
        prob = model.predict_proba(input_data)[0, 1]
        recommendation = get_recommendation(prob)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                <h2>Probabilitas Kegagalan</h2>
                <h1 style="font-size: 4rem; margin: 1rem 0;">{prob:.1%}</h1>
                <h3>{recommendation}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Level", 'font': {'size': 24}},
                delta={'reference': 30},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#0ba360'},
                        {'range': [30, 60], 'color': '#ffa751'},
                        {'range': [60, 100], 'color': '#f5576c'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Recommendation details
        if prob > 0.6:
            st.error("⚠️ **Tindakan Segera Diperlukan!** Mesin memerlukan perawatan berat segera untuk mencegah kegagalan.")
        elif prob > 0.3:
            st.warning("⚠️ **Perhatian!** Jadwalkan perawatan preventif dalam waktu dekat.")
        else:
            st.success("✅ **Kondisi Baik!** Mesin dalam kondisi operasional yang aman.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏭 Industrial Maintenance Recommender Dashboard | Powered by XGBoost & Streamlit</p>
</div>
""", unsafe_allow_html=True)