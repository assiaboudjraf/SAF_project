import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from scipy import stats

# Configuration de la page
st.set_page_config(
    page_title="Safran | Analyse Boursière Annuelle 2025-2026",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Couleurs thématiques Safran (rouge/bleu aéronautique)
SAFRAN_RED = "#E4002B"
SAFRAN_BLUE = "#003D7A"
BG_COLOR = "#0A1929"
SECOND_BG_COLOR = "#1A2332"
TEXT_COLOR = "#B0BEC5"
ACCENT_COLOR = "#00B8D4"

# Style CSS personnalisé
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    .main {{
        background: linear-gradient(135deg, {BG_COLOR} 0%, {SECOND_BG_COLOR} 100%);
        color: {TEXT_COLOR};
        font-family: 'Roboto', sans-serif;
    }}
    
    h1, h2, h3, h4 {{
        color: {SAFRAN_RED};
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stMetric {{
        background: linear-gradient(135deg, {SECOND_BG_COLOR} 0%, {BG_COLOR} 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {SAFRAN_RED};
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    
    .stMetric label {{
        color: {ACCENT_COLOR} !important;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        color: white !important;
        font-size: 2rem;
        font-weight: 700;
    }}
    
    .stButton>button {{
        background: linear-gradient(90deg, {SAFRAN_RED} 0%, {SAFRAN_BLUE} 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(228, 0, 43, 0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(228, 0, 43, 0.5);
    }}
    
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, {SECOND_BG_COLOR} 0%, {BG_COLOR} 100%);
    }}
    
    .header-container {{
        background: linear-gradient(90deg, {SAFRAN_BLUE} 0%, {SAFRAN_RED} 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }}
    
    .stat-box {{
        background: {SECOND_BG_COLOR};
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid {SAFRAN_RED};
        margin: 0.5rem 0;
        transition: all 0.3s;
    }}
    
    .stat-box:hover {{
        transform: scale(1.02);
        border-color: {ACCENT_COLOR};
        box-shadow: 0 4px 12px rgba(0, 184, 212, 0.3);
    }}
    
    .info-card {{
        background: linear-gradient(135deg, {SECOND_BG_COLOR} 0%, {BG_COLOR} 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid {ACCENT_COLOR};
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: {SECOND_BG_COLOR};
        border-radius: 10px;
        padding: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {TEXT_COLOR};
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {SAFRAN_RED} 0%, {SAFRAN_BLUE} 100%);
        color: white;
    }}
    
    /* Selectbox styling */
    .stSelectbox > div > div {{
        background-color: {SECOND_BG_COLOR};
        border: 2px solid {SAFRAN_BLUE};
        border-radius: 10px;
        color: white;
    }}
    
    .stSelectbox label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
    }}
    
    /* Alert boxes */
    .stAlert {{
        background-color: {SECOND_BG_COLOR};
        border-left: 4px solid {SAFRAN_RED};
    }}
    </style>
""", unsafe_allow_html=True)

# Chargement des données avec gestion d'erreur
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("SAFRAN_data_bourse.txt", sep="\t")
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
        df = df.sort_values('date')
        
        # Calcul des indicateurs techniques
        df['MA_20'] = df['clot'].rolling(window=20).mean()
        df['MA_50'] = df['clot'].rolling(window=50).mean()
        df['Volatility'] = df['clot'].rolling(window=20).std()
        df['Daily_Return'] = df['clot'].pct_change() * 100
        
        # Bandes de Bollinger
        df['BB_Middle'] = df['clot'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['clot'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['clot'].rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        delta = df['clot'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df, None
    except FileNotFoundError:
        return None, "❌ Erreur : Le fichier 'SAFRAN_data_bourse.txt' n'a pas été trouvé."
    except Exception as e:
        return None, f"❌ Erreur lors du chargement des données : {str(e)}"

# Chargement des données
df, error = load_data()

if error:
    st.error(error)
    st.info("💡 Assurez-vous que le fichier 'SAFRAN_data_bourse.txt' est présent dans le même répertoire.")
    st.stop()

# Header avec logo
LOGO_URL = "https://www.1min30.com/wp-content/uploads/2018/05/Couleur-logo-Safran.jpg"
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: 2rem 0;">
            <img src="{LOGO_URL}" style="max-width: 100%; max-height: 160px; object-fit: contain;">
        </div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown(f"""
        <div class="header-container">
            <h1 style="color: white; margin: 0;">SAFRAN</h1>
            <p style="color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9;">
                Analyse Boursière Annuelle | Janvier 2025 - Janvier 2026
            </p>
            <p style="color: white; font-size: 0.9rem; margin: 0.3rem 0 0 0; opacity: 0.7;">
                Leader mondial de l'aéronautique et de la défense
            </p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {SAFRAN_RED} 0%, {SAFRAN_BLUE} 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">SAFRAN</h2>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Analyse Boursière</p>
    </div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "NAVIGATION",
    ["Vue d'ensemble", "Analyse Technique", "Performance", "Indicateurs Avancés", "Données"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div class="info-card">
        <h4 style="color: {ACCENT_COLOR}; margin-top: 0;">À propos de Safran</h4>
        <p style="font-size: 0.85rem; line-height: 1.6;">
        Safran est un groupe international de haute technologie opérant dans les domaines de l'aéronautique, 
        de l'espace et de la défense. Leader mondial des moteurs d'avion et équipements aéronautiques.
        </p>
    </div>
""", unsafe_allow_html=True)

# Calcul des statistiques globales
current_price = df['clot'].iloc[-1]
start_price = df['clot'].iloc[0]
variation_total = ((current_price - start_price) / start_price) * 100
max_price = df['clot'].max()
min_price = df['clot'].min()
avg_volume = df['vol'].mean()
total_volume = df['vol'].sum()

# ===========================
# VUE D'ENSEMBLE
# ===========================
if section == "Vue d'ensemble":
    st.header("Vue d'Ensemble du Titre Safran")
    
    # Récapitulatif annuel
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, {SECOND_BG_COLOR} 0%, {BG_COLOR} 100%); 
                    padding: 2rem; border-radius: 15px; border: 2px solid {SAFRAN_RED}; margin-bottom: 2rem;">
            <h2 style="color: {SAFRAN_RED}; margin-top: 0; text-align: center;">
                ANNÉE BOURSIÈRE 2025-2026
            </h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <p style="color: {ACCENT_COLOR}; margin: 0; font-size: 0.9rem;">Période d'analyse</p>
                    <p style="color: white; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">12 mois</p>
                    <p style="color: {TEXT_COLOR}; margin: 0; font-size: 0.8rem;">Jan 2025 - Jan 2026</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: {ACCENT_COLOR}; margin: 0; font-size: 0.9rem;">Jours de cotation</p>
                    <p style="color: white; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">{len(df)}</p>
                    <p style="color: {TEXT_COLOR}; margin: 0; font-size: 0.8rem;">jours ouvrés</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: {ACCENT_COLOR}; margin: 0; font-size: 0.9rem;">Performance annuelle</p>
                    <p style="color: {'#4CAF50' if variation_total > 0 else '#F44336'}; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">
                        {variation_total:+.2f}%
                    </p>
                    <p style="color: {TEXT_COLOR}; margin: 0; font-size: 0.8rem;">{start_price:.2f}€ → {current_price:.2f}€</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: {ACCENT_COLOR}; margin: 0; font-size: 0.9rem;">Volume total annuel</p>
                    <p style="color: white; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">{total_volume/1000000:.1f}M</p>
                    <p style="color: {TEXT_COLOR}; margin: 0; font-size: 0.8rem;">actions échangées</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Cours Actuel",
            f"{current_price:.2f} €",
            f"{variation_total:+.2f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Plus Haut",
            f"{max_price:.2f} €",
            f"+{((max_price - current_price) / current_price * 100):.1f}%"
        )
    
    with col3:
        st.metric(
            "Plus Bas",
            f"{min_price:.2f} €",
            f"{((min_price - current_price) / current_price * 100):.1f}%"
        )
    
    with col4:
        st.metric(
            "Volume Moyen",
            f"{avg_volume/1000:.0f}K",
            "actions/jour"
        )
    
    with col5:
        amplitude = max_price - min_price
        st.metric(
            "Amplitude",
            f"{amplitude:.2f} €",
            f"{(amplitude/min_price*100):.1f}%"
        )
    
    st.markdown("---")
    
    # Graphique principal
    st.subheader("Évolution du Cours avec Moyennes Mobiles")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['ouv'],
        high=df['haut'],
        low=df['bas'],
        close=df['clot'],
        name='Safran',
        increasing_line_color=SAFRAN_RED,
        decreasing_line_color=SAFRAN_BLUE
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA_20'],
        name='MM 20 jours',
        line=dict(color=ACCENT_COLOR, width=2),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA_50'],
        name='MM 50 jours',
        line=dict(color='#FFD700', width=2),
        opacity=0.8
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=600,
        xaxis_title="Date",
        yaxis_title="Prix (€)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par période
    st.markdown("---")
    st.subheader("Analyse Annuelle par Trimestre")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df['Trimestre'] = df['date'].dt.to_period('Q').astype(str)
        vol_trimestriel = df.groupby('Trimestre')['vol'].sum().reset_index()
        
        fig_vol = px.bar(
            vol_trimestriel,
            x='Trimestre',
            y='vol',
            title="Volume de transactions par trimestre",
            labels={'vol': 'Volume total', 'Trimestre': 'Trimestre'},
            color='vol',
            color_continuous_scale=[[0, SAFRAN_BLUE], [1, SAFRAN_RED]],
            text='vol'
        )
        fig_vol.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_vol.update_layout(
            template='plotly_dark',
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=SECOND_BG_COLOR,
            showlegend=False,
            yaxis_title="Volume"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        df_temp = df.copy()
        df_temp['Trimestre'] = df_temp['date'].dt.to_period('Q')
        perf_trimestrielle = df_temp.groupby('Trimestre').agg({
            'clot': ['first', 'last'],
            'haut': 'max',
            'bas': 'min'
        }).reset_index()
        perf_trimestrielle.columns = ['Trimestre', 'Ouverture', 'Cloture', 'Plus_Haut', 'Plus_Bas']
        perf_trimestrielle['Performance'] = ((perf_trimestrielle['Cloture'] - perf_trimestrielle['Ouverture']) / perf_trimestrielle['Ouverture'] * 100)
        perf_trimestrielle['Trimestre'] = perf_trimestrielle['Trimestre'].astype(str)
        
        fig_perf = px.bar(
            perf_trimestrielle,
            x='Trimestre',
            y='Performance',
            title="Performance trimestrielle (%)",
            labels={'Performance': 'Performance (%)', 'Trimestre': 'Trimestre'},
            color='Performance',
            color_continuous_scale=[[0, SAFRAN_BLUE], [0.5, 'white'], [1, SAFRAN_RED]],
            text='Performance'
        )
        fig_perf.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_perf.update_layout(
            template='plotly_dark',
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=SECOND_BG_COLOR,
            showlegend=False,
            yaxis_title="Performance (%)"
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Statistiques trimestrielles
    st.markdown("---")
    st.subheader("Résumé Trimestriel Détaillé")
    
    trimestre_stats = df.groupby(df['date'].dt.to_period('Q')).agg({
        'clot': ['first', 'last', 'min', 'max', 'mean'],
        'vol': 'sum',
        'Daily_Return': 'std'
    }).reset_index()
    
    trimestre_stats.columns = ['Trimestre', 'Prix_Début', 'Prix_Fin', 'Plus_Bas', 'Plus_Haut', 'Prix_Moyen', 'Volume_Total', 'Volatilité']
    trimestre_stats['Performance_%'] = ((trimestre_stats['Prix_Fin'] - trimestre_stats['Prix_Début']) / trimestre_stats['Prix_Début'] * 100)
    trimestre_stats['Trimestre'] = trimestre_stats['Trimestre'].astype(str)
    
    def color_performance(val):
        if pd.isna(val):
            return ''
        color = '#4CAF50' if val > 0 else '#F44336' if val < 0 else '#FFA726'
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    st.dataframe(
        trimestre_stats.style.format({
            'Prix_Début': '{:.2f} €',
            'Prix_Fin': '{:.2f} €',
            'Plus_Bas': '{:.2f} €',
            'Plus_Haut': '{:.2f} €',
            'Prix_Moyen': '{:.2f} €',
            'Volume_Total': '{:,.0f}',
            'Volatilité': '{:.2f}%',
            'Performance_%': '{:+.2f}%'
        }).applymap(color_performance, subset=['Performance_%']),
        use_container_width=True,
        height=300
    )

# ===========================
# ANALYSE TECHNIQUE
# ===========================
elif section == "Analyse Technique":
    st.header("Analyse Technique Approfondie")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Sélectionnez la période d'analyse")
    with col2:
        period_options = {
            "1 Mois": 30,
            "3 Mois": 90,
            "6 Mois": 180,
            "1 An": 365
        }
        selected_period = st.selectbox("Période", list(period_options.keys()), index=3)
        days_back = period_options[selected_period]
    
    df_period = df.tail(min(days_back, len(df)))
    
    st.subheader("Bandes de Bollinger")
    
    fig_bb = go.Figure()
    
    fig_bb.add_trace(go.Scatter(
        x=df_period['date'],
        y=df_period['BB_Upper'],
        name='Bande Supérieure',
        line=dict(color='rgba(228, 0, 43, 0.3)', width=1),
        fill=None
    ))
    
    fig_bb.add_trace(go.Scatter(
        x=df_period['date'],
        y=df_period['BB_Lower'],
        name='Bande Inférieure',
        line=dict(color='rgba(228, 0, 43, 0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(228, 0, 43, 0.1)'
    ))
    
    fig_bb.add_trace(go.Scatter(
        x=df_period['date'],
        y=df_period['BB_Middle'],
        name='Moyenne Mobile',
        line=dict(color=ACCENT_COLOR, width=2)
    ))
    
    fig_bb.add_trace(go.Scatter(
        x=df_period['date'],
        y=df_period['clot'],
        name='Cours de clôture',
        line=dict(color='white', width=2)
    ))
    
    fig_bb.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=500,
        xaxis_title="Date",
        yaxis_title="Prix (€)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_bb, use_container_width=True)
    
    # RSI
    st.markdown("---")
    st.subheader("RSI (Relative Strength Index)")
    
    fig_rsi = go.Figure()
    
    fig_rsi.add_trace(go.Scatter(
        x=df_period['date'],
        y=df_period['RSI'],
        name='RSI',
        line=dict(color=SAFRAN_RED, width=2)
    ))
    
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Surachat (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Survente (30)")
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig_rsi.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=400,
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    current_rsi = df_period['RSI'].dropna().iloc[-1] if not df_period['RSI'].dropna().empty else 50
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI Actuel", f"{current_rsi:.1f}")
    
    with col2:
        if current_rsi > 70:
            rsi_signal = "⚠️ SURACHAT"
            rsi_color = "🔴"
        elif current_rsi < 30:
            rsi_signal = "💡 SURVENTE"
            rsi_color = "🟢"
        else:
            rsi_signal = "✅ NEUTRE"
            rsi_color = "⚪"
        st.metric("Signal RSI", f"{rsi_color} {rsi_signal}")
    
    with col3:
        rsi_avg = df_period['RSI'].mean()
        st.metric("RSI Moyen (période)", f"{rsi_avg:.1f}")
    
    # Volatilité
    st.markdown("---")
    st.subheader("Analyse de la Volatilité")
    
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Scatter(
        x=df_period['date'],
        y=df_period['Volatility'],
        name='Volatilité (écart-type 20j)',
        line=dict(color=ACCENT_COLOR, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(0, 184, 212, 0.2)'
    ))
    
    fig_vol.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=400,
        xaxis_title="Date",
        yaxis_title="Volatilité (€)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)

# ===========================
# PERFORMANCE
# ===========================
elif section == "Performance":
    st.header("Analyse de Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="stat-box">
                <h3 style="color: {SAFRAN_RED}; margin-top: 0;">Performance Annuelle</h3>
                <h1 style="color: {'#4CAF50' if variation_total > 0 else '#F44336'}; margin: 0.5rem 0;">
                    {variation_total:+.2f}%
                </h1>
                <p style="margin: 0; opacity: 0.8;">De {start_price:.2f}€ à {current_price:.2f}€</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if not df['Daily_Return'].dropna().empty:
            best_day = df.loc[df['Daily_Return'].idxmax()]
            st.markdown(f"""
                <div class="stat-box">
                    <h3 style="color: {SAFRAN_RED}; margin-top: 0;">Meilleure Journée</h3>
                    <h1 style="color: #4CAF50; margin: 0.5rem 0;">
                        +{best_day['Daily_Return']:.2f}%
                    </h1>
                    <p style="margin: 0; opacity: 0.8;">{best_day['date'].strftime('%d/%m/%Y')}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if not df['Daily_Return'].dropna().empty:
            worst_day = df.loc[df['Daily_Return'].idxmin()]
            st.markdown(f"""
                <div class="stat-box">
                    <h3 style="color: {SAFRAN_RED}; margin-top: 0;">Pire Journée</h3>
                    <h1 style="color: #F44336; margin: 0.5rem 0;">
                        {worst_day['Daily_Return']:.2f}%
                    </h1>
                    <p style="margin: 0; opacity: 0.8;">{worst_day['date'].strftime('%d/%m/%Y')}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribution des rendements
    st.subheader("Distribution des Rendements Quotidiens")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df.dropna(subset=['Daily_Return']),
            x='Daily_Return',
            nbins=50,
            title="Histogramme des rendements quotidiens",
            labels={'Daily_Return': 'Rendement quotidien (%)'},
            color_discrete_sequence=[SAFRAN_RED]
        )
        fig_hist.update_layout(
            template='plotly_dark',
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=SECOND_BG_COLOR,
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df.dropna(subset=['Daily_Return']),
            y='Daily_Return',
            title="Boîte à moustaches des rendements",
            labels={'Daily_Return': 'Rendement quotidien (%)'},
            color_discrete_sequence=[ACCENT_COLOR]
        )
        fig_box.update_layout(
            template='plotly_dark',
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=SECOND_BG_COLOR,
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Rendements cumulés
    st.markdown("---")
    st.subheader("Rendements Cumulés")
    
    df['Cumulative_Return'] = (1 + df['Daily_Return']/100).cumprod() - 1
    
    fig_cumul = go.Figure()
    
    fig_cumul.add_trace(go.Scatter(
        x=df['date'],
        y=df['Cumulative_Return'] * 100,
        name='Rendement cumulé',
        line=dict(color=SAFRAN_RED, width=3),
        fill='tozeroy',
        fillcolor=f'rgba(228, 0, 43, 0.2)'
    ))
    
    fig_cumul.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    
    fig_cumul.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=500,
        xaxis_title="Date",
        yaxis_title="Rendement cumulé (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_cumul, use_container_width=True)
    
    # Statistiques
    st.markdown("---")
    st.subheader("Statistiques Détaillées")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_return = df['Daily_Return'].mean()
        st.metric("Rendement Quotidien Moyen", f"{avg_return:.3f}%")
    
    with col2:
        std_return = df['Daily_Return'].std()
        st.metric("Écart-type des Rendements", f"{std_return:.3f}%")
    
    with col3:
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return != 0 else 0
        st.metric("Ratio de Sharpe (annualisé)", f"{sharpe:.2f}")
    
    with col4:
        positive_days = (df['Daily_Return'] > 0).sum()
        total_days = len(df.dropna(subset=['Daily_Return']))
        win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
        st.metric("Taux de Jours Positifs", f"{win_rate:.1f}%")

# ===========================
# INDICATEURS AVANCÉS
# ===========================
elif section == "Indicateurs Avancés":
    st.header("Indicateurs Avancés")
    
    st.subheader("Niveaux de Support et Résistance")
    
    recent_data = df.tail(min(60, len(df)))
    resistance_levels = recent_data.nlargest(3, 'haut')['haut'].values
    support_levels = recent_data.nsmallest(3, 'bas')['bas'].values
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="info-card">
                <h4 style="color: {SAFRAN_RED}; margin-top: 0;">🔴 Niveaux de Résistance</h4>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 0.5rem; margin: 0.3rem 0; background: {SECOND_BG_COLOR}; border-radius: 5px;">
                        R1: <strong>{resistance_levels[0]:.2f} €</strong>
                    </li>
                    <li style="padding: 0.5rem; margin: 0.3rem 0; background: {SECOND_BG_COLOR}; border-radius: 5px;">
                        R2: <strong>{resistance_levels[1]:.2f} €</strong>
                    </li>
                    <li style="padding: 0.5rem; margin: 0.3rem 0; background: {SECOND_BG_COLOR}; border-radius: 5px;">
                        R3: <strong>{resistance_levels[2]:.2f} €</strong>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-card">
                <h4 style="color: #4CAF50; margin-top: 0;">🟢 Niveaux de Support</h4>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 0.5rem; margin: 0.3rem 0; background: {SECOND_BG_COLOR}; border-radius: 5px;">
                        S1: <strong>{support_levels[0]:.2f} €</strong>
                    </li>
                    <li style="padding: 0.5rem; margin: 0.3rem 0; background: {SECOND_BG_COLOR}; border-radius: 5px;">
                        S2: <strong>{support_levels[1]:.2f} €</strong>
                    </li>
                    <li style="padding: 0.5rem; margin: 0.3rem 0; background: {SECOND_BG_COLOR}; border-radius: 5px;">
                        S3: <strong>{support_levels[2]:.2f} €</strong>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Graphique S&R
    fig_sr = go.Figure()
    
    fig_sr.add_trace(go.Candlestick(
        x=recent_data['date'],
        open=recent_data['ouv'],
        high=recent_data['haut'],
        low=recent_data['bas'],
        close=recent_data['clot'],
        name='Safran',
        increasing_line_color=SAFRAN_RED,
        decreasing_line_color=SAFRAN_BLUE
    ))
    
    for i, r in enumerate(resistance_levels, 1):
        fig_sr.add_hline(
            y=r,
            line_dash="dash",
            line_color="red",
            annotation_text=f"R{i}: {r:.2f}€",
            annotation_position="right"
        )
    
    for i, s in enumerate(support_levels, 1):
        fig_sr.add_hline(
            y=s,
            line_dash="dash",
            line_color="green",
            annotation_text=f"S{i}: {s:.2f}€",
            annotation_position="right"
        )
    
    fig_sr.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=600,
        xaxis_title="Date",
        yaxis_title="Prix (€)",
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig_sr, use_container_width=True)
    
    # Momentum
    st.markdown("---")
    st.subheader("Momentum et Tendance")
    
    df['Momentum'] = df['clot'] - df['clot'].shift(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_momentum = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in df['Momentum'].fillna(0)]
        
        fig_momentum.add_trace(go.Bar(
            x=df['date'],
            y=df['Momentum'],
            name='Momentum (10j)',
            marker_color=colors
        ))
        
        fig_momentum.add_hline(y=0, line_color="white", opacity=0.5)
        
        fig_momentum.update_layout(
            template='plotly_dark',
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=SECOND_BG_COLOR,
            title="Momentum sur 10 jours",
            xaxis_title="Date",
            yaxis_title="Momentum (€)",
            showlegend=False
        )
        
        st.plotly_chart(fig_momentum, use_container_width=True)
    
    with col2:
        recent_30 = df.tail(min(30, len(df)))
        x = np.arange(len(recent_30))
        y = recent_30['clot'].values
        
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend_line = slope * x + intercept
            
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=recent_30['date'],
                y=y,
                name='Cours',
                line=dict(color='white', width=2)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=recent_30['date'],
                y=trend_line,
                name='Tendance',
                line=dict(color=SAFRAN_RED, width=3, dash='dash')
            ))
            
            fig_trend.update_layout(
                template='plotly_dark',
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=SECOND_BG_COLOR,
                title=f"Tendance sur {len(recent_30)} jours (R²={r_value**2:.3f})",
                xaxis_title="Date",
                yaxis_title="Prix (€)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            if slope > 0:
                trend_text = "📈 HAUSSIÈRE"
                trend_color = "#4CAF50"
            else:
                trend_text = "📉 BAISSIÈRE"
                trend_color = "#F44336"
            
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {SECOND_BG_COLOR}; border-radius: 10px; border: 2px solid {trend_color};">
                    <h3 style="color: {trend_color}; margin: 0;">{trend_text}</h3>
                    <p style="margin: 0.5rem 0 0 0;">Pente: {slope:.3f} €/jour</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Volume
    st.markdown("---")
    st.subheader("Analyse des Volumes")
    
    df['Volume_MA'] = df['vol'].rolling(window=20).mean()
    
    fig_vol_analysis = go.Figure()
    
    colors_vol = [SAFRAN_RED if df['clot'].iloc[i] > df['ouv'].iloc[i] else SAFRAN_BLUE 
                  for i in range(len(df))]
    
    fig_vol_analysis.add_trace(go.Bar(
        x=df['date'],
        y=df['vol'],
        name='Volume',
        marker_color=colors_vol,
        opacity=0.5
    ))
    
    fig_vol_analysis.add_trace(go.Scatter(
        x=df['date'],
        y=df['Volume_MA'],
        name='Moyenne Mobile Volume (20j)',
        line=dict(color=ACCENT_COLOR, width=2)
    ))
    
    fig_vol_analysis.update_layout(
        template='plotly_dark',
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECOND_BG_COLOR,
        height=400,
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_vol_analysis, use_container_width=True)

# ===========================
# DONNÉES
# ===========================
elif section == "Données":
    st.header("Données Brutes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Date de début", df['date'].min().date())
    
    with col2:
        end_date = st.date_input("Date de fin", df['date'].max().date())
    
    with col3:
        st.write("")
        st.write("")
        export_format = st.radio("Format d'export", ["CSV", "Excel"], horizontal=True)
    
    mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    df_filtered = df[mask].copy()
    
    st.markdown("---")
    st.subheader("Statistiques de la période sélectionnée")
    
    if len(df_filtered) > 0:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Nombre de jours", len(df_filtered))
        
        with col2:
            st.metric("Prix moyen", f"{df_filtered['clot'].mean():.2f} €")
        
        with col3:
            if len(df_filtered) > 1:
                period_return = ((df_filtered['clot'].iloc[-1] - df_filtered['clot'].iloc[0]) / df_filtered['clot'].iloc[0] * 100)
                st.metric("Performance", f"{period_return:+.2f}%")
            else:
                st.metric("Performance", "N/A")
        
        with col4:
            st.metric("Volume total", f"{df_filtered['vol'].sum()/1000000:.2f}M")
        
        with col5:
            volatility_period = df_filtered['clot'].std()
            st.metric("Volatilité", f"{volatility_period:.2f} €")
        
        st.markdown("---")
        st.subheader("Tableau de données")
        
        df_display = df_filtered[['date', 'ouv', 'haut', 'bas', 'clot', 'vol']].copy()
        df_display.columns = ['Date', 'Ouverture', 'Plus Haut', 'Plus Bas', 'Clôture', 'Volume']
        df_display['Date'] = df_display['Date'].dt.strftime('%d/%m/%Y')
        
        st.dataframe(
            df_display.style.format({
                'Ouverture': '{:.2f} €',
                'Plus Haut': '{:.2f} €',
                'Plus Bas': '{:.2f} €',
                'Clôture': '{:.2f} €',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True,
            height=500
        )
        
        st.markdown("---")
        st.subheader("Export des données")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Téléchargez les données filtrées au format souhaité")
        
        with col2:
            if export_format == "CSV":
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name=f"safran_data_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("💡 Pour exporter en Excel, installez openpyxl: pip install openpyxl")
    else:
        st.warning("⚠️ Aucune donnée disponible pour la période sélectionnée.")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, {SAFRAN_BLUE} 0%, {SAFRAN_RED} 100%); border-radius: 10px; margin-top: 2rem;">
        <h3 style="color: white; margin: 0;">SAFRAN - Analyse Boursière Annuelle</h3>
        <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">
            Période : Janvier 2025 - Janvier 2026 | Données quotidiennes
        </p>
        <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem; opacity: 0.7;">
            © 2026 Assia BOUDJRAF - Analyse réalisée à des fins éducatives uniquement
        </p>
    </div>
""", unsafe_allow_html=True)