"""
Dashboard Interattivo Gestione Scorte - Moretti S.p.A.
Streamlit App per monitoraggio real-time e decisioni
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Configurazione pagina
st.set_page_config(
    page_title="Moretti S.p.A. - Gestione Scorte",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main {padding-top: 0px;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-critica {background-color: #ffcccc; border-left: 5px solid #ff0000;}
    .alert-alta {background-color: #ffe6cc; border-left: 5px solid #ff9900;}
    .alert-media {background-color: #ffffcc; border-left: 5px solid #ffcc00;}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SIMULAZIONE DATI (in produzione: connessione DB)
# =====================================================

@st.cache_data
def carica_dati_simulati():
    """Carica dati simulati per demo"""
    
    # Catalogo prodotti
    prodotti = pd.DataFrame({
        'codice': ['CRZ001', 'CRZ002', 'MAT001', 'MAT002', 'RIA001', 'ELT001'],
        'nome': [
            'Carrozzina Pieghevole Standard',
            'Carrozzina Elettrica Kyara', 
            'Materasso Antidecubito Aria',
            'Cuscino Antidecubito Memory',
            'Deambulatore Pieghevole',
            'Saturimetro Professionale'
        ],
        'categoria': [
            'Carrozzine', 'Carrozzine', 'Antidecubito',
            'Antidecubito', 'Riabilitazione', 'Elettromedicali'
        ],
        'scorte_attuali': [45, 12, 28, 67, 89, 34],
        'scorta_minima': [20, 5, 15, 30, 40, 25],
        'scorta_sicurezza': [10, 3, 8, 15, 20, 12],
        'prezzo_medio': [280, 1850, 450, 85, 65, 120],
        'lead_time': [15, 25, 10, 7, 5, 10],
        'criticita': [5, 5, 5, 4, 4, 5]
    })
    
    # Storico vendite (ultimi 90 giorni)
    date_range = pd.date_range(end=datetime.now(), periods=90, freq='D')
    vendite = pd.DataFrame()
    
    for codice in prodotti['codice']:
        base = np.random.randint(5, 30)
        vendite[codice] = np.random.poisson(base, 90) * (1 + 0.2*np.sin(np.arange(90)*2*np.pi/30))
    
    vendite['data'] = date_range
    vendite = vendite.set_index('data')
    
    # Previsioni (prossimi 30 giorni)
    future_dates = pd.date_range(start=datetime.now()+timedelta(days=1), periods=30, freq='D')
    previsioni = pd.DataFrame()
    
    for codice in prodotti['codice']:
        base = vendite[codice].mean()
        previsioni[codice] = np.random.poisson(base, 30) * (1 + 0.1*np.random.randn(30))
        previsioni[f'{codice}_lower'] = previsioni[codice] * 0.8
        previsioni[f'{codice}_upper'] = previsioni[codice] * 1.2
    
    previsioni['data'] = future_dates
    previsioni = previsioni.set_index('data')
    
    # Ordini in corso
    ordini = pd.DataFrame({
        'id_ordine': ['ORD001', 'ORD002', 'ORD003'],
        'prodotto': ['CRZ001', 'MAT001', 'ELT001'],
        'quantita': [30, 50, 40],
        'fornitore': ['MedSupply Italia', 'AntiDecubito Pro', 'DiagnosticPro'],
        'data_ordine': [
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=1)
        ],
        'data_consegna_prevista': [
            datetime.now() + timedelta(days=10),
            datetime.now() + timedelta(days=7),
            datetime.now() + timedelta(days=9)
        ],
        'stato': ['In transito', 'Confermato', 'In elaborazione'],
        'costo_totale': [8400, 21000, 4800]
    })
    
    return prodotti, vendite, previsioni, ordini


# =====================================================
# COMPONENTI DASHBOARD
# =====================================================

def mostra_kpi_principali(prodotti, vendite, ordini):
    """Mostra KPI principali in alto"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        valore_magazzino = (prodotti['scorte_attuali'] * prodotti['prezzo_medio']).sum()
        st.metric(
            "💰 Valore Magazzino",
            f"€{valore_magazzino:,.0f}",
            f"+{np.random.randint(1,10)}%"
        )
    
    with col2:
        vendite_mese = vendite.tail(30).sum().sum()
        st.metric(
            "📦 Vendite Ultimo Mese",
            f"{vendite_mese:,.0f}",
            f"+{np.random.randint(5,15)}%"
        )
    
    with col3:
        prodotti_sotto_scorta = len(prodotti[prodotti['scorte_attuali'] < prodotti['scorta_minima']])
        st.metric(
            "⚠️ Prodotti Sotto Scorta",
            prodotti_sotto_scorta,
            delta=None if prodotti_sotto_scorta == 0 else f"+{prodotti_sotto_scorta}",
            delta_color="inverse"
        )
    
    with col4:
        ordini_attivi = len(ordini[ordini['stato'] != 'Consegnato'])
        st.metric(
            "🚚 Ordini in Corso",
            ordini_attivi,
            delta=None
        )
    
    with col5:
        service_level = (1 - prodotti_sotto_scorta/len(prodotti)) * 100
        st.metric(
            "✅ Service Level",
            f"{service_level:.1f}%",
            f"+{np.random.uniform(0.5, 2):.1f}%"
        )


def mostra_alerts(prodotti):
    """Mostra alert critici"""
    
    st.subheader("🚨 Alert e Notifiche")
    
    alerts = []
    
    # Check scorte critiche
    for _, prod in prodotti.iterrows():
        if prod['scorte_attuali'] < prod['scorta_minima']:
            urgenza = 'CRITICA' if prod['scorte_attuali'] < prod['scorta_sicurezza'] else 'ALTA'
            alerts.append({
                'urgenza': urgenza,
                'tipo': 'Scorte Basse',
                'messaggio': f"{prod['nome']}: {prod['scorte_attuali']} unità rimanenti (minimo: {prod['scorta_minima']})",
                'azione': f"Ordinare almeno {prod['scorta_minima'] * 2} unità"
            })
    
    # Mostra alerts
    if alerts:
        for alert in sorted(alerts, key=lambda x: 0 if x['urgenza']=='CRITICA' else 1):
            css_class = 'alert-critica' if alert['urgenza'] == 'CRITICA' else 'alert-alta'
            st.markdown(f"""
            <div class='alert-box {css_class}'>
                <strong>[{alert['urgenza']}] {alert['tipo']}</strong><br>
                {alert['messaggio']}<br>
                <em>Azione suggerita: {alert['azione']}</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Nessun alert critico al momento")


def grafico_scorte_prodotto(prodotti):
    """Grafico a barre scorte vs soglie"""
    
    fig = go.Figure()
    
    # Scorte attuali
    fig.add_trace(go.Bar(
        name='Scorte Attuali',
        x=prodotti['nome'],
        y=prodotti['scorte_attuali'],
        marker_color='lightblue'
    ))
    
    # Scorta minima
    fig.add_trace(go.Bar(
        name='Scorta Minima',
        x=prodotti['nome'],
        y=prodotti['scorta_minima'],
        marker_color='orange'
    ))
    
    # Scorta sicurezza
    fig.add_trace(go.Bar(
        name='Scorta Sicurezza',
        x=prodotti['nome'],
        y=prodotti['scorta_sicurezza'],
        marker_color='red'
    ))
    
    fig.update_layout(
        title="Livelli Scorte per Prodotto",
        barmode='group',
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    return fig


def grafico_trend_vendite(vendite, prodotto_selezionato):
    """Grafico trend vendite con previsioni"""
    
    fig = go.Figure()
    
    # Storico
    fig.add_trace(go.Scatter(
        x=vendite.index,
        y=vendite[prodotto_selezionato],
        mode='lines',
        name='Vendite Storiche',
        line=dict(color='blue')
    ))
    
    # Media mobile
    ma7 = vendite[prodotto_selezionato].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=vendite.index,
        y=ma7,
        mode='lines',
        name='Media Mobile 7gg',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Trend Vendite - {prodotto_selezionato}",
        xaxis_title="Data",
        yaxis_title="Unità Vendute",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def grafico_previsioni(previsioni, prodotto_selezionato):
    """Grafico previsioni con intervalli confidenza"""
    
    fig = go.Figure()
    
    # Previsione centrale
    fig.add_trace(go.Scatter(
        x=previsioni.index,
        y=previsioni[prodotto_selezionato],
        mode='lines',
        name='Previsione',
        line=dict(color='green')
    ))
    
    # Intervallo confidenza
    fig.add_trace(go.Scatter(
        x=previsioni.index,
        y=previsioni[f'{prodotto_selezionato}_upper'],
        mode='lines',
        name='Limite Superiore',
        line=dict(color='lightgreen', dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=previsioni.index,
        y=previsioni[f'{prodotto_selezionato}_lower'],
        mode='lines',
        name='Limite Inferiore',
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(color='lightgreen', dash='dot')
    ))
    
    fig.update_layout(
        title=f"Previsioni 30 giorni - {prodotto_selezionato}",
        xaxis_title="Data",
        yaxis_title="Unità Previste",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def tabella_ordini(ordini):
    """Mostra tabella ordini in corso"""
    
    st.subheader("📋 Ordini in Corso")
    
    # Formatta date
    ordini_display = ordini.copy()
    ordini_display['data_ordine'] = ordini_display['data_ordine'].dt.strftime('%d/%m/%Y')
    ordini_display['data_consegna_prevista'] = ordini_display['data_consegna_prevista'].dt.strftime('%d/%m/%Y')
    ordini_display['costo_totale'] = ordini_display['costo_totale'].apply(lambda x: f"€{x:,.2f}")
    
    # Colora per stato
    def color_stato(stato):
        colors = {
            'In elaborazione': 'background-color: #ffffcc',
            'Confermato': 'background-color: #ccffcc',
            'In transito': 'background-color: #ccf2ff',
            'Consegnato': 'background-color: #e6e6e6'
        }
        return colors.get(stato, '')
    
    styled = ordini_display.style.applymap(
        color_stato,
        subset=['stato']
    )
    
    st.dataframe(styled, use_container_width=True)


def calcola_suggerimenti_riordino(prodotti, previsioni):
    """Calcola e mostra suggerimenti riordino"""
    
    st.subheader("💡 Suggerimenti Riordino")
    
    suggerimenti = []
    
    for _, prod in prodotti.iterrows():
        # Previsione domanda durante lead time
        if prod['codice'] in previsioni.columns:
            domanda_lead_time = previsioni[prod['codice']][:prod['lead_time']].sum()
            
            # Punto riordino
            punto_riordino = domanda_lead_time + prod['scorta_sicurezza']
            
            if prod['scorte_attuali'] <= punto_riordino:
                # EOQ semplificato
                domanda_media = previsioni[prod['codice']].mean()
                eoq = np.sqrt(2 * domanda_media * 365 * 50 / (prod['prezzo_medio'] * 0.2))
                
                suggerimenti.append({
                    'Prodotto': prod['nome'],
                    'Scorte Attuali': prod['scorte_attuali'],
                    'Punto Riordino': int(punto_riordino),
                    'Quantità Suggerita': int(eoq),
                    'Urgenza': '🔴 Alta' if prod['scorte_attuali'] < prod['scorta_sicurezza'] else '🟡 Media'
                })
    
    if suggerimenti:
        df_sugg = pd.DataFrame(suggerimenti)
        st.dataframe(df_sugg, use_container_width=True)
    else:
        st.info("📊 Nessun riordino necessario al momento")


# =====================================================
# PAGINA PRINCIPALE
# =====================================================

def main():
    # Header
    st.title("🏥 Moretti S.p.A. - Sistema Gestione Scorte Intelligente")
    st.markdown("---")
    
    # Carica dati
    prodotti, vendite, previsioni, ordini = carica_dati_simulati()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controlli")
        
        # Filtro categoria
        categorie = ['Tutte'] + list(prodotti['categoria'].unique())
        categoria_sel = st.selectbox("📂 Categoria", categorie)
        
        if categoria_sel != 'Tutte':
            prodotti_filtrati = prodotti[prodotti['categoria'] == categoria_sel]
        else:
            prodotti_filtrati = prodotti
        
        # Selezione prodotto per grafici
        prodotto_sel = st.selectbox(
            "📦 Prodotto per Analisi",
            prodotti_filtrati['codice'].tolist(),
            format_func=lambda x: prodotti[prodotti['codice']==x]['nome'].values[0]
        )
        
        st.markdown("---")
        
        # Parametri simulazione
        st.subheader("🎮 Parametri Simulazione")
        
        lead_time_mod = st.slider(
            "Lead Time Modifier (%)",
            min_value=50,
            max_value=150,
            value=100,
            step=10
        )
        
        domanda_mod = st.slider(
            "Domanda Modifier (%)",
            min_value=50,
            max_value=200,
            value=100,
            step=10
        )
        
        st.markdown("---")
        
        # Info sistema
        st.info("""
        **Sistema Features:**
        - Previsioni SARIMA/VAR
        - Ottimizzazione Multi-Fornitore
        - Alert Automatici
        - Integrazione Dati ISTAT
        - Analisi What-If
        """)
    
    # Layout principale
    
    # KPI Row
    mostra_kpi_principali(prodotti_filtrati, vendite, ordini)
    
    st.markdown("---")
    
    # Alert Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        mostra_alerts(prodotti_filtrati)
    
    with col2:
        st.subheader("📊 Livelli Scorte")
        st.plotly_chart(
            grafico_scorte_prodotto(prodotti_filtrati),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Grafici Analisi
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trend Vendite",
        "🔮 Previsioni",
        "📋 Ordini",
        "💡 Suggerimenti"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                grafico_trend_vendite(vendite, prodotto_sel),
                use_container_width=True
            )
        
        with col2:
            # Statistiche vendite
            st.subheader("📊 Statistiche Vendite")
            
            stats_vendite = pd.DataFrame({
                'Metrica': [
                    'Media Giornaliera',
                    'Deviazione Standard',
                    'Vendite Max',
                    'Vendite Min',
                    'Trend Ultimo Mese'
                ],
                'Valore': [
                    f"{vendite[prodotto_sel].mean():.1f}",
                    f"{vendite[prodotto_sel].std():.1f}",
                    f"{vendite[prodotto_sel].max():.0f}",
                    f"{vendite[prodotto_sel].min():.0f}",
                    f"{(vendite[prodotto_sel].tail(30).mean() - vendite[prodotto_sel].head(30).mean()) / vendite[prodotto_sel].head(30).mean() * 100:+.1f}%"
                ]
            })
            
            st.dataframe(stats_vendite, use_container_width=True, hide_index=True)
            
            # Distribuzione vendite
            fig_dist = px.histogram(
                vendite[prodotto_sel],
                nbins=20,
                title="Distribuzione Vendite",
                labels={'value': 'Unità', 'count': 'Frequenza'}
            )
            fig_dist.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                grafico_previsioni(previsioni, prodotto_sel),
                use_container_width=True
            )
        
        with col2:
            st.subheader("📈 Metriche Previsione")
            
            # Calcola metriche
            prev_totale = previsioni[prodotto_sel].sum()
            prev_media = previsioni[prodotto_sel].mean()
            prev_max = previsioni[prodotto_sel].max()
            confidence_range = (
                previsioni[f'{prodotto_sel}_upper'].mean() - 
                previsioni[f'{prodotto_sel}_lower'].mean()
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Totale 30gg", f"{prev_totale:.0f}")
                st.metric("Media Giornaliera", f"{prev_media:.1f}")
            with col_b:
                st.metric("Picco Previsto", f"{prev_max:.0f}")
                st.metric("Range Confidenza", f"±{confidence_range/2:.1f}")
            
            # Confronto con storico
            st.subheader("📊 Confronto con Storico")
            
            storico_media = vendite[prodotto_sel].tail(30).mean()
            variazione = (prev_media - storico_media) / storico_media * 100
            
            if variazione > 0:
                st.warning(f"⬆️ Aumento previsto del {variazione:.1f}% rispetto agli ultimi 30 giorni")
            else:
                st.success(f"⬇️ Diminuzione prevista del {abs(variazione):.1f}% rispetto agli ultimi 30 giorni")
    
    with tab3:
        tabella_ordini(ordini)
        
        # Timeline ordini
        st.subheader("📅 Timeline Consegne")
        
        fig_timeline = go.Figure()
        
        for _, ordine in ordini.iterrows():
            fig_timeline.add_trace(go.Scatter(
                x=[ordine['data_ordine'], ordine['data_consegna_prevista']],
                y=[ordine['prodotto'], ordine['prodotto']],
                mode='lines+markers',
                name=ordine['id_ordine'],
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig_timeline.update_layout(
            title="Timeline Ordini",
            xaxis_title="Data",
            yaxis_title="Prodotto",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab4:
        calcola_suggerimenti_riordino(prodotti_filtrati, previsioni)
        
        # Ottimizzazione fornitori
        st.subheader("🏭 Ottimizzazione Fornitori")
        
        # Simula confronto fornitori
        fornitori_comp = pd.DataFrame({
            'Fornitore': ['MedSupply Italia', 'EuroMedical', 'GlobalMed'],
            'Lead Time (gg)': [15, 12, 20],
            'Affidabilità': ['95%', '92%', '88%'],
            'Prezzo 1-10 unità': ['€300', '€310', '€295'],
            'Prezzo 50+ unità': ['€260', '€250', '€265'],
            'Rating': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐']
        })
        
        st.dataframe(fornitori_comp, use_container_width=True, hide_index=True)
        
        # Calcolo risparmio
        st.subheader("💰 Analisi Risparmio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risparmio Mensile Stimato",
                "€2,450",
                "+12%"
            )
        
        with col2:
            st.metric(
                "Riduzione Lead Time Medio",
                "-3 giorni",
                "-18%"
            )
        
        with col3:
            st.metric(
                "Miglioramento Service Level",
                "+5.2%",
                None
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
        Sistema Intelligente Gestione Scorte v2.0 | 
        Powered by ARIMA Forecaster | 
        © 2024 Moretti S.p.A.
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()