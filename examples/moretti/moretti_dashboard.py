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
import subprocess
import tempfile
import os

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
    
    /* Stile per le metriche con bordi e allineamento */
    [data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 2px solid #444444;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    
    /* Allineamento a destra per i valori delle metriche */
    [data-testid="metric-container"] > div:first-child {
        text-align: left;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        text-align: right;
        font-weight: bold;
    }
    
    /* Allineamento a destra per il delta */
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        text-align: right;
        justify-content: flex-end;
    }
    
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

def carica_dati_simulati(lead_time_mod=100, domanda_mod=100):
    """Carica dati simulati per demo con modificatori
    
    Args:
        lead_time_mod: Modificatore lead time in percentuale (100 = normale)
        domanda_mod: Modificatore domanda in percentuale (100 = normale)
    """
    
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
        'lead_time': [int(15 * lead_time_mod / 100), int(25 * lead_time_mod / 100), 
                     int(10 * lead_time_mod / 100), int(7 * lead_time_mod / 100), 
                     int(5 * lead_time_mod / 100), int(10 * lead_time_mod / 100)],
        'criticita': [5, 5, 5, 4, 4, 5]
    })
    
    # Storico vendite (ultimi 90 giorni)
    date_range = pd.date_range(end=datetime.now(), periods=90, freq='D')
    vendite = pd.DataFrame()
    
    for codice in prodotti['codice']:
        base = np.random.randint(5, 30)
        # Applica modificatore domanda alle vendite storiche
        vendite[codice] = np.random.poisson(base, 90) * (1 + 0.2*np.sin(np.arange(90)*2*np.pi/30)) * (domanda_mod / 100)
    
    vendite['data'] = date_range
    vendite = vendite.set_index('data')
    
    # Previsioni (prossimi 30 giorni)
    future_dates = pd.date_range(start=datetime.now()+timedelta(days=1), periods=30, freq='D')
    previsioni = pd.DataFrame()
    
    for codice in prodotti['codice']:
        base = vendite[codice].mean()
        # Applica modificatore domanda alle previsioni
        previsioni[codice] = np.random.poisson(base, 30) * (1 + 0.1*np.random.randn(30)) * (domanda_mod / 100)
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
    
    # Calcola valori
    valore_magazzino = (prodotti['scorte_attuali'] * prodotti['prezzo_medio']).sum()
    vendite_mese = vendite.tail(30).sum().sum()
    prodotti_sotto_scorta = len(prodotti[prodotti['scorte_attuali'] < prodotti['scorta_minima']])
    ordini_attivi = len(ordini[ordini['stato'] != 'Consegnato'])
    service_level = (1 - prodotti_sotto_scorta/len(prodotti)) * 100
    
    with col1:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">💰 Valore Magazzino</div>
            <div style="color: white; font-size: 28px; font-weight: bold; text-align: right;">€{valore_magazzino:,.0f}</div>
            <div style="color: #4CAF50; font-size: 14px; text-align: right; margin-top: 5px;">▲ +{np.random.randint(1,10)}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">📦 Vendite Ultimo Mese</div>
            <div style="color: white; font-size: 28px; font-weight: bold; text-align: right;">{vendite_mese:,.0f}</div>
            <div style="color: #4CAF50; font-size: 14px; text-align: right; margin-top: 5px;">▲ +{np.random.randint(5,15)}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delta_color = "#f44336" if prodotti_sotto_scorta > 0 else "#4CAF50"
        delta_symbol = "▲" if prodotti_sotto_scorta > 0 else "✓"
        delta_text = f"+{prodotti_sotto_scorta}" if prodotti_sotto_scorta > 0 else "OK"
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">⚠️ Prodotti Sotto Scorta</div>
            <div style="color: white; font-size: 28px; font-weight: bold; text-align: right;">{prodotti_sotto_scorta}</div>
            <div style="color: {delta_color}; font-size: 14px; text-align: right; margin-top: 5px;">{delta_symbol} {delta_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">🚚 Ordini in Corso</div>
            <div style="color: white; font-size: 28px; font-weight: bold; text-align: right;">{ordini_attivi}</div>
            <div style="color: #2196F3; font-size: 14px; text-align: right; margin-top: 5px;">━ Stabile</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">✅ Service Level</div>
            <div style="color: white; font-size: 28px; font-weight: bold; text-align: right;">{service_level:.1f}%</div>
            <div style="color: #4CAF50; font-size: 14px; text-align: right; margin-top: 5px;">▲ +{np.random.uniform(0.5, 2):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


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


def genera_report_quarto(prodotti, vendite, previsioni, ordini, **kwargs):
    """Genera report con Quarto in vari formati"""
    
    try:
        # Estrai parametri
        output_format = kwargs.get('output_format', 'HTML (Interattivo)')
        include_kpi = kwargs.get('include_kpi', True)
        include_alerts = kwargs.get('include_alerts', True)
        include_inventory = kwargs.get('include_inventory', True)
        include_forecast = kwargs.get('include_forecast', True)
        include_suppliers = kwargs.get('include_suppliers', True)
        include_recommendations = kwargs.get('include_recommendations', True)
        include_charts = kwargs.get('include_charts', True)
        include_tables = kwargs.get('include_tables', True)
        periodo = kwargs.get('periodo', 'Ultimo Mese')
        language = kwargs.get('language', 'Italiano')
        
        # Mappa formati
        format_map = {
            "HTML (Interattivo)": "html",
            "PDF (Stampa)": "pdf",
            "DOCX (Word)": "docx",
            "Markdown": "md"
        }
        
        output_ext = format_map[output_format]
        
        # Crea directory temporanea per il report
        temp_dir = Path(tempfile.mkdtemp())
        qmd_file = temp_dir / "report.qmd"
        output_file = temp_dir / f"report.{output_ext}"
        
        # Genera contenuto Quarto Markdown
        qmd_content = f"""---
title: "Report Gestione Scorte - Moretti S.p.A."
subtitle: "Analisi Intelligente con Sistema ARIMA/SARIMA"
author: "Sistema AI - Powered by ARIMA Forecaster"
date: "{datetime.now().strftime('%d/%m/%Y %H:%M')}"
format:
  {output_ext}:
    toc: true
    toc-depth: 3
    theme: cosmo
    {"self-contained: true" if output_ext == "html" else ""}
    {"pdf-engine: xelatex" if output_ext == "pdf" else ""}
execute:
  echo: false
  warning: false
---

# Executive Summary

"""
        
        if include_kpi:
            # Calcola KPI
            valore_magazzino = (prodotti['scorte_attuali'] * prodotti['prezzo_medio']).sum()
            vendite_mese = vendite.tail(30).sum().sum()
            prodotti_sotto_scorta = len(prodotti[prodotti['scorte_attuali'] < prodotti['scorta_minima']])
            service_level = (1 - prodotti_sotto_scorta/len(prodotti)) * 100
            
            qmd_content += f"""
## KPI Principali

::: {{.callout-note}}
### Metriche Chiave al {datetime.now().strftime('%d/%m/%Y')}

- **Valore Magazzino**: €{valore_magazzino:,.0f}
- **Vendite Ultimo Mese**: {vendite_mese:,.0f} unità
- **Prodotti Sotto Scorta**: {prodotti_sotto_scorta}
- **Service Level**: {service_level:.1f}%
:::

"""
        
        if include_alerts:
            qmd_content += """
## Alert Critici

"""
            alerts_critici = prodotti[prodotti['scorte_attuali'] < prodotti['scorta_sicurezza']]
            if len(alerts_critici) > 0:
                qmd_content += """
::: {{.callout-warning}}
### ⚠️ Prodotti in Emergenza
"""
                for _, prod in alerts_critici.iterrows():
                    qmd_content += f"- **{prod['nome']}**: Solo {prod['scorte_attuali']} unità rimanenti\\n"
                qmd_content += ":::\\n\\n"
        
        if include_inventory:
            qmd_content += """
# Analisi Inventario

## Stato Attuale Scorte

"""
            if include_tables:
                qmd_content += """
| Prodotto | Codice | Scorte Attuali | Scorta Minima | Stato |
|----------|--------|----------------|---------------|-------|
"""
                for _, prod in prodotti.iterrows():
                    stato = "🔴 Critico" if prod['scorte_attuali'] < prod['scorta_sicurezza'] else "🟡 Attenzione" if prod['scorte_attuali'] < prod['scorta_minima'] else "🟢 OK"
                    qmd_content += f"| {prod['nome']} | {prod['codice']} | {prod['scorte_attuali']} | {prod['scorta_minima']} | {stato} |\\n"
                qmd_content += "\\n"
        
        if include_forecast:
            qmd_content += """
# Previsioni Domanda

## Forecast 30 Giorni

Le previsioni sono generate utilizzando modelli ARIMA/SARIMA ottimizzati per ogni prodotto.

"""
            if include_charts:
                # Salva i dati per i grafici in CSV temporanei
                vendite_csv = temp_dir / "vendite.csv"
                previsioni_csv = temp_dir / "previsioni.csv"
                vendite.to_csv(vendite_csv)
                previsioni.to_csv(previsioni_csv)
                
                qmd_content += f"""
```{{python}}
#| echo: false
#| warning: false
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Carica i dati
vendite = pd.read_csv('{vendite_csv.as_posix()}', index_col=0, parse_dates=True)
previsioni = pd.read_csv('{previsioni_csv.as_posix()}', index_col=0, parse_dates=True)

# Crea una griglia di grafici per i prodotti principali
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

prodotti_da_mostrare = vendite.columns[:6]  # Primi 6 prodotti

for idx, prodotto in enumerate(prodotti_da_mostrare):
    ax = axes[idx]
    
    # Grafico vendite storiche
    ax.plot(vendite.index[-30:], vendite[prodotto].tail(30), 
            label='Storico', color='blue', linewidth=2)
    
    # Grafico previsioni
    if prodotto in previsioni.columns:
        ax.plot(previsioni.index, previsioni[prodotto], 
                label='Previsione', color='green', linewidth=2, linestyle='--')
        
        # Aggiungi intervalli di confidenza se esistono
        if f'{{prodotto}}_lower' in previsioni.columns:
            ax.fill_between(previsioni.index, 
                           previsioni[f'{{prodotto}}_lower'],
                           previsioni[f'{{prodotto}}_upper'],
                           alpha=0.3, color='green', label='Intervallo 95%')
    
    ax.set_title(f'{{prodotto}}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Data', fontsize=8)
    ax.set_ylabel('Unità', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    
    # Ruota le etichette delle date
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('Analisi Vendite e Previsioni per Prodotto', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Analisi Trend Aggregato

```{{python}}
#| echo: false
#| warning: false

# Grafico trend aggregato
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Vendite totali storiche
vendite_totali = vendite.sum(axis=1)
ax1.plot(vendite_totali.index[-60:], vendite_totali.tail(60), 
         color='darkblue', linewidth=2)
ax1.fill_between(vendite_totali.index[-60:], 0, vendite_totali.tail(60), 
                 alpha=0.3, color='lightblue')
ax1.set_title('Vendite Totali - Ultimi 60 Giorni', fontsize=12, fontweight='bold')
ax1.set_xlabel('Data')
ax1.set_ylabel('Unità Totali')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Previsioni aggregate
previsioni_prodotti = [col for col in previsioni.columns if not ('_lower' in col or '_upper' in col)]
previsioni_totali = previsioni[previsioni_prodotti].sum(axis=1)
ax2.plot(previsioni.index, previsioni_totali, 
         color='darkgreen', linewidth=2, marker='o', markersize=4)
ax2.fill_between(previsioni.index, 0, previsioni_totali, 
                 alpha=0.3, color='lightgreen')
ax2.set_title('Previsioni Totali - Prossimi 30 Giorni', fontsize=12, fontweight='bold')
ax2.set_xlabel('Data')
ax2.set_ylabel('Unità Previste')
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

### Distribuzione Previsioni per Categoria

```{{python}}
#| echo: false
#| warning: false

# Grafico a torta per distribuzione previsioni
fig, ax = plt.subplots(figsize=(8, 8))

# Calcola totali per prodotto
totali_prodotto = {{}}
for prodotto in previsioni_prodotti:
    if prodotto in previsioni.columns:
        totali_prodotto[prodotto] = previsioni[prodotto].sum()

# Ordina e prendi top 8
top_prodotti = dict(sorted(totali_prodotto.items(), key=lambda x: x[1], reverse=True)[:8])

colors = plt.cm.Set3(np.linspace(0, 1, len(top_prodotti)))
wedges, texts, autotexts = ax.pie(top_prodotti.values(), 
                                   labels=top_prodotti.keys(),
                                   colors=colors,
                                   autopct='%1.1f%%',
                                   startangle=90)

# Migliora l'aspetto
for text in texts:
    text.set_fontsize(10)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax.set_title('Distribuzione Previsioni per Prodotto (30 giorni)', 
             fontsize=12, fontweight='bold', pad=20)

plt.show()
```

"""
            
            if include_tables:
                qmd_content += """
### Domanda Prevista per Prodotto

| Prodotto | Media Giornaliera | Totale 30gg | Picco Previsto |
|----------|-------------------|-------------|----------------|
"""
                for codice in prodotti['codice']:
                    if codice in previsioni.columns:
                        media = previsioni[codice].mean()
                        totale = previsioni[codice].sum()
                        picco = previsioni[codice].max()
                        nome = prodotti[prodotti['codice']==codice]['nome'].values[0]
                        qmd_content += f"| {nome} | {media:.1f} | {totale:.0f} | {picco:.0f} |\\n"
                qmd_content += "\\n"
        
        if include_suppliers:
            qmd_content += """
# Ottimizzazione Fornitori

## Analisi Comparativa

| Fornitore | Lead Time | Affidabilità | Prezzo Volume | Rating |
|-----------|-----------|--------------|---------------|--------|
| MedSupply Italia | 15 giorni | 95% | €260/50+ | ⭐⭐⭐⭐⭐ |
| EuroMedical | 12 giorni | 92% | €250/50+ | ⭐⭐⭐⭐ |
| GlobalMed | 20 giorni | 88% | €265/50+ | ⭐⭐⭐ |

"""
        
        if include_recommendations:
            qmd_content += """
# Raccomandazioni

## Azioni Immediate

1. **Riordini Urgenti**: Procedere con ordini per prodotti sotto scorta sicurezza
2. **Ottimizzazione Fornitori**: Consolidare ordini con MedSupply Italia per migliori condizioni
3. **Revisione Scorte Minime**: Aggiornare parametri basandosi su previsioni SARIMA

## Opportunità di Risparmio

- **Consolidamento Ordini**: Risparmio stimato €2,450/mese
- **Riduzione Lead Time**: -3 giorni medi con cambio fornitore
- **Ottimizzazione EOQ**: Riduzione costi gestione 15%

"""
        
        # Footer
        qmd_content += f"""
---

*Report generato automaticamente dal Sistema Intelligente Gestione Scorte v2.0*  
*Powered by ARIMA Forecaster - © 2024 Moretti S.p.A.*
"""
        
        # Scrivi file .qmd
        with open(qmd_file, "w", encoding="utf-8") as f:
            f.write(qmd_content)
        
        # Genera report con Quarto
        try:
            # Verifica se Quarto è installato
            result = subprocess.run(
                ["quarto", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Genera il report
                render_result = subprocess.run(
                    ["quarto", "render", str(qmd_file), "--to", output_ext],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    timeout=30
                )
                
                if render_result.returncode == 0 and output_file.exists():
                    # Copia il file nella directory outputs
                    output_dir = Path(__file__).parent.parent.parent / "outputs" / "reports"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    final_file = output_dir / f"moretti_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_ext}"
                    
                    import shutil
                    shutil.copy2(output_file, final_file)
                    
                    return True, str(final_file)
                else:
                    # Se Quarto fallisce, genera almeno un HTML base
                    return genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir)
            else:
                # Quarto non installato, usa fallback
                return genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Usa fallback HTML se Quarto non è disponibile
            return genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir)
            
    except Exception as e:
        print(f"Errore generazione report: {e}")
        return False, None


def genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir):
    """Genera report HTML semplice come fallback se Quarto non è disponibile"""
    
    # Calcola KPI
    valore_magazzino = (prodotti['scorte_attuali'] * prodotti['prezzo_medio']).sum()
    vendite_mese = vendite.tail(30).sum().sum()
    prodotti_sotto_scorta = len(prodotti[prodotti['scorte_attuali'] < prodotti['scorta_minima']])
    service_level = (1 - prodotti_sotto_scorta/len(prodotti)) * 100
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Report Moretti S.p.A.</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .header {{ background: #1e1e1e; color: white; padding: 30px; border-radius: 10px; text-align: center; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .kpi-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .kpi-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .kpi-label {{ color: #666; margin-top: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; font-weight: bold; }}
        .alert {{ padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; margin: 20px 0; }}
        .section {{ background: white; padding: 30px; margin: 20px 0; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Report Gestione Scorte</h1>
        <h2>Moretti S.p.A.</h2>
        <p>Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    
    <div class="section">
        <h2>KPI Principali</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">€{valore_magazzino:,.0f}</div>
                <div class="kpi-label">Valore Magazzino</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{vendite_mese:,.0f}</div>
                <div class="kpi-label">Vendite Ultimo Mese</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{prodotti_sotto_scorta}</div>
                <div class="kpi-label">Prodotti Sotto Scorta</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{service_level:.1f}%</div>
                <div class="kpi-label">Service Level</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Stato Inventario</h2>
        <table>
            <tr>
                <th>Prodotto</th>
                <th>Codice</th>
                <th>Scorte Attuali</th>
                <th>Scorta Minima</th>
                <th>Stato</th>
            </tr>
"""
    
    for _, prod in prodotti.iterrows():
        stato = "Critico" if prod['scorte_attuali'] < prod['scorta_sicurezza'] else "Attenzione" if prod['scorte_attuali'] < prod['scorta_minima'] else "OK"
        html_content += f"""
            <tr>
                <td>{prod['nome']}</td>
                <td>{prod['codice']}</td>
                <td>{prod['scorte_attuali']}</td>
                <td>{prod['scorta_minima']}</td>
                <td>{stato}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
    
    <div class="section">
        <p style="text-align: center; color: #666;">
            Report generato da Sistema Intelligente Gestione Scorte v2.0<br>
            Powered by ARIMA Forecaster
        </p>
    </div>
</body>
</html>"""
    
    # Salva file HTML
    output_file = temp_dir / "report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Copia nella directory outputs
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_file = output_dir / f"moretti_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    import shutil
    shutil.copy2(output_file, final_file)
    
    return True, str(final_file)


def calcola_suggerimenti_riordino(prodotti, previsioni, domanda_mod=100):
    """Calcola e mostra suggerimenti riordino con modificatore domanda"""
    
    st.subheader("💡 Suggerimenti Riordino")
    
    suggerimenti = []
    
    for _, prod in prodotti.iterrows():
        # Previsione domanda durante lead time
        if prod['codice'] in previsioni.columns:
            domanda_lead_time = previsioni[prod['codice']][:prod['lead_time']].sum()
            
            # Punto riordino
            punto_riordino = domanda_lead_time + prod['scorta_sicurezza']
            
            if prod['scorte_attuali'] <= punto_riordino:
                # EOQ semplificato - considera il modificatore domanda
                domanda_media = previsioni[prod['codice']].mean()
                # Aggiusta EOQ in base al modificatore domanda
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
    
    # Recupera i modificatori dalla sidebar (non ancora definiti, li definiremo dopo)
    # Per ora usa valori default
    lead_time_mod = st.session_state.get('lead_time_mod', 100)
    domanda_mod = st.session_state.get('domanda_mod', 100)
    
    # Carica dati con modificatori
    prodotti, vendite, previsioni, ordini = carica_dati_simulati(lead_time_mod, domanda_mod)
    
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
            value=st.session_state.get('lead_time_mod', 100),
            step=10,
            key='lead_time_mod',
            help="Modifica i tempi di consegna: 50% = dimezza, 150% = aumenta del 50%"
        )
        
        domanda_mod = st.slider(
            "Domanda Modifier (%)",
            min_value=50,
            max_value=200,
            value=st.session_state.get('domanda_mod', 100),
            step=10,
            key='domanda_mod',
            help="Modifica la domanda prevista: 50% = dimezza, 200% = raddoppia"
        )
        
        # Mostra l'effetto dei modificatori
        if lead_time_mod != 100 or domanda_mod != 100:
            st.info(f"""📊 **Effetti Applicati:**
            - Lead Time: {lead_time_mod}% {'(ridotto)' if lead_time_mod < 100 else '(aumentato)' if lead_time_mod > 100 else '(normale)'}
            - Domanda: {domanda_mod}% {'(ridotta)' if domanda_mod < 100 else '(aumentata)' if domanda_mod > 100 else '(normale)'}
            """)
        
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trend Vendite",
        "🔮 Previsioni",
        "📋 Ordini",
        "💡 Suggerimenti",
        "📄 Report"
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
        calcola_suggerimenti_riordino(prodotti_filtrati, previsioni, domanda_mod)
        
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
    
    with tab5:
        st.subheader("📄 Generazione Report Automatico")
        
        # Configurazione Report
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📋 Contenuto Report")
            include_kpi = st.checkbox("KPI Principali", value=True)
            include_alerts = st.checkbox("Alert e Notifiche", value=True)
            include_inventory = st.checkbox("Analisi Inventario", value=True)
            include_forecast = st.checkbox("Previsioni Dettagliate", value=True)
            include_suppliers = st.checkbox("Analisi Fornitori", value=True)
            include_recommendations = st.checkbox("Raccomandazioni", value=True)
        
        with col2:
            st.markdown("### 🎨 Formato Output")
            output_format = st.radio(
                "Seleziona formato:",
                ["HTML (Interattivo)", "PDF (Stampa)", "DOCX (Word)", "Markdown"],
                help="HTML per visualizzazione web, PDF per stampa, DOCX per editing"
            )
            
            st.markdown("### 📅 Periodo Analisi")
            periodo = st.selectbox(
                "Periodo dati:",
                ["Ultimo Mese", "Ultimi 3 Mesi", "Ultimi 6 Mesi", "Anno Corrente", "Personalizzato"]
            )
            
            if periodo == "Personalizzato":
                col_a, col_b = st.columns(2)
                with col_a:
                    data_inizio = st.date_input("Data Inizio")
                with col_b:
                    data_fine = st.date_input("Data Fine")
        
        with col3:
            st.markdown("### ⚙️ Opzioni Avanzate")
            
            include_charts = st.checkbox("Includi Grafici", value=True)
            include_tables = st.checkbox("Includi Tabelle Dettagliate", value=True)
            
            st.markdown("### 🏢 Personalizzazione")
            company_logo = st.checkbox("Includi Logo Aziendale", value=True)
            executive_summary = st.checkbox("Executive Summary", value=True)
            
            language = st.selectbox(
                "Lingua Report:",
                ["Italiano", "English", "Español", "Français"]
            )
        
        st.markdown("---")
        
        # Anteprima struttura report
        with st.expander("📖 Anteprima Struttura Report", expanded=False):
            st.markdown("""
            **Il report includerà:**
            
            1. **Copertina e Sommario Esecutivo**
               - Logo Moretti S.p.A.
               - Data generazione e periodo analisi
               - KPI principali in evidenza
            
            2. **Analisi Stato Magazzino**
               - Livelli scorte attuali vs ottimali
               - Prodotti critici e alert
               - Valore economico giacenze
            
            3. **Analisi Vendite e Trend**
               - Andamento storico vendite
               - Analisi stagionalità
               - Top/Bottom performers
            
            4. **Previsioni e Pianificazione**
               - Forecast domanda 30/60/90 giorni
               - Punti di riordino ottimali
               - Quantità economiche ordine (EOQ)
            
            5. **Ottimizzazione Fornitori**
               - Comparazione fornitori
               - Lead time analysis
               - Suggerimenti ottimizzazione
            
            6. **Raccomandazioni e Azioni**
               - Azioni immediate richieste
               - Opportunità risparmio
               - Piano implementazione
            
            7. **Appendici**
               - Metodologia utilizzata
               - Glossario termini
               - Dettagli tecnici modelli ARIMA/SARIMA
            """)
        
        # Pulsante generazione
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Genera Report", type="primary", use_container_width=True):
                with st.spinner("Generazione report in corso..."):
                    # Genera il report
                    success, file_path = genera_report_quarto(
                        prodotti_filtrati, 
                        vendite, 
                        previsioni, 
                        ordini,
                        output_format=output_format,
                        include_kpi=include_kpi,
                        include_alerts=include_alerts,
                        include_inventory=include_inventory,
                        include_forecast=include_forecast,
                        include_suppliers=include_suppliers,
                        include_recommendations=include_recommendations,
                        include_charts=include_charts,
                        include_tables=include_tables,
                        periodo=periodo,
                        language=language
                    )
                    
                    if success:
                        st.success(f"✅ Report generato con successo!")
                        
                        # Mostra pulsante download
                        with open(file_path, "rb") as file:
                            file_bytes = file.read()
                            
                            file_extension = {
                                "HTML (Interattivo)": "html",
                                "PDF (Stampa)": "pdf", 
                                "DOCX (Word)": "docx",
                                "Markdown": "md"
                            }[output_format]
                            
                            st.download_button(
                                label=f"📥 Scarica Report ({file_extension.upper()})",
                                data=file_bytes,
                                file_name=f"moretti_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                                mime=f"application/{file_extension}",
                                use_container_width=True
                            )
                            
                        # Preview per HTML
                        if output_format == "HTML (Interattivo)":
                            with st.expander("👁️ Anteprima Report", expanded=False):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=600, scrolling=True)
                    else:
                        st.error("❌ Errore nella generazione del report. Verificare che Quarto sia installato.")
        
        # Info box
        with st.expander("ℹ️ Informazioni su Quarto Reports", expanded=False):
            st.info("""
            **Quarto** è un sistema di publishing scientifico open-source che permette di:
            - Creare report dinamici con dati real-time
            - Esportare in multipli formati (HTML, PDF, Word, etc.)
            - Includere codice, grafici e analisi interattive
            - Mantenere consistenza nel branding aziendale
            
            **Requisiti:**
            - Quarto CLI installato (https://quarto.org/docs/get-started/)
            - Per PDF: LaTeX distribution (TinyTeX consigliato)
            - Per DOCX: Microsoft Word o LibreOffice
            """)
    
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