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
import time
import sys

# Aggiungi il modulo arima_forecaster al path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from arima_forecaster.utils.translations import get_all_translations, translate
from arima_forecaster.core import SARIMAXAutoSelector
from arima_forecaster.utils.preprocessing import ExogenousPreprocessor, analyze_feature_relationships
from arima_forecaster.utils.exog_diagnostics import ExogDiagnostics

# Import modulo inventory balance optimizer
try:
    from inventory_balance_optimizer import (
        SafetyStockCalculator,
        TotalCostAnalyzer,
        InventoryAlertSystem,
        InventoryKPIDashboard,
        CostiGiacenza,
        AnalisiRischio,
        AlertLevel
    )
    INVENTORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    INVENTORY_OPTIMIZER_AVAILABLE = False
try:
    from arima_forecaster.core import ARIMAForecaster, SARIMAForecaster, ProphetForecaster, ProphetModelSelector
    from arima_forecaster.core.cold_start import ColdStartForecaster
    FORECASTING_AVAILABLE = True
    PROPHET_AVAILABLE = True
    COLD_START_AVAILABLE = True
except ImportError:
    try:
        from arima_forecaster.core import ARIMAForecaster, SARIMAForecaster
        from arima_forecaster.core.cold_start import ColdStartForecaster
        FORECASTING_AVAILABLE = True
        PROPHET_AVAILABLE = False
        COLD_START_AVAILABLE = True
    except ImportError:
        try:
            from arima_forecaster.core.cold_start import ColdStartForecaster
            FORECASTING_AVAILABLE = False
            PROPHET_AVAILABLE = False
            COLD_START_AVAILABLE = True
        except ImportError:
            FORECASTING_AVAILABLE = False
            PROPHET_AVAILABLE = False
            COLD_START_AVAILABLE = False

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
    
    /* Stile per le metriche con bordi e allineamento centrato */
    [data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 2px solid #444444;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        text-align: center;
    }
    
    /* Centratura titolo metrica */
    [data-testid="metric-container"] > div:first-child {
        text-align: center;
    }
    
    /* Centratura valore principale */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        text-align: center;
        font-weight: bold;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Centratura delta */
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        text-align: center;
        justify-content: center;
        display: flex;
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
        color: #000000;
        font-weight: 500;
    }
    .alert-critica {
        background-color: #ffffff; 
        border: 3px solid #dc3545; 
        box-shadow: 0 2px 6px rgba(220, 53, 69, 0.3);
    }
    .alert-alta {
        background-color: #ffffff; 
        border: 3px solid #fd7e14; 
        box-shadow: 0 2px 6px rgba(253, 126, 20, 0.3);
    }
    .alert-media {
        background-color: #ffffff; 
        border: 3px solid #ffc107; 
        box-shadow: 0 2px 6px rgba(255, 193, 7, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# SIMULAZIONE DATI (in produzione: connessione DB)
# =====================================================

def carica_dati_da_csv(lead_time_mod=100, domanda_mod=100, language='Italiano'):
    """Carica dati da file CSV con modificatori
    
    Args:
        lead_time_mod: Modificatore lead time in percentuale (100 = normale)
        domanda_mod: Modificatore domanda in percentuale (100 = normale)
        language: Lingua per le traduzioni
    """
    
    # Path to data directory
    data_dir = Path(__file__).parent / "data"
    
    try:
        # Carica prodotti da CSV
        prodotti_path = data_dir / "prodotti_dettaglio.csv"
        if prodotti_path.exists():
            prodotti = pd.read_csv(prodotti_path)
            # Applica modificatori al lead time
            prodotti['lead_time'] = (prodotti['lead_time'] * lead_time_mod / 100).astype(int)
        else:
            # Fallback ai dati hardcoded se il CSV non esiste
            prodotti = carica_dati_fallback_prodotti(lead_time_mod)
    except Exception as e:
        print(f"[WARNING] Errore caricamento prodotti CSV: {e}. Uso dati fallback.")
        prodotti = carica_dati_fallback_prodotti(lead_time_mod)
    
    # Carica storico vendite da CSV
    try:
        vendite_path = data_dir / "vendite_storiche_dettagliate.csv"
        if vendite_path.exists():
            vendite_csv = pd.read_csv(vendite_path, parse_dates=['data'])
            # Prendi gli ultimi 90 giorni
            vendite_csv = vendite_csv.tail(90).copy()
            # Applica modificatore domanda
            codici_prodotti = [col for col in vendite_csv.columns if col != 'data']
            for codice in codici_prodotti:
                vendite_csv[codice] = (vendite_csv[codice] * domanda_mod / 100).astype(int)
            
            vendite = vendite_csv.set_index('data')
        else:
            # Fallback ai dati generati
            vendite = carica_dati_fallback_vendite(prodotti, domanda_mod)
    except Exception as e:
        print(f"[WARNING] Errore caricamento vendite CSV: {e}. Uso dati fallback.")
        vendite = carica_dati_fallback_vendite(prodotti, domanda_mod)
    
    # Genera previsioni basate sui dati storici
    future_dates = pd.date_range(start=datetime.now()+timedelta(days=1), periods=30, freq='D')
    previsioni = pd.DataFrame()
    
    # Selezione algoritmo di forecasting nel sidebar
    if FORECASTING_AVAILABLE:
        with st.sidebar:
            st.markdown("---")
            st.subheader(translate("forecasting_model", language))
            
            forecasting_options = ["ARIMA (Veloce)"]
            if PROPHET_AVAILABLE:
                forecasting_options.extend([
                    "Prophet (Automatico)", 
                    "Prophet + Holidays IT",
                    "Confronto ARIMA vs Prophet"
                ])
            
            forecasting_method = st.selectbox(
                translate("select_model", language),
                forecasting_options,
                key="forecasting_method_moretti"
            )

    # Utilizza forecasting avanzato se disponibile
    if FORECASTING_AVAILABLE and len(vendite) >= 30:
        for codice in vendite.columns:
            if codice in prodotti['codice'].values:
                try:
                    # Prepara dati per forecasting
                    serie_temp = vendite[codice].asfreq('D').fillna(0)
                    
                    if "Prophet" in forecasting_method and PROPHET_AVAILABLE:
                        # 🆕 USA PROPHET
                        if "Holidays" in forecasting_method:
                            model = ProphetForecaster(
                                yearly_seasonality=True,
                                weekly_seasonality=True,
                                country_holidays='IT'  # Festività italiane
                            )
                        elif "Confronto" in forecasting_method:
                            # Confronta ARIMA vs Prophet
                            arima_model = ARIMAForecaster(order=(1,1,1))
                            arima_model.fit(serie_temp)
                            arima_result = arima_model.forecast(steps=30, confidence_intervals=True)
                            
                            # Extract ARIMA forecast values properly
                            if isinstance(arima_result, dict):
                                arima_pred = arima_result.get('forecast', arima_result.get('mean', serie_temp.mean() * np.ones(30)))
                                arima_lower = arima_result.get('confidence_intervals', {}).get('lower', arima_pred * 0.8)
                                arima_upper = arima_result.get('confidence_intervals', {}).get('upper', arima_pred * 1.2)
                            elif isinstance(arima_result, tuple):
                                arima_pred, arima_intervals = arima_result
                                if isinstance(arima_intervals, dict):
                                    arima_lower = arima_intervals.get('lower', arima_pred * 0.8)
                                    arima_upper = arima_intervals.get('upper', arima_pred * 1.2)
                                else:
                                    arima_lower = arima_pred * 0.8
                                    arima_upper = arima_pred * 1.2
                            else:
                                arima_pred = arima_result
                                arima_lower = arima_pred * 0.8
                                arima_upper = arima_pred * 1.2
                            
                            prophet_model = ProphetForecaster(yearly_seasonality=True, weekly_seasonality=True)
                            prophet_model.fit(serie_temp)
                            prophet_result = prophet_model.forecast(steps=30, confidence_intervals=True)
                            
                            # Extract Prophet forecast values properly
                            if isinstance(prophet_result, tuple):
                                prophet_forecast, prophet_intervals = prophet_result
                                if isinstance(prophet_intervals, dict):
                                    prophet_lower = prophet_intervals.get('lower', prophet_forecast * 0.8)
                                    prophet_upper = prophet_intervals.get('upper', prophet_forecast * 1.2)
                                elif hasattr(prophet_intervals, 'iloc'):
                                    prophet_lower = prophet_intervals.iloc[:, 0] if prophet_intervals.shape[1] > 0 else prophet_forecast * 0.8
                                    prophet_upper = prophet_intervals.iloc[:, 1] if prophet_intervals.shape[1] > 1 else prophet_forecast * 1.2
                                else:
                                    prophet_lower = prophet_forecast * 0.8
                                    prophet_upper = prophet_forecast * 1.2
                            else:
                                prophet_forecast = prophet_result
                                prophet_lower = prophet_forecast * 0.8
                                prophet_upper = prophet_forecast * 1.2
                            
                            # Media pesata (60% Prophet, 40% ARIMA) - now with proper array operations
                            try:
                                predictions = 0.6 * prophet_forecast + 0.4 * arima_pred
                                previsioni[f'{codice}_lower'] = 0.6 * prophet_lower + 0.4 * arima_lower
                                previsioni[f'{codice}_upper'] = 0.6 * prophet_upper + 0.4 * arima_upper
                            except Exception as e:
                                # Emergency fallback if there are still type issues
                                predictions = (prophet_forecast + arima_pred) / 2  # Simple average
                                previsioni[f'{codice}_lower'] = predictions * 0.8
                                previsioni[f'{codice}_upper'] = predictions * 1.2
                        else:
                            # Prophet standard
                            model = ProphetForecaster(
                                yearly_seasonality='auto',
                                weekly_seasonality='auto'
                            )
                            
                        if "Confronto" not in forecasting_method:
                            model.fit(serie_temp)
                            forecast_result = model.forecast(steps=30, confidence_intervals=True)
                            
                            if isinstance(forecast_result, tuple):
                                predictions, confidence_intervals = forecast_result
                                if isinstance(confidence_intervals, dict):
                                    previsioni[f'{codice}_lower'] = confidence_intervals.get('lower', predictions * 0.8)
                                    previsioni[f'{codice}_upper'] = confidence_intervals.get('upper', predictions * 1.2)
                                elif hasattr(confidence_intervals, 'iloc'):
                                    # DataFrame format
                                    previsioni[f'{codice}_lower'] = confidence_intervals.iloc[:, 0] if confidence_intervals.shape[1] > 0 else predictions * 0.8
                                    previsioni[f'{codice}_upper'] = confidence_intervals.iloc[:, 1] if confidence_intervals.shape[1] > 1 else predictions * 1.2
                                else:
                                    # Fallback
                                    previsioni[f'{codice}_lower'] = predictions * 0.8
                                    previsioni[f'{codice}_upper'] = predictions * 1.2
                            elif isinstance(forecast_result, dict):
                                # Handle dict format (some models might return this)
                                predictions = forecast_result.get('forecast', forecast_result.get('mean', serie_temp.mean() * np.ones(30)))
                                previsioni[f'{codice}_lower'] = forecast_result.get('lower', predictions * 0.8)
                                previsioni[f'{codice}_upper'] = forecast_result.get('upper', predictions * 1.2)
                            else:
                                predictions = forecast_result
                                # Fallback per intervalli di confidenza
                                if hasattr(predictions, '__len__') and len(predictions) > 0:
                                    std_err = serie_temp.std()
                                    previsioni[f'{codice}_lower'] = predictions * 0.8  # Safe fallback
                                    previsioni[f'{codice}_upper'] = predictions * 1.2  # Safe fallback
                                else:
                                    predictions = np.array([serie_temp.mean()] * 30)
                                    previsioni[f'{codice}_lower'] = predictions * 0.8
                                    previsioni[f'{codice}_upper'] = predictions * 1.2
                    else:
                        # 🔄 USA ARIMA (Default)
                        model = ARIMAForecaster(order=(1,1,1))
                        model.fit(serie_temp)
                        forecast_result = model.forecast(steps=30, confidence_intervals=True)
                        
                        # Handle ARIMA forecast result
                        if isinstance(forecast_result, tuple):
                            predictions, confidence_intervals = forecast_result
                            if isinstance(confidence_intervals, dict):
                                previsioni[f'{codice}_lower'] = confidence_intervals.get('lower', predictions * 0.8)
                                previsioni[f'{codice}_upper'] = confidence_intervals.get('upper', predictions * 1.2)
                            elif hasattr(confidence_intervals, 'iloc'):
                                previsioni[f'{codice}_lower'] = confidence_intervals.iloc[:, 0] if confidence_intervals.shape[1] > 0 else predictions * 0.8
                                previsioni[f'{codice}_upper'] = confidence_intervals.iloc[:, 1] if confidence_intervals.shape[1] > 1 else predictions * 1.2
                            else:
                                previsioni[f'{codice}_lower'] = predictions * 0.8
                                previsioni[f'{codice}_upper'] = predictions * 1.2
                        elif isinstance(forecast_result, dict):
                            predictions = forecast_result.get('forecast', forecast_result.get('mean', serie_temp.mean() * np.ones(30)))
                            previsioni[f'{codice}_lower'] = forecast_result.get('lower', predictions * 0.8)
                            previsioni[f'{codice}_upper'] = forecast_result.get('upper', predictions * 1.2)
                        else:
                            predictions = forecast_result
                            previsioni[f'{codice}_lower'] = predictions * 0.8
                            previsioni[f'{codice}_upper'] = predictions * 1.2
                    
                    # Salva previsioni principali
                    previsioni[codice] = predictions
                    
                    # Applica modificatore domanda
                    previsioni[codice] *= (domanda_mod / 100)
                    previsioni[f'{codice}_lower'] *= (domanda_mod / 100)
                    previsioni[f'{codice}_upper'] *= (domanda_mod / 100)
                    
                except Exception as e:
                    # Fallback su metodo semplice
                    st.warning(f"Errore forecasting per {codice}: {str(e)[:50]}... Uso metodo semplice.")
                    base = vendite[codice].mean()
                    previsioni[codice] = np.random.poisson(max(base, 0.1), 30) * (1 + 0.1*np.random.randn(30)) * (domanda_mod / 100)
                    previsioni[f'{codice}_lower'] = previsioni[codice] * 0.8
                    previsioni[f'{codice}_upper'] = previsioni[codice] * 1.2
    else:
        # Fallback su metodo simulato originale
        for codice in vendite.columns:
            if codice in prodotti['codice'].values:
                base = vendite[codice].mean()
                # Applica modificatore domanda alle previsioni
                previsioni[codice] = np.random.poisson(max(base, 0.1), 30) * (1 + 0.1*np.random.randn(30)) * (domanda_mod / 100)
                previsioni[f'{codice}_lower'] = previsioni[codice] * 0.8
                previsioni[f'{codice}_upper'] = previsioni[codice] * 1.2
    
    previsioni['data'] = future_dates
    previsioni = previsioni.set_index('data')
    
    # Carica ordini da CSV
    try:
        ordini_path = data_dir / "ordini_attivi.csv"
        if ordini_path.exists():
            ordini = pd.read_csv(ordini_path, parse_dates=['data_ordine', 'data_consegna_prevista'])
            # Seleziona solo colonne necessarie per compatibilità
            ordini = ordini[['id_ordine', 'prodotto_codice', 'quantita', 'fornitore', 
                           'data_ordine', 'data_consegna_prevista', 'stato', 'costo_totale']].copy()
            ordini.rename(columns={'prodotto_codice': 'prodotto'}, inplace=True)
        else:
            # Fallback ai dati hardcoded
            ordini = carica_dati_fallback_ordini()
    except Exception as e:
        print(f"[WARNING] Errore caricamento ordini CSV: {e}. Uso dati fallback.")
        ordini = carica_dati_fallback_ordini()
    
    return prodotti, vendite, previsioni, ordini


def carica_dati_fallback_prodotti(lead_time_mod=100):
    """Dati fallback per prodotti se CSV non disponibile"""
    return pd.DataFrame({
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


def carica_dati_fallback_vendite(prodotti, domanda_mod=100):
    """Dati fallback per vendite se CSV non disponibile"""
    date_range = pd.date_range(end=datetime.now(), periods=90, freq='D')
    vendite = pd.DataFrame()
    
    for codice in prodotti['codice']:
        base = np.random.randint(1, 8)
        vendite[codice] = np.random.poisson(base, 90) * (1 + 0.2*np.sin(np.arange(90)*2*np.pi/30)) * (domanda_mod / 100)
    
    vendite['data'] = date_range
    return vendite.set_index('data')


def carica_dati_fallback_ordini():
    """Dati fallback per ordini se CSV non disponibile"""
    return pd.DataFrame({
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


# Alias per compatibilità con il codice esistente
def carica_scenari_whatif():
    """Carica scenari what-if da CSV"""
    data_dir = Path(__file__).parent / "data"
    scenari_path = data_dir / "scenari_whatif.csv"
    
    try:
        if scenari_path.exists():
            return pd.read_csv(scenari_path)
        else:
            # Scenari di default
            return pd.DataFrame({
                'scenario_nome': ['Scenario_Base', 'Crisi_Fornitori', 'Boom_Domanda'],
                'descrizione': ['Situazione Attuale', 'Problemi Supply Chain', 'Crescita Post-Pandemia'],
                'lead_time_modifier': [100, 150, 100],
                'domanda_modifier': [100, 100, 180],
                'impact_description': [
                    'Baseline normale operativo',
                    'Lead time aumentati del 50%',
                    'Aumento domanda 80%'
                ]
            })
    except Exception as e:
        print(f"[WARNING] Errore caricamento scenari: {e}")
        return pd.DataFrame({
            'scenario_nome': ['Scenario_Base'],
            'descrizione': ['Situazione Attuale'],
            'lead_time_modifier': [100],
            'domanda_modifier': [100],
            'impact_description': ['Baseline normale operativo']
        })


def carica_dati_simulati(lead_time_mod=100, domanda_mod=100, language='Italiano'):
    """Alias per compatibilità - ora carica da CSV"""
    return carica_dati_da_csv(lead_time_mod, domanda_mod, language)


# =====================================================
# TRADUZIONI - Sistema Centralizzato
# =====================================================

# NOTA: Traduzioni ora gestite dal sistema centralizzato in src/arima_forecaster/utils/translations.py
# Le traduzioni sono caricate dinamicamente dai file JSON in assets/locales/

def get_translations_dict(language: str) -> dict:
    """Ottieni traduzioni usando il sistema centralizzato per compatibilità."""
    return get_all_translations(language)

# Sistema centralizzato attivo - traduzioni caricate dinamicamente


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
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">💰 Valore Magazzino</div>
            <div style="color: white; font-size: 28px; font-weight: bold;">€{valore_magazzino:,.0f}</div>
            <div style="color: #4CAF50; font-size: 14px; margin-top: 5px;">▲ +{np.random.randint(1,10)}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">📦 Vendite Ultimo Mese</div>
            <div style="color: white; font-size: 28px; font-weight: bold;">{vendite_mese:,.0f}</div>
            <div style="color: #4CAF50; font-size: 14px; margin-top: 5px;">▲ +{np.random.randint(5,15)}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delta_color = "#f44336" if prodotti_sotto_scorta > 0 else "#4CAF50"
        delta_symbol = "▲" if prodotti_sotto_scorta > 0 else "✓"
        delta_text = f"+{prodotti_sotto_scorta}" if prodotti_sotto_scorta > 0 else "OK"
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">⚠️ Prodotti Sotto Scorta</div>
            <div style="color: white; font-size: 28px; font-weight: bold;">{prodotti_sotto_scorta}</div>
            <div style="color: {delta_color}; font-size: 14px; margin-top: 5px;">{delta_symbol} {delta_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">🚚 Ordini in Corso</div>
            <div style="color: white; font-size: 28px; font-weight: bold;">{ordini_attivi}</div>
            <div style="color: #2196F3; font-size: 14px; margin-top: 5px;">━ Stabile</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
            <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">✅ Service Level</div>
            <div style="color: white; font-size: 28px; font-weight: bold;">{service_level:.1f}%</div>
            <div style="color: #4CAF50; font-size: 14px; margin-top: 5px;">▲ +{np.random.uniform(0.5, 2):.1f}%</div>
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
            urgenza_color = '#dc3545' if alert['urgenza'] == 'CRITICA' else '#fd7e14'
            
            st.markdown(f"""
            <div class='alert-box {css_class}'>
                <strong style='color: {urgenza_color}; font-size: 16px;'>[{alert['urgenza']}] {alert['tipo']}</strong><br>
                <span style='color: #000000; font-size: 14px;'>{alert['messaggio']}</span><br>
                <em style='color: #333333; font-size: 13px;'>Azione suggerita: {alert['azione']}</em>
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


def grafico_previsioni(previsioni, prodotto_selezionato, translations=None):
    """Grafico previsioni con intervalli confidenza"""
    if translations is None:
        translations = get_all_translations('Italiano')
    
    fig = go.Figure()
    
    # Previsione centrale
    fig.add_trace(go.Scatter(
        x=previsioni.index,
        y=previsioni[prodotto_selezionato],
        mode='lines',
        name=translations.get('forecast', 'Previsione'),
        line=dict(color='green')
    ))
    
    # Intervallo confidenza
    fig.add_trace(go.Scatter(
        x=previsioni.index,
        y=previsioni[f'{prodotto_selezionato}_upper'],
        mode='lines',
        name=translations.get('upper_limit', 'Limite Superiore'),
        line=dict(color='lightgreen', dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=previsioni.index,
        y=previsioni[f'{prodotto_selezionato}_lower'],
        mode='lines',
        name=translations.get('lower_limit', 'Limite Inferiore'),
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
    
    # Colora per stato con contrasto migliorato e testo centrato
    def color_stato(stato):
        colors = {
            'In elaborazione': 'background-color: #ffffff; color: #856404; font-weight: bold; border: 2px solid #ffc107; border-radius: 4px; padding: 4px; text-align: center;',
            'Confermato': 'background-color: #ffffff; color: #155724; font-weight: bold; border: 2px solid #28a745; border-radius: 4px; padding: 4px; text-align: center;',
            'In transito': 'background-color: #ffffff; color: #004085; font-weight: bold; border: 2px solid #007bff; border-radius: 4px; padding: 4px; text-align: center;',
            'In produzione': 'background-color: #ffffff; color: #721c24; font-weight: bold; border: 2px solid #dc3545; border-radius: 4px; padding: 4px; text-align: center;',
            'Consegnato': 'background-color: #ffffff; color: #383d41; font-weight: bold; border: 2px solid #6c757d; border-radius: 4px; padding: 4px; text-align: center;'
        }
        return colors.get(stato, 'background-color: #ffffff; color: #000000; font-weight: bold; text-align: center;')
    
    styled = ordini_display.style.map(
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
        
        # Ottieni traduzioni per la lingua selezionata
        translations = get_all_translations(language)
        
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
title: "{translations['title']}"
subtitle: "{translations['subtitle']}"
author: "{translations['author']}"
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

# {translations['executive_summary']}

"""
        
        if include_kpi:
            # Calcola KPI
            valore_magazzino = (prodotti['scorte_attuali'] * prodotti['prezzo_medio']).sum()
            vendite_mese = vendite.tail(30).sum().sum()
            prodotti_sotto_scorta = len(prodotti[prodotti['scorte_attuali'] < prodotti['scorta_minima']])
            service_level = (1 - prodotti_sotto_scorta/len(prodotti)) * 100
            
            qmd_content += f"""
## {translations['key_metrics']}

::: {{.callout-note}}
### {translations['metrics_title']} {datetime.now().strftime('%d/%m/%Y')}

- **{translations['warehouse_value']}**: €{valore_magazzino:,.0f}
- **{translations['sales_last_month']}**: {vendite_mese:,.0f} {translations['units']}
- **{translations['products_low_stock']}**: {prodotti_sotto_scorta}
- **{translations['service_level']}**: {service_level:.1f}%
:::

"""
        
        if include_alerts:
            qmd_content += f"""
## {translations['critical_alerts']}

"""
            alerts_critici = prodotti[prodotti['scorte_attuali'] < prodotti['scorta_sicurezza']]
            if len(alerts_critici) > 0:
                qmd_content += f"""
::: {{.callout-warning}}
### {translations['emergency_products']}
"""
                for _, prod in alerts_critici.iterrows():
                    qmd_content += f"- **{prod['nome']}**: {translations['only_remaining']} {prod['scorte_attuali']} {translations['units_remaining']}\\n"
                qmd_content += ":::\\n\\n"
        
        if include_inventory:
            qmd_content += f"""
# {translations['inventory_analysis']}

## {translations['current_stock_status']}

"""
            if include_tables:
                qmd_content += f"""
| {translations['product']} | {translations['code']} | {translations['current_stock']} | {translations['min_stock']} | {translations['status']} |
|----------|--------|----------------|---------------|-------|
"""
                for _, prod in prodotti.iterrows():
                    stato = translations['critical'] if prod['scorte_attuali'] < prod['scorta_sicurezza'] else translations['warning'] if prod['scorte_attuali'] < prod['scorta_minima'] else translations['ok']
                    qmd_content += f"| {prod['nome']} | {prod['codice']} | {prod['scorte_attuali']} | {prod['scorta_minima']} | {stato} |\\n"
                qmd_content += "\\n"
        
        if include_forecast:
            qmd_content += f"""
# {translations['demand_forecast']}

## {translations['forecast_30_days']}

{translations['forecast_description']}

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

# Traduzioni per grafici
translations = {translations}

# Crea una griglia di grafici per i prodotti principali
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

prodotti_da_mostrare = vendite.columns[:6]  # Primi 6 prodotti

for idx, prodotto in enumerate(prodotti_da_mostrare):
    ax = axes[idx]
    
    # Grafico vendite storiche
    ax.plot(vendite.index[-30:], vendite[prodotto].tail(30), 
            label=translations.get('historical', 'Storico'), color='blue', linewidth=2)
    
    # Grafico previsioni
    if prodotto in previsioni.columns:
        ax.plot(previsioni.index, previsioni[prodotto], 
                label=translations.get('forecast', 'Previsione'), color='green', linewidth=2, linestyle='--')
        
        # Aggiungi intervalli di confidenza se esistono
        if f'{{prodotto}}_lower' in previsioni.columns:
            ax.fill_between(previsioni.index, 
                           previsioni[f'{{prodotto}}_lower'],
                           previsioni[f'{{prodotto}}_upper'],
                           alpha=0.3, color='green', label=f"{{translations.get('confidence_interval', 'Intervallo 95%')}}")
    
    ax.set_title(f'{{prodotto}}', fontsize=10, fontweight='bold')
    ax.set_xlabel(translations.get('date', 'Data'), fontsize=8)
    ax.set_ylabel(translations.get('units', 'Unità'), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    
    # Ruota le etichette delle date
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle(f'{translations["sales_analysis_forecast"]}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### {translations['aggregate_trend_analysis']}

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
ax1.set_title(f'{translations["total_sales_last_60"]}', fontsize=12, fontweight='bold')
ax1.set_xlabel(f'{translations["date"]}')
ax1.set_ylabel(f'{translations["total_units"]}')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Previsioni aggregate
previsioni_prodotti = [col for col in previsioni.columns if not ('_lower' in col or '_upper' in col)]
previsioni_totali = previsioni[previsioni_prodotti].sum(axis=1)
ax2.plot(previsioni.index, previsioni_totali, 
         color='darkgreen', linewidth=2, marker='o', markersize=4)
ax2.fill_between(previsioni.index, 0, previsioni_totali, 
                 alpha=0.3, color='lightgreen')
ax2.set_title(f'{translations["total_forecast_next_30"]}', fontsize=12, fontweight='bold')
ax2.set_xlabel(f'{translations["date"]}')
ax2.set_ylabel(f'{translations["expected_units"]}')
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

### {translations['forecast_distribution_category']}

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

ax.set_title(f'{translations["forecast_distribution_category"]}', 
             fontsize=12, fontweight='bold', pad=20)

plt.show()
```

"""
            
            if include_tables:
                qmd_content += f"""
### {translations['expected_demand_per_product']}

| {translations['product']} | {translations['daily_average']} | {translations['total_30_days']} | {translations['expected_peak']} |
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
            qmd_content += f"""
# {translations['supplier_optimization']}

## {translations['comparative_analysis']}

| {translations['supplier']} | {translations['lead_time']} | {translations['reliability']} | {translations['volume_price']} | {translations['rating']} |
|-----------|-----------|--------------|---------------|--------|
| MedSupply Italia | 15 {translations['days']} | 95% | €260/50+ | ⭐⭐⭐⭐⭐ |
| EuroMedical | 12 {translations['days']} | 92% | €250/50+ | ⭐⭐⭐⭐ |
| GlobalMed | 20 {translations['days']} | 88% | €265/50+ | ⭐⭐⭐ |

"""
        
        if include_recommendations:
            qmd_content += f"""
# {translations['recommendations']}

## {translations['immediate_actions']}

1. **{translations['urgent_reorders']}**
2. **{translations['supplier_optimization_action']}**
3. **{translations['min_stock_review']}**

## {translations['savings_opportunities']}

- **{translations['order_consolidation']}**
- **{translations['lead_time_reduction']}**
- **{translations['eoq_optimization']}**

"""
        
        # Footer
        qmd_content += f"""
---

*{translations['report_footer']}*  
*{translations['powered_by']}*
"""
        
        # Scrivi file .qmd
        with open(qmd_file, "w", encoding="utf-8") as f:
            f.write(qmd_content)
        
        # Genera report con Quarto
        try:
            # Configura ambiente per UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Verifica se Quarto è installato
            result = subprocess.run(
                ["quarto", "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env,
                timeout=5
            )
            
            if result.returncode == 0:
                # Genera il report
                render_result = subprocess.run(
                    ["quarto", "render", str(qmd_file), "--to", output_ext],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    env=env,
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
                    return genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir, language)
            else:
                # Quarto non installato, usa fallback
                return genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir, language)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Usa fallback HTML se Quarto non è disponibile
            return genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir, language)
            
    except Exception as e:
        print(f"Errore generazione report: {e}")
        return False, None


def genera_report_html_fallback(prodotti, vendite, previsioni, ordini, temp_dir, language='Italiano'):
    """Genera report HTML semplice come fallback se Quarto non è disponibile"""
    
    # Ottieni traduzioni
    translations = get_all_translations(language)
    
    # Calcola KPI
    valore_magazzino = (prodotti['scorte_attuali'] * prodotti['prezzo_medio']).sum()
    vendite_mese = vendite.tail(30).sum().sum()
    prodotti_sotto_scorta = len(prodotti[prodotti['scorte_attuali'] < prodotti['scorta_minima']])
    service_level = (1 - prodotti_sotto_scorta/len(prodotti)) * 100
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{translations['title']}</title>
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
        <h1>{translations['title']}</h1>
        <h2>{translations['subtitle']}</h2>
        <p>Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    
    <div class="section">
        <h2>{translations['key_metrics']}</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">€{valore_magazzino:,.0f}</div>
                <div class="kpi-label">{translations['warehouse_value']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{vendite_mese:,.0f}</div>
                <div class="kpi-label">{translations['sales_last_month']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{prodotti_sotto_scorta}</div>
                <div class="kpi-label">{translations['products_low_stock']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{service_level:.1f}%</div>
                <div class="kpi-label">{translations['service_level']}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>{translations['inventory_analysis']}</h2>
        <table>
            <tr>
                <th>{translations['product']}</th>
                <th>{translations['code']}</th>
                <th>{translations['current_stock']}</th>
                <th>{translations['min_stock']}</th>
                <th>{translations['status']}</th>
            </tr>
"""
    
    for _, prod in prodotti.iterrows():
        stato = translations['critical'] if prod['scorte_attuali'] < prod['scorta_sicurezza'] else translations['warning'] if prod['scorte_attuali'] < prod['scorta_minima'] else translations['ok']
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
            {translations['report_footer']}<br>
            {translations['powered_by']}
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
    # Inizializza lingua di default in session state se non presente
    if 'dashboard_language' not in st.session_state:
        st.session_state.dashboard_language = 'Italiano'
    
    # Selettore lingua nella sidebar
    with st.sidebar:
        st.markdown("### 🌍 Impostazioni Lingua")
        dashboard_language = st.selectbox(
            "Lingua Dashboard:",
            ["Italiano", "English", "Español", "Français", "中文"],
            index=["Italiano", "English", "Español", "Français", "中文"].index(st.session_state.dashboard_language),
            key='lang_selector'
        )
        st.session_state.dashboard_language = dashboard_language
    
    # Ottieni traduzioni per la dashboard
    dashboard_translations = get_all_translations(dashboard_language)
    
    # Header
    st.title("🏥 Moretti S.p.A. - Sistema Gestione Scorte Intelligente")
    
    # Info caricamento dati
    data_dir = Path(__file__).parent / "data"
    csv_files_exist = all([
        (data_dir / "prodotti_dettaglio.csv").exists(),
        (data_dir / "vendite_storiche_dettagliate.csv").exists(),
        (data_dir / "ordini_attivi.csv").exists(),
        (data_dir / "fornitori_dettaglio.csv").exists()
    ])
    
    if csv_files_exist:
        st.success("✅ **Dati caricati da file CSV esterni** - Dashboard pronta per demo clienti!")
    else:
        st.warning("⚠️ Alcuni file CSV mancanti - Utilizzo dati fallback simulati")
    
    st.markdown("---")
    
    # Recupera i modificatori dalla sidebar (non ancora definiti, li definiremo dopo)
    # Per ora usa valori default
    lead_time_mod = st.session_state.get('lead_time_mod', 100)
    domanda_mod = st.session_state.get('domanda_mod', 100)
    
    # Carica dati con modificatori
    prodotti, vendite, previsioni, ordini = carica_dati_simulati(lead_time_mod, domanda_mod, dashboard_language)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controlli Dashboard")
        
        # Carica scenari what-if
        scenari_df = carica_scenari_whatif()
        
        # Selezione scenario
        st.subheader("🎭 Scenari What-If")
        scenario_options = dict(zip(scenari_df['scenario_nome'], scenari_df['descrizione']))
        scenario_selected = st.selectbox(
            "Seleziona Scenario:",
            options=list(scenario_options.keys()),
            format_func=lambda x: f"{scenario_options[x]} ({x})",
            help="Scenari predefiniti per analisi what-if"
        )
        
        # Ottieni parametri del scenario selezionato
        scenario_row = scenari_df[scenari_df['scenario_nome'] == scenario_selected].iloc[0]
        scenario_lead_time = int(scenario_row['lead_time_modifier'])
        scenario_domanda = int(scenario_row['domanda_modifier'])
        
        # Mostra dettagli scenario
        with st.expander(f"📋 Dettagli {scenario_selected}", expanded=False):
            st.write(f"**Descrizione:** {scenario_row['descrizione']}")
            st.write(f"**Impact:** {scenario_row['impact_description']}")
            if 'business_case' in scenario_row:
                st.write(f"**Business Case:** {scenario_row['business_case']}")
        
        st.markdown("---")
        
        # Parametri personalizzati (override scenario)
        st.subheader("🎮 Override Parametri")
        use_custom = st.checkbox("Usa parametri personalizzati", value=False)
        
        if use_custom:
            lead_time_mod = st.slider(
                "Lead Time Modifier (%)",
                min_value=50,
                max_value=200,
                value=scenario_lead_time,
                step=5,
                help="Modifica i tempi di consegna"
            )
            
            domanda_mod = st.slider(
                "Domanda Modifier (%)",
                min_value=30,
                max_value=250,
                value=scenario_domanda,
                step=5,
                help="Modifica la domanda prevista"
            )
        else:
            lead_time_mod = scenario_lead_time
            domanda_mod = scenario_domanda
            
            # Mostra i parametri del scenario corrente
            st.write(f"**Lead Time:** {lead_time_mod}%")
            st.write(f"**Domanda:** {domanda_mod}%")
        
        st.markdown("---")
        
        # Filtro categoria
        st.subheader("📂 Filtri Dati")
        categorie = ['Tutte'] + list(prodotti['categoria'].unique())
        categoria_sel = st.selectbox("Categoria", categorie, key="categoria_filter")
        
        if categoria_sel != 'Tutte':
            prodotti_filtrati = prodotti[prodotti['categoria'] == categoria_sel]
        else:
            prodotti_filtrati = prodotti
        
        # Selezione prodotto per grafici
        # Aggiungo "Tutti" come prima opzione
        opzioni_prodotti = ['Tutti'] + prodotti_filtrati['codice'].tolist()
        
        # Reset prodotto a "Tutti" quando cambia categoria
        if 'last_categoria' not in st.session_state:
            st.session_state.last_categoria = categoria_sel
        
        if st.session_state.last_categoria != categoria_sel:
            st.session_state.last_categoria = categoria_sel
            # Forza il reset a "Tutti"
            prodotto_sel = st.selectbox(
                "Prodotto per Analisi",
                opzioni_prodotti,
                index=0,  # Seleziona "Tutti"
                format_func=lambda x: x if x == 'Tutti' else prodotti[prodotti['codice']==x]['nome'].values[0],
                key="prodotto_filter_reset"
            )
        else:
            prodotto_sel = st.selectbox(
                "Prodotto per Analisi",
                opzioni_prodotti,
                format_func=lambda x: x if x == 'Tutti' else prodotti[prodotti['codice']==x]['nome'].values[0],
                key="prodotto_filter"
            )
        
        st.markdown("---")
        
        # Info dati
        st.subheader("📊 Info Dati")
        data_info = f"""
        **Prodotti Totali:** {len(prodotti)}  
        **Prodotti Visualizzati:** {len(prodotti_filtrati)}  
        **Giorni Storico:** 120  
        **Giorni Previsione:** 30  
        **Fonte Dati:** File CSV  
        """
        st.info(data_info)
        
        # Sistema features
        with st.expander("🔧 Sistema Features"):
            st.write("""
            - ✅ Caricamento dati da CSV
            - ✅ Scenari What-If predefiniti 
            - ✅ Forecasting ARIMA/SARIMA
            - ✅ Dashboard interattiva
            - ✅ Report automatizzati
            - ✅ Analisi fornitori
            - ✅ Alert intelligenti
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Trend Vendite",
        "🔮 Previsioni", 
        "📋 Ordini",
        "🏪 Depositi",
        "💡 Suggerimenti",
        "📄 Report",
        "🗃️ Dati CSV",
        "🔬 Advanced Exog",
        "🚀 Cold Start"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        # Gestione caso "Tutti" vs prodotto singolo
        if prodotto_sel == 'Tutti':
            # Aggregazione dati per tutti i prodotti
            vendite_aggregate = vendite[prodotti_filtrati['codice'].tolist()].sum(axis=1)
            
            with col1:
                # Grafico aggregato
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=vendite_aggregate.index,
                    y=vendite_aggregate.values,
                    mode='lines',
                    name='Vendite Totali',
                    line=dict(color='blue', width=2)
                ))
                
                # Media mobile
                ma7 = vendite_aggregate.rolling(7).mean()
                fig_trend.add_trace(go.Scatter(
                    x=ma7.index,
                    y=ma7.values,
                    mode='lines',
                    name='Media Mobile 7gg',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_trend.update_layout(
                    title="Trend Vendite - Tutti i Prodotti",
                    xaxis_title="Data",
                    yaxis_title="Unità Vendute",
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Statistiche vendite aggregate
                st.subheader("📊 Statistiche Vendite Aggregate")
                
                stats_vendite = pd.DataFrame({
                    'Metrica': [
                        'Media Giornaliera Totale',
                        'Deviazione Standard',
                        'Vendite Max Giornaliere',
                        'Vendite Min Giornaliere',
                        'Trend Ultimo Mese'
                    ],
                    'Valore': [
                        f"{vendite_aggregate.mean():.1f}",
                        f"{vendite_aggregate.std():.1f}",
                        f"{vendite_aggregate.max():.0f}",
                        f"{vendite_aggregate.min():.0f}",
                        f"{(vendite_aggregate.tail(30).mean() - vendite_aggregate.head(30).mean()) / vendite_aggregate.head(30).mean() * 100:+.1f}%"
                    ]
                })
                
                st.dataframe(stats_vendite, use_container_width=True, hide_index=True)
                
                # Distribuzione vendite aggregate
                fig_dist = px.histogram(
                    vendite_aggregate,
                    nbins=20,
                    title="Distribuzione Vendite Totali",
                    labels={'value': 'Unità', 'count': 'Frequenza'}
                )
                fig_dist.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            # Caso prodotto singolo (codice esistente)
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
        
        if prodotto_sel == 'Tutti':
            # Previsioni aggregate per tutti i prodotti
            prodotti_list = prodotti_filtrati['codice'].tolist()
            previsioni_aggregate = previsioni[prodotti_list].sum(axis=1)
            
            # Calcolo intervalli di confidenza aggregati
            upper_cols = [f'{cod}_upper' for cod in prodotti_list if f'{cod}_upper' in previsioni.columns]
            lower_cols = [f'{cod}_lower' for cod in prodotti_list if f'{cod}_lower' in previsioni.columns]
            
            if upper_cols and lower_cols:
                previsioni_upper = previsioni[upper_cols].sum(axis=1)
                previsioni_lower = previsioni[lower_cols].sum(axis=1)
            else:
                # Se non ci sono intervalli, usa +/- 10% come stima
                previsioni_upper = previsioni_aggregate * 1.1
                previsioni_lower = previsioni_aggregate * 0.9
            
            with col1:
                # Grafico previsioni aggregate
                fig_prev = go.Figure()
                
                # Previsione principale
                fig_prev.add_trace(go.Scatter(
                    x=previsioni_aggregate.index,
                    y=previsioni_aggregate.values,
                    mode='lines',
                    name=dashboard_translations.get('total_forecast', 'Previsione Totale'),
                    line=dict(color='green', width=3)
                ))
                
                # Intervallo di confidenza
                fig_prev.add_trace(go.Scatter(
                    x=previsioni_upper.index,
                    y=previsioni_upper.values,
                    mode='lines',
                    name=dashboard_translations.get('upper_limit', 'Limite Superiore'),
                    line=dict(color='lightgreen', width=1, dash='dash'),
                    showlegend=False
                ))
                
                fig_prev.add_trace(go.Scatter(
                    x=previsioni_lower.index,
                    y=previsioni_lower.values,
                    mode='lines',
                    name=dashboard_translations.get('lower_limit', 'Limite Inferiore'),
                    line=dict(color='lightgreen', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)',
                    showlegend=False
                ))
                
                fig_prev.update_layout(
                    title="Previsioni 30 giorni - Tutti i Prodotti",
                    xaxis_title="Data",
                    yaxis_title="Unità Previste",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig_prev, use_container_width=True)
            
            with col2:
                st.subheader("📈 Metriche Previsione Aggregate")
                
                # Calcola metriche aggregate
                prev_totale = previsioni_aggregate.sum()
                prev_media = previsioni_aggregate.mean()
                prev_max = previsioni_aggregate.max()
                confidence_range = (previsioni_upper.mean() - previsioni_lower.mean())
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Totale 30gg", f"{prev_totale:.0f}")
                    st.metric("Media Giornaliera", f"{prev_media:.1f}")
                with col_b:
                    st.metric("Picco Previsto", f"{prev_max:.0f}")
                    st.metric("Range Confidenza", f"±{confidence_range/2:.1f}")
                
                # Confronto con storico
                st.subheader("📊 Confronto con Storico")
                
                vendite_aggregate = vendite[prodotti_list].sum(axis=1)
                storico_media = vendite_aggregate.tail(30).mean()
                variazione = (prev_media - storico_media) / storico_media * 100
                
                if variazione > 0:
                    st.warning(f"⬆️ Aumento previsto del {variazione:.1f}% rispetto agli ultimi 30 giorni")
                else:
                    st.success(f"⬇️ Diminuzione prevista del {abs(variazione):.1f}% rispetto agli ultimi 30 giorni")
        else:
            # Caso prodotto singolo (codice esistente)
            with col1:
                st.plotly_chart(
                    grafico_previsioni(previsioni, prodotto_sel, dashboard_translations),
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
        # TAB DEPOSITI - BILANCIAMENTO SCORTE
        st.subheader("🏪 Gestione Depositi e Bilanciamento Scorte")
        
        if INVENTORY_OPTIMIZER_AVAILABLE:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("⚙️ Configurazione")
                
                # Parametri configurabili
                service_level = st.selectbox(
                    "Livello Servizio Target",
                    options=[0.85, 0.90, 0.95, 0.99],
                    index=2,
                    format_func=lambda x: f"{x:.0%}"
                )
                
                deposito_sel = st.selectbox(
                    "Deposito",
                    options=["Centrale Milano", "Filiale Roma", "Filiale Napoli", "Hub Logistico Bologna"],
                    index=0
                )
                
                # Simula dati deposito
                np.random.seed(42)
                stock_levels = {
                    "CRZ001": np.random.randint(200, 400),
                    "MAT001": np.random.randint(150, 300),
                    "ELT001": np.random.randint(80, 150)
                }
                
                st.markdown("### 📊 Stock Correnti")
                for codice, stock in stock_levels.items():
                    prod_info = prodotti_filtrati[prodotti_filtrati['codice'] == codice]
                    if not prod_info.empty:
                        nome = prod_info.iloc[0]['nome']
                        st.metric(f"{codice}", f"{stock} unità", f"{nome[:20]}...")
            
            with col1:
                st.subheader("📈 Analisi Bilanciamento Scorte")
                
                # Calcola KPI per ogni prodotto
                calculator = SafetyStockCalculator()
                alert_system = InventoryAlertSystem()
                
                results_data = []
                
                for codice in prodotti_filtrati['codice']:
                    if codice in vendite.columns and codice in stock_levels:
                        serie_vendite = vendite[codice].dropna()
                        stock_corrente = stock_levels[codice]
                        
                        # Calcola safety stock
                        safety_stock = calculator.calculate_dynamic_safety_stock(
                            demand_mean=serie_vendite.mean(),
                            demand_std=serie_vendite.std(),
                            lead_time_days=15,  # Default lead time
                            service_level=service_level,
                            criticality_factor=1.2
                        )
                        
                        # Calcola reorder point
                        reorder_point = calculator.calculate_reorder_point(
                            demand_mean=serie_vendite.mean(),
                            lead_time_days=15,
                            safety_stock=safety_stock['dynamic_safety_stock']
                        )
                        
                        # Analisi rischio
                        analisi = alert_system.check_inventory_status(
                            current_stock=stock_corrente,
                            safety_stock=safety_stock['dynamic_safety_stock'],
                            reorder_point=reorder_point,
                            max_stock=stock_corrente * 2,  # Simulated max
                            daily_demand=serie_vendite.mean(),
                            lead_time_days=15
                        )
                        
                        results_data.append({
                            'Codice': codice,
                            'Stock': stock_corrente,
                            'Safety Stock': safety_stock['dynamic_safety_stock'],
                            'Reorder Point': int(reorder_point),
                            'Giorni Copertura': analisi.giorni_copertura,
                            'Alert': analisi.livello_alert.value[1],
                            'Stockout Risk': f"{analisi.probabilita_stockout:.1%}",
                            'Overstock Risk': f"{analisi.probabilita_overstock:.1%}",
                            'Turnover': f"{analisi.inventory_turnover:.1f}x"
                        })
                
                # Tabella risultati
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                
                # Grafici visualizzazioni
                st.subheader("📊 Dashboard KPI Deposito")
                
                # Metriche aggregate
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                if results_data:
                    avg_turnover = np.mean([float(r['Turnover'].replace('x', '')) for r in results_data])
                    avg_coverage = np.mean([r['Giorni Copertura'] for r in results_data])
                    critical_items = sum(1 for r in results_data if 'Critico' in r['Alert'] or 'Stockout' in r['Alert'])
                    overstock_items = sum(1 for r in results_data if 'Overstock' in r['Alert'] or 'Eccesso' in r['Alert'])
                    
                    with col_m1:
                        st.metric("Inventory Turnover", f"{avg_turnover:.1f}x", "Medio deposito")
                    
                    with col_m2:
                        st.metric("Days of Supply", f"{avg_coverage:.0f}gg", "Media copertura")
                    
                    with col_m3:
                        color = "🔴" if critical_items > 0 else "🟢"
                        st.metric("Articoli Critici", f"{critical_items}", f"{color} Alert")
                    
                    with col_m4:
                        color = "🟣" if overstock_items > 0 else "🟢"
                        st.metric("Overstock Items", f"{overstock_items}", f"{color} Eccessi")
                
                # Grafico alert levels
                if results_data:
                    st.subheader("🚨 Mappa Alert Deposito")
                    
                    # Conta alert per tipo
                    alert_counts = {}
                    for r in results_data:
                        alert = r['Alert']
                        alert_counts[alert] = alert_counts.get(alert, 0) + 1
                    
                    # Grafico a torta alert
                    fig_alerts = px.pie(
                        values=list(alert_counts.values()),
                        names=list(alert_counts.keys()),
                        title="Distribuzione Alert per Stato Stock"
                    )
                    st.plotly_chart(fig_alerts, use_container_width=True)
                
                # Raccomandazioni automatiche
                st.subheader("💡 Raccomandazioni Automatiche")
                
                for result in results_data[:3]:  # Mostra top 3
                    codice = result['Codice']
                    alert_type = result['Alert']
                    
                    if 'Critico' in alert_type or 'Stockout' in alert_type:
                        st.error(f"🚨 **{codice}**: Ordine urgente richiesto! Stock: {result['Stock']}, Reorder: {result['Reorder Point']}")
                    elif 'Basse' in alert_type or 'diminuzione' in alert_type:
                        st.warning(f"⚠️ **{codice}**: Pianificare riordino. Copertura: {result['Giorni Copertura']} giorni")
                    elif 'Overstock' in alert_type or 'Eccesso' in alert_type:
                        st.info(f"🟣 **{codice}**: Eccesso scorte. Considerare promozioni o sospendere ordini")
                    else:
                        st.success(f"✅ **{codice}**: Livelli ottimali")
        
        else:
            st.warning("⚠️ Modulo Inventory Balance Optimizer non disponibile. Installare dipendenze.")
            st.info("Per utilizzare questa funzionalità, eseguire: `pip install scipy pydantic`")
    
    with tab5:
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
            st.markdown(f"""
            <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
                <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">💰 Risparmio Mensile Stimato</div>
                <div style="color: white; font-size: 28px; font-weight: bold;">€2,450</div>
                <div style="color: #4CAF50; font-size: 14px; margin-top: 5px;">▲ +12%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
                <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">⏱️ Riduzione Lead Time Medio</div>
                <div style="color: white; font-size: 28px; font-weight: bold;">-3 giorni</div>
                <div style="color: #f44336; font-size: 14px; margin-top: 5px;">▼ -18%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="border: 2px solid #444; border-radius: 10px; padding: 20px; background-color: #1e1e1e; height: 120px; text-align: center;">
                <div style="color: #aaa; font-size: 14px; margin-bottom: 5px;">📈 Miglioramento Service Level</div>
                <div style="color: white; font-size: 28px; font-weight: bold;">+5.2%</div>
                <div style="color: #4CAF50; font-size: 14px; margin-top: 5px;">▲ Miglioramento</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab6:
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
                ["Italiano", "English", "Español", "Français", "中文"]
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
    
    with tab7:
        st.subheader("🗃️ Gestione File CSV")
        
        # Mostra status dei file CSV
        st.markdown("### 📁 Status File Dati")
        
        data_files = {
            'prodotti_dettaglio.csv': 'Catalogo prodotti con scorte e parametri',
            'vendite_storiche_dettagliate.csv': 'Storico vendite ultimi 120 giorni',
            'ordini_attivi.csv': 'Ordini in corso con fornitori',
            'fornitori_dettaglio.csv': 'Database fornitori e condizioni',
            'scenari_whatif.csv': 'Scenari predefiniti per analisi',
            'categorie_config.csv': 'Configurazione categorie prodotti'
        }
        
        for filename, description in data_files.items():
            file_path = data_dir / filename
            if file_path.exists():
                try:
                    # Leggi file info
                    df = pd.read_csv(file_path)
                    file_size = file_path.stat().st_size
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%d/%m/%Y %H:%M')
                    
                    st.markdown(f"""
                    **✅ {filename}**  
                    📝 {description}  
                    📊 {len(df)} righe × {len(df.columns)} colonne | 📁 {file_size:,} bytes | 🕐 {mod_time}
                    """)
                    
                    # Mostra anteprima in expander
                    with st.expander(f"👁️ Anteprima {filename}", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        if len(df) > 10:
                            st.info(f"Mostrate prime 10 righe di {len(df)} totali")
                        
                        # Per file vendite, mostra statistiche aggiuntive
                        if 'vendite_storiche' in filename:
                            st.markdown("**📈 Statistiche Vendite:**")
                            vendite_cols = [col for col in df.columns if col != 'data']
                            if vendite_cols:
                                stats_df = pd.DataFrame({
                                    'Prodotto': vendite_cols,
                                    'Media Giornaliera': [df[col].mean() for col in vendite_cols],
                                    'Max Giornaliero': [df[col].max() for col in vendite_cols],
                                    'Totale Periodo': [df[col].sum() for col in vendite_cols]
                                })
                                st.dataframe(stats_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Errore lettura {filename}: {e}")
            else:
                st.markdown(f"""
                **❌ {filename}**  
                📝 {description}  
                ⚠️ File non trovato - Utilizzo dati fallback
                """)
        
        st.markdown("---")
        
        # Sezione per personalizzare i dati
        st.markdown("### 🎨 Personalizzazione Dati per Demo")
        
        st.info("""
        **💡 Suggerimenti per Demo Clienti:**
        
        1. **Modifica prodotti_dettaglio.csv** per includere prodotti specifici del cliente
        2. **Aggiorna vendite_storiche_dettagliate.csv** con pattern realistici del settore
        3. **Personalizza fornitori_dettaglio.csv** con fornitori reali del territorio
        4. **Crea scenari_whatif.csv** specifici per le sfide del cliente
        5. **Configura categorie_config.csv** per le categorie del client
        
        ✨ **Risultato:** Dashboard completamente brandizzata per il cliente!
        """)
        
        # Tabella format requirements
        with st.expander("📋 Formato File CSV Richiesti", expanded=False):
            st.markdown("""
            **prodotti_dettaglio.csv:**
            ```
            codice,nome,categoria,scorte_attuali,scorta_minima,scorta_sicurezza,prezzo_medio,lead_time,criticita
            ```
            
            **vendite_storiche_dettagliate.csv:**
            ```
            data,CRZ001,CRZ002,MAT001,... (una colonna per prodotto)
            ```
            
            **ordini_attivi.csv:**
            ```
            id_ordine,prodotto_codice,quantita,fornitore,data_ordine,data_consegna_prevista,stato,costo_totale
            ```
            
            **fornitori_dettaglio.csv:**
            ```
            nome_fornitore,categoria_specializzazione,lead_time_medio,affidabilita_percentuale,prezzo_1_10_unita
            ```
            
            **scenari_whatif.csv:**
            ```
            scenario_nome,descrizione,lead_time_modifier,domanda_modifier,impact_description
            ```
            """)
        
        # Azioni file
        st.markdown("### ⚡ Azioni Rapide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Ricarica Dati", help="Ricarica tutti i dati dai CSV"):
                st.experimental_rerun()
        
        with col2:
            if st.button("📊 Rigenerazione Vendite", help="Rigenera il file vendite storiche"):
                try:
                    # Rigenera vendite storiche
                    exec(open(data_dir.parent / "generate_vendite_storiche.py").read())
                    st.success("✅ Vendite storiche rigenerate!")
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Errore rigenerazione: {e}")
        
        with col3:
            st.download_button(
                "📥 Download Template CSV",
                data="codice,nome,categoria,scorte_attuali,scorta_minima\nEXAMPLE001,Prodotto Esempio,Categoria1,100,50",
                file_name="template_prodotti.csv",
                mime="text/csv",
                help="Scarica template per creare nuovi file prodotti"
            )
    
    with tab8:
        st.subheader("🔬 Advanced Exogenous Analysis")
        
        if not FORECASTING_AVAILABLE:
            st.warning("⚠️ Advanced forecasting modules not available. Install full ARIMA Forecaster package.")
        else:
            # Info box
            st.info("""
            **Advanced Exogenous Analysis** consente di utilizzare variabili esterne (exogenous)
            per migliorare la precisione delle previsioni SARIMAX. Analizza relazioni, preprocessa
            i dati e seleziona automaticamente le features più rilevanti.
            """)
            
            # Simula alcune variabili exog per demo
            np.random.seed(42)
            n_days = len(vendite)
            exog_demo = pd.DataFrame({
                'temperatura': np.random.normal(20, 5, n_days),
                'promocioni': np.random.binomial(1, 0.3, n_days),
                'festivi': np.random.binomial(1, 0.1, n_days),
                'marketing_spend': np.random.exponential(1000, n_days)
            }, index=vendite.index)
            
            st.markdown("**Demo Exogenous Variables:**")
            st.dataframe(exog_demo.head(10), use_container_width=True)
            
            st.info("🔧 Advanced SARIMAX Analysis: Seleziona un prodotto specifico e usa i controlli per testare la nuova funzionalità Advanced Exogenous Handling.")
            
            if prodotto_sel != 'Tutti':
                if st.button("🚀 Demo Advanced SARIMAX", type="primary"):
                    with st.spinner("Running SARIMAX Auto-Selection demo..."):
                        try:
                            # Demo semplificato
                            target_series = vendite[prodotto_sel].asfreq('D').fillna(0)
                            
                            # Demo correlazioni
                            st.success("✅ Advanced Exog Demo completato!")
                            
                            st.markdown("**📊 Feature Correlations:**")
                            corr_data = []
                            for feature in exog_demo.columns:
                                corr = target_series.corr(exog_demo[feature])
                                corr_data.append({
                                    'Feature': feature,
                                    'Correlation': f"{corr:.3f}",
                                    'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                                })
                            
                            corr_df = pd.DataFrame(corr_data)
                            st.dataframe(corr_df, use_container_width=True)
                            
                            # Grafico correlazioni
                            fig_corr = go.Figure()
                            fig_corr.add_trace(go.Bar(
                                x=exog_demo.columns,
                                y=[target_series.corr(exog_demo[col]) for col in exog_demo.columns],
                                marker_color=['green' if abs(target_series.corr(exog_demo[col])) > 0.5 else 'orange' for col in exog_demo.columns]
                            ))
                            fig_corr.update_layout(
                                title=f'Feature Correlations with {prodotto_sel}',
                                xaxis_title='Exogenous Variables',
                                yaxis_title='Correlation Coefficient'
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Demo error: {e}")
            else:
                st.info("Seleziona un prodotto specifico per testare Advanced Exog Analysis.")
    
    with tab9:
        st.subheader("🚀 Cold Start Problem - Nuovo Prodotto")
        
        if not COLD_START_AVAILABLE:
            st.warning("⚠️ Cold Start modules not available. Check arima_forecaster installation.")
        else:
            st.info("""
            **Cold Start Problem Solution** - Genera previsioni per prodotti nuovi senza dati storici,
            utilizzando pattern e caratteristiche di prodotti simili esistenti. Perfetto per 
            lanci di nuovi prodotti o estensioni di gamma.
            """)
            
            # Configurazione nuovo prodotto
            st.markdown("### 📝 Configurazione Nuovo Prodotto")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Caratteristiche Prodotto:**")
                new_product_code = st.text_input("Codice Prodotto", "NUOVO001", help="Codice del nuovo prodotto")
                new_product_name = st.text_input("Nome Prodotto", "Nuovo Dispositivo Medicale")
                new_product_category = st.selectbox(
                    "Categoria", 
                    ["Carrozzine", "Materassi Antidecubito", "Elettromedicali", "Altri"],
                    help="Categoria del nuovo prodotto"
                )
                new_product_price = st.number_input("Prezzo (€)", min_value=10.0, max_value=5000.0, value=150.0, step=10.0)
                
            with col2:
                st.markdown("**Parametri Forecasting:**")
                forecast_method = st.selectbox(
                    "Metodo Cold Start",
                    ["hybrid", "pattern", "analogical"],
                    help="Metodo per generare previsioni"
                )
                forecast_days_cs = st.slider("Giorni da prevedere", 7, 90, 30)
                similarity_threshold = st.slider("Soglia Similarità", 0.1, 0.9, 0.7, 0.1)
                
                # Caratteristiche opzionali aggiuntive
                with st.expander("Caratteristiche Aggiuntive"):
                    new_product_weight = st.number_input("Peso (kg)", min_value=0.1, max_value=100.0, value=2.0)
                    new_product_volume = st.number_input("Volume (L)", min_value=0.1, max_value=1000.0, value=10.0)
                    new_expected_demand = st.number_input("Domanda Attesa (unità/giorno)", min_value=0.1, max_value=100.0, value=5.0)
            
            st.markdown("---")
            
            # Genera forecast
            if st.button("🎯 Genera Forecast Cold Start", type="primary"):
                with st.spinner("Analizzando prodotti simili e generando previsioni..."):
                    try:
                        # Prepara dati del nuovo prodotto
                        target_product_info = {
                            'codice': new_product_code,
                            'nome': new_product_name,
                            'categoria': new_product_category,
                            'prezzo': new_product_price,
                            'peso': new_product_weight,
                            'volume': new_product_volume,
                            'expected_demand': new_expected_demand,
                            'features': {}
                        }
                        
                        # Estrai features del prodotto target (simulato)
                        target_features = {
                            'price': new_product_price,
                            'category_encoded': hash(new_product_category) % 1000,
                            'weight': new_product_weight,
                            'volume': new_product_volume,
                            'expected_demand_level': new_expected_demand
                        }
                        target_product_info['features'] = target_features
                        
                        # Prepara database prodotti esistenti
                        products_database = {}
                        
                        # Usa i dati prodotti già caricati all'avvio
                        # Convertiamo prodotti (che è un DataFrame) in formato con indice sui codici
                        prodotti_info = prodotti.set_index('codice') if 'codice' in prodotti.columns else pd.DataFrame()
                        
                        for codice in vendite.columns:
                            if not prodotti_info.empty and codice in prodotti_info.index:
                                # Dati vendite
                                product_sales = vendite[codice].dropna()
                                if len(product_sales) < 30:  # Skip prodotti con pochi dati
                                    continue
                                
                                # Info prodotto
                                product_info = prodotti_info.loc[codice].to_dict()
                                
                                # Estrai features per similarità
                                cold_start_forecaster = ColdStartForecaster(
                                    similarity_threshold=similarity_threshold
                                )
                                
                                product_features = cold_start_forecaster.extract_product_features(
                                    product_sales, product_info
                                )
                                
                                products_database[codice] = {
                                    'vendite': product_sales,
                                    'info': product_info,
                                    'features': product_features
                                }
                        
                        if not products_database:
                            st.error("❌ Nessun prodotto con dati sufficienti per l'analisi")
                        else:
                            # Inizializza Cold Start Forecaster
                            cold_start_forecaster = ColdStartForecaster(
                                similarity_threshold=similarity_threshold
                            )
                            
                            # Genera forecast
                            forecast_series, metadata = cold_start_forecaster.cold_start_forecast(
                                target_product_info=target_product_info,
                                products_database=products_database,
                                forecast_days=forecast_days_cs,
                                method=forecast_method
                            )
                            
                            st.success(f"✅ Forecast generato con metodo: {metadata.get('method', 'unknown')}")
                            
                            # Risultati
                            st.markdown("### 📊 Risultati Cold Start")
                            
                            # Metriche riepilogo
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_demand = forecast_series.mean()
                                st.metric("Domanda Media", f"{avg_demand:.1f}", help="Unità al giorno previste")
                            
                            with col2:
                                total_demand = forecast_series.sum()
                                st.metric("Domanda Totale", f"{total_demand:.0f}", help=f"Unità totali in {forecast_days_cs} giorni")
                            
                            with col3:
                                max_demand = forecast_series.max()
                                st.metric("Picco Domanda", f"{max_demand:.1f}", help="Giorno con maggiore domanda")
                            
                            with col4:
                                confidence = metadata.get('confidence', 'medium')
                                confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                                st.metric("Affidabilità", f"{confidence_color.get(confidence, '⚪')} {confidence.title()}")
                            
                            # Grafico forecast
                            st.markdown("### 📈 Previsioni Giornaliere")
                            
                            fig_cs = go.Figure()
                            fig_cs.add_trace(go.Scatter(
                                x=forecast_series.index,
                                y=forecast_series.values,
                                mode='lines+markers',
                                name=f'{new_product_name}',
                                line=dict(color='#1f77b4', width=3),
                                marker=dict(size=6)
                            ))
                            
                            fig_cs.update_layout(
                                title=f'Cold Start Forecast - {new_product_name} ({forecast_days_cs} giorni)',
                                xaxis_title='Data',
                                yaxis_title='Unità Previste',
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig_cs, use_container_width=True)
                            
                            # Prodotti simili utilizzati
                            if 'source_products' in metadata:
                                st.markdown("### 🔍 Prodotti Simili Utilizzati")
                                
                                similar_products_data = []
                                source_products = metadata.get('source_products', [])
                                similarity_scores = metadata.get('similarity_scores', [])
                                
                                for i, source_product in enumerate(source_products):
                                    if source_product in prodotti_info.index:
                                        product_name = prodotti_info.loc[source_product, 'nome']
                                        similarity_score = similarity_scores[i] if i < len(similarity_scores) else 0
                                        
                                        similar_products_data.append({
                                            'Codice': source_product,
                                            'Nome': product_name,
                                            'Similarità': f"{similarity_score:.3f}",
                                            'Categoria': prodotti_info.loc[source_product, 'categoria']
                                        })
                                
                                if similar_products_data:
                                    similar_df = pd.DataFrame(similar_products_data)
                                    st.dataframe(similar_df, use_container_width=True)
                                else:
                                    st.info("Nessun prodotto simile identificato nel database")
                            
                            # Download forecast CSV
                            st.markdown("### 📁 Export Risultati")
                            
                            # Prepara CSV export
                            forecast_export = forecast_series.reset_index()
                            forecast_export.columns = ['Data', 'Domanda_Prevista']
                            forecast_export['Prodotto_Codice'] = new_product_code
                            forecast_export['Prodotto_Nome'] = new_product_name
                            forecast_export['Metodo'] = metadata.get('method', 'unknown')
                            forecast_export['Affidabilità'] = metadata.get('confidence', 'medium')
                            
                            csv_export = forecast_export.to_csv(index=False)
                            
                            st.download_button(
                                "📥 Download Forecast CSV",
                                data=csv_export,
                                file_name=f"cold_start_forecast_{new_product_code}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                            
                            # Metadata tecnico (espandibile)
                            with st.expander("🔧 Dettagli Tecnici"):
                                st.json(metadata)
                    
                    except Exception as e:
                        st.error(f"❌ Errore durante generazione forecast: {str(e)}")
                        st.exception(e)
                        
            # Esempio pratico
            st.markdown("---")
            st.markdown("### 💡 Esempio Scenario")
            
            with st.expander("📖 Caso d'Uso: Lancio Carrozzina Ultra-Light"):
                st.markdown("""
                **Scenario**: Moretti vuole lanciare una nuova carrozzina ultra-leggera "CRZ-ULTRA-001"
                
                **Parametri**:
                - **Prezzo**: €1,200 (premium rispetto a CRZ001 standard)
                - **Categoria**: Carrozzine 
                - **Peso**: 8kg (vs 12kg della standard)
                - **Target**: Clienti mobility-conscious
                
                **Il sistema**:
                1. Analizza CRZ001 (carrozzina standard) come prodotto più simile
                2. Applica scaling per prezzo premium (-20% domanda stimata)
                3. Considera stagionalità e trend da prodotti simili
                4. Genera forecast 30-90 giorni per pianificazione scorte iniziali
                
                **Output**: Domanda stimata 15-18 unità/giorno primi 30 giorni
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