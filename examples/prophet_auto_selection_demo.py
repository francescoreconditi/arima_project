#!/usr/bin/env python3
"""
Esempio dimostrativo di Prophet Auto-Selection.

Questo esempio mostra come utilizzare ProphetModelSelector per la ricerca automatica
dei parametri ottimali per modelli Facebook Prophet, includendo tutti e tre i metodi
di ottimizzazione disponibili.

Esegui con: uv run python prophet_auto_selection_demo.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import dalla libreria
try:
    from arima_forecaster.core import ProphetForecaster, ProphetModelSelector
    from arima_forecaster.visualization import ForecastPlotter
except ImportError as e:
    print(f"[ERROR] Errore import: {e}")
    print("💡 Installa con: uv add prophet")
    exit(1)


def generate_demo_data(n_days: int = 200) -> pd.Series:
    """Genera dati demo con pattern stagionale e trend."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # Trend lineare
    trend = np.linspace(100, 150, n_days)
    
    # Stagionalità settimanale
    weekly_pattern = 10 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
    
    # Stagionalità mensile
    monthly_pattern = 5 * np.sin(np.arange(n_days) * 2 * np.pi / 30)
    
    # Rumore
    noise = np.random.normal(0, 3, n_days)
    
    # Serie finale
    values = trend + weekly_pattern + monthly_pattern + noise
    values = np.maximum(values, 0)  # Evita valori negativi
    
    return pd.Series(values, index=dates, name='vendite')


def demo_manual_prophet():
    """Demo modello Prophet manuale."""
    print("\n[CONFIG] 1. MODELLO PROPHET MANUALE")
    print("=" * 50)
    
    # Genera dati
    series = generate_demo_data(150)
    
    # Modello manuale con parametri fissi
    model = ProphetForecaster(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    
    # Training
    model.fit(series)
    
    # Forecast
    forecast = model.forecast(steps=30, confidence_intervals=True)
    
    if isinstance(forecast, tuple):
        forecast_series, conf_int = forecast
        print(f"[FORECAST] Forecast 30 giorni:")
        print(f"   Media: {forecast_series.mean():.2f}")
        print(f"   Min-Max: {forecast_series.min():.2f} - {forecast_series.max():.2f}")
        print(f"   Incertezza media: +/-{(conf_int['upper'] - conf_int['lower']).mean()/2:.2f}")
    
    return series, model


def demo_auto_selection():
    """Demo selezione automatica parametri."""
    print("\n[ROBOT] 2. PROPHET AUTO-SELECTION")
    print("=" * 50)
    
    # Genera dati più lunghi per auto-selection
    series = generate_demo_data(180)
    
    # Configurazione selector
    selector = ProphetModelSelector(
        changepoint_prior_scales=[0.01, 0.05, 0.1],  # Limitato per demo
        seasonality_prior_scales=[1.0, 10.0],
        seasonality_modes=['additive', 'multiplicative'],
        yearly_seasonalities=[False],  # Serie troppo corta per yearly
        weekly_seasonalities=[True, 'auto'],
        daily_seasonalities=[False],
        scoring='mape',
        max_models=8,  # Demo veloce
        verbose=True
    )
    
    print(f"[DATA] Serie dati: {len(series)} giorni ({series.index[0].date()} - {series.index[-1].date()})")
    
    # Test Grid Search
    print(f"\n[SEARCH] GRID SEARCH:")
    try:
        best_model, results = selector.search(series, method='grid_search')
        print(f"[OK] Grid Search completato:")
        print(f"   Modelli testati: {len(results)}")
        print(f"   Miglior MAPE: {selector.get_best_score():.3f}%")
        print(f"   Migliori parametri: {selector.get_best_params()}")
    except Exception as e:
        print(f"[ERROR] Grid Search fallito: {e}")
        return None, None
    
    # Test Bayesian Search (se Optuna disponibile)
    print(f"\n[BAYESIAN] BAYESIAN OPTIMIZATION:")
    try:
        selector_bayesian = ProphetModelSelector(
            changepoint_prior_scales=[0.001, 0.01, 0.05, 0.1, 0.5],
            seasonality_modes=['additive', 'multiplicative'],
            max_models=5,  # Demo veloce
            verbose=False
        )
        
        best_model_bay, results_bay = selector_bayesian.search(series, method='bayesian')
        print(f"[OK] Bayesian Search completato:")
        print(f"   Modelli testati: {len(results_bay)}")
        print(f"   Miglior MAPE: {selector_bayesian.get_best_score():.3f}%")
        
    except ImportError:
        print("[WARN]  Optuna non installato, Bayesian search saltato")
    except Exception as e:
        print(f"[ERROR] Bayesian Search fallito: {e}")
    
    # Summary finale
    print(f"\n[SUMMARY] SUMMARY RICERCA:")
    print(selector.summary())
    
    return series, best_model


def demo_forecast_comparison(series, auto_model):
    """Confronta forecast automatico vs manuale."""
    print("\n[COMPARE] 3. CONFRONTO FORECAST")
    print("=" * 50)
    
    if auto_model is None:
        print("[WARN] Modello automatico non disponibile per confronto")
        return
    
    # Modello manuale base
    manual_model = ProphetForecaster(
        growth='linear',
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05  # Parametro fisso
    )
    manual_model.fit(series)
    
    # Forecast entrambi
    steps = 30
    manual_forecast = manual_model.forecast(steps=steps, confidence_intervals=False)
    auto_forecast = auto_model.forecast(steps=steps, confidence_intervals=False)
    
    # Confronto
    manual_mean = manual_forecast.mean()
    auto_mean = auto_forecast.mean()
    difference = abs(auto_mean - manual_mean)
    
    print(f"[FORECAST] Forecast {steps} giorni:")
    print(f"   Manuale (parametri fissi): {manual_mean:.2f}")
    print(f"   Automatico (ottimizzato):  {auto_mean:.2f}")
    print(f"   Differenza assoluta:       {difference:.2f}")
    
    if difference > 5:
        print(f"[OK] Auto-selection ha trovato parametri significativamente diversi!")
    else:
        print(f"[INFO]  I parametri ottimizzati sono simili a quelli manuali")


def main():
    """Esegue demo completa Prophet Auto-Selection."""
    print("[ROCKET] DEMO PROPHET AUTO-SELECTION")
    print("=" * 60)
    print("Questo esempio dimostra la selezione automatica dei parametri")
    print("ottimali per modelli Facebook Prophet con diversi algoritmi.")
    print()
    
    try:
        # Demo 1: Modello manuale
        series_manual, manual_model = demo_manual_prophet()
        
        # Demo 2: Auto-selection 
        series_auto, auto_model = demo_auto_selection()
        
        # Demo 3: Confronto
        if series_auto is not None and auto_model is not None:
            demo_forecast_comparison(series_auto, auto_model)
        
        print(f"\n[SUCCESS] DEMO COMPLETATA!")
        print("=" * 60)
        print("[NEXT] Prossimi step:")
        print("   1. Prova con i tuoi dati reali")
        print("   2. Sperimenta con method='bayesian' per dataset grandi")
        print("   3. Usa cross-validation per valutazioni più robuste")
        print("   4. Integra nel tuo pipeline di forecasting")
        
    except Exception as e:
        print(f"\n[ERROR] DEMO FALLITA: {e}")
        raise


if __name__ == "__main__":
    main()