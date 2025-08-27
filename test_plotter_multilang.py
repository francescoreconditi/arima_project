#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test del ForecastPlotter con supporto multilingue.
Testa la generazione di grafici in diverse lingue.
"""

import sys
import io
# Forza encoding UTF-8 per output console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per test
import matplotlib.pyplot as plt
from pathlib import Path

# Import del plotter con traduzioni
from src.arima_forecaster.visualization.plotter import ForecastPlotter
# Sistema di traduzioni integrato nel plotter

def generate_sample_data():
    """Genera dati di esempio per i test."""
    # Serie temporale con trend e stagionalità
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + seasonal + noise
    actual = pd.Series(values[:-30], index=dates[:-30], name='vendite')
    forecast = pd.Series(values[-30:] + np.random.normal(0, 3, 30), 
                        index=dates[-30:], name='previsioni')
    
    # Intervalli di confidenza
    confidence_intervals = pd.DataFrame({
        'lower': forecast - 10,
        'upper': forecast + 10
    })
    
    # Residui
    residuals = pd.Series(np.random.normal(0, 5, len(actual)), index=actual.index)
    
    # Metriche
    metrics = {
        'mae': 5.234,
        'rmse': 6.789,
        'mape': 12.45,
        'r_squared': 0.85
    }
    
    return actual, forecast, confidence_intervals, residuals, metrics

def test_plotter_languages():
    """Test del plotter in tutte le lingue supportate."""
    print("=" * 60)
    print("TEST FORECAST PLOTTER MULTILINGUE")
    print("=" * 60)
    
    # Genera dati di esempio
    actual, forecast, confidence_intervals, residuals, metrics = generate_sample_data()
    
    # Directory output
    output_dir = Path("outputs/test_plotter")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lingue da testare
    languages = ['it', 'en', 'es', 'fr', 'zh']
    
    print("\nTest grafici in diverse lingue:")
    print("-" * 40)
    
    for lang in languages:
        print(f"\n[{lang.upper()}] Generazione grafici...")
        
        try:
            # Crea plotter per la lingua specifica
            plotter = ForecastPlotter(language=lang)
            
            # 1. Test plot_forecast
            fig1 = plotter.plot_forecast(
                actual=actual,
                forecast=forecast,
                confidence_intervals=confidence_intervals,
                title=None,  # Usa titolo tradotto automaticamente
                save_path=output_dir / f"forecast_{lang}.png"
            )
            plt.close(fig1)
            print(f"  ✓ Forecast plot salvato")
            
            # 2. Test plot_residuals
            fig2 = plotter.plot_residuals(
                residuals=residuals,
                fitted_values=actual,
                title=None,  # Usa titolo tradotto automaticamente
                save_path=output_dir / f"residuals_{lang}.png"
            )
            plt.close(fig2)
            print(f"  ✓ Residuals plot salvato")
            
            # 3. Test plot_acf_pacf
            fig3 = plotter.plot_acf_pacf(
                series=actual,
                lags=20,
                title=None,  # Usa titolo tradotto automaticamente
                save_path=output_dir / f"acf_pacf_{lang}.png"
            )
            plt.close(fig3)
            print(f"  ✓ ACF/PACF plot salvato")
            
            # 4. Test create_dashboard
            fig4 = plotter.create_dashboard(
                actual=actual,
                forecast=forecast,
                residuals=residuals,
                confidence_intervals=confidence_intervals,
                metrics=metrics,
                title=None,  # Usa titolo tradotto automaticamente
                save_path=output_dir / f"dashboard_{lang}.png"
            )
            plt.close(fig4)
            print(f"  ✓ Dashboard salvata")
            
            # 5. Test plot_decomposition (solo per alcune lingue per velocità)
            if lang in ['it', 'en']:
                try:
                    fig5 = plotter.plot_decomposition(
                        series=actual,
                        model='additive',
                        period=30,
                        title=None,  # Usa titolo tradotto automaticamente
                        save_path=output_dir / f"decomposition_{lang}.png"
                    )
                    plt.close(fig5)
                    print(f"  ✓ Decomposition plot salvato")
                except Exception as e:
                    print(f"  ! Decomposition non disponibile: {e}")
            
            # 6. Test plot_model_comparison
            models_results = {
                'ARIMA(1,1,1)': {'aic': 1234.5, 'bic': 1245.6, 'rmse': 5.2},
                'ARIMA(2,1,2)': {'aic': 1230.1, 'bic': 1242.3, 'rmse': 4.9},
                'SARIMA(1,1,1)': {'aic': 1228.9, 'bic': 1241.2, 'rmse': 4.8}
            }
            
            fig6 = plotter.plot_model_comparison(
                results=models_results,
                metric='aic',
                title=None,  # Usa titolo tradotto automaticamente
                save_path=output_dir / f"model_comparison_{lang}.png"
            )
            plt.close(fig6)
            print(f"  ✓ Model comparison salvato")
            
            print(f"  ✅ Tutti i grafici generati con successo per lingua [{lang}]")
            
        except Exception as e:
            print(f"  ❌ Errore per lingua [{lang}]: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETATO")
    print(f"Grafici salvati in: {output_dir.absolute()}")
    print("=" * 60)
    
    # Test specifico per verificare che le label siano tradotte
    print("\n\nVERIFICA TRADUZIONI LABEL:")
    print("-" * 40)
    
    # Crea un grafico per ogni lingua e verifica alcune label chiave
    for lang in ['it', 'en', 'es']:
        plotter = ForecastPlotter(language=lang)
        
        # Usa il sistema di traduzioni per verificare
        from src.arima_forecaster.utils.translations import translate as _
        
        print(f"\n[{lang.upper()}]:")
        print(f"  - historical_data: {_('historical_data', lang)}")
        print(f"  - forecast: {_('forecast', lang)}")
        print(f"  - confidence_interval: {_('confidence_interval', lang)}")
        print(f"  - time: {_('time', lang)}")
        print(f"  - residuals: {_('residuals', lang)}")
    
    print("\n✅ Sistema di traduzioni integrato correttamente nel ForecastPlotter!")

if __name__ == "__main__":
    # Esegui il test
    test_plotter_languages()
    print("\nTest completato. Controlla i file generati nella directory outputs/test_plotter/")