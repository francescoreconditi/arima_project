"""
Esempio completo di utilizzo del modello SARIMAX con variabili esogene.

Questo esempio dimostra come:
1. Caricare e preprocessare dati con variabili esogene
2. Addestrare un modello SARIMAX
3. Fare previsioni con variabili esogene future
4. Valutare le performance e visualizzare i risultati
5. Usare la selezione automatica dei parametri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Assicuriamoci che il package sia trovabile
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from arima_forecaster import (
    SARIMAXForecaster,
    SARIMAXModelSelector,
    TimeSeriesPreprocessor,
    ModelEvaluator,
    ForecastPlotter
)


def generate_sample_data_with_exog(n_periods: int = 200) -> tuple[pd.Series, pd.DataFrame]:
    """
    Genera dati di esempio con variabili esogene per dimostrare SARIMAX.
    
    Returns:
        Tupla di (serie_target, variabili_esogene)
    """
    print("Generazione dati di esempio con variabili esogene...")
    
    # Indice temporale
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Variabili esogene
    # 1. Temperatura (influenza stagionale)
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25) + np.random.normal(0, 2, n_periods)
    
    # 2. Marketing spend (trend crescente con rumore)
    marketing = 1000 + 50 * np.arange(n_periods) + np.random.normal(0, 200, n_periods)
    marketing = np.maximum(marketing, 0)  # Non può essere negativo
    
    # 3. Day of week effect (variabile categorica convertita in numerica)
    dow_effect = np.sin(2 * np.pi * np.arange(n_periods) / 7) * 5
    
    # 4. Economic indicator (Random walk with trend)
    econ_indicator = np.cumsum(np.random.normal(0.1, 1, n_periods)) + 100
    
    # DataFrame variabili esogene
    exog = pd.DataFrame({
        'temperature': temperature,
        'marketing_spend': marketing,
        'day_of_week_effect': dow_effect,
        'economic_indicator': econ_indicator
    }, index=dates)
    
    # Serie target (influenzata dalle variabili esogene)
    # Base trend
    trend = 500 + 2 * np.arange(n_periods)
    
    # Stagionalità
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25) + 20 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
    
    # Influenza delle variabili esogene
    temp_effect = 0.8 * temperature  # Correlazione positiva con temperatura
    marketing_effect = 0.02 * marketing  # Effetto positivo del marketing
    dow_effect_target = 0.5 * dow_effect  # Effetto giorno della settimana
    econ_effect = 0.3 * (econ_indicator - 100)  # Effetto indicatore economico
    
    # Componente ARIMA
    arima_component = np.random.normal(0, 20, n_periods)
    for i in range(1, n_periods):
        arima_component[i] = 0.3 * arima_component[i-1] + np.random.normal(0, 15)
    
    # Serie finale
    target = trend + seasonal + temp_effect + marketing_effect + dow_effect_target + econ_effect + arima_component
    target_series = pd.Series(target, index=dates, name='target_variable')
    
    print(f"Generati {n_periods} giorni di dati")
    print(f"   Variabili esogene: {list(exog.columns)}")
    print(f"   Range serie target: {target_series.min():.2f} - {target_series.max():.2f}")
    
    return target_series, exog


def demonstrate_sarimax_basic():
    """Dimostra l'uso base del modello SARIMAX."""
    print("\n" + "="*60)
    print("DEMO: Modello SARIMAX Base")
    print("="*60)
    
    # Genera dati
    series, exog = generate_sample_data_with_exog(n_periods=150)
    
    # Split train/test
    train_size = int(len(series) * 0.8)
    series_train = series[:train_size]
    series_test = series[train_size:]
    exog_train = exog[:train_size]
    exog_test = exog[train_size:]
    
    print(f"\nSplit dati:")
    print(f"   Training: {len(series_train)} giorni")
    print(f"   Test: {len(series_test)} giorni")
    
    # Preprocessing
    print("\nPreprocessing dati...")
    preprocessor = TimeSeriesPreprocessor()
    
    # Preprocessa serie target e variabili esogene insieme
    series_processed, exog_processed, metadata = preprocessor.preprocess_pipeline_with_exog(
        series=series_train,
        exog=exog_train,
        handle_missing=True,
        make_stationary_flag=True,
        scale_exog=True,  # Scala le variabili esogene
        exog_scaling_method='standardize'
    )
    
    print(f"Preprocessing completato:")
    print(f"   Serie: {len(series_train)} -> {len(series_processed)} osservazioni")
    print(f"   Variabili esogene: {exog.shape} -> {exog_processed.shape}")
    
    # Crea e addestra modello SARIMAX
    print("\nAddestramento modello SARIMAX...")
    model = SARIMAXForecaster(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),  # Stagionalità settimanale
        exog_names=list(exog.columns)
    )
    
    model.fit(series_processed, exog=exog_processed)
    print("Modello addestrato con successo!")
    
    # Informazioni modello
    model_info = model.get_model_info()
    print(f"\nInfo modello:")
    print(f"   AIC: {model_info['aic']:.2f}")
    print(f"   BIC: {model_info['bic']:.2f}")
    print(f"   Variabili esogene: {model_info['n_exog']}")
    
    # Analisi importanza variabili esogene
    try:
        exog_importance = model.get_exog_importance()
        if not exog_importance.empty:
            print(f"\nImportanza variabili esogene:")
            for _, row in exog_importance.iterrows():
                significance = "OK" if row['significant'] else "NO"
                print(f"   {row['variable']}: coeff={row['coefficient']:.4f}, p-value={row['pvalue']:.4f} {significance}")
    except Exception as e:
        print(f"Errore analisi importanza: {e}")
    
    # Previsione (Nota: richiede variabili esogene future)
    print("\nGenerazione previsioni...")
    
    # Per la previsione, dobbiamo fornire le variabili esogene future
    # Simuliamo questo prendendo i valori effettivi del test set
    forecast_steps = min(30, len(series_test))
    exog_future = exog_test[:forecast_steps].copy()
    
    # Applica stesso preprocessing alle variabili esogene future
    exog_future_processed, _ = preprocessor.preprocess_exog_data(
        exog=exog_future,
        handle_missing=True,
        scale_data=True,
        scaling_method='standardize'
    )
    
    # Assicurati che le colonne siano nell'ordine giusto
    exog_future_processed = exog_future_processed[exog_processed.columns]
    
    try:
        forecast_result = model.forecast(
            steps=forecast_steps,
            exog_future=exog_future_processed,
            confidence_intervals=True
        )
        
        if isinstance(forecast_result, tuple):
            forecast_values, conf_int = forecast_result
            print(f"Previsioni generate: {len(forecast_values)} punti")
        else:
            forecast_values = forecast_result
            conf_int = None
            print(f"Previsioni generate: {len(forecast_values)} punti (senza intervalli confidenza)")
        
        # Valutazione su dati disponibili
        if len(forecast_values) <= len(series_test):
            evaluator = ModelEvaluator()
            actual_values = series_test[:len(forecast_values)]
            
            # Nota: Le previsioni sono sulla serie differenziata, dobbiamo "de-differenziare"
            # Per semplicità, valutiamo direttamente (in pratica servirebbe una trasformazione inversa)
            try:
                metrics = evaluator.calculate_forecast_metrics(
                    actual=actual_values.values, 
                    predicted=forecast_values.values
                )
                print(f"\nMetriche performance:")
                print(f"   MAE: {metrics['mae']:.2f}")
                print(f"   RMSE: {metrics['rmse']:.2f}")
                print(f"   MAPE: {metrics['mape']:.2f}%")
            except Exception as e:
                print(f"Errore calcolo metriche: {e}")
        
    except Exception as e:
        print(f"Errore previsione: {e}")
        print("Suggerimento: SARIMAX richiede variabili esogene per fare previsioni")


def demonstrate_sarimax_model_selection():
    """Dimostra la selezione automatica del modello SARIMAX."""
    print("\n" + "="*60)
    print("🔍 DEMO: Selezione Automatica Modello SARIMAX")
    print("="*60)
    
    # Genera dati più piccoli per velocizzare la ricerca
    series, exog = generate_sample_data_with_exog(n_periods=100)
    
    print(f"\n🎯 Selezione automatica parametri su {len(series)} osservazioni")
    
    # Preprocessing rapido
    preprocessor = TimeSeriesPreprocessor()
    series_processed, exog_processed, _ = preprocessor.preprocess_pipeline_with_exog(
        series=series,
        exog=exog,
        make_stationary_flag=True,
        scale_exog=True
    )
    
    # Selezione automatica modello
    print("\n🔍 Ricerca parametri ottimali...")
    selector = SARIMAXModelSelector(
        p_range=(0, 2),  # Range ridotto per velocità
        d_range=(0, 1),
        q_range=(0, 2),
        P_range=(0, 1),
        D_range=(0, 1),
        Q_range=(0, 1),
        seasonal_periods=[7],  # Solo stagionalità settimanale
        exog_names=list(exog.columns),
        max_models=10,  # Limita numero modelli per demo
        information_criterion='aic'
    )
    
    # Esegui ricerca
    selector.search(
        series=series_processed,
        exog=exog_processed,
        verbose=True
    )
    
    # Risultati
    best_model = selector.get_best_model()
    if best_model:
        model_info = best_model.get_model_info()
        print(f"\n🏆 Miglior modello trovato:")
        print(f"   Ordine: {model_info['order']}")
        print(f"   Ordine stagionale: {model_info['seasonal_order']}")
        print(f"   AIC: {model_info['aic']:.2f}")
        print(f"   Variabili esogene: {model_info['n_exog']}")
        
        # Mostra risultati top 5
        results_summary = selector.get_results_summary(top_n=5)
        if not results_summary.empty:
            print(f"\n📋 Top 5 modelli:")
            for i, row in results_summary.iterrows():
                print(f"   {i+1}. {row['order']}x{row['seasonal_order']} - AIC: {row['aic']:.2f}")
        
        # Analisi variabili esogene
        try:
            exog_analysis = selector.get_exog_analysis()
            if exog_analysis is not None:
                print(f"\n🧬 Analisi variabili esogene:")
                significant_vars = exog_analysis[exog_analysis['significant'] == True]['variable'].unique()
                print(f"   Variabili significative: {list(significant_vars)}")
        except Exception as e:
            print(f"⚠️  Errore analisi esogene: {e}")
    else:
        print("❌ Nessun modello valido trovato")


def demonstrate_sarimax_visualization():
    """Dimostra le visualizzazioni specifiche per SARIMAX."""
    print("\n" + "="*60)
    print("📊 DEMO: Visualizzazioni SARIMAX")
    print("="*60)
    
    # Genera dati
    series, exog = generate_sample_data_with_exog(n_periods=120)
    
    # Split e preprocessing
    train_size = int(len(series) * 0.8)
    series_train = series[:train_size]
    exog_train = exog[:train_size]
    
    preprocessor = TimeSeriesPreprocessor()
    series_processed, exog_processed, _ = preprocessor.preprocess_pipeline_with_exog(
        series=series_train,
        exog=exog_train,
        scale_exog=True
    )
    
    # Addestra modello
    print("\n🤖 Addestramento modello per visualizzazioni...")
    model = SARIMAXForecaster(
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        exog_names=list(exog.columns)
    )
    model.fit(series_processed, exog=exog_processed)
    
    # Crea visualizzazioni
    print("\n📈 Creazione visualizzazioni...")
    plotter = ForecastPlotter()
    
    try:
        # 1. Analisi variabili esogene
        exog_importance = model.get_exog_importance()
        
        fig1 = plotter.plot_exog_analysis(
            exog_data=exog_train,
            exog_importance=exog_importance,
            target_series=series_train,
            title="Analisi Variabili Esogene SARIMAX"
        )
        
        output_dir = Path("outputs/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.savefig(output_dir / "sarimax_exog_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"✅ Grafico analisi esogene salvato in: outputs/plots/sarimax_exog_analysis.png")
        
        # 2. Dashboard SARIMAX completa
        # Genera previsioni per la dashboard
        forecast_steps = 20
        exog_test = exog[train_size:train_size+forecast_steps]
        exog_test_processed, _ = preprocessor.preprocess_exog_data(exog_test, scale_data=True)
        exog_test_processed = exog_test_processed[exog_processed.columns]
        
        forecast_series = model.forecast(forecast_steps, exog_future=exog_test_processed)
        
        # Residui in-sample
        predictions = model.predict()
        residuals = series_processed - predictions
        
        # Dashboard
        fig2 = plotter.create_sarimax_dashboard(
            actual=series_train,
            forecast=forecast_series,
            exog_data=exog_train,
            exog_importance=exog_importance,
            residuals=residuals,
            title="Dashboard SARIMAX Completa"
        )
        
        fig2.savefig(output_dir / "sarimax_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"✅ Dashboard SARIMAX salvata in: outputs/plots/sarimax_dashboard.png")
        
        print(f"\n🎨 Visualizzazioni completate!")
        print(f"   📁 Directory output: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Errore creazione visualizzazioni: {e}")


def main():
    """Funzione principale che esegue tutte le demo."""
    print("*** ESEMPI SARIMAX - Modelli con Variabili Esogene ***")
    print("="*80)
    
    try:
        # Demo 1: Uso base
        demonstrate_sarimax_basic()
        
        # Demo 2: Selezione automatica
        demonstrate_sarimax_model_selection()
        
        # Demo 3: Visualizzazioni
        demonstrate_sarimax_visualization()
        
        print("\n" + "="*80)
        print("🎯 RIASSUNTO SARIMAX")
        print("="*80)
        print("✅ SARIMAX estende SARIMA con variabili esogene")
        print("✅ Richiede variabili esogene sia per training che per previsioni")
        print("✅ Supporta selezione automatica parametri")
        print("✅ Include visualizzazioni dedicate per analisi variabili esogene")
        print("✅ Integrato con preprocessing avanzato e scaling")
        print("✅ Fornisce analisi importanza e significatività variabili")
        
        print("\n💡 SUGGERIMENTI:")
        print("• Le variabili esogene devono essere note anche per i periodi futuri")
        print("• Usa il preprocessing integrato per gestire missing values e scaling")
        print("• Analizza sempre la significatività delle variabili esogene")
        print("• Considera correlazioni elevate tra variabili esogene (multicollinearità)")
        
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")
        raise


if __name__ == "__main__":
    main()