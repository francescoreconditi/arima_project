"""
Dimostrazione completa delle funzionalità avanzate di forecasting ARIMA.

Questo script dimostra:
- Modelli SARIMA con supporto stagionale
- Modelli VAR multivariati
- Ottimizzazione iperparametri Auto-ML
- Ensemble forecasting
- Confronto e valutazione modelli
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per Windows
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arima_forecaster import (
from utils import get_plots_path, get_models_path, get_reports_path
    ARIMAForecaster, SARIMAForecaster, VARForecaster,
    ARIMAModelSelector, SARIMAModelSelector,
    TimeSeriesPreprocessor, ModelEvaluator,
    ARIMAOptimizer, SARIMAOptimizer, VAROptimizer,
    HyperparameterTuner, optimize_model
)


def generate_sample_data():
    """Genera dati di serie temporali di esempio per la dimostrazione."""
    print("📊 Generazione dataset di esempio...")
    
    # Serie temporale univariata con trend, stagionalità e rumore
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='M')
    n = len(dates)
    
    # Trend di base
    trend = np.linspace(100, 200, n)
    
    # Componente stagionale (ciclo annuale)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12)
    
    # Componente random walk
    np.random.seed(42)
    random_walk = np.cumsum(np.random.normal(0, 2, n))
    
    # Rumore
    noise = np.random.normal(0, 3, n)
    
    # Combina componenti
    univariate_series = pd.Series(
        trend + seasonal + random_walk + noise,
        index=dates,
        name='sales'
    )
    
    # Dati multivariati (es. vendite, spesa marketing, indice competitor)
    marketing_spend = 50 + 0.3 * trend + 5 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 5, n)
    competitor_index = 80 + 0.1 * trend - 3 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 4, n)
    
    multivariate_data = pd.DataFrame({
        'sales': univariate_series.values,
        'marketing_spend': marketing_spend,
        'competitor_index': competitor_index
    }, index=dates)
    
    print(f"✅ Serie univariata generata: {len(univariate_series)} osservazioni")
    print(f"✅ Dati multivariati generati: {multivariate_data.shape[0]} × {multivariate_data.shape[1]}")
    
    return univariate_series, multivariate_data


def demonstrate_sarima_models(data):
    """Dimostra modelli SARIMA con supporto stagionale."""
    print("\n🌊 Modelli SARIMA con Supporto Stagionale")
    print("=" * 50)
    
    # Specifica SARIMA manuale
    print("\n1. Modello SARIMA Manuale (1,1,1)(1,1,1,12)")
    sarima_manual = SARIMAForecaster(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    sarima_manual.fit(data)
    
    model_info = sarima_manual.get_model_info()
    print(f"   AIC: {model_info['aic']:.2f}")
    print(f"   BIC: {model_info['bic']:.2f}")
    
    # Generate forecast
    forecast, conf_int = sarima_manual.forecast(steps=12, return_conf_int=True)
    print(f"   Range previsione 12 mesi: {forecast.min():.2f} a {forecast.max():.2f}")
    
    # Selezione SARIMA automatica
    print("\n2. Selezione Automatica Modello SARIMA")
    sarima_selector = SARIMAModelSelector(
        p_range=(0, 2),
        d_range=(0, 2),
        q_range=(0, 2),
        P_range=(0, 1),
        D_range=(0, 1),
        Q_range=(0, 1),
        seasonal_periods=[12],
        max_models=20
    )
    
    sarima_selector.search(data, verbose=False)
    best_sarima = sarima_selector.get_best_model()
    
    print(f"   Ordine SARIMA migliore: {sarima_selector.best_order}")
    print(f"   Ordine stagionale migliore: {sarima_selector.best_seasonal_order}")
    
    best_info = best_sarima.get_model_info()
    print(f"   AIC migliore: {best_info['aic']:.2f}")
    
    # Decomposizione stagionale
    print("\n3. Decomposizione Stagionale")
    decomposition = best_sarima.get_seasonal_decomposition()
    print(f"   Range componente trend: {decomposition['trend'].min():.2f} a {decomposition['trend'].max():.2f}")
    print(f"   Range componente stagionale: {decomposition['seasonal'].min():.2f} a {decomposition['seasonal'].max():.2f}")
    
    return best_sarima


def demonstrate_var_models(data):
    """Dimostra modelli Vector Autoregression (VAR)."""
    print("\n📈 Modelli Vector Autoregression (VAR)")
    print("=" * 50)
    
    # Prima adatta il modello VAR per controllare la stazionarietà
    print("\n1. Analisi di Stazionarietà")
    var_model = VARForecaster(maxlags=4)
    var_model.fit(data)  # Fit first so we can check stationarity
    
    stationarity_results = var_model.check_stationarity()
    
    for variable, result in stationarity_results.items():
        status = "✅" if result.get('is_stationary', False) else "❌"
        print(f"   {variable}: {status} (p-value: {result.get('p_value', 'N/A'):.4f})")
    
    # Rendi i dati stazionari se necessario
    stationary_data = data.diff().dropna()  # Prima differenza
    
    # Adatta modello VAR con selezione automatica lag su dati stazionari
    print("\n2. Modello VAR con Selezione Automatica Lag")
    var_model = VARForecaster(ic='aic')
    var_model.fit(stationary_data)
    
    model_info = var_model.get_model_info()
    print(f"   Ordine lag selezionato: {model_info['lag_order']}")
    print(f"   AIC: {model_info['aic']:.2f}")
    print(f"   BIC: {model_info['bic']:.2f}")
    
    # Genera previsione VAR
    print("\n3. Previsione VAR")
    var_forecast = var_model.forecast(steps=6, alpha=0.05)
    
    forecast_df = var_forecast['forecast']
    print(f"   Previsione 6 periodi generata per {len(forecast_df.columns)} variabili")
    
    for var in forecast_df.columns:
        forecast_range = f"{forecast_df[var].min():.2f} to {forecast_df[var].max():.2f}"
        print(f"   Range previsione {var}: {forecast_range}")
    
    # Funzioni di Risposta all'Impulso
    print("\n4. Analisi di Risposta all'Impulso")
    try:
        irf = var_model.impulse_response(periods=10)
        print(f"   IRF calcolata per {irf.shape[1]} combinazioni di variabili")
        
        # Esempio: Risposta vendite a shock spesa marketing
        if 'sales' in data.columns and 'marketing_spend' in data.columns:
            sales_response = var_model.impulse_response(
                periods=10,
                impulse='marketing_spend',
                response='sales'
            )
            max_response = sales_response.max()
            print(f"   Massima risposta vendite a shock marketing: {max_response:.4f}")
        
    except Exception as e:
        print(f"   Analisi IRF fallita: {e}")
    
    # Test di Causalità di Granger
    print("\n5. Test di Causalità di Granger")
    try:
        if 'sales' in data.columns and len(data.columns) > 1:
            other_vars = [col for col in data.columns if col != 'sales']
            causality_results = var_model.granger_causality('sales', other_vars)
            
            for test, result in causality_results.items():
                significance = "✅" if result['p_value'] < 0.05 else "❌"
                print(f"   {test}: {significance} (p-value: {result['p_value']:.4f})")
                
    except Exception as e:
        print(f"   Test causalità Granger falliti: {e}")
    
    return var_model


def demonstrate_automl_optimization(data):
    """Dimostra ottimizzazione iperparametri Auto-ML."""
    print("\n🤖 Ottimizzazione Iperparametri Auto-ML")
    print("=" * 50)
    
    # Ottimizzazione single-objective con algoritmi diversi
    print("\n1. Confronto Ottimizzazione Single-Objective")
    
    algorithms = ['optuna']  # Focus su Optuna perché è più probabile che sia disponibile
    results = {}
    
    for algorithm in algorithms:
        try:
            print(f"\n   Test {algorithm.upper()}:")
            
            if algorithm == 'optuna':
                optimizer = ARIMAOptimizer(objective_metric='aic')
                result = optimizer.optimize_optuna(data, n_trials=20)
            
            results[algorithm] = result
            print(f"   ✅ Parametri migliori: {result['best_params']}")
            print(f"   ✅ Punteggio migliore: {result['best_score']:.2f}")
            print(f"   ✅ Trial completati: {result.get('n_trials', 'N/A')}")
            
        except ImportError as e:
            print(f"   ❌ {algorithm} non disponibile: {e}")
        except Exception as e:
            print(f"   ❌ {algorithm} fallito: {e}")
    
    # Ottimizzazione multi-objective
    print("\n2. Ottimizzazione Multi-Objective")
    try:
        tuner = HyperparameterTuner(
            objective_metrics=['aic', 'bic'],
            ensemble_method='weighted_average'
        )
        
        multi_result = tuner.multi_objective_optimization(
            'arima', data, n_trials=15
        )
        
        print(f"   ✅ Soluzioni Pareto-ottimali trovate: {multi_result['n_pareto_solutions']}")
        
        if multi_result['best_solution']:
            best_params = multi_result['best_solution']['params']
            print(f"   ✅ Soluzione di compromesso migliore: {best_params}")
            
            # Mostra punteggi per obiettivi multipli
            scores = multi_result['best_solution']['scores']
            for metric, score in scores.items():
                print(f"      {metric.upper()}: {score:.2f}")
        
    except Exception as e:
        print(f"   ❌ Ottimizzazione multi-objective fallita: {e}")
    
    # Ottimizzazione ensemble
    print("\n3. Creazione Modello Ensemble")
    try:
        ensemble_result = tuner.ensemble_optimization(
            'arima', data, n_models=3, diversity_threshold=0.2
        )
        
        n_models = ensemble_result['n_models']
        diversity = ensemble_result['diversity_metrics']
        
        print(f"   ✅ Ensemble creato con {n_models} modelli")
        print(f"   ✅ Diversità media: {diversity['avg_diversity']:.3f}")
        print(f"   ✅ Punteggio ensemble: {ensemble_result['ensemble_score']:.2f}")
        
        # Genera previsione ensemble
        ensemble_forecast = tuner.forecast_ensemble(steps=6, method='weighted')
        print(f"   ✅ Previsione ensemble generata (6 passi)")
        print(f"      Range previsione: {ensemble_forecast['forecast'].min():.2f} a {ensemble_forecast['forecast'].max():.2f}")
        
    except Exception as e:
        print(f"   ❌ Ottimizzazione ensemble fallita: {e}")
    
    # Ottimizzazione adattiva
    print("\n4. Ottimizzazione Adattiva")
    try:
        adaptive_result = tuner.adaptive_optimization(
            'arima', data, max_iterations=5, improvement_threshold=0.01
        )
        
        iterations = adaptive_result['total_iterations']
        final_score = adaptive_result['final_score']
        converged = adaptive_result['converged']
        
        print(f"   ✅ Completate {iterations} iterazioni")
        print(f"   ✅ Punteggio finale: {final_score:.2f}")
        print(f"   ✅ Convergenza: {'Sì' if converged else 'No'}")
        
        if adaptive_result['best_iteration']:
            best_iter = adaptive_result['best_iteration']
            print(f"   ✅ Iterazione migliore: {best_iter['iteration']} (punteggio: {best_iter['score']:.2f})")
        
    except Exception as e:
        print(f"   ❌ Ottimizzazione adattiva fallita: {e}")
    
    return results


def comprehensive_model_comparison(data):
    """Confronta tutti i tipi di modello in modo completo."""
    print("\n🏆 Confronto Completo Modelli")
    print("=" * 50)
    
    models = {}
    
    # Prepara split train/test
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"\nTraining su {len(train_data)} osservazioni, test su {len(test_data)} osservazioni")
    
    # 1. ARIMA Base
    print("\n1. Modello ARIMA Base")
    try:
        arima = ARIMAForecaster(order=(2, 1, 2))
        arima.fit(train_data)
        
        arima_forecast = arima.forecast(steps=len(test_data), confidence_intervals=False)
        arima_mse = np.mean((arima_forecast[:len(test_data)] - test_data[:len(arima_forecast)]) ** 2)
        
        models['ARIMA'] = {
            'model': arima,
            'forecast': arima_forecast,
            'mse': arima_mse,
            'aic': arima.get_model_info()['aic']
        }
        print(f"   ✅ MSE: {arima_mse:.2f}, AIC: {models['ARIMA']['aic']:.2f}")
        
    except Exception as e:
        print(f"   ❌ ARIMA fallito: {e}")
    
    # 2. Modello SARIMA
    print("\n2. Modello SARIMA")
    try:
        sarima = SARIMAForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        sarima.fit(train_data)
        
        sarima_forecast = sarima.forecast(steps=len(test_data), confidence_intervals=False)
        sarima_mse = np.mean((sarima_forecast[:len(test_data)] - test_data[:len(sarima_forecast)]) ** 2)
        
        models['SARIMA'] = {
            'model': sarima,
            'forecast': sarima_forecast,
            'mse': sarima_mse,
            'aic': sarima.get_model_info()['aic']
        }
        print(f"   ✅ MSE: {sarima_mse:.2f}, AIC: {models['SARIMA']['aic']:.2f}")
        
    except Exception as e:
        print(f"   ❌ SARIMA fallito: {e}")
    
    # 3. ARIMA Auto-ottimizzato
    print("\n3. ARIMA Auto-Ottimizzato")
    try:
        optimizer = ARIMAOptimizer(objective_metric='aic')
        opt_result = optimizer.optimize_optuna(train_data, n_trials=15)
        
        opt_model = optimizer.best_model
        opt_forecast = opt_model.forecast(steps=len(test_data), confidence_intervals=False)
        opt_mse = np.mean((opt_forecast[:len(test_data)] - test_data[:len(opt_forecast)]) ** 2)
        
        models['Auto-ARIMA'] = {
            'model': opt_model,
            'forecast': opt_forecast,
            'mse': opt_mse,
            'aic': opt_result['best_score']
        }
        print(f"   ✅ MSE: {opt_mse:.2f}, AIC: {models['Auto-ARIMA']['aic']:.2f}")
        print(f"      Parametri migliori: {opt_result['best_params']}")
        
    except Exception as e:
        print(f"   ❌ Auto-ARIMA fallito: {e}")
    
    # Classifica modelli
    print("\n🏅 Classifica Modelli (per MSE):")
    if models:
        ranked_models = sorted(models.items(), key=lambda x: x[1]['mse'])
        
        for i, (name, info) in enumerate(ranked_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"   {medal} {name}: MSE={info['mse']:.2f}, AIC={info['aic']:.2f}")
    
    return models


def create_visualization_dashboard(data, models):
    """Crea dashboard di visualizzazione completa."""
    print("\n📊 Creazione Dashboard di Visualizzazione")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dashboard Funzionalità Avanzate ARIMA Forecasting', fontsize=16, fontweight='bold')
        
        # Grafico 1: Dati originali con trend
        ax1 = axes[0, 0]
        ax1.plot(data.index, data.values, 'b-', alpha=0.7, label='Dati Originali')
        ax1.set_title('Dati Serie Temporale')
        ax1.set_ylabel('Valore')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Grafico 2: Confronto modelli (se modelli disponibili)
        ax2 = axes[0, 1]
        if models:
            # Mostra ultimi 24 mesi di dati + previsioni
            recent_data = data[-24:]
            ax2.plot(recent_data.index, recent_data.values, 'k-', label='Effettivo', linewidth=2)
            
            colors = ['red', 'green', 'orange', 'purple']
            for i, (name, info) in enumerate(models.items()):
                if 'forecast' in info:
                    forecast = info['forecast']
                    # Crea indice previsione
                    last_date = data.index[-1]
                    freq = pd.infer_freq(data.index)
                    forecast_index = pd.date_range(
                        start=last_date + pd.tseries.frequencies.to_offset(freq),
                        periods=len(forecast),
                        freq=freq
                    )
                    
                    color = colors[i % len(colors)]
                    ax2.plot(forecast_index[:len(forecast)], forecast.values[:len(forecast_index)], 
                            '--', color=color, label=f'{name} Previsione', alpha=0.8)
        
        ax2.set_title('Confronto Previsioni Modelli')
        ax2.set_ylabel('Valore')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Grafico 3: Decomposizione stagionale (se disponibile)
        ax3 = axes[1, 0]
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(data, model='additive', period=12)
            ax3.plot(decomposition.seasonal[-48:], 'g-', label='Componente Stagionale')
            ax3.set_title('Componente Stagionale (Ultimi 4 Anni)')
            ax3.set_ylabel('Effetto Stagionale')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except Exception:
            ax3.text(0.5, 0.5, 'Decomposizione stagionale\nnon disponibile', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Analisi Stagionale')
        
        # Grafico 4: Metriche performance modelli
        ax4 = axes[1, 1]
        if models:
            model_names = list(models.keys())
            mse_values = [models[name]['mse'] for name in model_names]
            aic_values = [models[name]['aic'] for name in model_names]
            
            # Normalizza valori per confronto
            mse_norm = np.array(mse_values) / max(mse_values)
            aic_norm = np.array(aic_values) / max(aic_values)
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax4.bar(x - width/2, mse_norm, width, label='MSE (normalizzato)', alpha=0.7)
            ax4.bar(x + width/2, aic_norm, width, label='AIC (normalizzato)', alpha=0.7)
            
            ax4.set_xlabel('Modelli')
            ax4.set_ylabel('Punteggio Normalizzato')
            ax4.set_title('Confronto Performance Modelli')
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Nessun risultato\nmodello disponibile', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Modelli')
        
        plt.tight_layout()
        
        # Salva il grafico
        output_path = Path(__file__).parent.parent / "outputs" / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path / "advanced_forecasting_dashboard.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard salvata in: {output_path / 'advanced_forecasting_dashboard.png'}")
        print("Grafico salvato come 'outputs/plots/advanced_forecast_showcase.png'")
        
        # plt.show()  # Disabilitato per compatibilità Windows
        
    except ImportError:
        print("❌ Matplotlib non disponibile per la visualizzazione")
    except Exception as e:
        print(f"❌ Visualizzazione fallita: {e}")


def main():
    """Funzione principale di dimostrazione."""
    print("🚀 Dimostrazione Funzionalità Avanzate di ARIMA Forecaster")
    print("=" * 60)
    print("Questo script dimostra le funzionalità avanzate aggiunte alla libreria ARIMA:")
    print("• Modelli SARIMA con supporto stagionale")
    print("• Modelli VAR multivariati")  
    print("• Ottimizzazione iperparametri Auto-ML")
    print("• Metodi ensemble e tuning avanzato")
    print("• Confronto modelli completo")
    print("=" * 60)
    
    # Genera dati di esempio
    univariate_data, multivariate_data = generate_sample_data()
    
    # Dimostra modelli SARIMA
    best_sarima = demonstrate_sarima_models(univariate_data)
    
    # Dimostra modelli VAR
    var_model = demonstrate_var_models(multivariate_data)
    
    # Dimostra ottimizzazione Auto-ML
    automl_results = demonstrate_automl_optimization(univariate_data)
    
    # Confronto completo modelli
    model_comparison = comprehensive_model_comparison(univariate_data)
    
    # Crea dashboard visualizzazione
    create_visualization_dashboard(univariate_data, model_comparison)
    
    print("\n✅ Dimostrazione Funzionalità Avanzate Completata!")
    print("\nProssimi passi:")
    print("• Avvia il server API: python scripts/run_api.py")
    print("• Lancia la dashboard: python scripts/run_dashboard.py")
    print("• Controlla i grafici generati in outputs/plots/")
    
    return {
        'univariate_data': univariate_data,
        'multivariate_data': multivariate_data,
        'best_sarima': best_sarima,
        'var_model': var_model,
        'automl_results': automl_results,
        'model_comparison': model_comparison
    }


if __name__ == "__main__":
    results = main()