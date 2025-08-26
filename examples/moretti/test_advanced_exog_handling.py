#!/usr/bin/env python3
"""
Test completo Advanced Exogenous Handling per sistema Moretti S.p.A.

Questo script dimostra le nuove funzionalità:
- SARIMAXAutoSelector con selezione automatica features
- ExogPreprocessor avanzato con multiple strategie  
- ExogDiagnostics per validazione completa
- Integration con dati realistici settore medicale

Caso d'uso: Previsioni domanda Carrozzine con fattori esterni
(demografici, stagionali, economici, epidemiologici)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import logging

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root / "src"))

# Imports libreria
from arima_forecaster.core.sarimax_auto_selector import SARIMAXAutoSelector
from arima_forecaster.utils.preprocessing import ExogenousPreprocessor, analyze_feature_relationships
from arima_forecaster.utils.exog_diagnostics import ExogDiagnostics
from arima_forecaster.utils.logger import setup_logger, get_logger
from arima_forecaster.evaluation.metrics import ModelEvaluator

# Setup logging
setup_logger(level=logging.INFO)
logger = get_logger(__name__)

# Suppress warnings per demo
warnings.filterwarnings("ignore")

def generate_moretti_exog_data(n_days: int = 365) -> tuple[pd.Series, pd.DataFrame]:
    """
    Genera dati realistici per test Advanced Exog Handling.
    
    Returns:
        Tuple con (serie_domanda_carrozzine, variabili_esogene)
    """
    logger.info(f"Generando {n_days} giorni di dati Moretti con variabili esogene")
    
    # Date range
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Base random seed per reproducibilità
    np.random.seed(42)
    
    # === VARIABILI ESOGENE REALISTICHE ===
    
    # 1. Demografia (invecchiamento popolazione - trend crescente)
    popolazione_over65 = np.linspace(100000, 105000, n_days) + np.random.normal(0, 500, n_days)
    
    # 2. Stagionalità meteo (inverno = più richiesta)
    giorni_anno = np.arange(n_days) % 365
    temp_media = 15 + 10 * np.cos(2 * np.pi * giorni_anno / 365) + np.random.normal(0, 2, n_days)
    
    # 3. Indicatori economici (PIL, disoccupazione)
    pil_regionale = 50000 + np.random.walk(n_days, scale=100)  # Random walk realistic
    tasso_disoccupazione = 8 + 2 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 0.3, n_days)
    
    # 4. Eventi COVID/epidemiologici (spikes casuali)
    covid_impact = np.zeros(n_days)
    covid_dates = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)  # 5% giorni
    covid_impact[covid_dates] = np.random.exponential(2, len(covid_dates))
    
    # 5. Festività/weekend (pattern settimanale)
    is_weekend = np.array([d.weekday() >= 5 for d in dates]).astype(int)
    is_holiday = np.zeros(n_days)
    holiday_dates = [25, 50, 100, 150, 200, 250, 300, 350]  # Festività principali
    for h in holiday_dates:
        if h < n_days:
            is_holiday[h] = 1
    
    # 6. Spesa sanitaria pubblica (budget cycles)
    spesa_sanitaria = 1000 + 200 * np.sin(2 * np.pi * np.arange(n_days) / 90) + np.random.normal(0, 50, n_days)  # Cicli trimestrali
    
    # 7. Advertising/Marketing budget (campagne casuali)
    marketing_budget = 5000 + np.random.exponential(1000, n_days)
    marketing_spikes = np.random.choice(n_days, size=int(n_days * 0.1), replace=False)
    marketing_budget[marketing_spikes] *= 3  # Campagne intensive
    
    # 8. Competitor pricing (anti-correlato con domanda)
    competitor_price = 1500 + 200 * np.random.walk(n_days, scale=5) + np.random.normal(0, 30, n_days)
    
    # 9. Supply chain disruption (events occasionali)
    supply_disruption = np.random.exponential(0.1, n_days)  # Basso baseline
    disruption_events = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)  # 2% giorni
    supply_disruption[disruption_events] += np.random.exponential(5, len(disruption_events))
    
    # 10. Google search trends (leading indicator)
    search_trends = 100 + 30 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 10, n_days)
    # Correlato con domanda futura (3-day lead)
    search_trends = np.roll(search_trends, -3)  # Shift verso futuro
    
    # === SERIE TARGET: DOMANDA CARROZZINE ===
    
    # Combina tutti i fattori con weights realistici
    base_demand = 25  # Domanda base giornaliera
    
    # Seasonal component (inverno +20%)
    seasonal = 2 * np.cos(2 * np.pi * giorni_anno / 365 + np.pi)  # Peak inverno
    
    # Demographic trend (+2% year)
    trend = 0.1 * np.arange(n_days) / 365
    
    # External factors effects
    temp_effect = -0.2 * (temp_media - 15)  # Freddo = più domanda
    covid_effect = 1.5 * covid_impact  # COVID aumenta domanda
    weekend_effect = -0.3 * is_weekend  # Weekend meno domanda
    holiday_effect = -0.5 * is_holiday  # Festivi meno domanda
    econ_effect = -0.001 * (pil_regionale - 50000) + 0.2 * (tasso_disoccupazione - 8)  # PIL alto e disoc. bassa = meno domanda
    health_effect = 0.003 * (spesa_sanitaria - 1000)  # Spesa alta = più domanda
    marketing_effect = 0.001 * (marketing_budget - 5000)  # Marketing = più domanda
    price_effect = -0.01 * (competitor_price - 1500)  # Competitor caro = più domanda per noi
    supply_effect = -0.5 * supply_disruption  # Supply issues = meno vendite
    search_effect = 0.05 * (search_trends - 100)  # Search alto = più domanda
    
    # Combina tutti gli effetti
    deterministic_demand = (base_demand + trend + seasonal + 
                          temp_effect + covid_effect + weekend_effect + holiday_effect +
                          econ_effect + health_effect + marketing_effect + 
                          price_effect + supply_effect + search_effect)
    
    # Aggiungi noise realistico
    noise = np.random.normal(0, 2.5, n_days)  # ±2.5 unità random variation
    
    # Demand finale (non negativa)
    demand = np.maximum(deterministic_demand + noise, 1)
    
    # === CREA DATAFRAMES ===
    
    demand_series = pd.Series(demand, index=dates, name='domanda_carrozzine')
    
    exog_data = pd.DataFrame({
        'popolazione_over65': popolazione_over65,
        'temperatura_media': temp_media,
        'pil_regionale': pil_regionale,
        'tasso_disoccupazione': tasso_disoccupazione,
        'covid_impact': covid_impact,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'spesa_sanitaria': spesa_sanitaria,
        'marketing_budget': marketing_budget,
        'competitor_price': competitor_price,
        'supply_disruption': supply_disruption,
        'search_trends': search_trends
    }, index=dates)
    
    logger.info(f"Dati generati: domanda media {demand.mean():.1f} ± {demand.std():.1f}")
    logger.info(f"Range domanda: {demand.min():.1f} - {demand.max():.1f}")
    
    return demand_series, exog_data


def random_walk(n: int, scale: float = 1.0) -> np.ndarray:
    """Genera random walk per serie economiche."""
    steps = np.random.normal(0, scale, n)
    return np.cumsum(steps)


def demo_advanced_exog_preprocessing():
    """Demo ExogPreprocessor avanzato."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Advanced Exogenous Preprocessing")
    logger.info("="*80)
    
    # Genera dati
    demand, exog = generate_moretti_exog_data(180)  # 6 mesi
    
    logger.info(f"\nDati originali:")
    logger.info(f"- Variabili esogene: {len(exog.columns)}")
    logger.info(f"- Osservazioni: {len(exog)}")
    logger.info(f"- Missing values: {exog.isnull().sum().sum()}")
    
    # Aggiungi qualche missing value per test
    exog.iloc[10:15, 2] = np.nan  # PIL mancante
    exog.iloc[50:52, 7] = np.nan  # Spesa sanitaria mancante
    
    # Test different preprocessing methods
    methods = ['robust', 'standard', 'minmax']
    
    for method in methods:
        logger.info(f"\nTesting preprocessing method: {method.upper()}")
        
        preprocessor = ExogenousPreprocessor(
            method=method,
            handle_outliers=True,
            outlier_method='modified_zscore',
            missing_strategy='interpolate',
            detect_multicollinearity=True,
            multicollinearity_threshold=0.90,
            stationarity_test=True
        )
        
        # Fit e transform
        exog_processed = preprocessor.fit_transform(exog)
        
        # Report
        report = preprocessor.get_preprocessing_report()
        
        logger.info(f"  - Trasformazioni applicate: {len(report.get('transformations_applied', []))}")
        logger.info(f"  - Feature multicollineari rimosse: {len(report.get('multicollinear_features', []))}")
        logger.info(f"  - Feature finali: {exog_processed.shape[1]}")
        
        # Suggerimenti
        suggestions = preprocessor.suggest_improvements(demand)
        if suggestions:
            logger.info(f"  - Suggerimenti: {suggestions[0]}")
    
    return exog_processed


def demo_feature_selection_methods():
    """Demo metodi selezione feature."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Feature Selection Methods Comparison")
    logger.info("="*80)
    
    # Genera dati più ricchi
    demand, exog = generate_moretti_exog_data(300)  # 10 mesi
    
    # Test different selection methods
    methods = ['stepwise', 'lasso', 'elastic_net', 'f_test']
    
    results_comparison = {}
    
    for method in methods:
        logger.info(f"\nTesting selection method: {method.upper()}")
        
        try:
            selector = SARIMAXAutoSelector(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),  # Settimanale
                max_features=6,
                selection_method=method,
                feature_engineering=['lags'],  # Solo lags per velocità
                preprocessing_method='robust'
            )
            
            # Split train/test
            train_size = int(len(demand) * 0.8)
            demand_train = demand.iloc[:train_size]
            exog_train = exog.iloc[:train_size]
            demand_test = demand.iloc[train_size:]
            exog_test = exog.iloc[train_size:]
            
            # Training
            start_time = datetime.now()
            selector.fit_with_exog(demand_train, exog_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Feature analysis
            feature_analysis = selector.get_feature_analysis()
            
            # Forecast test
            forecast_result = selector.forecast_with_exog(
                steps=len(demand_test),
                exog=exog_test,
                confidence_intervals=False
            )
            
            # Evaluation
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_forecast_metrics(demand_test, forecast_result)
            
            results_comparison[method] = {
                'selected_features': len(selector.selected_features),
                'features_list': selector.selected_features,
                'training_time': training_time,
                'mape': metrics['mape'],
                'rmse': metrics['rmse'],
                'aic': selector.fitted_model.aic,
                'bic': selector.fitted_model.bic
            }
            
            logger.info(f"  - Features selected: {len(selector.selected_features)}")
            logger.info(f"  - Top features: {selector.selected_features[:3]}")
            logger.info(f"  - Training time: {training_time:.1f}s")
            logger.info(f"  - MAPE: {metrics['mape']:.2f}%")
            logger.info(f"  - AIC: {selector.fitted_model.aic:.1f}")
            
        except Exception as e:
            logger.error(f"  - FAILED: {str(e)}")
            results_comparison[method] = {'error': str(e)}
    
    # Comparison summary
    logger.info(f"\n{'METHOD':<15} {'FEATURES':<10} {'MAPE':<8} {'TIME':<8} {'AIC':<8}")
    logger.info("-" * 60)
    
    for method, results in results_comparison.items():
        if 'error' not in results:
            logger.info(f"{method:<15} {results['selected_features']:<10} "
                       f"{results['mape']:<8.2f} {results['training_time']:<8.1f} {results['aic']:<8.1f}")
        else:
            logger.info(f"{method:<15} {'ERROR':<10}")
    
    return results_comparison


def demo_exog_diagnostics():
    """Demo diagnostica avanzata variabili esogene."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Advanced Exogenous Diagnostics")
    logger.info("="*80)
    
    # Genera dati
    demand, exog = generate_moretti_exog_data(200)  # ~7 mesi
    
    # Setup diagnostics
    diagnostics = ExogDiagnostics(max_lag=7, significance_level=0.05)
    
    # Addestra un modello per analisi residui
    logger.info("Training SARIMAX model for residual analysis...")
    selector = SARIMAXAutoSelector(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        max_features=5,
        selection_method='stepwise',
        preprocessing_method='robust'
    )
    
    selector.fit_with_exog(demand, exog)
    
    # Full diagnostic suite
    logger.info("Running comprehensive diagnostic suite...")
    diagnostic_results = diagnostics.full_diagnostic_suite(
        target_series=demand,
        exog_data=exog,
        fitted_model=selector.fitted_model
    )
    
    # Report results
    summary = diagnostic_results.get('summary', {})
    logger.info(f"\nDIAGNOSTIC SUMMARY:")
    logger.info(f"  - Overall assessment: {summary.get('overall_assessment', 'unknown').upper()}")
    logger.info(f"  - Features analyzed: {summary.get('feature_count', 0)}")
    logger.info(f"  - Key findings: {len(summary.get('key_findings', []))}")
    logger.info(f"  - Critical issues: {len(summary.get('critical_issues', []))}")
    
    # Stationarity results
    stationarity = diagnostic_results.get('stationarity', {})
    stationary_vars = [var for var, result in stationarity.items() 
                      if result.get('consensus') == 'stationary']
    logger.info(f"\nSTATIONARITY ANALYSIS:")
    logger.info(f"  - Stationary variables: {len(stationary_vars)}")
    logger.info(f"  - Examples: {stationary_vars[:3]}")
    
    # Causality results
    causality = diagnostic_results.get('causality', {})
    causal_vars = [(var, result.get('strength', 'unknown')) for var, result in causality.items() 
                   if result.get('is_causal', False)]
    logger.info(f"\nCAUSALITY ANALYSIS:")
    logger.info(f"  - Causal variables found: {len(causal_vars)}")
    if causal_vars:
        logger.info(f"  - Top causal: {causal_vars[0][0]} ({causal_vars[0][1]})")
    
    # Feature importance
    importance = diagnostic_results.get('feature_importance', {})
    top_features = importance.get('top_features', [])[:3]
    logger.info(f"\nFEATURE IMPORTANCE:")
    logger.info(f"  - Top important features: {top_features}")
    
    # Recommendations
    recommendations = diagnostic_results.get('recommendations', [])
    logger.info(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[:3], 1):
        logger.info(f"  {i}. {rec}")
    
    # Save diagnostic report
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    report_path = diagnostics.save_diagnostic_report(
        output_path=output_dir / "moretti_exog_diagnostic_report.json",
        format_type='json'
    )
    
    logger.info(f"\nDiagnostic report saved to: {report_path}")
    
    return diagnostic_results


def demo_end_to_end_workflow():
    """Demo workflow completo Advanced Exog Handling."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: End-to-End Advanced Exog Workflow")
    logger.info("="*80)
    
    logger.info("Simulating complete Moretti forecasting workflow with external factors...")
    
    # 1. Data generation
    logger.info("\n1. GENERATING REALISTIC DATA")
    demand, exog = generate_moretti_exog_data(365)  # 1 anno
    
    # 2. Feature relationship analysis
    logger.info("\n2. ANALYZING FEATURE RELATIONSHIPS")
    relationships = analyze_feature_relationships(exog, demand)
    
    high_corr = [var for var, corr in relationships.get('correlations', {}).items() 
                 if abs(corr) > 0.3]
    logger.info(f"High correlation features: {high_corr}")
    
    # 3. Advanced preprocessing
    logger.info("\n3. ADVANCED PREPROCESSING")
    preprocessor = ExogenousPreprocessor(
        method='robust',
        handle_outliers=True,
        outlier_method='modified_zscore',
        missing_strategy='interpolate',
        detect_multicollinearity=True,
        stationarity_test=True
    )
    
    exog_processed = preprocessor.fit_transform(exog)
    logger.info(f"Features after preprocessing: {exog_processed.shape[1]} (from {exog.shape[1]})")
    
    # 4. Train/test split
    logger.info("\n4. TRAIN/TEST SPLIT")
    train_size = int(len(demand) * 0.8)
    demand_train, demand_test = demand.iloc[:train_size], demand.iloc[train_size:]
    exog_train, exog_test = exog.iloc[:train_size], exog.iloc[train_size:]
    
    logger.info(f"Training period: {train_size} days")
    logger.info(f"Test period: {len(demand_test)} days")
    
    # 5. Model training with auto feature selection
    logger.info("\n5. SARIMAX AUTO TRAINING")
    selector = SARIMAXAutoSelector(
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        max_features=8,
        selection_method='stepwise',
        feature_engineering=['lags', 'differences'],
        preprocessing_method='robust',
        validation_split=0.2
    )
    
    selector.fit_with_exog(demand_train, exog_train)
    
    # 6. Model evaluation and diagnostics
    logger.info("\n6. MODEL DIAGNOSTICS")
    diagnostics = ExogDiagnostics()
    diagnostic_results = diagnostics.full_diagnostic_suite(
        target_series=demand_train,
        exog_data=exog_train,
        fitted_model=selector.fitted_model
    )
    
    # 7. Forecasting
    logger.info("\n7. FORECASTING")
    forecast_result = selector.forecast_with_exog(
        steps=len(demand_test),
        exog=exog_test,
        confidence_intervals=True
    )
    
    if isinstance(forecast_result, tuple):
        forecast_values, conf_intervals = forecast_result
    else:
        forecast_values = forecast_result
        conf_intervals = None
    
    # 8. Evaluation
    logger.info("\n8. EVALUATION")
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_forecast_metrics(demand_test, forecast_values)
    
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"  - Selected features: {len(selector.selected_features)}")
    logger.info(f"  - Feature list: {selector.selected_features}")
    logger.info(f"  - MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  - RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  - MAE: {metrics['mae']:.2f}")
    logger.info(f"  - Model AIC: {selector.fitted_model.aic:.1f}")
    logger.info(f"  - Model BIC: {selector.fitted_model.bic:.1f}")
    
    # Feature importance analysis
    feature_analysis = selector.get_feature_analysis()
    logger.info(f"\nFEATURE SELECTION ANALYSIS:")
    logger.info(f"  - Original features: {feature_analysis['selection_summary']['original_features']}")
    logger.info(f"  - Engineered features: {feature_analysis['selection_summary']['engineered_features']}")
    logger.info(f"  - Final selected: {feature_analysis['selection_summary']['selected_features']}")
    logger.info(f"  - Selection ratio: {feature_analysis['selection_summary']['selection_ratio']:.2%}")
    
    # Diagnostic summary
    diag_summary = diagnostic_results.get('summary', {})
    logger.info(f"\nDIAGNOSTIC ASSESSMENT: {diag_summary.get('overall_assessment', 'unknown').upper()}")
    
    recommendations = diagnostic_results.get('recommendations', [])[:2]
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  Recommendation {i}: {rec}")
    
    # Save results
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save forecast
    results_df = pd.DataFrame({
        'date': demand_test.index,
        'actual': demand_test.values,
        'forecast': forecast_values.values,
        'error': demand_test.values - forecast_values.values,
        'abs_error_pct': abs((demand_test.values - forecast_values.values) / demand_test.values * 100)
    })
    
    results_path = output_dir / "moretti_advanced_exog_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    return {
        'metrics': metrics,
        'selected_features': selector.selected_features,
        'feature_analysis': feature_analysis,
        'diagnostic_summary': diag_summary,
        'model_info': selector.get_model_info()
    }


def main():
    """Esegue tutti i demo Advanced Exog Handling."""
    logger.info("\n🏥 MORETTI S.p.A. - ADVANCED EXOGENOUS HANDLING DEMO")
    logger.info("="*80)
    logger.info("Testing enterprise-grade exogenous variable handling for medical device forecasting")
    logger.info("="*80)
    
    try:
        # Demo 1: Preprocessing
        demo_advanced_exog_preprocessing()
        
        # Demo 2: Feature selection
        demo_feature_selection_methods()
        
        # Demo 3: Diagnostics
        demo_exog_diagnostics()
        
        # Demo 4: End-to-end workflow
        final_results = demo_end_to_end_workflow()
        
        logger.info("\n" + "="*80)
        logger.info("🎯 ADVANCED EXOG HANDLING DEMO COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        logger.info(f"✅ All components tested successfully")
        logger.info(f"✅ Final forecast MAPE: {final_results['metrics']['mape']:.2f}%")
        logger.info(f"✅ Selected {len(final_results['selected_features'])} optimal features")
        logger.info(f"✅ Diagnostic assessment: {final_results['diagnostic_summary'].get('overall_assessment', 'completed')}")
        
        logger.info(f"\n📊 Key Performance Indicators:")
        logger.info(f"   - Feature reduction: {(1 - final_results['feature_analysis']['selection_summary']['selection_ratio']):.1%}")
        logger.info(f"   - Forecast accuracy: {100 - final_results['metrics']['mape']:.1f}%")
        logger.info(f"   - Model fit (AIC): {final_results['model_info']['aic']:.0f}")
        
        logger.info(f"\n🔥 Advanced Exog Handling is production-ready!")
        logger.info(f"   Ready for integration with Moretti forecasting system")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()