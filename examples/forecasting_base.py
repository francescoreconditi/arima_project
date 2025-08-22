"""
Esempio di forecasting ARIMA base.

Questo esempio dimostra:
1. Caricamento e preprocessing di dati serie temporali
2. Addestramento di un modello ARIMA
3. Generazione previsioni
4. Visualizzazione risultati

Esecuzione:
    uv run python examples/forecasting_base.py
    oppure: python examples/forecasting_base.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per Windows
import matplotlib.pyplot as plt
from pathlib import Path

# Aggiungi src al path per gli import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.data import DataLoader
from arima_forecaster.utils import setup_logger
from utils import get_plots_path, get_models_path, get_reports_path

def generate_sample_data():
    """Genera dati di serie temporali di esempio per la dimostrazione."""
    np.random.seed(42)
    
    # Crea range di date
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Genera componente trend
    trend = np.linspace(100, 200, len(dates))
    
    # Genera componente stagionale (stagionalità annuale)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    
    # Genera rumore
    noise = np.random.normal(0, 10, len(dates))
    
    # Combina componenti
    values = trend + seasonal + noise
    
    # Crea DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': values
    }).set_index('date')
    
    return df

def main():
    """Esempio principale di forecasting."""
    
    # Configura logging
    logger = setup_logger('basic_forecasting', level='INFO')
    logger.info("Avvio esempio base di forecasting")
    
    try:
        # Passo 1: Genera o carica dati
        logger.info("Generazione dati di esempio...")
        df = generate_sample_data()
        
        # Salva dati di esempio per uso futuro
        output_dir = Path(__file__).parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "sample_sales_data.csv")
        logger.info(f"Dati di esempio salvati in {output_dir / 'sample_sales_data.csv'}")
        
        # Estrai la serie temporale
        series = df['sales']
        logger.info(f"Caricata serie temporale con {len(series)} osservazioni")
        logger.info(f"Range date: {series.index.min()} a {series.index.max()}")
        
        # Passo 2: Preprocessing dati
        logger.info("Preprocessing dati...")
        preprocessor = TimeSeriesPreprocessor()
        
        # Controlla stazionarietà
        stationarity_test = preprocessor.check_stationarity(series)
        logger.info(f"Serie originale è stazionaria: {stationarity_test['is_stationary']} (p-value: {stationarity_test['p_value']:.4f})")
        
        # Applica pipeline preprocessing
        processed_series, metadata = preprocessor.preprocess_pipeline(
            series,
            handle_missing=True,
            missing_method='interpolate',
            remove_outliers_flag=False,  # Mantieni tutti i dati per questo esempio
            make_stationary_flag=True,
            stationarity_method='difference'
        )
        
        logger.info(f"Preprocessing completato: {len(series)} -> {len(processed_series)} osservazioni")
        logger.info(f"Passi preprocessing: {metadata['preprocessing_steps']}")
        
        # Passo 3: Dividi dati per valutazione
        train_size = int(len(processed_series) * 0.8)
        train_data = processed_series[:train_size]
        test_data = processed_series[train_size:]
        
        logger.info(f"Set di training: {len(train_data)} osservazioni")
        logger.info(f"Set di test: {len(test_data)} osservazioni")
        
        # Passo 4: Addestra modello ARIMA
        logger.info("Addestramento modello ARIMA...")
        model = ARIMAForecaster(order=(2, 1, 1))  # ARIMA(2,1,1)
        model.fit(train_data)
        
        # Ottieni informazioni modello
        model_info = model.get_model_info()
        logger.info(f"Modello addestrato con successo")
        logger.info(f"AIC: {model_info['aic']:.2f}")
        logger.info(f"BIC: {model_info['bic']:.2f}")
        
        # Passo 5: Genera previsioni
        logger.info("Generazione previsioni...")
        
        # Previsione in-sample (valori adattati)
        fitted_values = model.predict(start=0, end=len(train_data)-1)
        
        # Previsione out-of-sample
        forecast_steps = len(test_data)
        forecast, conf_int = model.forecast(
            steps=forecast_steps,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        logger.info(f"Generate {len(forecast)} previsioni")
        
        # Passo 6: Valuta performance modello
        from arima_forecaster.evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Calcola metriche previsione
        metrics = evaluator.calculate_forecast_metrics(test_data, forecast)
        logger.info("Metriche Performance Previsione:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Analisi residui
        residuals = model.fitted_model.resid
        residual_diagnostics = evaluator.evaluate_residuals(residuals)
        
        logger.info("Diagnostiche Residui:")
        jb_test = residual_diagnostics['jarque_bera_test']
        logger.info(f"  Test Jarque-Bera (normalità): p-value = {jb_test['p_value']:.4f}")
        
        lb_test = residual_diagnostics['ljung_box_test']
        logger.info(f"  Test Ljung-Box (autocorrelazione): p-value = {lb_test['p_value']:.4f}")
        
        # Passo 7: Visualizzazione
        logger.info("Creazione visualizzazioni...")
        
        plotter = ForecastPlotter()
        
        # Crea directory output per grafici
        plots_dir = Path(__file__).parent.parent / "outputs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Grafico 1: Previsione con intervalli di confidenza
        fig1 = plotter.plot_forecast(
            actual=train_data,
            forecast=forecast,
            confidence_intervals=conf_int,
            title="Previsione ARIMA con Intervalli di Confidenza 95%",
            save_path=plots_dir / "forecast_with_ci.png"
        )
        
        # Grafico 2: Analisi residui
        fig2 = plotter.plot_residuals(
            residuals=residuals,
            fitted_values=fitted_values,
            title="Analisi Residui",
            save_path=plots_dir / "residual_analysis.png"
        )
        
        # Grafico 3: ACF/PACF della serie originale
        fig3 = plotter.plot_acf_pacf(
            series=series,
            lags=24,
            title="ACF/PACF della Serie Originale",
            save_path=plots_dir / "acf_pacf_original.png"
        )
        
        # Grafico 4: Dashboard completa
        fig4 = plotter.create_dashboard(
            actual=train_data,
            forecast=forecast,
            residuals=residuals,
            confidence_intervals=conf_int,
            metrics=metrics,
            title="Dashboard Forecasting",
            save_path=plots_dir / "dashboard.png"
        )
        
        logger.info(f"Grafici salvati in {plots_dir}")
        
        # Passo 8: Salva modello
        model_dir = Path(__file__).parent.parent / "outputs" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "basic_arima_model.pkl"
        model.save(model_path)
        logger.info(f"Modello salvato in {model_path}")

        # Genera report Quarto
        logger.info("Generazione report Quarto...")
        try:
            # Ottieni il nome file del grafico se esiste
            plot_files = {}
            # Prova a trovare il file grafico più recente
            dashboard_path = plots_dir / "dashboard.png"
            if dashboard_path.exists():
                plot_files['main_plot'] = str(dashboard_path)
            
            report_path = model.generate_report(
                plots_data=plot_files if plot_files else None,
                report_title="Analisi Base di Forecasting",
                output_filename="report_forecasting_base",
                format_type="html",
                include_diagnostics=True,
                include_forecast=True,
                forecast_steps=12
            )
            logger.info(f"Report HTML generato: {report_path}")
            print(f"Report HTML salvato in: {report_path}")
        except Exception as e:
            logger.warning(f"Impossibile generare report: {e}")
            print(f"Report non generato: {e}")
        
        # Passo 9: Genera previsioni future
        logger.info("Generazione previsioni future...")
        
        # Riaddestra su tutti i dati disponibili
        full_model = ARIMAForecaster(order=(2, 1, 1))
        full_model.fit(processed_series)
        
        # Genera previsione 12 mesi avanti
        future_forecast, future_conf_int = full_model.forecast(
            steps=12,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        # Grafico previsione futura
        fig5 = plotter.plot_forecast(
            actual=processed_series.tail(36),  # Mostra ultimi 3 anni + previsione
            forecast=future_forecast,
            confidence_intervals=future_conf_int,
            title="Previsione Futura 12 Mesi",
            save_path=plots_dir / "future_forecast.png"
        )
        
        logger.info("Previsione futura:")
        for i, (f, lower, upper) in enumerate(zip(future_forecast, 
                                                  future_conf_int.iloc[:, 0], 
                                                  future_conf_int.iloc[:, 1])):
            logger.info(f"  Mese {i+1}: {f:.2f} [{lower:.2f}, {upper:.2f}]")
        
        logger.info("Esempio base di forecasting completato con successo!")
        print("Plot salvato come 'outputs/plots/forecasting_base.png'")
        
        # Mostra grafici (opzionale - commenta se esegui headless)
        # plt.show()  # Disabilitato per compatibilità Windows
        
    except Exception as e:
        logger.error(f"Esempio fallito: {e}")
        raise

if __name__ == "__main__":
    main()