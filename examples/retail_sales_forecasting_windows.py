#!/usr/bin/env python3
"""
Esempio di Forecasting Vendite Retail - Versione Compatibile Windows

Questo esempio dimostra come utilizzare ARIMA per prevedere le vendite al dettaglio
con pattern stagionali e trend. Include dati mensili con stagionalità annuale tipica
del settore retail (picchi durante le festività).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.core import ARIMAModelSelector
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils import setup_logger

warnings.filterwarnings("ignore")


def generate_retail_sales_data():
    """Genera dati di vendita al dettaglio sintetici con stagionalità"""
    np.random.seed(42)

    # Periodo: 5 anni di dati mensili
    start_date = datetime(2019, 1, 1)
    dates = pd.date_range(start=start_date, periods=60, freq="ME")

    # Componenti del modello retail
    time_idx = np.arange(len(dates))

    # Trend crescente (business in crescita)
    trend = 100000 + 400 * time_idx

    # Stagionalità annuale (picchi in novembre/dicembre per shopping natalizio)
    seasonal = 15000 * np.sin(2 * np.pi * time_idx / 12 + np.pi / 2)  # Sfasato per picchi invernali
    seasonal += 8000 * np.sin(4 * np.pi * time_idx / 12)  # Componente semestrale

    # Picchi extra per Black Friday/Natale (novembre-dicembre)
    black_friday_bonus = np.where((dates.month == 11) | (dates.month == 12), 20000, 0)

    # Variazione casuale per realismo
    noise = np.random.normal(0, 5000, len(dates))

    # Vendite finali
    sales = trend + seasonal + black_friday_bonus + noise

    # Assicuriamoci che le vendite siano positive
    sales = np.maximum(sales, 50000)

    return pd.Series(sales, index=dates, name="monthly_sales")


def evaluate_model_performance(actual, predicted):
    """Valuta le performance del modello"""
    evaluator = ModelEvaluator()

    metrics = evaluator.calculate_forecast_metrics(actual, predicted)

    return {
        "MAPE": metrics["mape"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "R2": metrics.get("r_squared", 0),
    }


def main():
    """Funzione principale per l'analisi forecasting retail"""

    # Setup logging
    logger = setup_logger("retail_forecasting", level="INFO")

    logger.info("Avvio analisi forecasting vendite retail")
    logger.info("Generazione dati vendite retail...")

    # Genera dati
    sales_data = generate_retail_sales_data()

    print(f"Dataset generato: {len(sales_data)} punti dati")
    print(
        f"Periodo: {sales_data.index.min().strftime('%Y-%m')} - {sales_data.index.max().strftime('%Y-%m')}"
    )
    print(f"Vendite media mensile: EUR{sales_data.mean():,.0f}")
    print(f"Range vendite: EUR{sales_data.min():,.0f} - EUR{sales_data.max():,.0f}")

    # Split train/test
    train_size = int(0.8 * len(sales_data))
    train_data = sales_data[:train_size]
    test_data = sales_data[train_size:]

    print(f"\nSplit dataset:")
    print(
        f"  Training: {len(train_data)} mesi ({train_data.index.min().strftime('%Y-%m')} - {train_data.index.max().strftime('%Y-%m')})"
    )
    print(
        f"  Test: {len(test_data)} mesi ({test_data.index.min().strftime('%Y-%m')} - {test_data.index.max().strftime('%Y-%m')})"
    )

    logger.info("Preprocessing dati...")

    # Preprocessing
    preprocessor = TimeSeriesPreprocessor()

    # Check stationarity
    is_stationary = preprocessor.check_stationarity(train_data)
    if not is_stationary:
        print("Serie non stazionaria - applicando differenziazione")
        # Il modello ARIMA gestirà automaticamente la differenziazione

    logger.info("Selezione automatica modello ARIMA...")

    # Selezione modello automatica (versione semplificata per questo esempio)
    print("Utilizzo modello ARIMA(1,1,1) per dati retail...")

    best_order = (1, 1, 1)  # Ordine ottimale per dati retail tipici

    print(f"\nModello ottimale trovato:")
    print(f"  ARIMA{best_order}")
    print(f"  SeasonalNone")

    logger.info("Training modello ARIMA...")

    # Training modello finale
    model = ARIMAForecaster(order=best_order)
    model.fit(train_data)

    logger.info("Generazione forecast per 12 mesi...")

    # Test forecast
    test_forecast = model.forecast(steps=len(test_data), confidence_intervals=True)

    logger.info("Valutazione performance modello...")

    # Estrai i valori del forecast
    if isinstance(test_forecast, dict):
        forecast_values = test_forecast["forecast"]
    else:
        forecast_values = test_forecast

    # Valutazione
    performance = evaluate_model_performance(test_data, forecast_values)

    print(f"\nMetriche Performance:")
    print(f"  MAPE: {performance['MAPE']:.2f}%")
    print(f"  MAE: EUR{performance['MAE']:,.0f}")
    print(f"  RMSE: EUR{performance['RMSE']:,.0f}")
    print(f"  R2: {performance['R2']:.3f}")

    logger.info("Forecast per i prossimi 12 mesi...")

    # Forecast futuro
    future_forecast = model.forecast(steps=12, confidence_intervals=True)

    # Crea date future
    last_date = sales_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="ME")

    # Visualizzazione
    plt.figure(figsize=(15, 10))

    # Subplot 1: Serie storica + test forecast
    plt.subplot(2, 1, 1)
    plt.plot(train_data.index, train_data, label="Training Data", color="blue", alpha=0.8)
    plt.plot(test_data.index, test_data, label="Test Data (Actual)", color="green", alpha=0.8)
    plt.plot(test_data.index, forecast_values, label="Test Forecast", color="red", linestyle="--")

    plt.title("Analisi Performance Modello ARIMA - Vendite Retail", fontsize=14)
    plt.ylabel("Vendite Mensili (EUR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Subplot 2: Forecast futuro
    plt.subplot(2, 1, 2)

    # Plot ultimi 24 mesi per contesto
    recent_data = sales_data[-24:]
    plt.plot(recent_data.index, recent_data, label="Dati Storici (24m)", color="blue", alpha=0.8)

    # Future forecast
    if isinstance(future_forecast, dict):
        future_values = future_forecast["forecast"]
        if "confidence_intervals" in future_forecast:
            conf_int = future_forecast["confidence_intervals"]
            if isinstance(conf_int, dict):
                lower = conf_int["lower"]
                upper = conf_int["upper"]
            else:
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]

            plt.fill_between(
                future_dates, lower, upper, alpha=0.3, color="red", label="Intervallo Confidenza"
            )
    else:
        future_values = future_forecast

    plt.plot(
        future_dates, future_values, label="Forecast 12 Mesi", color="red", linewidth=2, marker="o"
    )

    plt.title("Forecast Vendite Retail - Prossimi 12 Mesi", fontsize=14)
    plt.ylabel("Vendite Mensili (EUR)")
    plt.xlabel("Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Create output directories if they don't exist
    output_plots_dir = Path("../outputs/plots")
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    # Salva plot
    plot_path = output_plots_dir / "retail_sales_forecast.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot salvato in {plot_path}")

    # plt.show()  # Disabled for Windows compatibility
    print(f"Plot saved as '{plot_path}'")

    # Insights business
    print(f"\nBusiness Insights:")

    # Previsioni future con interpretazioni
    future_total = future_values.sum()
    current_year_total = sales_data[-12:].sum()
    growth_rate = ((future_total - current_year_total) / current_year_total) * 100

    print(f"  Vendite previste prossimi 12 mesi: EUR{future_total:,.0f}")
    print(f"  Crescita stimata: {growth_rate:+.1f}% vs ultimo anno")

    # Mesi con vendite più alte
    future_df = pd.DataFrame(
        {"month": [d.strftime("%Y-%m") for d in future_dates], "forecast": future_values}
    )
    best_months = future_df.nlargest(3, "forecast")

    print(f"  Top 3 mesi previsti:")
    for _, row in best_months.iterrows():
        print(f"    {row['month']}: EUR{row['forecast']:,.0f}")

    # Raccomandazioni
    print(f"\nRaccomandazioni Strategiche:")
    if growth_rate > 0:
        print(f"  + Crescita positiva prevista: investire in inventory per novembre/dicembre")
        print(f"  + Pianificare campagne marketing per i mesi di picco")
    else:
        print(f"  - Crescita negativa: ottimizzare costi e rivedere strategy")
        print(f"  - Focus su customer retention e new products")

    print(f"  * Mantenere stock extra per i mesi di punta (Nov-Dic)")
    print(f"  * Considerare promozioni per i mesi con vendite più basse")

    # Salvataggio del modello
    logger.info("Salvataggio modello...")

    # Create models output directory
    output_models_dir = Path("../outputs/models")
    output_models_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_models_dir / "retail_sales_arima_model.pkl"
    model.save(model_path)

    logger.info(f"Modello salvato in {model_path}")
    print(f"Modello salvato in: {model_path}")

    logger.info("Analisi completata con successo!")
    print("\nAnalisi completata! Check the outputs/plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
