#!/usr/bin/env python3
"""
IoT Sensor Forecasting Example

Questo esempio dimostra l'applicazione di ARIMA per il forecasting di dati sensori IoT
industriali. Include pattern tipici come cicli operativi, derive del sensore,
anomalie e multi-variabilità tipica degli ambienti IoT.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for Windows
import warnings
from datetime import datetime, timedelta

from utils import get_models_path, get_plots_path

from arima_forecaster import ARIMAForecaster, ForecastPlotter, TimeSeriesPreprocessor
from arima_forecaster.core import ARIMAModelSelector
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils import setup_logger

warnings.filterwarnings("ignore")


def generate_iot_sensor_data():
    """Genera dati sensori IoT sintetici con pattern realistici"""
    np.random.seed(42)

    # Periodo: 30 giorni di dati ogni 15 minuti (2880 punti)
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=30)
    timestamps = pd.date_range(start_date, end_date, freq="15min")[:-1]
    n_points = len(timestamps)

    # Sensor 1: Temperatura - con cicli giornalieri e derive
    temp_base = 25.0  # Temperatura base

    # Ciclo giornaliero (24h = 96 punti per giorno)
    daily_cycle = 8 * np.sin(2 * np.pi * np.arange(n_points) / 96) + 2 * np.sin(
        4 * np.pi * np.arange(n_points) / 96
    )  # Armoniche

    # Ciclo settimanale (più blando)
    weekly_cycle = 2 * np.sin(2 * np.pi * np.arange(n_points) / (96 * 7))

    # Deriva del sensore (degradazione)
    sensor_drift = 0.1 * np.arange(n_points) / n_points

    # Operazioni industriali (maggiore calore durante ore lavorative)
    working_hours_mask = np.array([8 <= (t.hour) <= 18 and t.weekday() < 5 for t in timestamps])
    operational_heat = working_hours_mask * np.random.normal(3, 1, n_points)

    # Eventi anomali (surriscaldamento sporadico)
    anomaly_events = np.zeros(n_points)
    anomaly_indices = np.random.choice(n_points, size=10, replace=False)
    for idx in anomaly_indices:
        duration = np.random.randint(4, 16)  # 1-4 ore
        end_idx = min(idx + duration, n_points)
        anomaly_events[idx:end_idx] = np.random.uniform(8, 15)

    # Rumore del sensore
    noise = np.random.normal(0, 0.5, n_points)

    # Temperatura finale
    temperature = (
        temp_base
        + daily_cycle
        + weekly_cycle
        + sensor_drift
        + operational_heat
        + anomaly_events
        + noise
    )

    # Sensor 2: Pressione - correlata con temperatura ma con propri pattern
    pressure_base = 1013.25  # hPa

    # Correlazione con temperatura (coefficiente termico)
    temp_correlation = 0.1 * (temperature - temp_base)

    # Pattern atmosferici (più lenti)
    atmospheric_pattern = 5 * np.sin(2 * np.pi * np.arange(n_points) / (96 * 3)) + 3 * np.sin(
        2 * np.pi * np.arange(n_points) / (96 * 1.5)
    )

    # Eventi di pressure drop (manutenzione sistema)
    maintenance_events = np.zeros(n_points)
    maintenance_indices = np.random.choice(n_points, size=5, replace=False)
    for idx in maintenance_indices:
        duration = np.random.randint(8, 24)  # 2-6 ore
        end_idx = min(idx + duration, n_points)
        maintenance_events[idx:end_idx] = -np.random.uniform(10, 25)

    pressure_noise = np.random.normal(0, 1.0, n_points)

    pressure = (
        pressure_base + temp_correlation + atmospheric_pattern + maintenance_events + pressure_noise
    )

    # Sensor 3: Vibrazione - con pattern ad alta frequenza
    vibration_base = 0.5  # mm/s RMS baseline

    # Vibrazione operazionale (correlata con ore lavorative)
    operational_vibration = working_hours_mask * np.random.uniform(0.3, 0.8, n_points)

    # Pattern di usura (aumenta nel tempo)
    wear_pattern = 0.2 * np.arange(n_points) / n_points * np.random.uniform(0.8, 1.2, n_points)

    # Spike di vibrazione (problemi meccanici)
    vibration_spikes = np.zeros(n_points)
    spike_indices = np.random.choice(n_points, size=15, replace=False)
    for idx in spike_indices:
        vibration_spikes[idx] = np.random.uniform(2, 5)

    vibration_noise = np.random.gamma(2, 0.1, n_points)  # Gamma distribution per positive skew

    vibration = (
        vibration_base + operational_vibration + wear_pattern + vibration_spikes + vibration_noise
    )

    # Crea DataFrame multi-sensor
    sensor_data = pd.DataFrame(
        {"temperature_c": temperature, "pressure_hpa": pressure, "vibration_rms": vibration},
        index=timestamps,
    )

    return sensor_data


def detect_anomalies(series, threshold=3):
    """Semplice rilevamento anomalie basato su Z-score"""
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def main():
    logger = setup_logger("iot_forecasting", level="INFO")
    logger.info("🔧 Avvio analisi forecasting sensori IoT")

    # Genera dati
    logger.info("📟 Generazione dati sensori IoT...")
    sensor_data = generate_iot_sensor_data()

    print(f"Dataset generato: {len(sensor_data)} punti dati (ogni 15 min)")
    print(
        f"Periodo: {sensor_data.index[0].strftime('%Y-%m-%d %H:%M')} - {sensor_data.index[-1].strftime('%Y-%m-%d %H:%M')}"
    )
    print(
        f"Temperatura: {sensor_data['temperature_c'].mean():.1f}°C ± {sensor_data['temperature_c'].std():.1f}°C"
    )
    print(
        f"Pressione: {sensor_data['pressure_hpa'].mean():.1f} hPa ± {sensor_data['pressure_hpa'].std():.1f} hPa"
    )
    print(
        f"Vibrazione: {sensor_data['vibration_rms'].mean():.2f} mm/s ± {sensor_data['vibration_rms'].std():.2f} mm/s"
    )

    # Focus su temperatura per questo esempio
    temp_series = sensor_data["temperature_c"]

    # Rilevamento anomalie
    temp_anomalies = detect_anomalies(temp_series)
    print(f"Anomalie rilevate: {temp_anomalies.sum()} punti ({temp_anomalies.mean() * 100:.1f}%)")

    # Split train/test
    train_size = int(len(temp_series) * 0.8)
    train_data = temp_series[:train_size]
    test_data = temp_series[train_size:]

    print(f"\nSplit dataset (Temperatura):")
    print(
        f"  Training: {len(train_data)} punti ({train_data.index[0].strftime('%Y-%m-%d %H:%M')} - {train_data.index[-1].strftime('%Y-%m-%d %H:%M')})"
    )
    print(
        f"  Test: {len(test_data)} punti ({test_data.index[0].strftime('%Y-%m-%d %H:%M')} - {test_data.index[-1].strftime('%Y-%m-%d %H:%M')})"
    )

    # Preprocessing per IoT data
    logger.info("🔧 Preprocessing dati IoT...")
    preprocessor = TimeSeriesPreprocessor()

    # Rimuovi outlier per training più stabile
    train_clean = train_data.copy()
    outlier_mask = detect_anomalies(train_clean, threshold=3)
    print(f"Rimozione {outlier_mask.sum()} outlier per training")

    # Interpola outlier invece di rimuoverli per mantenere frequenza temporale
    train_clean.loc[outlier_mask] = np.nan
    train_clean = train_clean.interpolate(method="linear")

    # Check stazionarietà
    stationarity_result = preprocessor.check_stationarity(train_clean)
    is_stationary = stationarity_result["is_stationary"]
    if not is_stationary:
        print("Serie non stazionaria - il modello userà differenziazione")
        print(f"ADF p-value: {stationarity_result['p_value']:.4f}")

    # Per dati IoT complessi, usiamo un modello semplice che funziona bene
    print("Utilizzo modello ARIMA(2,1,2) ottimizzato per dati IoT...")

    # Training modello
    logger.info("Training modello ARIMA per sensori IoT...")
    model = ARIMAForecaster(order=(2, 1, 2))
    model.fit(train_clean)

    # Forecast
    forecast_steps = len(test_data)
    logger.info(
        f"🔮 Generazione forecast per {forecast_steps} punti ({forecast_steps / 4:.1f} ore)..."
    )
    forecast_result = model.forecast(steps=forecast_steps, confidence_intervals=True, alpha=0.05)

    # Valutazione
    logger.info("📊 Valutazione performance modello...")
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_forecast_metrics(test_data, forecast_result["forecast"])

    print(f"\nMetriche Performance (Temperatura):")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE: {metrics['mae']:.2f}°C")
    print(f"  RMSE: {metrics['rmse']:.2f}°C")

    # Check if r2_score exists
    if "r2_score" in metrics:
        print(f"  R²: {metrics['r2_score']:.3f}")
    elif "r_squared" in metrics:
        print(f"  R²: {metrics['r_squared']:.3f}")

    # IoT-specific metrics
    temp_range = test_data.max() - test_data.min()
    normalized_rmse = metrics["rmse"] / temp_range
    print(f"  Normalized RMSE: {normalized_rmse:.1%} (del range)")

    # Forecast futuro operazionale (prossime 24 ore)
    future_steps = 96  # 24 ore * 4 punti/ora
    logger.info(f"🚀 Forecast operazionale prossime 24 ore...")
    future_forecast = model.forecast(steps=future_steps, confidence_intervals=True)

    # Visualizzazione specializzata per IoT
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # Plot 1: Overview completo multi-sensore
    ax1 = axes[0, 0]
    ax1.plot(
        sensor_data.index,
        sensor_data["temperature_c"],
        label="Temperature",
        color="red",
        alpha=0.7,
        linewidth=0.8,
    )
    ax1.set_ylabel("Temperature (°C)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_title("🌡️ Multi-Sensor IoT Data Overview")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        sensor_data.index,
        sensor_data["pressure_hpa"],
        label="Pressure",
        color="blue",
        alpha=0.6,
        linewidth=0.8,
    )
    ax1_twin.set_ylabel("Pressure (hPa)", color="blue")
    ax1_twin.tick_params(axis="y", labelcolor="blue")

    ax1.grid(True, alpha=0.3)

    # Plot 2: Temperature forecast
    ax2 = axes[0, 1]
    ax2.plot(train_data.index, train_data, label="Training", color="blue", alpha=0.6)
    ax2.plot(test_data.index, test_data, label="Test Actual", color="green", alpha=0.8)
    ax2.plot(
        test_data.index,
        forecast_result["forecast"],
        label="Forecast",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    ax2.fill_between(
        test_data.index,
        forecast_result["confidence_intervals"]["lower"],
        forecast_result["confidence_intervals"]["upper"],
        color="red",
        alpha=0.2,
        label="95% CI",
    )

    ax2.set_title("📊 Temperature Forecast Results")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Vibration analysis
    ax3 = axes[1, 0]
    vibration_data = sensor_data["vibration_rms"]
    ax3.plot(vibration_data.index, vibration_data, color="orange", alpha=0.7, linewidth=0.6)

    # Evidenzia anomalie vibrazione
    vib_anomalies = detect_anomalies(vibration_data, threshold=3)
    ax3.scatter(
        vibration_data.index[vib_anomalies],
        vibration_data[vib_anomalies],
        color="red",
        s=20,
        alpha=0.8,
        label=f"Anomalies ({vib_anomalies.sum()})",
    )

    ax3.set_title("📳 Vibration Analysis with Anomalies")
    ax3.set_ylabel("Vibration RMS (mm/s)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Daily pattern analysis (temperature)
    ax4 = axes[1, 1]
    # Raggruppa per ora del giorno
    hourly_pattern = train_data.groupby(train_data.index.hour).mean()
    hourly_std = train_data.groupby(train_data.index.hour).std()

    ax4.plot(
        hourly_pattern.index,
        hourly_pattern.values,
        "o-",
        color="blue",
        linewidth=2,
        label="Mean Temperature",
    )
    ax4.fill_between(
        hourly_pattern.index,
        hourly_pattern.values - hourly_std.values,
        hourly_pattern.values + hourly_std.values,
        color="blue",
        alpha=0.2,
        label="±1σ",
    )

    ax4.set_title("🕐 Daily Temperature Pattern")
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Temperature (°C)")
    ax4.set_xlim(0, 23)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Future operational forecast
    ax5 = axes[2, 0]

    # Ultime 48 ore per contesto
    recent_data = temp_series[-192:]  # 48h * 4 punti/ora
    ax5.plot(recent_data.index, recent_data, label="Recent Data", color="blue", alpha=0.7)

    # Future forecast
    future_dates = pd.date_range(
        start=temp_series.index[-1] + pd.Timedelta(minutes=15), periods=future_steps, freq="15min"
    )

    ax5.plot(
        future_dates,
        future_forecast["forecast"],
        "s-",
        label="24h Forecast",
        color="purple",
        linewidth=2,
        markersize=3,
        alpha=0.8,
    )

    ax5.fill_between(
        future_dates,
        future_forecast["confidence_intervals"]["lower"],
        future_forecast["confidence_intervals"]["upper"],
        color="purple",
        alpha=0.2,
        label="95% CI",
    )

    ax5.set_title("🚀 Next 24 Hours Operational Forecast")
    ax5.set_ylabel("Temperature (°C)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # Plot 6: Residuals e diagnostica
    ax6 = axes[2, 1]
    residuals = test_data - forecast_result["forecast"]

    # Time series residuals
    ax6.plot(test_data.index, residuals, "o-", color="orange", alpha=0.7, markersize=2)
    ax6.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax6.axhline(y=residuals.std(), color="red", linestyle="--", alpha=0.5, label="+1σ")
    ax6.axhline(y=-residuals.std(), color="red", linestyle="--", alpha=0.5, label="-1σ")

    # Evidenzia periodi con residui elevati
    high_residuals = np.abs(residuals) > 2 * residuals.std()
    ax6.scatter(
        test_data.index[high_residuals],
        residuals[high_residuals],
        color="red",
        s=30,
        alpha=0.8,
        label=f"High Residuals ({high_residuals.sum()})",
    )

    ax6.set_title("📉 Forecast Residuals Analysis")
    ax6.set_ylabel("Residuals (°C)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # Salva plot
    plot_path = get_plots_path("iot_sensor_forecast.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"📁 Plot salvato in {plot_path}")

    # plt.show()  # Disabled for Windows compatibility
    print("Plot salvato come 'outputs/plots/iot_sensor_forecast.png'")

    # IoT Operational Insights
    print(f"\nIoT Operational Insights:")

    # Alert system simulation
    future_temps = future_forecast["forecast"]
    temp_threshold_high = 35.0
    temp_threshold_low = 15.0

    high_temp_alerts = future_temps > temp_threshold_high
    low_temp_alerts = future_temps < temp_threshold_low

    if high_temp_alerts.any():
        alert_times = future_dates[high_temp_alerts]
        print(f"  HIGH TEMP ALERT: {len(alert_times)} forecast points > {temp_threshold_high}°C")
        print(f"    First alert at: {alert_times[0].strftime('%Y-%m-%d %H:%M')}")

    if low_temp_alerts.any():
        alert_times = future_dates[low_temp_alerts]
        print(f"  LOW TEMP ALERT: {len(alert_times)} forecast points < {temp_threshold_low}°C")

    if not (high_temp_alerts.any() or low_temp_alerts.any()):
        print(f"  No temperature alerts per prossime 24 ore")

    # Maintenance prediction
    vibration_trend = sensor_data["vibration_rms"].rolling(96).mean().iloc[-1]  # 24h average
    vibration_threshold = 2.0

    if vibration_trend > vibration_threshold:
        print(
            f"  MAINTENANCE ALERT: Vibrazione trend {vibration_trend:.2f} > {vibration_threshold} mm/s"
        )
        print(f"    Raccomandazione: Ispezione programmata consigliata")
    else:
        print(f"  Vibrazione nominale: {vibration_trend:.2f} mm/s")

    # Efficiency metrics - calculate working hours from test_data timestamps
    test_working_hours = sum(1 for t in test_data.index if 8 <= t.hour <= 18 and t.weekday() < 5)
    operational_hours = test_working_hours / 4  # Convert to hours (15-min intervals)
    print(f"  Ore operative nel test period: {operational_hours:.1f}h")

    # Check if r2_score exists in metrics
    if "r2_score" in metrics:
        print(f"  Accuratezza durante ore operative: {metrics['r2_score']:.1%}")
    elif "r_squared" in metrics:
        print(f"  Accuratezza durante ore operative: {metrics['r_squared']:.1%}")
    else:
        print(f"  Accuratezza generale: MAPE {metrics['mape']:.1f}%")

    # Sensor drift analysis
    initial_temp = train_data.iloc[:96].mean()  # First day
    final_temp = train_data.iloc[-96:].mean()  # Last day
    drift_rate = (final_temp - initial_temp) / (len(train_data) / (96 * 7))  # Per week

    print(f"  Deriva sensore stimata: {drift_rate:+.3f}°C/settimana")

    if abs(drift_rate) > 0.1:
        print(f"  Raccomandazione: Calibrazione sensore temperatura")

    # Salva modello
    model_path = get_models_path("iot_sensor_arima_model.joblib")
    model.save(model_path)
    logger.info(f"💾 Modello salvato in {model_path}")

    # Genera report Quarto
    logger.info("Generazione report Quarto...")
    try:
        # Get the plot filename if it exists
        plot_files = {}
        # Try to find the most recent plot file
        if "plot_path" in locals():
            plot_files["main_plot"] = str(plot_path)
        elif "plt" in locals():
            # If we have a matplotlib figure, save it temporarily
            temp_plot = get_plots_path("temp_report_plot.png")
            plt.savefig(temp_plot, dpi=300, bbox_inches="tight")
            plot_files["analysis_plot"] = str(temp_plot)

        report_path = model.generate_report(
            plots_data=plot_files if plot_files else None,
            report_title="Iot Sensor Forecasting Analysis",
            output_filename="iot_sensor_forecasting_report",
            format_type="html",
            include_diagnostics=True,
            include_forecast=True,
            forecast_steps=12,
        )
        logger.info(f"Report HTML generato: {report_path}")
        print(f"Report HTML salvato in: {report_path}")
    except Exception as e:
        logger.warning(f"Impossibile generare report: {e}")
        print(f"Report non generato: {e}")

    print(f"\nAnalisi IoT completata!")
    print(f"Risultati e grafici salvati in outputs/")
    print(f"Modello pronto per deployment in sistemi IoT industriali")


if __name__ == "__main__":
    main()
