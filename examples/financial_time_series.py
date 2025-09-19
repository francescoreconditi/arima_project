#!/usr/bin/env python3
"""
Esempio di Forecasting Serie Temporali Finanziarie

Questo esempio dimostra l'utilizzo di ARIMA per serie temporali finanziarie giornaliere.
Include volatilità clustering, trend e pattern tipici dei mercati finanziari come
returns, volatilità e regime changes.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.core import ARIMAModelSelector
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils import setup_logger
from utils import get_plots_path, get_models_path, get_reports_path

warnings.filterwarnings("ignore")


def generate_financial_data():
    """Genera dati finanziari sintetici con caratteristiche realistiche"""
    np.random.seed(42)

    # Periodo: 2 anni di dati giornalieri (escludendo weekend)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.bdate_range(start_date, end_date)  # Solo giorni lavorativi

    n_days = len(dates)

    # Parametri modello
    initial_price = 100.0
    drift = 0.0002  # Piccola drift positiva giornaliera

    # GARCH-like volatility clustering
    volatility = np.zeros(n_days)
    volatility[0] = 0.02

    for i in range(1, n_days):
        # GARCH(1,1) semplificato
        volatility[i] = np.sqrt(
            0.000005 + 0.05 * (volatility[i - 1] ** 2) + 0.90 * volatility[i - 1] ** 2
        )

    # Shock di volatilità in periodi specifici (eventi di mercato)
    shock_periods = [
        (datetime(2022, 3, 1), datetime(2022, 3, 15)),  # Geopolitical shock
        (datetime(2022, 9, 15), datetime(2022, 10, 1)),  # Market correction
        (datetime(2023, 3, 10), datetime(2023, 3, 20)),  # Banking sector stress
    ]

    for start_shock, end_shock in shock_periods:
        mask = (dates >= start_shock) & (dates <= end_shock)
        volatility[mask] *= np.random.uniform(2.0, 3.0, sum(mask))

    # Genera returns con fat tails (distribuzione t)
    returns = np.random.standard_t(df=5, size=n_days) * volatility + drift

    # Aggiunge autocorrelazione ai returns per simulare momentum/mean reversion
    for i in range(1, n_days):
        returns[i] += 0.05 * returns[i - 1]  # Piccolo momentum

    # Calcola prezzi
    log_prices = np.cumsum(returns) + np.log(initial_price)
    prices = np.exp(log_prices)

    # Aggiunge trend deterministico lieve
    trend = 1 + np.linspace(0, 0.1, n_days)  # 10% in 2 anni
    prices *= trend

    return pd.Series(prices, index=dates, name="stock_price")


def calculate_financial_metrics(prices):
    """Calcola metriche finanziarie standard"""
    returns = prices.pct_change().dropna()

    metrics = {
        "annual_return": returns.mean() * 252,
        "annual_volatility": returns.std() * np.sqrt(252),
        "sharpe_ratio": (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        "max_drawdown": ((prices / prices.cummax()) - 1).min(),
        "var_95": returns.quantile(0.05),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }

    return metrics, returns


def main():
    logger = setup_logger("financial_forecasting", level="INFO")
    logger.info("📈 Avvio analisi forecasting serie temporali finanziarie")

    # Genera dati
    logger.info("💰 Generazione dati finanziari...")
    price_data = generate_financial_data()

    # Calcola metriche finanziarie
    financial_metrics, returns = calculate_financial_metrics(price_data)

    print(f"📊 Dataset generato: {len(price_data)} giorni lavorativi")
    print(
        f"📅 Periodo: {price_data.index[0].strftime('%Y-%m-%d')} - {price_data.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"💵 Prezzo iniziale: ${price_data.iloc[0]:.2f}")
    print(f"💵 Prezzo finale: ${price_data.iloc[-1]:.2f}")
    print(f"📈 Return totale: {((price_data.iloc[-1] / price_data.iloc[0]) - 1) * 100:.2f}%")

    print(f"\n📊 Metriche Finanziarie:")
    print(f"  📈 Return annualizzato: {financial_metrics['annual_return'] * 100:.2f}%")
    print(f"  📉 Volatilità annualizzata: {financial_metrics['annual_volatility'] * 100:.2f}%")
    print(f"  ⭐ Sharpe Ratio: {financial_metrics['sharpe_ratio']:.2f}")
    print(f"  📉 Max Drawdown: {financial_metrics['max_drawdown'] * 100:.2f}%")
    print(f"  📊 VaR 95%: {financial_metrics['var_95'] * 100:.2f}%")

    # Split train/test
    train_size = int(len(price_data) * 0.8)
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]

    print(f"\n🔄 Split dataset:")
    print(
        f"  📚 Training: {len(train_data)} giorni ({train_data.index[0].strftime('%Y-%m-%d')} - {train_data.index[-1].strftime('%Y-%m-%d')})"
    )
    print(
        f"  🧪 Test: {len(test_data)} giorni ({test_data.index[0].strftime('%Y-%m-%d')} - {test_data.index[-1].strftime('%Y-%m-%d')})"
    )

    # Preprocessing con log transformation per stabilizzare varianza
    logger.info("🔧 Preprocessing dati finanziari...")
    preprocessor = TimeSeriesPreprocessor()

    # Log transformation per prezzi
    log_train = np.log(train_data)
    log_test = np.log(test_data)

    # Check stazionarietà su log prices
    stationarity_result = preprocessor.check_stationarity(log_train)
    is_stationary = stationarity_result["is_stationary"]
    if not is_stationary:
        print("📈 Log-prices non stazionarie - il modello userà differenziazione")

    # Analisi returns per confronto
    train_returns = train_data.pct_change().dropna()
    returns_stationarity = preprocessor.check_stationarity(train_returns)
    is_returns_stationary = returns_stationarity["is_stationary"]
    print(f"📊 Returns stazionari: {is_returns_stationary}")

    # Selezione automatica modello su log-prices
    logger.info("🔍 Selezione automatica modello ARIMA su log-prices...")
    # Usa modello ARIMA semplice per dati finanziari
    print("Utilizzo modello ARIMA(2,1,2) per dati finanziari...")
    best_order = (2, 1, 2)
    seasonal_order = None

    print(f"\n✅ Modello ottimale trovato:")
    print(f"  📊 ARIMA{best_order}")

    # Training modello
    logger.info("🎯 Training modello ARIMA...")
    model = ARIMAForecaster(order=best_order)
    model.fit(log_train)

    # Forecast su log-scale
    forecast_steps = len(test_data)
    logger.info(f"🔮 Generazione forecast per {forecast_steps} giorni...")
    log_forecast_result = model.forecast(
        steps=forecast_steps, confidence_intervals=True, alpha=0.05
    )

    # Converti di nuovo a livello prezzi
    forecast_prices = np.exp(log_forecast_result["forecast"])
    forecast_lower = np.exp(log_forecast_result["confidence_intervals"]["lower"])
    forecast_upper = np.exp(log_forecast_result["confidence_intervals"]["upper"])

    forecast_result = {
        "forecast": pd.Series(forecast_prices, index=test_data.index),
        "confidence_intervals": {
            "lower": pd.Series(forecast_lower, index=test_data.index),
            "upper": pd.Series(forecast_upper, index=test_data.index),
        },
    }

    # Valutazione
    logger.info("📊 Valutazione performance modello...")
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_forecast_metrics(test_data, forecast_result["forecast"])

    print(f"\n📊 Metriche Performance (Prezzi):")
    print(f"  📈 MAPE: {metrics['mape']:.2f}%")
    print(f"  📉 MAE: ${metrics['mae']:.2f}")
    print(f"  🎯 RMSE: ${metrics['rmse']:.2f}")

    # Controlla punteggio R² con diversi nomi chiave possibili
    if "r2_score" in metrics:
        print(f"  📊 R²: {metrics['r2_score']:.3f}")
    elif "r_squared" in metrics:
        print(f"  📊 R²: {metrics['r_squared']:.3f}")
    else:
        print(f"  📊 R²: N/A")

    # Valuta anche accuracy direzionale
    actual_direction = np.sign(test_data.pct_change().dropna())
    forecast_direction = np.sign(forecast_result["forecast"].pct_change().dropna())
    directional_accuracy = (actual_direction == forecast_direction).mean()
    print(f"  🧭 Directional Accuracy: {directional_accuracy:.1%}")

    # Forecast futuro
    logger.info("🚀 Forecast per i prossimi 30 giorni lavorativi...")
    future_log_forecast = model.forecast(steps=30, confidence_intervals=True)
    future_prices = np.exp(future_log_forecast["forecast"])
    future_lower = np.exp(future_log_forecast["confidence_intervals"]["lower"])
    future_upper = np.exp(future_log_forecast["confidence_intervals"]["upper"])

    # Visualizzazione
    plt.figure(figsize=(18, 12))

    # Subplot 1: Prezzi completi
    plt.subplot(2, 3, 1)
    plt.plot(train_data.index, train_data, label="Training Data", color="blue", alpha=0.7)
    plt.plot(test_data.index, test_data, label="Test Data (Actual)", color="green", alpha=0.8)
    plt.plot(
        test_data.index,
        forecast_result["forecast"],
        label="Forecast",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.fill_between(
        test_data.index,
        forecast_result["confidence_intervals"]["lower"],
        forecast_result["confidence_intervals"]["upper"],
        color="red",
        alpha=0.2,
        label="95% CI",
    )

    plt.title("📊 Serie Temporali Finanziarie - Previsione Prezzi")
    plt.ylabel("Prezzo Azione ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Subplot 2: Dettaglio periodo test
    plt.subplot(2, 3, 2)
    plt.plot(test_data.index, test_data, "o-", label="Actual", color="green", markersize=3)
    plt.plot(
        test_data.index,
        forecast_result["forecast"],
        "s-",
        label="Forecast",
        color="red",
        markersize=2,
    )

    plt.fill_between(
        test_data.index,
        forecast_result["confidence_intervals"]["lower"],
        forecast_result["confidence_intervals"]["upper"],
        color="red",
        alpha=0.2,
        label="95% CI",
    )

    plt.title("🎯 Test Period Performance")
    plt.ylabel("Prezzo Azione ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Subplot 3: Returns comparison
    plt.subplot(2, 3, 3)
    actual_returns = test_data.pct_change().dropna()
    forecast_returns = forecast_result["forecast"].pct_change().dropna()

    plt.plot(
        actual_returns.index,
        actual_returns,
        "o-",
        label="Actual Returns",
        color="green",
        alpha=0.7,
        markersize=2,
    )
    plt.plot(
        forecast_returns.index,
        forecast_returns,
        "s-",
        label="Forecast Returns",
        color="red",
        alpha=0.7,
        markersize=2,
    )

    plt.title("📈 Confronto Returns")
    plt.ylabel("Returns Giornalieri")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Subplot 4: Volatility analysis
    plt.subplot(2, 3, 4)
    actual_vol = actual_returns.rolling(20).std()
    forecast_vol = forecast_returns.rolling(20).std()

    plt.plot(actual_vol.index, actual_vol, label="Actual Volatility (20d)", color="green")
    plt.plot(forecast_vol.index, forecast_vol, label="Forecast Volatility (20d)", color="red")

    plt.title("📊 Volatilità Mobile (20 giorni)")
    plt.ylabel("Volatilità")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Subplot 5: Future forecast
    plt.subplot(2, 3, 5)

    # Ultime 60 osservazioni per contesto
    recent_data = price_data[-60:]
    plt.plot(recent_data.index, recent_data, label="Recent Data", color="blue", alpha=0.7)

    # Future forecast dates
    future_dates = pd.bdate_range(start=price_data.index[-1] + pd.Timedelta(days=1), periods=30)

    plt.plot(
        future_dates,
        future_prices,
        "s-",
        label="Future Forecast",
        color="purple",
        linewidth=2,
        markersize=4,
    )

    plt.fill_between(
        future_dates, future_lower, future_upper, color="purple", alpha=0.2, label="95% CI"
    )

    plt.title("🚀 Future 30-Day Forecast")
    plt.ylabel("Prezzo Azione ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Subplot 6: Residuals analysis
    plt.subplot(2, 3, 6)
    residuals = test_data - forecast_result["forecast"]
    plt.plot(test_data.index, residuals, "o-", color="orange", alpha=0.7, markersize=3)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    plt.axhline(y=residuals.std(), color="red", linestyle="--", alpha=0.5, label="+1σ")
    plt.axhline(y=-residuals.std(), color="red", linestyle="--", alpha=0.5, label="-1σ")

    plt.title("📉 Forecast Residuals")
    plt.ylabel("Residui ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Salva plot
    plot_path = get_plots_path("financial_forecast.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("📁 Plot salvato in outputs/plots/financial_forecast.png")

    # plt.show()  # Disabled for Windows compatibility
    print("Plot salvato come 'outputs/plots/financial_forecast.png'")

    # Financial insights
    print(f"\n💼 Financial Insights:")

    # Prezzo target e risk metrics
    future_final_price = future_prices.iloc[-1]
    current_price = price_data.iloc[-1]
    expected_return_30d = (future_final_price - current_price) / current_price

    print(f"  💵 Prezzo attuale: ${current_price:.2f}")
    print(f"  🎯 Prezzo previsto (30gg): ${future_final_price:.2f}")
    print(f"  📈 Return atteso 30gg: {expected_return_30d * 100:+.2f}%")
    print(
        f"  📊 Confidence interval 30gg: ${future_lower.iloc[-1]:.2f} - ${future_upper.iloc[-1]:.2f}"
    )

    # Risk metrics
    forecast_volatility = forecast_returns.std() * np.sqrt(252)
    print(f"  📉 Volatilità prevista (annualizzata): {forecast_volatility * 100:.1f}%")

    # Trading signals (semplificati)
    short_ma = future_prices.rolling(5).mean().iloc[-1]
    if future_final_price > short_ma * 1.02:
        signal = "🟢 BULLISH"
    elif future_final_price < short_ma * 0.98:
        signal = "🔴 BEARISH"
    else:
        signal = "🟡 NEUTRAL"

    print(f"  📊 Signal tecnico: {signal}")

    # Salva modello
    model_path = get_models_path("financial_arima_model.joblib")
    model.save(model_path)
    logger.info(f"💾 Modello salvato in {model_path}")

    # Genera report Quarto
    logger.info("Generazione report Quarto...")
    try:
        # Passa il percorso dell'immagine salvata
        plot_files = {
            "main_plot": str(plot_path)  # plot_path è definito sopra quando salviamo l'immagine
        }

        report_path = model.generate_report(
            plots_data=plot_files,
            report_title="Analisi Serie Temporali Finanziarie",
            output_filename="report_serie_temporali_finanziarie",
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

    print(f"\n✅ Analisi finanziaria completata!")
    print(f"📁 Risultati salvati in outputs/")
    print(f"⚠️  Disclaimer: Questo è un modello dimostrativo per scopi educativi.")
    print(f"    Non utilizzare per decisioni di investimento reali.")


if __name__ == "__main__":
    main()
