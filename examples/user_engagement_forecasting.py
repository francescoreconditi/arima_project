#!/usr/bin/env python3
"""
Esempio di Forecasting Engagement Utenti

Questo esempio dimostra l'applicazione di ARIMA per il forecasting delle metriche
di engagement utenti. Include DAU/MAU, session duration, retention rates,
feature adoption, e pattern comportamentali tipici delle applicazioni digitali.
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


def generate_user_engagement_data():
    """Genera dati engagement utenti con pattern realistici"""
    np.random.seed(42)

    # Periodo: 1 anno di dati giornalieri (365 giorni)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    # User base iniziale e crescita
    initial_users = 10000
    monthly_growth_rate = 0.08  # 8% monthly growth

    # 1. Daily Active Users (DAU)
    base_dau_rate = 0.25  # 25% of total users active daily

    # Crescita user base nel tempo
    total_users = initial_users * (1 + monthly_growth_rate) ** (np.arange(n_days) / 30)

    # Pattern settimanali DAU (più alto durante weekdays)
    weekly_dau_pattern = np.array(
        [
            1.1 if date.weekday() < 5 else 0.7  # Weekday vs Weekend
            for date in dates
        ]
    )

    # Pattern mensili (variazioni stagionali)
    monthly_dau_pattern = np.array(
        [
            1.2
            if date.month in [1, 9]
            # New Year, Back to school
            else 0.8
            if date.month in [7, 8, 12]
            # Summer holidays, Christmas
            else 1.0  # Normal months
            for date in dates
        ]
    )

    # Feature releases impact (boost engagement)
    feature_releases = np.ones(n_days)
    release_dates = [30, 95, 150, 210, 280, 330]  # Major releases
    for release_day in release_dates:
        if release_day < n_days:
            # Gradual adoption curve
            adoption_period = 30
            end_period = min(release_day + adoption_period, n_days)
            boost_curve = np.linspace(1.0, 1.4, end_period - release_day)
            boost_curve = boost_curve * np.exp(
                -np.linspace(0, 2, end_period - release_day)
            )  # Decay
            feature_releases[release_day:end_period] *= boost_curve

    # Marketing campaigns (periodic boosts)
    marketing_campaigns = np.ones(n_days)
    # Monthly campaigns (first Monday of each month)
    for month in range(1, 13):
        first_monday = dates[dates.month == month][dates[dates.month == month].dayofweek == 0]
        if len(first_monday) > 0:
            campaign_day = (first_monday[0] - start_date).days
            campaign_duration = 7  # 1 week campaigns
            if campaign_day < n_days:
                campaign_end = min(campaign_day + campaign_duration, n_days)
                campaign_boost = np.random.uniform(1.3, 1.8)
                marketing_campaigns[campaign_day:campaign_end] *= campaign_boost

    # Churn events (negative events affecting engagement)
    churn_events = np.ones(n_days)

    # Competitor launches
    competitor_impacts = [80, 200, 290]  # Days with competitor impact
    for impact_day in competitor_impacts:
        if impact_day < n_days:
            impact_duration = 20  # 3 weeks impact
            impact_end = min(impact_day + impact_duration, n_days)
            impact_severity = np.random.uniform(0.7, 0.9)
            recovery_curve = np.linspace(impact_severity, 1.0, impact_end - impact_day)
            churn_events[impact_day:impact_end] *= recovery_curve

    # App store policy changes / platform issues
    platform_issues = [45, 180, 320]
    for issue_day in platform_issues:
        if issue_day < n_days:
            issue_duration = 10
            issue_end = min(issue_day + issue_duration, n_days)
            churn_events[issue_day:issue_end] *= np.random.uniform(0.6, 0.8)

    # Random noise and daily variations
    daily_noise = np.random.normal(1.0, 0.1, n_days)

    # Calculate DAU
    dau = (
        total_users
        * base_dau_rate
        * weekly_dau_pattern
        * monthly_dau_pattern
        * feature_releases
        * marketing_campaigns
        * churn_events
        * daily_noise
    )
    dau = np.maximum(dau, total_users * 0.1)  # Min 10% DAU

    # 2. Session Duration (minuti per sessione)
    base_session_duration = 8.5  # minuti

    # Session duration correlata con DAU (più users = più casual users = session più brevi)
    user_dilution_effect = 1 - 0.3 * (dau / total_users - 0.1) / 0.4  # Normalizzato

    # Feature richness (nuove features = session più lunghe)
    feature_richness = 1 + 0.2 * np.cumsum(np.diff(feature_releases, prepend=1) > 0.1) / 6

    # Weekend effect (session più lunghe)
    weekend_session_boost = np.array([1.0 if date.weekday() < 5 else 1.3 for date in dates])

    # Stagionalità (estate = session più brevi, inverno = più lunghe)
    seasonal_session = 1 + 0.2 * np.sin(2 * np.pi * (np.arange(n_days) + 365 / 4) / 365)

    session_noise = np.random.normal(1.0, 0.15, n_days)

    session_duration = (
        base_session_duration
        * user_dilution_effect
        * feature_richness
        * weekend_session_boost
        * seasonal_session
        * session_noise
    )
    session_duration = np.maximum(session_duration, 2.0)  # Min 2 min

    # 3. Retention Rate (% utenti che ritornano dopo 7 giorni)
    base_retention = 0.35  # 35% retention rate

    # Retention migliora con feature releases
    retention_boost = 1 + 0.3 * (feature_releases - 1)

    # Retention diminuisce con competizione
    retention_impact = churn_events**0.5  # Attenuated impact

    # Onboarding improvements (step changes)
    onboarding_improvements = np.ones(n_days)
    improvement_dates = [60, 180, 300]
    for imp_date in improvement_dates:
        if imp_date < n_days:
            onboarding_improvements[imp_date:] *= np.random.uniform(1.1, 1.2)

    retention_noise = np.random.normal(1.0, 0.08, n_days)

    retention_rate = (
        base_retention
        * retention_boost
        * retention_impact
        * onboarding_improvements
        * retention_noise
    )
    retention_rate = np.clip(retention_rate, 0.1, 0.8)  # 10% - 80%

    # 4. Feature Adoption Rate (% utenti che usano feature premium)
    base_adoption = 0.15  # 15% adoption rate

    # Adoption cresce gradualmente nel tempo (awareness + word of mouth)
    adoption_growth = 1 + 0.5 * (1 - np.exp(-np.arange(n_days) / 180))  # 180-day half-life

    # Pricing changes impact
    pricing_changes = np.ones(n_days)
    price_change_dates = [120, 250]
    for price_date in price_change_dates:
        if price_date < n_days:
            # Immediate drop then gradual recovery
            pricing_changes[price_date:] *= np.random.uniform(0.7, 0.9)
            # Recovery over 60 days
            if price_date + 60 < n_days:
                recovery = np.linspace(1.0, np.random.uniform(0.9, 1.1), 60)
                pricing_changes[price_date : price_date + 60] *= recovery

    # A/B testing effects (periodic fluctuations)
    ab_testing_effects = 1 + 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 30) * np.random.choice(
        [0, 1], n_days, p=[0.7, 0.3]
    )  # Only during testing

    adoption_noise = np.random.normal(1.0, 0.12, n_days)

    feature_adoption = (
        base_adoption * adoption_growth * pricing_changes * ab_testing_effects * adoption_noise
    )
    feature_adoption = np.clip(feature_adoption, 0.05, 0.4)  # 5% - 40%

    # 5. Revenue per User (RPU) - correlato con engagement
    base_rpu = 2.50  # $2.50 per DAU

    # RPU correlato con session duration e retention
    engagement_multiplier = (session_duration / base_session_duration) * (
        retention_rate / base_retention
    ) * 0.5 + 0.5

    # Seasonal spending patterns
    seasonal_spending = np.array(
        [
            1.5
            if date.month == 12
            # December holiday spending
            else 1.2
            if date.month in [6, 7]
            # Summer spending
            else 0.9
            if date.month in [1, 2]
            # Post-holiday drop
            else 1.0
            for date in dates
        ]
    )

    # Economic factors (simplified)
    economic_trend = 1 - 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi)  # Economic cycle

    rpu_noise = np.random.normal(1.0, 0.2, n_days)

    revenue_per_user = (
        base_rpu * engagement_multiplier * seasonal_spending * economic_trend * rpu_noise
    )
    revenue_per_user = np.maximum(revenue_per_user, 0.5)  # Min $0.50

    # Create comprehensive dataset
    engagement_data = pd.DataFrame(
        {
            "daily_active_users": dau.astype(int),
            "session_duration_min": session_duration,
            "retention_rate_7d": retention_rate,
            "feature_adoption_rate": feature_adoption,
            "revenue_per_user": revenue_per_user,
            "total_revenue": (dau * revenue_per_user).astype(int),
            "total_user_base": total_users.astype(int),
        },
        index=dates,
    )

    # Metadata for analysis
    metadata = pd.DataFrame(
        {
            "weekly_pattern": weekly_dau_pattern,
            "monthly_pattern": monthly_dau_pattern,
            "feature_releases": feature_releases,
            "marketing_campaigns": marketing_campaigns,
            "churn_events": churn_events,
            "is_weekend": [date.weekday() >= 5 for date in dates],
            "is_holiday_month": [date.month in [7, 8, 12] for date in dates],
        },
        index=dates,
    )

    return engagement_data, metadata


def calculate_engagement_kpis(engagement_data, metadata):
    """Calcola KPI di engagement standard"""

    # Core metrics
    avg_dau = engagement_data["daily_active_users"].mean()
    avg_mau = (
        engagement_data["daily_active_users"].rolling(30).mean().iloc[-1] * 30
    )  # Approssimazione
    dau_mau_ratio = avg_dau / (avg_mau / 30) if avg_mau > 0 else 0

    # Session metrics
    avg_session_duration = engagement_data["session_duration_min"].mean()
    session_duration_trend = (
        engagement_data["session_duration_min"].iloc[-30:].mean()
        - engagement_data["session_duration_min"].iloc[:30].mean()
    )

    # Retention analysis
    avg_retention = engagement_data["retention_rate_7d"].mean()
    retention_trend = (
        engagement_data["retention_rate_7d"].iloc[-30:].mean()
        - engagement_data["retention_rate_7d"].iloc[:30].mean()
    )

    # Feature adoption
    final_adoption_rate = engagement_data["feature_adoption_rate"].iloc[-1]
    adoption_growth = final_adoption_rate - engagement_data["feature_adoption_rate"].iloc[0]

    # Revenue metrics
    total_revenue = engagement_data["total_revenue"].sum()
    avg_rpu = engagement_data["revenue_per_user"].mean()
    arpu = total_revenue / engagement_data["daily_active_users"].sum()  # Average Revenue Per User

    # Growth metrics
    user_growth = (
        engagement_data["total_user_base"].iloc[-1] - engagement_data["total_user_base"].iloc[0]
    ) / engagement_data["total_user_base"].iloc[0]

    # Volatility metrics
    dau_volatility = engagement_data["daily_active_users"].std() / avg_dau
    revenue_volatility = (
        engagement_data["total_revenue"].std() / engagement_data["total_revenue"].mean()
    )

    kpis = {
        "avg_dau": avg_dau,
        "dau_mau_ratio": dau_mau_ratio,
        "avg_session_duration": avg_session_duration,
        "session_duration_trend": session_duration_trend,
        "avg_retention_rate": avg_retention,
        "retention_trend": retention_trend,
        "final_adoption_rate": final_adoption_rate,
        "adoption_growth": adoption_growth,
        "total_revenue": total_revenue,
        "avg_rpu": avg_rpu,
        "arpu": arpu,
        "user_growth": user_growth,
        "dau_volatility": dau_volatility,
        "revenue_volatility": revenue_volatility,
    }

    return kpis


def main():
    logger = setup_logger("user_engagement_forecasting", level="INFO")
    logger.info("👥 Avvio analisi forecasting user engagement")

    # Genera dati
    logger.info("📱 Generazione dati user engagement...")
    engagement_data, metadata = generate_user_engagement_data()

    # Calcola KPI
    kpis = calculate_engagement_kpis(engagement_data, metadata)

    print(f"📊 Dataset generato: {len(engagement_data)} giorni di engagement data")
    print(
        f"📅 Periodo: {engagement_data.index[0].strftime('%Y-%m-%d')} - {engagement_data.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"👥 DAU medio: {kpis['avg_dau']:,.0f} utenti")
    print(f"📈 DAU/MAU ratio: {kpis['dau_mau_ratio']:.1%}")
    print(f"⏱️  Session duration media: {kpis['avg_session_duration']:.1f} minuti")
    print(f"🔄 Retention rate media: {kpis['avg_retention_rate']:.1%}")

    print(f"\n📊 KPI User Engagement:")
    print(f"  👥 Crescita user base: {kpis['user_growth']:.1%}")
    print(f"  ⏱️  Trend session duration: {kpis['session_duration_trend']:+.1f} min")
    print(f"  🔄 Trend retention: {kpis['retention_trend']:+.1%}")
    print(f"  🎯 Feature adoption finale: {kpis['final_adoption_rate']:.1%}")
    print(f"  💰 Revenue totale: ${kpis['total_revenue']:,.0f}")
    print(f"  💵 ARPU: ${kpis['arpu']:.2f}")
    print(f"  📊 DAU volatility: {kpis['dau_volatility']:.1%}")

    # Focus su DAU per forecasting principale
    dau_series = engagement_data["daily_active_users"]

    # Split train/test
    train_size = int(len(dau_series) * 0.8)
    train_data = dau_series[:train_size]
    test_data = dau_series[train_size:]

    print(f"\n🔄 Split dataset (DAU):")
    print(
        f"  📚 Training: {len(train_data)} giorni ({train_data.index[0].strftime('%Y-%m-%d')} - {train_data.index[-1].strftime('%Y-%m-%d')})"
    )
    print(
        f"  🧪 Test: {len(test_data)} giorni ({test_data.index[0].strftime('%Y-%m-%d')} - {test_data.index[-1].strftime('%Y-%m-%d')})"
    )

    # Preprocessing per user engagement data
    logger.info("🔧 Preprocessing dati engagement...")
    preprocessor = TimeSeriesPreprocessor()

    # Log transform per stabilizzare varianza (user counts hanno growth trend)
    log_train = np.log(train_data)

    # Check stazionarietà
    stationarity_result = preprocessor.check_stationarity(log_train)
    is_stationary = stationarity_result["is_stationary"]
    if not is_stationary:
        print("📈 Serie non stazionaria - il modello userà differenziazione")

    # Selezione automatica modello per user engagement
    logger.info("🔍 Selezione automatica modello ARIMA per user engagement...")
    # Use simple ARIMA model for user engagement data
    print("Utilizzo modello ARIMA(2,1,2) per dati user engagement...")
    best_order = (2, 1, 2)
    seasonal_order = None

    print(f"\nModello selezionato:")
    print(f"  ARIMA{best_order}")

    # Training modello
    logger.info("🎯 Training modello ARIMA per DAU...")
    model = ARIMAForecaster(order=best_order)
    model.fit(log_train)

    # Forecast
    forecast_steps = len(test_data)
    logger.info(f"🔮 Generazione forecast per {forecast_steps} giorni...")
    log_forecast_result = model.forecast(
        steps=forecast_steps, confidence_intervals=True, alpha=0.05
    )

    # Trasforma back da log-scale
    forecast_dau = np.exp(log_forecast_result["forecast"])
    forecast_lower = np.exp(log_forecast_result["confidence_intervals"]["lower"])
    forecast_upper = np.exp(log_forecast_result["confidence_intervals"]["upper"])

    forecast_result = {
        "forecast": pd.Series(forecast_dau, index=test_data.index),
        "confidence_intervals": {
            "lower": pd.Series(forecast_lower, index=test_data.index),
            "upper": pd.Series(forecast_upper, index=test_data.index),
        },
    }

    # Valutazione
    logger.info("📊 Valutazione performance modello...")
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_forecast_metrics(test_data, forecast_result["forecast"])

    print(f"\n📊 Metriche Performance (DAU):")
    print(f"  📈 MAPE: {metrics['mape']:.2f}%")
    print(f"  📉 MAE: {metrics['mae']:.0f} utenti")
    print(f"  🎯 RMSE: {metrics['rmse']:.0f} utenti")

    # Check for R² score with different possible key names
    if "r2_score" in metrics:
        print(f"  📊 R²: {metrics['r2_score']:.3f}")
    elif "r_squared" in metrics:
        print(f"  📊 R²: {metrics['r_squared']:.3f}")
    else:
        print(f"  📊 R²: N/A")

    # User engagement specific metrics
    avg_forecast_growth = (
        forecast_result["forecast"].iloc[-1] - forecast_result["forecast"].iloc[0]
    ) / len(test_data)
    avg_actual_growth = (test_data.iloc[-1] - test_data.iloc[0]) / len(test_data)
    growth_accuracy = 1 - abs(avg_forecast_growth - avg_actual_growth) / abs(avg_actual_growth)

    print(f"  📈 Growth Trend Accuracy: {growth_accuracy:.1%}")

    # Weekend vs weekday accuracy
    test_metadata = metadata.iloc[train_size : train_size + len(test_data)]
    weekend_mask = test_metadata["is_weekend"]

    weekend_mape = (
        np.mean(
            np.abs(
                (test_data[weekend_mask] - forecast_result["forecast"][weekend_mask])
                / test_data[weekend_mask]
            )
        )
        * 100
    )
    weekday_mape = (
        np.mean(
            np.abs(
                (test_data[~weekend_mask] - forecast_result["forecast"][~weekend_mask])
                / test_data[~weekend_mask]
            )
        )
        * 100
    )

    print(f"  📅 Weekday MAPE: {weekday_mape:.1f}% | Weekend MAPE: {weekend_mape:.1f}%")

    # Future forecast (prossimi 30 giorni)
    future_steps = 30
    logger.info("🚀 Forecast engagement prossimi 30 giorni...")
    future_log_forecast = model.forecast(steps=future_steps, confidence_intervals=True)
    future_dau = np.exp(future_log_forecast["forecast"])
    future_lower = np.exp(future_log_forecast["confidence_intervals"]["lower"])
    future_upper = np.exp(future_log_forecast["confidence_intervals"]["upper"])

    # Forecast per altre metriche basato su correlazioni storiche
    # Session Duration forecast (semplificato basato su correlazione con DAU)
    dau_session_corr = np.corrcoef(
        engagement_data["daily_active_users"], engagement_data["session_duration_min"]
    )[0, 1]

    current_session_avg = engagement_data["session_duration_min"].iloc[-30:].mean()
    dau_change_factor = future_dau.mean() / train_data.iloc[-30:].mean()
    future_session_duration = current_session_avg * (1 + dau_session_corr * (dau_change_factor - 1))

    # Retention forecast
    current_retention = engagement_data["retention_rate_7d"].iloc[-30:].mean()
    retention_dau_corr = np.corrcoef(
        engagement_data["daily_active_users"], engagement_data["retention_rate_7d"]
    )[0, 1]
    future_retention = current_retention * (1 + retention_dau_corr * (dau_change_factor - 1))
    future_retention = np.clip(future_retention, 0.1, 0.8)

    # Visualizzazione completa engagement metrics
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # Plot 1: DAU Forecast principale
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(train_data.index, train_data, label="Training DAU", color="blue", alpha=0.7)
    ax1.plot(test_data.index, test_data, label="Test Actual", color="green", linewidth=2)
    ax1.plot(
        test_data.index,
        forecast_result["forecast"],
        label="Forecast",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    ax1.fill_between(
        test_data.index,
        forecast_result["confidence_intervals"]["lower"],
        forecast_result["confidence_intervals"]["upper"],
        color="red",
        alpha=0.2,
        label="95% CI",
    )

    ax1.set_title("👥 Daily Active Users (DAU) Forecast")
    ax1.set_ylabel("Daily Active Users")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Multi-metric overview
    ax2 = fig.add_subplot(gs[1, 0])

    # Normalizza metriche per comparazione
    metrics_to_plot = ["daily_active_users", "session_duration_min", "retention_rate_7d"]
    normalized_data = engagement_data[metrics_to_plot].div(engagement_data[metrics_to_plot].iloc[0])

    for i, metric in enumerate(metrics_to_plot):
        ax2.plot(
            normalized_data.index,
            normalized_data[metric],
            label=metric.replace("_", " ").title(),
            linewidth=2,
            alpha=0.8,
        )

    ax2.set_title("📊 Multi-Metric Trends (Normalized)")
    ax2.set_ylabel("Normalized Value (Day 1 = 1.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Weekly pattern analysis
    ax3 = fig.add_subplot(gs[1, 1])

    weekly_dau = train_data.groupby(train_data.index.dayofweek).mean()
    weekly_test = test_data.groupby(test_data.index.dayofweek).mean()
    weekly_forecast = (
        forecast_result["forecast"].groupby(forecast_result["forecast"].index.dayofweek).mean()
    )

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = range(7)
    width = 0.25

    ax3.bar(
        [i - width for i in x], weekly_dau.values, width, label="Training", alpha=0.7, color="blue"
    )
    ax3.bar(x, weekly_test.values, width, label="Test Actual", alpha=0.7, color="green")
    ax3.bar(
        [i + width for i in x],
        weekly_forecast.values,
        width,
        label="Forecast",
        alpha=0.7,
        color="red",
    )

    ax3.set_title("📅 Weekly DAU Pattern")
    ax3.set_xlabel("Day of Week")
    ax3.set_ylabel("Avg DAU")
    ax3.set_xticks(x)
    ax3.set_xticklabels(days)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Revenue analysis
    ax4 = fig.add_subplot(gs[1, 2])

    monthly_revenue = engagement_data["total_revenue"].resample("M").sum()
    monthly_dau = engagement_data["daily_active_users"].resample("M").mean()

    ax4_twin = ax4.twinx()

    bars = ax4.bar(
        monthly_revenue.index,
        monthly_revenue.values / 1000,
        alpha=0.7,
        color="green",
        label="Monthly Revenue ($k)",
    )
    ax4.set_ylabel("Revenue ($k)", color="green")
    ax4.tick_params(axis="y", labelcolor="green")

    line = ax4_twin.plot(
        monthly_dau.index, monthly_dau.values / 1000, "ro-", color="red", label="Avg DAU (k)"
    )
    ax4_twin.set_ylabel("DAU (thousands)", color="red")
    ax4_twin.tick_params(axis="y", labelcolor="red")

    ax4.set_title("💰 Revenue vs DAU Correlation")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Plot 5: Feature adoption trends
    ax5 = fig.add_subplot(gs[2, 0])

    ax5.plot(
        engagement_data.index,
        engagement_data["feature_adoption_rate"] * 100,
        color="purple",
        linewidth=2,
        label="Feature Adoption %",
    )

    # Evidenzia periodi marketing campaigns
    campaign_mask = metadata["marketing_campaigns"] > 1.2
    if campaign_mask.any():
        ax5.fill_between(
            engagement_data.index,
            0,
            50,
            where=campaign_mask,
            alpha=0.3,
            color="orange",
            label="Marketing Campaigns",
        )

    ax5.set_title("🎯 Feature Adoption Rate")
    ax5.set_ylabel("Adoption Rate (%)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis="x", rotation=45)

    # Plot 6: Retention analysis
    ax6 = fig.add_subplot(gs[2, 1])

    # Retention rate with smoothing
    retention_smooth = engagement_data["retention_rate_7d"].rolling(7).mean()
    ax6.plot(
        engagement_data.index,
        retention_smooth * 100,
        color="orange",
        linewidth=2,
        label="7-Day Retention % (7d avg)",
    )

    # Mark significant retention drops
    retention_drops = engagement_data["retention_rate_7d"].diff() < -0.05
    if retention_drops.any():
        ax6.scatter(
            engagement_data.index[retention_drops],
            (engagement_data["retention_rate_7d"][retention_drops] * 100),
            color="red",
            s=50,
            alpha=0.8,
            label="Retention Drops",
            zorder=5,
        )

    ax6.set_title("🔄 User Retention Analysis")
    ax6.set_ylabel("7-Day Retention (%)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis="x", rotation=45)

    # Plot 7: Session duration patterns
    ax7 = fig.add_subplot(gs[2, 2])

    # Session duration con pattern weekend
    weekend_mask_all = [d.weekday() >= 5 for d in engagement_data.index]

    weekday_sessions = engagement_data["session_duration_min"][~np.array(weekend_mask_all)]
    weekend_sessions = engagement_data["session_duration_min"][np.array(weekend_mask_all)]

    ax7.hist(
        weekday_sessions,
        bins=30,
        alpha=0.7,
        color="blue",
        label=f"Weekdays (μ={weekday_sessions.mean():.1f}m)",
        density=True,
    )
    ax7.hist(
        weekend_sessions,
        bins=30,
        alpha=0.7,
        color="green",
        label=f"Weekends (μ={weekend_sessions.mean():.1f}m)",
        density=True,
    )

    ax7.set_title("⏱️ Session Duration Distribution")
    ax7.set_xlabel("Session Duration (minutes)")
    ax7.set_ylabel("Density")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Future forecast dashboard
    ax8 = fig.add_subplot(gs[3, 0])

    # Ultimi 30 giorni + future forecast
    recent_dau = dau_series[-30:]
    ax8.plot(recent_dau.index, recent_dau, label="Recent DAU", color="blue", linewidth=2)

    # Future dates
    future_dates = pd.date_range(
        start=dau_series.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq="D"
    )

    ax8.plot(
        future_dates,
        future_dau,
        "o-",
        label="30-Day Forecast",
        color="purple",
        linewidth=2,
        markersize=4,
    )

    ax8.fill_between(
        future_dates, future_lower, future_upper, color="purple", alpha=0.2, label="95% CI"
    )

    ax8.set_title("🚀 Next 30 Days DAU Forecast")
    ax8.set_ylabel("Daily Active Users")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis="x", rotation=45)

    # Plot 9: Forecast vs actual scatter
    ax9 = fig.add_subplot(gs[3, 1])

    ax9.scatter(test_data, forecast_result["forecast"], alpha=0.6, color="blue", s=30)

    # Perfect prediction line
    min_val = min(test_data.min(), forecast_result["forecast"].min())
    max_val = max(test_data.max(), forecast_result["forecast"].max())
    ax9.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="Perfect Prediction")

    # Correlation info
    correlation = np.corrcoef(test_data, forecast_result["forecast"])[0, 1]
    ax9.text(
        0.05,
        0.95,
        f"R = {correlation:.3f}",
        transform=ax9.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax9.set_title("📊 Forecast vs Actual")
    ax9.set_xlabel("Actual DAU")
    ax9.set_ylabel("Forecast DAU")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot 10: Key metrics forecast
    ax10 = fig.add_subplot(gs[3, 2])

    # Forecast summary bars
    current_metrics = {
        "DAU": train_data.iloc[-30:].mean(),
        "Session (min)": engagement_data["session_duration_min"].iloc[-30:].mean(),
        "Retention %": engagement_data["retention_rate_7d"].iloc[-30:].mean() * 100,
        "Adoption %": engagement_data["feature_adoption_rate"].iloc[-30:].mean() * 100,
    }

    future_metrics = {
        "DAU": future_dau.mean(),
        "Session (min)": future_session_duration,
        "Retention %": future_retention * 100,
        "Adoption %": engagement_data["feature_adoption_rate"].iloc[-1] * 100,  # Assume stable
    }

    metrics_names = list(current_metrics.keys())
    current_values = list(current_metrics.values())
    future_values = list(future_metrics.values())

    x = range(len(metrics_names))
    width = 0.35

    bars1 = ax10.bar(
        [i - width / 2 for i in x],
        current_values,
        width,
        label="Current (30d avg)",
        alpha=0.7,
        color="blue",
    )
    bars2 = ax10.bar(
        [i + width / 2 for i in x],
        future_values,
        width,
        label="Forecast (30d)",
        alpha=0.7,
        color="red",
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax10.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax10.set_title("📈 Key Metrics Forecast Summary")
    ax10.set_ylabel("Metric Value")
    ax10.set_xticks(x)
    ax10.set_xticklabels(metrics_names)
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis="y")

    # Salva plot
    plot_path = get_plots_path("user_engagement_forecast.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("📁 Plot salvato in outputs/plots/user_engagement_forecast.png")

    # plt.show()  # Disabled for Windows compatibility
    print("Plot saved as 'outputs/plots/user_engagement_forecast.png'")

    # User Engagement Strategic Insights
    print(f"\n👥 User Engagement Strategic Insights:")

    # Growth projections
    current_monthly_dau = train_data.iloc[-30:].mean()
    future_monthly_dau = future_dau.mean()
    monthly_growth = ((future_monthly_dau - current_monthly_dau) / current_monthly_dau) * 100

    print(f"  📊 DAU previsto (30gg): {future_monthly_dau:,.0f} utenti")
    print(f"  📈 Crescita DAU: {monthly_growth:+.1f}% vs media corrente")

    # Retention insights
    current_retention_avg = engagement_data["retention_rate_7d"].iloc[-30:].mean()
    retention_change = (future_retention - current_retention_avg) / current_retention_avg * 100

    print(f"  🔄 Retention prevista: {future_retention:.1%}")
    print(f"  📈 Cambio retention: {retention_change:+.1f}%")

    if future_retention < 0.30:
        print("  🚨 RETENTION ALERT: Retention sotto 30% - focus su onboarding")
    elif future_retention > 0.45:
        print("  ✅ RETENTION EXCELLENT: Retention > 45% - ottima user experience")

    # Session engagement
    session_change = (future_session_duration - current_session_avg) / current_session_avg * 100
    print(f"  ⏱️  Session duration prevista: {future_session_duration:.1f} minuti")
    print(f"  📈 Cambio session: {session_change:+.1f}%")

    # Revenue projections
    future_revenue_estimate = (
        future_monthly_dau * engagement_data["revenue_per_user"].iloc[-30:].mean() * 30
    )
    current_revenue_30d = engagement_data["total_revenue"].iloc[-30:].sum()
    revenue_growth = ((future_revenue_estimate - current_revenue_30d) / current_revenue_30d) * 100

    print(f"  💰 Revenue stimate 30gg: ${future_revenue_estimate:,.0f}")
    print(f"  📈 Crescita revenue: {revenue_growth:+.1f}%")

    # User acquisition needs
    if monthly_growth < 5:
        print("  📢 RACCOMANDAZIONE: Intensificare user acquisition (crescita < 5%)")
    elif monthly_growth > 15:
        print("  🔥 EXCELLENT GROWTH: Crescita > 15% - preparare scaling infrastructure")

    # Seasonal patterns
    future_dates_series = pd.Series(future_dates)
    future_weekends = sum(d.weekday() >= 5 for d in future_dates)
    weekend_ratio = future_weekends / len(future_dates)

    if weekend_ratio > 0.3:  # More than 30% weekends
        print("  📅 Periodo con molti weekend - considerare campagne weekend-specific")

    # Feature adoption opportunities
    current_adoption = engagement_data["feature_adoption_rate"].iloc[-1]
    if current_adoption < 0.20:
        print(
            f"  🎯 Feature adoption bassa ({current_adoption:.1%}) - ottimizzare onboarding premium"
        )
    elif current_adoption > 0.35:
        print(f"  🎯 Ottima feature adoption ({current_adoption:.1%}) - considerare upselling")

    # Churn risk assessment
    recent_volatility = train_data.iloc[-30:].std() / train_data.iloc[-30:].mean()
    if recent_volatility > 0.15:
        print(
            f"  🚨 VOLATILITY ALERT: DAU volatility {recent_volatility:.1%} - investigate churn factors"
        )
    else:
        print(f"  ✅ DAU stability buona: volatility {recent_volatility:.1%}")

    # Salva modello
    model_path = get_models_path("user_engagement_arima_model.joblib")
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
            report_title="Analisi Forecasting Engagement Utenti",
            output_filename="user_engagement_forecasting_report",
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

    # Salva insights per reporting
    insights_summary = {
        "forecast_period": "30 days",
        "avg_dau_forecast": future_monthly_dau,
        "growth_rate": monthly_growth,
        "retention_forecast": future_retention,
        "session_duration_forecast": future_session_duration,
        "revenue_estimate": future_revenue_estimate,
        "model_accuracy": metrics.get("r2_score", metrics.get("r_squared", "N/A")),
        "recommendations": [],
    }

    if monthly_growth < 5:
        insights_summary["recommendations"].append("Intensificare user acquisition")
    if future_retention < 0.30:
        insights_summary["recommendations"].append("Focus su retention e onboarding")
    if current_adoption < 0.20:
        insights_summary["recommendations"].append("Ottimizzare feature adoption")

    print(f"\n✅ Analisi user engagement completata!")
    print(f"📁 Risultati e insights salvati in outputs/")
    print(f"👥 Modello pronto per integrazione con analytics platforms e dashboards")


if __name__ == "__main__":
    main()
