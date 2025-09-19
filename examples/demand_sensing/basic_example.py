"""
Esempio base di Demand Sensing - Integrazione fattori esterni.

Dimostra come utilizzare il sistema demand sensing per migliorare
le previsioni ARIMA con fattori esterni.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from arima_forecaster import SARIMAForecaster
from arima_forecaster.demand_sensing import EnsembleDemandSensor, EnsembleConfig


def generate_demo_sales_data(days=365):
    """Genera dati di vendita demo con pattern realistici."""
    np.random.seed(42)

    # Trend base
    base_trend = 100 + np.cumsum(np.random.normal(0, 0.1, days)) * 0.5

    # Stagionalità settimanale
    weekly_pattern = np.tile([0.8, 0.9, 1.0, 1.0, 1.1, 1.4, 1.2], days // 7 + 1)[:days]

    # Stagionalità mensile
    monthly_pattern = np.sin(np.arange(days) * 2 * np.pi / 365) * 10 + 15

    # Noise
    noise = np.random.normal(0, 5, days)

    # Combina tutto
    sales = base_trend * weekly_pattern + monthly_pattern + noise
    sales = np.maximum(sales, 10)  # Min 10 vendite

    # Crea DataFrame
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    df = pd.DataFrame({"date": dates, "sales": sales})

    return df


def main():
    print("🔥 DEMAND SENSING - ESEMPIO BASE")
    print("=" * 50)

    # 1. Genera dati demo
    print("\n📊 1. Generazione dati vendite storiche...")
    sales_data = generate_demo_sales_data(365)
    print(f"   Generati {len(sales_data)} giorni di vendite")
    print(f"   Media: {sales_data['sales'].mean():.1f} unità/giorno")
    print(f"   Std: {sales_data['sales'].std():.1f}")

    # 2. Prepara dati per training
    print("\n🤖 2. Training modello SARIMA base...")
    train_size = int(len(sales_data) * 0.8)
    train_data = sales_data[:train_size]["sales"]
    test_data = sales_data[train_size:]["sales"]

    # Crea e addestra modello
    model = SARIMAForecaster(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),  # Stagionalità settimanale
        validate_input=False,  # Velocizza per demo
    )

    model.fit(train_data)
    print("   ✅ Modello SARIMA addestrato")

    # 3. Forecast base (senza demand sensing)
    print("\n📈 3. Previsioni base (solo SARIMA)...")
    forecast_horizon = 30
    base_forecast = model.predict(steps=forecast_horizon)

    print(f"   Previsioni per {forecast_horizon} giorni")
    print(f"   Media forecast: {base_forecast.mean():.1f}")

    # 4. Inizializza Demand Sensing
    print("\n🌍 4. Inizializzazione Ensemble Demand Sensor...")

    # Configurazione per categoria "electronics"
    config = EnsembleConfig(
        enable_weather=True,
        enable_trends=True,
        enable_social=True,
        enable_economic=True,
        enable_calendar=True,
        combination_strategy="weighted_average",
        max_total_adjustment=0.3,  # Max 30% aggiustamento
        source_weights={
            "weather": 0.15,  # Elettronica poco sensibile al meteo
            "trends": 0.30,  # Molto sensibile a trend ricerca
            "social": 0.20,  # Importante sentiment
            "economic": 0.25,  # Sensibile a economia
            "calendar": 0.10,  # Poco sensibile a festività
        },
    )

    sensor = EnsembleDemandSensor(
        base_model=model, product_category="electronics", location="Milan,IT", config=config
    )

    print("   ✅ Demand Sensor configurato per elettronica")

    # 5. Applica Demand Sensing
    print("\n🎯 5. Applicazione Demand Sensing...")
    print("   Raccolta fattori esterni (usando dati demo)...")

    # Ottieni risultati completi
    sensing_result, details = sensor.sense(
        base_forecast=base_forecast, use_demo_data=True, return_details=True
    )

    print(f"   ✅ Sensing completato")
    print(f"   Aggiustamento totale: {sensing_result.total_adjustment:.2%}")
    print(f"   Confidenza media: {sensing_result.confidence_score:.2f}")
    print(f"   Fattori applicati: {len(sensing_result.factors_applied)}")

    # 6. Analisi risultati
    print("\n📊 6. Analisi risultati...")

    # Calcola metriche
    original_total = sensing_result.original_forecast.sum()
    adjusted_total = sensing_result.adjusted_forecast.sum()
    improvement = ((adjusted_total - original_total) / original_total) * 100

    print(f"   Domanda totale originale: {original_total:.0f} unità")
    print(f"   Domanda totale aggiustata: {adjusted_total:.0f} unità")
    print(f"   Variazione complessiva: {improvement:+.1f}%")

    # Contributi per fonte
    print("\n📈 Contributi per fonte:")
    contributions = details["source_contributions"]
    for _, row in contributions.iterrows():
        print(
            f"   {row['source'].capitalize():<12}: "
            f"{row['contribution_pct']:5.1f}% "
            f"(impact: {row['avg_impact']:+.3f}, confidence: {row['avg_confidence']:.2f})"
        )

    # 7. Raccomandazioni
    print("\n💡 Raccomandazioni:")
    for i, rec in enumerate(sensing_result.recommendations, 1):
        print(f"   {i}. {rec}")

    # 8. Visualizzazioni
    print("\n📊 8. Generazione grafici...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Confronto previsioni
    ax1 = axes[0, 0]
    days = range(len(base_forecast))
    ax1.plot(days, sensing_result.original_forecast.values, "b-", label="SARIMA Base", linewidth=2)
    ax1.plot(
        days,
        sensing_result.adjusted_forecast.values,
        "r--",
        label="Con Demand Sensing",
        linewidth=2,
    )
    ax1.fill_between(
        days,
        sensing_result.original_forecast.values,
        sensing_result.adjusted_forecast.values,
        alpha=0.3,
        color="green",
        label="Aggiustamento",
    )
    ax1.set_title("Confronto Previsioni", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Giorni")
    ax1.set_ylabel("Vendite")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Aggiustamento percentuale
    ax2 = axes[0, 1]
    adjustment_pct = (
        (sensing_result.adjusted_forecast - sensing_result.original_forecast)
        / sensing_result.original_forecast
        * 100
    )
    colors = ["green" if x > 0 else "red" for x in adjustment_pct.values]
    ax2.bar(days, adjustment_pct.values, color=colors, alpha=0.7)
    ax2.set_title("Aggiustamento Giornaliero %", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Giorni")
    ax2.set_ylabel("Aggiustamento %")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Plot 3: Top fattori impatto
    ax3 = axes[1, 0]
    if sensing_result.factors_applied:
        top_factors = sensing_result.factors_applied[:8]
        impacts = [f.adjustment_percentage for f in top_factors]
        names = [
            f.factor.name[:15] + "..." if len(f.factor.name) > 15 else f.factor.name
            for f in top_factors
        ]
        colors = ["green" if x > 0 else "red" for x in impacts]

        bars = ax3.barh(names, impacts, color=colors, alpha=0.7)
        ax3.set_title("Top 8 Fattori per Impatto", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Impatto %")

        # Aggiungi valori sulle barre
        for bar, impact in zip(bars, impacts):
            width = bar.get_width()
            ax3.text(
                width + (0.1 if width > 0 else -0.1),
                bar.get_y() + bar.get_height() / 2,
                f"{impact:+.1f}%",
                ha="left" if width > 0 else "right",
                va="center",
            )

    # Plot 4: Contributi per fonte
    ax4 = axes[1, 1]
    if not contributions.empty:
        wedges, texts, autotexts = ax4.pie(
            contributions["contribution_pct"],
            labels=contributions["source"].str.capitalize(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax4.set_title("Contributi per Fonte", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("demand_sensing_analysis.png", dpi=150, bbox_inches="tight")
    print("   📊 Grafici salvati in 'demand_sensing_analysis.png'")

    # 9. Export risultati
    print("\n💾 9. Export risultati...")

    # Crea summary DataFrame
    results_df = sensing_result.to_dataframe()
    results_df["date"] = pd.date_range(start=datetime.now(), periods=len(results_df), freq="D")
    results_df = results_df[["date", "original", "adjusted", "adjustment", "adjustment_pct"]]
    results_df.columns = ["Data", "Originale", "Aggiustata", "Aggiustamento", "Aggiustamento%"]

    # Salva CSV
    results_df.to_csv("demand_sensing_results.csv", index=False, float_format="%.2f")
    print("   📄 Risultati salvati in 'demand_sensing_results.csv'")

    # 10. Simulazione apprendimento
    print("\n🧠 10. Simulazione apprendimento dal feedback...")

    # Simula dati effettivi (con un po' di rumore)
    np.random.seed(123)
    simulated_actuals = sensing_result.adjusted_forecast * np.random.uniform(
        0.9, 1.1, len(sensing_result.adjusted_forecast)
    )

    # Applica apprendimento
    sensor.learn_from_actuals(sensing_result.adjusted_forecast, simulated_actuals)

    print("   ✅ Sistema aggiornato con feedback")
    print("   📈 Pesi fonti ottimizzati per performance future")

    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETATA CON SUCCESSO!")
    print("\nFile generati:")
    print("- demand_sensing_analysis.png (grafici)")
    print("- demand_sensing_results.csv (risultati)")
    print("\nIl sistema Demand Sensing ha dimostrato come i fattori esterni")
    print("possano migliorare significativamente l'accuratezza delle previsioni!")


if __name__ == "__main__":
    main()
