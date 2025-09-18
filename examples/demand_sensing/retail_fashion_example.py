"""
Esempio Demand Sensing per Retail Fashion.

Caso d'uso specifico per abbigliamento/moda con focus su:
- Stagionalità e meteo
- Trend social media e influencer
- Eventi moda e saldi
- Sentiment brand
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from arima_forecaster import SARIMAForecaster
from arima_forecaster.demand_sensing import (
    EnsembleDemandSensor,
    EnsembleConfig,
    CalendarEvents,
    Event,
    EventType,
)


def generate_fashion_sales_data(days=300):
    """Genera dati vendita abbigliamento con pattern stagionali."""
    np.random.seed(42)

    # Trend base in crescita
    base_trend = 50 + np.cumsum(np.random.normal(0.05, 0.2, days))

    # Stagionalità annuale forte (picchi autunno/inverno)
    yearly_season = np.sin(np.arange(days) * 2 * np.pi / 365 + np.pi) * 25 + 25

    # Pattern settimanale (weekend più alti)
    weekly_pattern = np.tile([0.7, 0.8, 0.9, 0.9, 1.0, 1.4, 1.3], days // 7 + 1)[:days]

    # Eventi speciali (saldi, black friday)
    special_events = np.zeros(days)

    # Saldi invernali (gennaio)
    if days > 10:
        special_events[5:35] = 40  # Boom saldi

    # Black Friday (fine novembre)
    if days > 250:
        special_events[250:255] = 60  # Mega boom

    # Noise
    noise = np.random.normal(0, 8, days)

    # Combina tutto
    sales = (base_trend + yearly_season) * weekly_pattern + special_events + noise
    sales = np.maximum(sales, 5)  # Min 5 vendite

    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    df = pd.DataFrame({"date": dates, "sales": sales})

    return df


def main():
    print("👗 DEMAND SENSING - RETAIL FASHION")
    print("=" * 50)

    # 1. Setup scenario
    print("\n🎯 1. Setup Scenario Fashion Retail...")
    brand_name = "StyleMilano"
    product_category = "clothing"
    location = "Milan,IT"

    print(f"   Brand: {brand_name}")
    print(f"   Categoria: Abbigliamento donna")
    print(f"   Location: {location}")
    print(f"   Periodo analisi: Gennaio 2024 - Ottobre 2024")

    # 2. Genera dati storici
    print("\n📊 2. Generazione storico vendite...")
    sales_data = generate_fashion_sales_data(300)

    print(f"   Dati: {len(sales_data)} giorni")
    print(f"   Media giornaliera: {sales_data['sales'].mean():.1f} pezzi")
    print(f"   Picco massimo: {sales_data['sales'].max():.0f} pezzi")
    print(f"   Stagionalità rilevata: ✅")

    # 3. Training modello
    print("\n🤖 3. Training modello SARIMA stagionale...")
    train_size = int(len(sales_data) * 0.85)
    train_data = sales_data[:train_size]["sales"]
    test_data = sales_data[train_size:]["sales"]

    model = SARIMAForecaster(
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),  # Stagionalità settimanale
        validate_input=False,
    )

    model.fit(train_data)
    print("   ✅ Modello addestrato con successo")

    # 4. Setup Demand Sensing per Fashion
    print("\n👔 4. Configurazione Demand Sensing Fashion...")

    # Configurazione ottimizzata per fashion
    config = EnsembleConfig(
        source_weights={
            "weather": 0.25,  # Molto importante per abbigliamento
            "trends": 0.20,  # Trend moda importanti
            "social": 0.25,  # Influencer e social key
            "economic": 0.15,  # Meno critico per fashion
            "calendar": 0.15,  # Eventi e saldi importanti
        },
        combination_strategy="weighted_average",
        max_total_adjustment=0.4,  # Fashion molto volatile
        enable_learning=True,
    )

    sensor = EnsembleDemandSensor(
        base_model=model, product_category="clothing", location=location, config=config
    )

    # Aggiungi eventi fashion personalizzati
    milan_fashion_week = datetime(2024, 9, 17)
    winter_sales_start = datetime(2025, 1, 7)

    sensor.calendar.add_custom_event(
        name="Milano Fashion Week",
        date=milan_fashion_week,
        event_type=EventType.FAIR,
        expected_impact=0.30,  # +30% durante fashion week
        impact_radius_days=10,
        duration_days=7,
    )

    sensor.calendar.add_custom_event(
        name="Saldi Invernali 2025",
        date=winter_sales_start,
        event_type=EventType.SEASONAL,
        expected_impact=0.50,  # +50% durante saldi
        impact_radius_days=5,
        duration_days=45,
    )

    print("   ✅ Configurazione fashion completata")
    print("   👑 Eventi aggiunti: Fashion Week, Saldi Invernali")

    # 5. Forecast con Demand Sensing
    print("\n🔮 5. Previsioni con Demand Sensing...")

    forecast_horizon = 45  # 45 giorni (include eventi importanti)
    base_forecast = model.predict(steps=forecast_horizon)

    print(f"   Orizzonte: {forecast_horizon} giorni")
    print("   Raccolta fattori esterni...")

    # Applica sensing con dettagli
    sensing_result, details = sensor.sense(
        base_forecast=base_forecast, use_demo_data=True, return_details=True
    )

    print(f"   ✅ Sensing completato")
    print(f"   Aggiustamento totale: {sensing_result.total_adjustment:+.1%}")
    print(f"   Confidenza: {sensing_result.confidence_score:.2f}")

    # 6. Analisi dettagliata
    print("\n📈 6. Analisi Impact Factors...")

    # Contributi per fonte
    contributions = details["source_contributions"]
    print("\n   Contributi per fonte:")
    for _, row in contributions.iterrows():
        emoji_map = {
            "weather": "🌤️",
            "trends": "📱",
            "social": "💬",
            "economic": "💰",
            "calendar": "📅",
        }
        emoji = emoji_map.get(row["source"], "📊")
        print(
            f"   {emoji} {row['source'].capitalize():<10}: "
            f"{row['contribution_pct']:5.1f}% "
            f"(confidence: {row['avg_confidence']:.2f})"
        )

    # Analisi periodi critici
    adjustment_pct = (
        (sensing_result.adjusted_forecast - sensing_result.original_forecast)
        / sensing_result.original_forecast
        * 100
    )

    high_impact_days = adjustment_pct[abs(adjustment_pct) > 10]
    if len(high_impact_days) > 0:
        print(f"\n   🎯 Giorni ad alto impatto: {len(high_impact_days)}")
        print(f"   📈 Max aumento: {adjustment_pct.max():.1f}%")
        print(f"   📉 Max calo: {adjustment_pct.min():.1f}%")

    # 7. Business Intelligence
    print("\n💼 7. Business Intelligence Insights...")

    # Calcola metriche business
    total_forecast_original = sensing_result.original_forecast.sum()
    total_forecast_adjusted = sensing_result.adjusted_forecast.sum()
    revenue_impact = total_forecast_adjusted - total_forecast_original

    avg_price = 45  # €45 prezzo medio capo
    revenue_diff = revenue_impact * avg_price

    print(f"   📦 Volume previsto (originale): {total_forecast_original:.0f} pezzi")
    print(f"   📦 Volume previsto (aggiustato): {total_forecast_adjusted:.0f} pezzi")
    print(f"   💰 Impatto ricavi stimato: {revenue_diff:+.0f} €")

    # Inventory planning
    safety_stock_original = total_forecast_original * 0.15
    safety_stock_adjusted = total_forecast_adjusted * 0.15
    inventory_diff = safety_stock_adjusted - safety_stock_original

    print(
        f"   📦 Safety stock consigliato: {safety_stock_adjusted:.0f} pezzi ({inventory_diff:+.0f})"
    )

    # 8. Raccomandazioni strategiche
    print("\n💡 8. Raccomandazioni Strategiche...")

    for i, rec in enumerate(sensing_result.recommendations, 1):
        print(f"   {i}. {rec}")

    # Aggiungi raccomandazioni fashion-specific
    fashion_recommendations = []

    # Analizza weather impact
    weather_factors = details["factors_by_source"].get("weather", [])
    if weather_factors:
        avg_weather_impact = np.mean([f.impact for f in weather_factors])
        if avg_weather_impact > 0.05:
            fashion_recommendations.append(
                "🌞 Meteo favorevole previsto - aumentare stock capi leggeri/estivi"
            )
        elif avg_weather_impact < -0.05:
            fashion_recommendations.append(
                "🌧️ Meteo sfavorevole - focus su capi indoor e comfort wear"
            )

    # Analizza social impact
    social_factors = details["factors_by_source"].get("social", [])
    if social_factors:
        avg_social = np.mean([f.impact for f in social_factors])
        if avg_social > 0.1:
            fashion_recommendations.append(
                "📱 Forte buzz social positivo - intensificare campagne marketing"
            )

    # Eventi calendar
    calendar_factors = details["factors_by_source"].get("calendar", [])
    if calendar_factors:
        fashion_recommendations.append(
            "📅 Eventi importanti in programma - preparare stock e staff extra"
        )

    if fashion_recommendations:
        print("\n   👗 Raccomandazioni Fashion-Specific:")
        for i, rec in enumerate(fashion_recommendations, 1):
            print(f"   {i}. {rec}")

    # 9. Visualizzazioni avanzate
    print("\n📊 9. Generazione dashboard visuale...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Trend storico + previsioni
    ax1 = axes[0, 0]
    # Ultimi 60 giorni storici
    recent_history = sales_data.tail(60)
    ax1.plot(range(-60, 0), recent_history["sales"].values, "b-", label="Storico", linewidth=2)

    # Previsioni
    forecast_days = range(0, len(base_forecast))
    ax1.plot(
        forecast_days, sensing_result.original_forecast.values, "g-", label="SARIMA", linewidth=2
    )
    ax1.plot(
        forecast_days,
        sensing_result.adjusted_forecast.values,
        "r--",
        label="Con Demand Sensing",
        linewidth=2,
    )

    ax1.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Oggi")
    ax1.set_title("Trend Vendite + Previsioni", fontweight="bold")
    ax1.set_xlabel("Giorni")
    ax1.set_ylabel("Vendite (pezzi)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heatmap impatti giornalieri
    ax2 = axes[0, 1]
    daily_impacts = adjustment_pct.values.reshape(1, -1)
    im = ax2.imshow(daily_impacts, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)
    ax2.set_title("Heatmap Impatti Giornalieri (%)", fontweight="bold")
    ax2.set_xlabel("Giorni")
    ax2.set_yticks([])
    plt.colorbar(im, ax=ax2)

    # Plot 3: Contributi fonte (pie)
    ax3 = axes[0, 2]
    if not contributions.empty:
        wedges, texts, autotexts = ax3.pie(
            contributions["contribution_pct"],
            labels=contributions["source"].str.capitalize(),
            autopct="%1.1f%%",
            colors=["skyblue", "lightgreen", "salmon", "gold", "plum"],
        )
    ax3.set_title("Contributi Fonti Esterne", fontweight="bold")

    # Plot 4: Impatto economico cumulativo
    ax4 = axes[1, 0]
    cumulative_impact = (
        sensing_result.adjusted_forecast - sensing_result.original_forecast
    ).cumsum() * avg_price
    ax4.plot(forecast_days, cumulative_impact.values, "purple", linewidth=3)
    ax4.fill_between(forecast_days, 0, cumulative_impact.values, color="purple", alpha=0.3)
    ax4.set_title("Impatto Ricavi Cumulativo (€)", fontweight="bold")
    ax4.set_xlabel("Giorni")
    ax4.set_ylabel("Euro (€)")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Plot 5: Top eventi impattanti
    ax5 = axes[1, 1]
    top_factors = sensing_result.factors_applied[:10]
    if top_factors:
        impacts = [f.adjustment_percentage for f in top_factors]
        names = [f.factor.name.replace("_", " ")[:12] for f in top_factors]
        colors = ["green" if x > 0 else "red" for x in impacts]

        bars = ax5.barh(names, impacts, color=colors, alpha=0.7)
        ax5.set_title("Top 10 Fattori Impattanti", fontweight="bold")
        ax5.set_xlabel("Impatto (%)")

    # Plot 6: Confidence timeline
    ax6 = axes[1, 2]
    if sensing_result.factors_applied:
        confidences = [
            f.factor.confidence for f in sensing_result.factors_applied[: len(forecast_days)]
        ]
        ax6.plot(
            forecast_days[: len(confidences)],
            confidences,
            "orange",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax6.set_title("Timeline Confidenza Fattori", fontweight="bold")
        ax6.set_xlabel("Giorni")
        ax6.set_ylabel("Confidenza (0-1)")
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fashion_demand_sensing.png", dpi=150, bbox_inches="tight")
    print("   📊 Dashboard salvato: 'fashion_demand_sensing.png'")

    # 10. Export business report
    print("\n📋 10. Export Business Report...")

    # Crea report dettagliato
    business_report = {
        "Periodo": f"{datetime.now().strftime('%Y-%m-%d')} + {forecast_horizon} giorni",
        "Brand": brand_name,
        "Categoria": "Abbigliamento Donna",
        "Location": location,
        "Volume_Originale": int(total_forecast_original),
        "Volume_Aggiustato": int(total_forecast_adjusted),
        "Variazione_Pezzi": int(revenue_impact),
        "Variazione_Ricavi_EUR": int(revenue_diff),
        "Safety_Stock_Consigliato": int(safety_stock_adjusted),
        "Confidenza_Media": round(sensing_result.confidence_score, 2),
        "Aggiustamento_Totale_Pct": round(sensing_result.total_adjustment * 100, 1),
        "Top_Fonte_Impatto": contributions.iloc[0]["source"] if not contributions.empty else "N/A",
    }

    # Salva in CSV
    report_df = pd.DataFrame([business_report])
    report_df.to_csv("fashion_business_report.csv", index=False)

    # Salva dettagli previsioni
    forecast_detail = sensing_result.to_dataframe()
    forecast_detail["data"] = pd.date_range(
        start=datetime.now(), periods=len(forecast_detail), freq="D"
    )
    forecast_detail["ricavi_originali"] = forecast_detail["original"] * avg_price
    forecast_detail["ricavi_aggiustati"] = forecast_detail["adjusted"] * avg_price
    forecast_detail["impatto_ricavi"] = (
        forecast_detail["ricavi_aggiustati"] - forecast_detail["ricavi_originali"]
    )

    forecast_detail.to_csv("fashion_forecast_detail.csv", index=False, float_format="%.2f")

    print("   💼 Report business: 'fashion_business_report.csv'")
    print("   📊 Dettagli forecast: 'fashion_forecast_detail.csv'")

    print("\n" + "=" * 50)
    print("🎉 ANALISI FASHION COMPLETATA!")
    print(f"\n📈 RISULTATI CHIAVE:")
    print(f"• Impatto Demand Sensing: {sensing_result.total_adjustment:+.1%}")
    print(f"• Variazione ricavi stimata: {revenue_diff:+,.0f} €")
    print(f"• Giorni ad alto impatto: {len(high_impact_days)}")
    print(f"• Confidenza sistema: {sensing_result.confidence_score:.1%}")

    print(f"\n📁 FILE GENERATI:")
    print(f"• fashion_demand_sensing.png (dashboard)")
    print(f"• fashion_business_report.csv (KPI)")
    print(f"• fashion_forecast_detail.csv (previsioni)")


if __name__ == "__main__":
    main()
