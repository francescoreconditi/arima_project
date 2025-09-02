"""
Esempio Demand Sensing per Food Delivery.

Caso d'uso per delivery/ristoranti con focus su:
- Meteo (pioggia = più ordini)
- Eventi sportivi e festività
- Trends ricerca cibo
- Sentiment ristoranti
- Indicatori economici (potere d'acquisto)
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
    EventType
)


def generate_food_delivery_data(days=200):
    """Genera dati ordini food delivery realistici."""
    np.random.seed(42)
    
    # Base trend in crescita (settore in espansione)
    base_trend = 120 + np.cumsum(np.random.normal(0.15, 0.3, days))
    
    # Pattern settimanale forte (weekend picchi, lunedì basso)
    weekly_multipliers = [0.7, 0.9, 1.0, 1.0, 1.2, 1.4, 1.3]  # Lun-Dom
    weekly_pattern = np.tile(weekly_multipliers, days // 7 + 1)[:days]
    
    # Pattern orario simulato (cena > pranzo)
    # Simuliamo con variazione casuale che rappresenta mix pranzo/cena
    meal_variation = np.random.uniform(0.8, 1.2, days)
    
    # Effetto meteo (pioggia/freddo = più ordini)
    weather_effect = np.random.choice([1.0, 1.2, 1.0, 0.9, 1.3], days, 
                                     p=[0.4, 0.2, 0.2, 0.1, 0.1])  # 30% giorni con boost meteo
    
    # Eventi speciali
    special_days = np.ones(days)
    
    # Domeniche partite importanti
    sundays = [i for i in range(6, days, 7)]  # Ogni domenica
    for sunday in sundays[:min(len(sundays), 10)]:
        if np.random.random() < 0.4:  # 40% domeniche con partite
            special_days[sunday] = 1.4
    
    # Festività (Natale, Capodanno, etc.)
    holiday_boost_days = np.random.choice(range(days), min(8, days//25), replace=False)
    for day in holiday_boost_days:
        special_days[day] = 1.6
    
    # Noise
    noise = np.random.normal(0, 12, days)
    
    # Combina tutto
    orders = (base_trend * weekly_pattern * meal_variation * 
              weather_effect * special_days) + noise
    orders = np.maximum(orders, 20)  # Min 20 ordini/giorno
    
    dates = pd.date_range(start='2024-06-01', periods=days, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'orders': orders
    })
    
    return df


def main():
    print("[PIZZA] DEMAND SENSING - FOOD DELIVERY")
    print("=" * 50)
    
    # 1. Setup scenario
    print("\n[TARGET] 1. Setup Scenario Food Delivery...")
    restaurant_name = "Roma Pizza Express"
    location = "Rome,IT"
    cuisine_type = "Italian"
    
    print(f"   Ristorante: {restaurant_name}")
    print(f"   Cucina: {cuisine_type}")
    print(f"   Location: {location}")
    print(f"   Servizio: Delivery + Takeaway")
    
    # 2. Genera storico ordini
    print("\n[CHART] 2. Generazione storico ordini...")
    orders_data = generate_food_delivery_data(200)
    
    print(f"   Periodo: {len(orders_data)} giorni")
    print(f"   Media ordini/giorno: {orders_data['orders'].mean():.1f}")
    print(f"   Picco massimo: {orders_data['orders'].max():.0f} ordini")
    print(f"   Weekend factor: {orders_data[orders_data['date'].dt.weekday >= 5]['orders'].mean() / orders_data[orders_data['date'].dt.weekday < 5]['orders'].mean():.1f}x")
    
    # 3. Training modello
    print("\n[ROBOT] 3. Training modello SARIMA per food delivery...")
    train_size = int(len(orders_data) * 0.8)
    train_data = orders_data[:train_size]['orders']
    
    model = SARIMAForecaster(
        order=(2, 1, 1),
        seasonal_order=(1, 1, 0, 7)  # Stagionalità settimanale forte
    )
    
    model.fit(train_data)
    print("   [OK] Modello addestrato (pattern settimanali rilevati)")
    
    # 4. Configurazione Demand Sensing per Food
    print("\n[FOOD] 4. Configurazione Demand Sensing Food...")
    
    # Configurazione ottimizzata per food delivery
    config = EnsembleConfig(
        source_weights={
            'weather': 0.35,     # CRITICO per delivery (pioggia/caldo)
            'trends': 0.15,      # Meno importante per cibo base
            'social': 0.20,      # Reviews e sentiment importanti
            'economic': 0.10,    # Cibo essenziale meno sensibile
            'calendar': 0.20     # Eventi sportivi/festività cruciali
        },
        combination_strategy="weighted_average",
        max_total_adjustment=0.5,  # Food può variare molto
        min_sources_for_adjustment=2
    )
    
    sensor = EnsembleDemandSensor(
        base_model=model,
        product_category="food",
        location=location,
        config=config
    )
    
    # Eventi custom per food delivery
    champions_final = datetime(2024, 6, 1)
    summer_promotion = datetime(2024, 7, 15)
    
    sensor.calendar.add_custom_event(
        name="Champions League Final",
        date=champions_final,
        event_type=EventType.SPORT,
        expected_impact=0.40,  # +40% ordini durante finale
        impact_radius_days=1,
        duration_days=1
    )
    
    sensor.calendar.add_custom_event(
        name="Promozione Estate 2024",
        date=summer_promotion,
        event_type=EventType.CUSTOM,
        expected_impact=0.25,
        impact_radius_days=3,
        duration_days=14
    )
    
    print("   [OK] Configurazione food delivery completata")
    print("   [SPORT] Eventi aggiunti: Champions Final, Promo Estate")
    
    # 5. Weather-specific configuration
    print("\n[WEATHER] 5. Configurazione sensibilità meteo...")
    
    # Food delivery è MOLTO sensibile al meteo
    # Pioggia/freddo = più ordini, bel tempo = meno ordini
    weather_sensitivity = {
        'rain_boost': 1.3,      # +30% con pioggia
        'hot_boost': 1.2,       # +20% con caldo estremo
        'cold_boost': 1.25,     # +25% con freddo
        'nice_weather_penalty': 0.9  # -10% con bel tempo
    }
    
    print("   [RAIN] Pioggia: +30% ordini")
    print("   [HOT] Caldo estremo: +20% ordini")  
    print("   [COLD] Freddo: +25% ordini")
    print("   [SUN] Bel tempo: -10% ordini")
    
    # 6. Previsioni con Demand Sensing
    print("\n[FORECAST] 6. Previsioni ordini prossimi 21 giorni...")
    
    forecast_horizon = 21  # 3 settimane
    base_forecast = model.forecast(steps=forecast_horizon)
    
    print(f"   Orizzonte: {forecast_horizon} giorni")
    print("   Analisi fattori esterni in corso...")
    
    # Applica sensing
    sensing_result, details = sensor.sense(
        base_forecast=base_forecast,
        use_demo_data=True,
        return_details=True
    )
    
    print(f"   [OK] Sensing completato")
    print(f"   Aggiustamento: {sensing_result.total_adjustment:+.1%}")
    print(f"   Confidenza: {sensing_result.confidence_score:.2f}")
    
    # 7. Analisi operativa
    print("\n[SHOP] 7. Analisi Operativa & Logistica...")
    
    # Metriche business
    total_orders_base = sensing_result.original_forecast.sum()
    total_orders_adjusted = sensing_result.adjusted_forecast.sum()
    order_impact = total_orders_adjusted - total_orders_base
    
    avg_order_value = 18.50  # €18.50 valore medio ordine
    revenue_impact = order_impact * avg_order_value
    
    print(f"   [BOX] Ordini previsti (base): {total_orders_base:.0f}")
    print(f"   [BOX] Ordini previsti (adjusted): {total_orders_adjusted:.0f}")
    print(f"   [UP] Differenza ordini: {order_impact:+.0f}")
    print(f"   [MONEY] Impatto ricavi: {revenue_impact:+.0f} €")
    
    # Pianificazione staff
    orders_per_staff = 25  # Un rider gestisce ~25 ordini/turno
    additional_staff = max(0, order_impact / orders_per_staff)
    staff_cost = additional_staff * 8 * 15  # 8h x €15/h
    
    print(f"   [PEOPLE] Staff aggiuntivo necessario: {additional_staff:.1f} rider")
    print(f"   [COST] Costo staff extra: {staff_cost:.0f} €")
    print(f"   [PROFIT] Profitto netto: {revenue_impact - staff_cost:+.0f} €")
    
    # Inventory planning per ingredienti
    adjustment_pct = ((sensing_result.adjusted_forecast - sensing_result.original_forecast) / 
                     sensing_result.original_forecast * 100)
    high_demand_days = len(adjustment_pct[adjustment_pct > 15])
    
    if high_demand_days > 0:
        print(f"   [CART] Giorni alta domanda: {high_demand_days}")
        print(f"   [TOMATO] Scorte extra consigliate: +20% ingredienti principali")
    
    # 8. Breakdown per fonte
    print("\n[CHART] 8. Analisi Contributi Esterni...")
    
    contributions = details['source_contributions']
    print("   Contributi per fonte:")
    
    for _, row in contributions.iterrows():
        source_emoji = {
            'weather': '[WEATHER]',
            'calendar': '[CALENDAR]', 
            'social': '[SOCIAL]',
            'economic': '[MONEY]',
            'trends': '[TREND]'
        }
        emoji = source_emoji.get(row['source'], '[CHART]')
        print(f"   {emoji} {row['source'].capitalize():<10}: "
              f"{row['contribution_pct']:5.1f}% "
              f"(impact: {row['avg_impact']:+.3f})")
    
    # Analisi giorni critici
    peak_days = adjustment_pct[adjustment_pct > 20]
    valley_days = adjustment_pct[adjustment_pct < -10]
    
    if len(peak_days) > 0:
        print(f"\n   [FIRE] GIORNI PICCO ({len(peak_days)} giorni):")
        for i, day_impact in enumerate(peak_days.head(3).items()):
            day_idx, impact = day_impact
            date = datetime.now() + timedelta(days=day_idx)
            print(f"      {date.strftime('%d/%m')}: {impact:+.0f}% ordini")
    
    if len(valley_days) > 0:
        print(f"\n   [DOWN] GIORNI BASSI ({len(valley_days)} giorni):")
        for i, day_impact in enumerate(valley_days.head(2).items()):
            day_idx, impact = day_impact
            date = datetime.now() + timedelta(days=day_idx)
            print(f"      {date.strftime('%d/%m')}: {impact:.0f}% ordini")
    
    # 9. Raccomandazioni operative
    print("\n[BULB] 9. Raccomandazioni Operative...")
    
    # Raccomandazioni standard
    for i, rec in enumerate(sensing_result.recommendations[:3], 1):
        print(f"   {i}. {rec}")
    
    # Raccomandazioni food-specific
    food_recs = []
    
    # Analizza meteo
    weather_factors = details['factors_by_source'].get('weather', [])
    if weather_factors:
        avg_weather = np.mean([f.impact for f in weather_factors])
        if avg_weather > 0.1:
            food_recs.append(
                "[RAIN] Meteo favorevole per delivery - aumentare staff e scorte"
            )
        elif avg_weather < -0.05:
            food_recs.append(
                "[SUN] Bel tempo previsto - focus su promozioni takeaway"
            )
    
    # Analizza eventi
    calendar_factors = details['factors_by_source'].get('calendar', [])
    if calendar_factors and np.mean([f.impact for f in calendar_factors]) > 0.1:
        food_recs.append(
            "[CALENDAR] Eventi importanti - preparare menu speciali e extra staff"
        )
    
    # Analizza social sentiment
    social_factors = details['factors_by_source'].get('social', [])
    if social_factors:
        avg_social = np.mean([f.impact for f in social_factors])
        if avg_social < -0.05:
            food_recs.append(
                "[SOCIAL] Sentiment negativo rilevato - intensificare customer care"
            )
        elif avg_social > 0.1:
            food_recs.append(
                "[SOCIAL] Buzz positivo - capitalizzare con campagne social"
            )
    
    if food_recs:
        print(f"\n   [PIZZA] Raccomandazioni Food-Specific:")
        for i, rec in enumerate(food_recs, 1):
            print(f"   {i}. {rec}")
    
    # 10. Dashboard operativo
    print("\n[CHART] 10. Generazione Dashboard Operativo...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Trend ordini + previsioni
    ax1 = axes[0, 0]
    recent_data = orders_data.tail(30)
    ax1.plot(range(-30, 0), recent_data['orders'].values, 
            'b-', label='Storico', linewidth=2)
    
    forecast_range = range(0, forecast_horizon)
    ax1.plot(forecast_range, sensing_result.original_forecast.values,
            'g-', label='SARIMA Base', linewidth=2)
    ax1.plot(forecast_range, sensing_result.adjusted_forecast.values,
            'r--', label='Con Demand Sensing', linewidth=2)
    
    ax1.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    ax1.set_title('Trend Ordini + Previsioni', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Giorni')
    ax1.set_ylabel('Ordini')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pattern settimanale
    ax2 = axes[0, 1]
    days_of_week = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    weekly_avg = []
    for dow in range(7):
        dow_data = orders_data[orders_data['date'].dt.dayofweek == dow]['orders']
        weekly_avg.append(dow_data.mean() if len(dow_data) > 0 else 0)
    
    bars = ax2.bar(days_of_week, weekly_avg, 
                  color=['lightcoral' if x in ['Sab', 'Dom'] else 'skyblue' 
                        for x in days_of_week])
    ax2.set_title('Pattern Settimanale Medio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Ordini Medi')
    
    # Aggiungi valori sopra barre
    for bar, val in zip(bars, weekly_avg):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Impatti giornalieri
    ax3 = axes[0, 2]
    colors = ['green' if x > 0 else 'red' if x < -5 else 'orange' 
              for x in adjustment_pct.values]
    bars = ax3.bar(forecast_range, adjustment_pct.values, 
                   color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('Aggiustamenti Giornalieri (%)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Giorni')
    ax3.set_ylabel('Aggiustamento %')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Revenue impact cumulativo
    ax4 = axes[1, 0]
    revenue_daily = (sensing_result.adjusted_forecast - 
                    sensing_result.original_forecast) * avg_order_value
    revenue_cumulative = revenue_daily.cumsum()
    
    ax4.plot(forecast_range, revenue_cumulative.values, 
            'purple', linewidth=3, marker='o', markersize=4)
    ax4.fill_between(forecast_range, 0, revenue_cumulative.values,
                    alpha=0.3, color='purple')
    ax4.set_title('Impatto Ricavi Cumulativo (€)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Giorni')
    ax4.set_ylabel('Euro (€)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 5: Heatmap sources contribution
    ax5 = axes[1, 1]
    if not contributions.empty:
        source_names = contributions['source'].str.capitalize().tolist()
        source_values = contributions['contribution_pct'].tolist()
        
        wedges, texts, autotexts = ax5.pie(
            source_values,
            labels=source_names,
            autopct='%1.1f%%',
            colors=['lightblue', 'lightgreen', 'salmon', 'gold', 'plum']
        )
        ax5.set_title('Contributi Fonti Esterne', fontweight='bold', fontsize=12)
    
    # Plot 6: Staff planning
    ax6 = axes[1, 2]
    base_staff = sensing_result.original_forecast.values / orders_per_staff
    adjusted_staff = sensing_result.adjusted_forecast.values / orders_per_staff
    
    ax6.plot(forecast_range, base_staff, 'b-', label='Staff Base', linewidth=2)
    ax6.plot(forecast_range, adjusted_staff, 'r--', 
            label='Staff Aggiustato', linewidth=2)
    ax6.fill_between(forecast_range, base_staff, adjusted_staff,
                    alpha=0.3, color='orange', label='Extra Staff')
    ax6.set_title('Pianificazione Staff (Rider)', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Giorni')
    ax6.set_ylabel('Rider Necessari')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('food_delivery_dashboard.png', dpi=150, bbox_inches='tight')
    print("   [CHART] Dashboard salvato: 'food_delivery_dashboard.png'")
    
    # 11. Export planning operativo
    print("\n[CLIPBOARD] 11. Export Planning Operativo...")
    
    # Report giornaliero per operations
    daily_plan = pd.DataFrame({
        'Data': pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='D'),
        'Giorno': pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='D').strftime('%A'),
        'Ordini_Base': sensing_result.original_forecast.values.astype(int),
        'Ordini_Previsti': sensing_result.adjusted_forecast.values.astype(int),
        'Aggiustamento_Pct': adjustment_pct.values.round(1),
        'Rider_Necessari': np.ceil(sensing_result.adjusted_forecast.values / orders_per_staff).astype(int),
        'Ricavi_Stimati': (sensing_result.adjusted_forecast.values * avg_order_value).astype(int),
        'Alert_Level': ['[FIRE] ALTO' if x > 20 else '[WARN] MEDIO' if x > 10 else '[OK] NORMALE' 
                       for x in adjustment_pct.values]
    })
    
    daily_plan.to_csv('food_delivery_planning.csv', index=False)
    print("   [CALENDAR] Planning giornaliero: 'food_delivery_planning.csv'")
    
    # Summary business
    business_summary = {
        'Ristorante': restaurant_name,
        'Periodo_Analisi': f"{forecast_horizon} giorni",
        'Ordini_Totali_Base': int(total_orders_base),
        'Ordini_Totali_Previsti': int(total_orders_adjusted),
        'Impatto_Ricavi_EUR': int(revenue_impact),
        'Staff_Extra_Giorni': int(high_demand_days),
        'Costo_Staff_Extra_EUR': int(staff_cost),
        'Profitto_Netto_EUR': int(revenue_impact - staff_cost),
        'ROI_Demand_Sensing': f"{((revenue_impact - staff_cost) / 1000):.1f}x",
        'Giorni_Picco': int(len(peak_days)),
        'Giorni_Bassi': int(len(valley_days)),
        'Top_Fonte_Impatto': contributions.iloc[0]['source'] if not contributions.empty else 'N/A'
    }
    
    summary_df = pd.DataFrame([business_summary])
    summary_df.to_csv('food_delivery_summary.csv', index=False)
    print("   [BUSINESS] Summary business: 'food_delivery_summary.csv'")
    
    print("\n" + "="*50)
    print("[PARTY] ANALISI FOOD DELIVERY COMPLETATA!")
    
    print(f"\n[CHART] RISULTATI CHIAVE:")
    print(f"• Impatto ordini: {order_impact:+.0f} ordini ({sensing_result.total_adjustment:+.1%})")
    print(f"• Impatto ricavi: {revenue_impact:+,.0f} €")
    print(f"• Profitto netto: {revenue_impact - staff_cost:+,.0f} €")
    print(f"• Giorni ad alta domanda: {high_demand_days}")
    print(f"• Staff extra necessario: {additional_staff:.1f} rider medi")
    
    print(f"\n[TARGET] TOP INSIGHTS:")
    if contributions.iloc[0]['source'] == 'weather':
        print(f"• Meteo è il fattore #1 ({contributions.iloc[0]['contribution_pct']:.1f}%)")
    if len(peak_days) > 0:
        print(f"• {len(peak_days)} giorni con boost >20% - preparare scorte extra")
    print(f"• Confidenza previsioni: {sensing_result.confidence_score:.0%}")
    
    print(f"\n[FOLDER] FILE GENERATI:")
    print(f"• food_delivery_dashboard.png")
    print(f"• food_delivery_planning.csv (planning giornaliero)")
    print(f"• food_delivery_summary.csv (KPI business)")


if __name__ == "__main__":
    main()