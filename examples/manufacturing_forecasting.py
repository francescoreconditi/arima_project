#!/usr/bin/env python3
"""
Manufacturing Production Forecasting Example

Questo esempio dimostra l'applicazione di ARIMA per il forecasting della produzione
industriale con downtime programmati e non programmati, cicli di produzione,
variazioni di efficienza e pattern di manutenzione.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.core import ARIMAModelSelector
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils import setup_logger
from utils import get_plots_path, get_models_path, get_reports_path

warnings.filterwarnings('ignore')

def generate_manufacturing_data():
    """Genera dati produzione manifatturiera con downtime e cicli realistici"""
    np.random.seed(42)
    
    # Periodo: 90 giorni di dati orari (2160 punti)
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=90)
    timestamps = pd.date_range(start_date, end_date, freq='h')[:-1]
    n_points = len(timestamps)
    
    # Produzione base teorica (unità/ora)
    base_production = 100.0
    
    # 1. Pattern operativi - 3 turni giornalieri con efficienza diversa
    shift_patterns = np.zeros(n_points)
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if 6 <= hour < 14:  # Turno mattino (più efficiente)
            shift_patterns[i] = 1.0
        elif 14 <= hour < 22:  # Turno pomeriggio
            shift_patterns[i] = 0.95
        elif 22 <= hour < 6:  # Turno notte (meno efficiente)
            shift_patterns[i] = 0.85
        else:
            shift_patterns[i] = 0.9
    
    # 2. Pattern settimanali (produzione ridotta weekend)
    weekend_reduction = np.array([
        0.3 if ts.weekday() >= 5 else 1.0 
        for ts in timestamps
    ])
    
    # 3. Downtime programmati (manutenzione settimanale)
    planned_downtime = np.ones(n_points)
    for week in range(13):  # 13 settimane circa
        # Manutenzione ogni domenica 2-6 AM
        maintenance_start = week * 168 + 6 * 24 + 2  # Domenica 2 AM
        if maintenance_start < n_points:
            maintenance_end = min(maintenance_start + 4, n_points)  # 4 ore
            planned_downtime[maintenance_start:maintenance_end] = 0.0
    
    # 4. Downtime non programmati (guasti casuali)
    unplanned_downtime = np.ones(n_points)
    failure_probability = 0.001  # 0.1% probability per hour
    
    failure_events = np.random.random(n_points) < failure_probability
    for i, failure in enumerate(failure_events):
        if failure:
            # Durata guasto variabile (1-12 ore)
            failure_duration = np.random.choice([1, 2, 3, 4, 6, 8, 12], 
                                              p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
            end_failure = min(i + failure_duration, n_points)
            unplanned_downtime[i:end_failure] = 0.0
    
    # 5. Efficienza linea (degrada nel tempo, migliora dopo manutenzione)
    line_efficiency = np.ones(n_points)
    efficiency_degradation = 0.0001  # Degrado per ora
    
    for i in range(1, n_points):
        # Degrado continuo
        line_efficiency[i] = max(0.7, line_efficiency[i-1] - efficiency_degradation)
        
        # Reset efficienza dopo manutenzione programmata
        if planned_downtime[i-1] == 0.0 and planned_downtime[i] == 1.0:
            line_efficiency[i] = np.random.uniform(0.95, 1.0)
    
    # 6. Variazioni demand-driven (ordini speciali)
    demand_spikes = np.ones(n_points)
    special_orders = np.random.choice(n_points, size=15, replace=False)
    for order_start in special_orders:
        duration = np.random.randint(8, 48)  # 8-48 ore
        order_end = min(order_start + duration, n_points)
        boost_factor = np.random.uniform(1.3, 1.8)
        demand_spikes[order_start:order_end] = boost_factor
    
    # 7. Seasonal effects (variazioni mensili domanda)
    monthly_seasonal = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 30))
    
    # 8. Raw material availability (shortage occasionali)
    material_availability = np.ones(n_points)
    shortage_events = np.random.choice(n_points, size=8, replace=False)
    for shortage_start in shortage_events:
        duration = np.random.randint(4, 24)  # 4-24 ore
        shortage_end = min(shortage_start + duration, n_points)
        reduction_factor = np.random.uniform(0.4, 0.8)
        material_availability[shortage_start:shortage_end] = reduction_factor
    
    # 9. Quality issues (riduzioni produzione per problemi qualità)
    quality_issues = np.ones(n_points)
    quality_events = np.random.choice(n_points, size=12, replace=False)
    for quality_start in quality_events:
        duration = np.random.randint(2, 8)  # 2-8 ore
        quality_end = min(quality_start + duration, n_points)
        quality_reduction = np.random.uniform(0.6, 0.9)
        quality_issues[quality_start:quality_end] = quality_reduction
    
    # 10. Random noise operazionale
    operational_noise = np.random.normal(1.0, 0.05, n_points)
    
    # Calcola produzione finale
    production = (base_production * shift_patterns * weekend_reduction * 
                 planned_downtime * unplanned_downtime * line_efficiency *
                 demand_spikes * monthly_seasonal * material_availability *
                 quality_issues * operational_noise)
    
    # Assicurati che non sia negativa
    production = np.maximum(production, 0)
    
    # Informazioni aggiuntive per analisi
    metadata = pd.DataFrame({
        'shift_efficiency': shift_patterns,
        'weekend_factor': weekend_reduction,
        'planned_downtime': planned_downtime,
        'unplanned_downtime': unplanned_downtime,
        'line_efficiency': line_efficiency,
        'demand_factor': demand_spikes,
        'material_availability': material_availability,
        'quality_factor': quality_issues
    }, index=timestamps)
    
    production_series = pd.Series(production, index=timestamps, name='production_units_per_hour')
    
    return production_series, metadata

def calculate_manufacturing_kpis(production, metadata):
    """Calcola KPI manifatturieri standard"""
    
    # Overall Equipment Effectiveness (OEE) components
    availability = (metadata['planned_downtime'] * metadata['unplanned_downtime']).mean()
    performance = metadata['line_efficiency'].mean()
    quality = metadata['quality_factor'].mean()
    oee = availability * performance * quality
    
    # Utilization metrics
    theoretical_max = 100 * len(production)  # Max possible production
    actual_production = production.sum()
    utilization = actual_production / theoretical_max
    
    # Downtime analysis
    planned_downtime_hours = (1 - metadata['planned_downtime']).sum()
    unplanned_downtime_hours = (1 - metadata['unplanned_downtime']).sum()
    total_downtime_hours = planned_downtime_hours + unplanned_downtime_hours
    
    # Shift performance
    shift_performance = {}
    for shift_name, hours in [('Morning', (6, 14)), ('Afternoon', (14, 22)), ('Night', (22, 6))]:
        if shift_name == 'Night':
            mask = (production.index.hour >= hours[0]) | (production.index.hour < hours[1])
        else:
            mask = (production.index.hour >= hours[0]) & (production.index.hour < hours[1])
        
        shift_production = production[mask].mean()
        shift_performance[shift_name] = shift_production
    
    kpis = {
        'oee': oee,
        'availability': availability,
        'performance': performance,
        'quality': quality,
        'utilization': utilization,
        'planned_downtime_hours': planned_downtime_hours,
        'unplanned_downtime_hours': unplanned_downtime_hours,
        'total_downtime_hours': total_downtime_hours,
        'mtbf': len(production) / max(1, (1 - metadata['unplanned_downtime']).sum()),  # Mean Time Between Failures
        'mttr': unplanned_downtime_hours / max(1, len(metadata[metadata['unplanned_downtime'] < 1])),  # Mean Time To Repair
        'shift_performance': shift_performance
    }
    
    return kpis

def main():
    logger = setup_logger('manufacturing_forecasting', level='INFO')
    logger.info("🏭 Avvio analisi forecasting produzione manifatturiera")
    
    # Genera dati
    logger.info("🔧 Generazione dati produzione con downtime...")
    production_data, metadata = generate_manufacturing_data()
    
    # Calcola KPI
    kpis = calculate_manufacturing_kpis(production_data, metadata)
    
    print(f"📊 Dataset generato: {len(production_data)} ore di produzione")
    print(f"📅 Periodo: {production_data.index[0].strftime('%Y-%m-%d %H:%M')} - {production_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"🏭 Produzione totale: {production_data.sum():,.0f} unità")
    print(f"📈 Produzione media: {production_data.mean():.1f} unità/ora")
    print(f"⚡ Picco produzione: {production_data.max():.1f} unità/ora")
    
    print(f"\n📊 KPI Manifatturieri:")
    print(f"  🎯 OEE (Overall Equipment Effectiveness): {kpis['oee']:.1%}")
    print(f"    📍 Availability: {kpis['availability']:.1%}")
    print(f"    📍 Performance: {kpis['performance']:.1%}")
    print(f"    📍 Quality: {kpis['quality']:.1%}")
    print(f"  📊 Utilization: {kpis['utilization']:.1%}")
    print(f"  ⏰ Downtime programmato: {kpis['planned_downtime_hours']:.0f}h")
    print(f"  🚨 Downtime non programmato: {kpis['unplanned_downtime_hours']:.0f}h")
    print(f"  📈 MTBF: {kpis['mtbf']:.0f}h")
    print(f"  🔧 MTTR: {kpis['mttr']:.1f}h")
    
    print(f"\n👥 Performance per turno:")
    for shift, performance in kpis['shift_performance'].items():
        print(f"  {shift}: {performance:.1f} unità/ora")
    
    # Split train/test
    train_size = int(len(production_data) * 0.8)
    train_data = production_data[:train_size]
    test_data = production_data[train_size:]
    
    print(f"\n🔄 Split dataset:")
    print(f"  📚 Training: {len(train_data)} ore ({train_data.index[0].strftime('%Y-%m-%d')} - {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"  🧪 Test: {len(test_data)} ore ({test_data.index[0].strftime('%Y-%m-%d')} - {test_data.index[-1].strftime('%Y-%m-%d')})")
    
    # Preprocessing specifico per manufacturing
    logger.info("🔧 Preprocessing dati produzione...")
    preprocessor = TimeSeriesPreprocessor()
    
    # Gestione degli zeri (downtime)
    zero_production = (train_data == 0).sum()
    print(f"🔍 Ore con produzione zero: {zero_production} ({zero_production/len(train_data)*100:.1f}%)")
    
    # Per modelling, sostituisci 0 con piccolo valore per evitare problemi log
    train_adjusted = train_data.replace(0, 0.1)
    
    # Check stazionarietà
    stationarity_result = preprocessor.check_stationarity(train_adjusted)
    is_stationary = stationarity_result['is_stationary']
    if not is_stationary:
        print("📈 Serie non stazionaria - il modello userà differenziazione")
    
    # Selezione automatica modello per manufacturing
    logger.info("🔍 Selezione automatica modello ARIMA per produzione...")
    # Use simple ARIMA model for manufacturing data
    print("Utilizzo modello ARIMA(2,1,2) per dati produzione...")
    best_order = (2, 1, 2)
    seasonal_order = None
    
    print(f"\nModello selezionato:")
    print(f"  ARIMA{best_order}")
    
    # Training modello
    logger.info("🎯 Training modello ARIMA per produzione...")
    model = ARIMAForecaster(order=best_order)
    model.fit(train_adjusted)
    
    # Forecast
    forecast_steps = len(test_data)
    logger.info(f"🔮 Generazione forecast per {forecast_steps} ore ({forecast_steps/24:.1f} giorni)...")
    forecast_result = model.forecast(
        steps=forecast_steps, 
        confidence_intervals=True,
        alpha=0.05
    )
    
    # Aggiusta forecast per downtime conosciuti nel test period
    test_metadata = metadata.iloc[train_size:train_size+len(test_data)]
    known_planned_downtime = test_metadata['planned_downtime'] == 0
    
    forecast_adjusted = forecast_result['forecast'].copy()
    forecast_adjusted[known_planned_downtime] = 0
    
    print(f"🔧 Aggiustato forecast per {known_planned_downtime.sum()} ore downtime programmato")
    
    # Valutazione
    logger.info("📊 Valutazione performance modello...")
    evaluator = ModelEvaluator()
    
    # Valuta sia forecast originale che adjusted
    metrics_original = evaluator.calculate_forecast_metrics(test_data, forecast_result['forecast'])
    metrics_adjusted = evaluator.calculate_forecast_metrics(test_data, forecast_adjusted)
    
    print(f"\n📊 Metriche Performance (Forecast Originale):")
    print(f"  📈 MAPE: {metrics_original['mape']:.2f}%")
    print(f"  📉 MAE: {metrics_original['mae']:.1f} unità/ora")
    print(f"  🎯 RMSE: {metrics_original['rmse']:.1f} unità/ora")
    
    # Check for R² score with different possible key names
    if 'r2_score' in metrics_original:
        print(f"  📊 R²: {metrics_original['r2_score']:.3f}")
    elif 'r_squared' in metrics_original:
        print(f"  📊 R²: {metrics_original['r_squared']:.3f}")
    else:
        print(f"  📊 R²: N/A")
    
    print(f"\n📊 Metriche Performance (Forecast Adjusted):")
    print(f"  📈 MAPE: {metrics_adjusted['mape']:.2f}%")
    print(f"  📉 MAE: {metrics_adjusted['mae']:.1f} unità/ora")
    print(f"  🎯 RMSE: {metrics_adjusted['rmse']:.1f} unità/ora")
    
    # Check for R² score with different possible key names
    if 'r2_score' in metrics_adjusted:
        print(f"  📊 R²: {metrics_adjusted['r2_score']:.3f}")
    elif 'r_squared' in metrics_adjusted:
        print(f"  📊 R²: {metrics_adjusted['r_squared']:.3f}")
    else:
        print(f"  📊 R²: N/A")
    
    # Manufacturing-specific metrics
    forecast_production = forecast_adjusted.sum()
    actual_production = test_data.sum()
    production_accuracy = 1 - abs(forecast_production - actual_production) / actual_production
    print(f"  🏭 Production Volume Accuracy: {production_accuracy:.1%}")
    
    # Forecast operazionale futuro (prossimi 7 giorni)
    future_steps = 24 * 7  # 7 giorni
    logger.info("🚀 Forecast operazionale prossimi 7 giorni...")
    future_forecast = model.forecast(steps=future_steps, confidence_intervals=True)
    
    # Visualizzazione specializzata manufacturing
    fig = plt.figure(figsize=(20, 16))
    
    # Layout grid 3x3
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Overview produzione con downtime
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Colora background per downtime
    downtime_periods = metadata['planned_downtime'] == 0
    ax1.fill_between(production_data.index, 0, production_data.max() * 1.1,
                     where=downtime_periods, alpha=0.2, color='red', 
                     label='Planned Downtime')
    
    ax1.plot(train_data.index, train_data, label='Training', color='blue', alpha=0.7, linewidth=0.8)
    ax1.plot(test_data.index, test_data, label='Test Actual', color='green', alpha=0.8, linewidth=1)
    ax1.plot(test_data.index, forecast_adjusted, 
             label='Forecast (Adjusted)', color='red', linestyle='--', linewidth=2)
    
    ax1.fill_between(test_data.index,
                     forecast_result['confidence_intervals']['lower'],
                     forecast_result['confidence_intervals']['upper'],
                     color='red', alpha=0.2, label='95% CI')
    
    ax1.set_title('🏭 Manufacturing Production Forecast with Downtime')
    ax1.set_ylabel('Production (units/hour)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: OEE Components nel tempo
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Rolling OEE calculation
    window = 24 * 7  # 1 week
    rolling_availability = (metadata['planned_downtime'] * metadata['unplanned_downtime']).rolling(window).mean()
    rolling_performance = metadata['line_efficiency'].rolling(window).mean()
    rolling_quality = metadata['quality_factor'].rolling(window).mean()
    
    ax2.plot(rolling_availability.index, rolling_availability, label='Availability', alpha=0.8)
    ax2.plot(rolling_performance.index, rolling_performance, label='Performance', alpha=0.8)
    ax2.plot(rolling_quality.index, rolling_quality, label='Quality', alpha=0.8)
    
    ax2.set_title('📊 OEE Components (7-day rolling)')
    ax2.set_ylabel('Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Hourly production pattern
    ax3 = fig.add_subplot(gs[1, 0])
    
    hourly_avg = train_data.groupby(train_data.index.hour).mean()
    hourly_std = train_data.groupby(train_data.index.hour).std()
    
    ax3.bar(hourly_avg.index, hourly_avg.values, 
            yerr=hourly_std.values, alpha=0.7, capsize=3)
    ax3.axhline(y=hourly_avg.mean(), color='red', linestyle='--', alpha=0.8, label='Daily Average')
    
    # Evidenzia turni
    ax3.axvspan(6, 14, alpha=0.2, color='green', label='Morning Shift')
    ax3.axvspan(14, 22, alpha=0.2, color='orange', label='Afternoon Shift')
    ax3.axvspan(22, 24, alpha=0.2, color='blue', label='Night Shift')
    ax3.axvspan(0, 6, alpha=0.2, color='blue')
    
    ax3.set_title('⏰ Hourly Production Pattern')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Avg Production')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weekly pattern
    ax4 = fig.add_subplot(gs[1, 1])
    
    weekly_avg = train_data.groupby(train_data.index.dayofweek).mean()
    weekly_std = train_data.groupby(train_data.index.dayofweek).std()
    
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    ax4.bar(range(7), weekly_avg.values, 
            yerr=weekly_std.values, alpha=0.7, capsize=3,
            tick_label=weekdays)
    ax4.axhline(y=weekly_avg.mean(), color='red', linestyle='--', alpha=0.8, label='Weekly Average')
    
    ax4.set_title('📅 Weekly Production Pattern')
    ax4.set_ylabel('Avg Production')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Downtime analysis
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Downtime events duration distribution
    downtime_events = []
    in_downtime = False
    start_downtime = None
    
    for i, is_down in enumerate(metadata['unplanned_downtime'] == 0):
        if is_down and not in_downtime:
            in_downtime = True
            start_downtime = i
        elif not is_down and in_downtime:
            in_downtime = False
            downtime_events.append(i - start_downtime)
    
    if downtime_events:
        ax5.hist(downtime_events, bins=10, alpha=0.7, color='red')
        ax5.axvline(x=np.mean(downtime_events), color='blue', 
                   linestyle='--', label=f'Mean: {np.mean(downtime_events):.1f}h')
    
    ax5.set_title('🚨 Unplanned Downtime Duration')
    ax5.set_xlabel('Duration (hours)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Efficiency trends
    ax6 = fig.add_subplot(gs[2, 0])
    
    efficiency_trend = metadata['line_efficiency'].rolling(24).mean()  # Daily rolling
    
    ax6.plot(efficiency_trend.index, efficiency_trend, 
             color='purple', linewidth=2, label='Line Efficiency (24h)')
    
    # Evidenzia periodi manutenzione
    maintenance_periods = metadata['planned_downtime'] == 0
    ax6.fill_between(metadata.index, 0.6, 1.0, 
                     where=maintenance_periods, alpha=0.3, color='green',
                     label='Maintenance Periods')
    
    ax6.set_title('📈 Line Efficiency Trends')
    ax6.set_ylabel('Efficiency Factor')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    # Plot 7: Future forecast operazionale
    ax7 = fig.add_subplot(gs[2, 1])
    
    # Ultimi 7 giorni per contesto
    recent_data = production_data[-168:]  # 7 giorni * 24 ore
    ax7.plot(recent_data.index, recent_data, 
             label='Recent Data', color='blue', alpha=0.7)
    
    # Future forecast
    future_dates = pd.date_range(
        start=production_data.index[-1] + pd.Timedelta(hours=1),
        periods=future_steps, 
        freq='h'
    )
    
    ax7.plot(future_dates, future_forecast['forecast'], 
             's-', label='7-Day Forecast', color='purple', 
             linewidth=2, markersize=2, alpha=0.8)
    
    ax7.fill_between(future_dates,
                     future_forecast['confidence_intervals']['lower'],
                     future_forecast['confidence_intervals']['upper'],
                     color='purple', alpha=0.2, label='95% CI')
    
    ax7.set_title('🚀 Next 7 Days Production Forecast')
    ax7.set_ylabel('Production (units/hour)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', rotation=45)
    
    # Plot 8: Residuals con cause analysis
    ax8 = fig.add_subplot(gs[2, 2])
    
    residuals = test_data - forecast_adjusted
    ax8.plot(test_data.index, residuals, 'o-', 
             color='orange', alpha=0.7, markersize=2)
    
    # Identifica residui alti e loro cause
    high_residuals = np.abs(residuals) > 2 * residuals.std()
    test_meta = metadata.iloc[train_size:train_size+len(test_data)]
    
    # Colora residui per cause
    unplanned_down_mask = test_meta['unplanned_downtime'] < 1
    material_shortage_mask = test_meta['material_availability'] < 0.9
    quality_issue_mask = test_meta['quality_factor'] < 0.9
    
    ax8.scatter(test_data.index[unplanned_down_mask], 
                residuals[unplanned_down_mask],
                color='red', s=20, alpha=0.8, label='Unplanned Downtime')
    ax8.scatter(test_data.index[material_shortage_mask], 
                residuals[material_shortage_mask],
                color='brown', s=20, alpha=0.8, label='Material Shortage')
    ax8.scatter(test_data.index[quality_issue_mask], 
                residuals[quality_issue_mask],
                color='yellow', s=20, alpha=0.8, label='Quality Issues')
    
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax8.axhline(y=residuals.std(), color='red', linestyle='--', alpha=0.5, label='+1σ')
    ax8.axhline(y=-residuals.std(), color='red', linestyle='--', alpha=0.5, label='-1σ')
    
    ax8.set_title('📉 Forecast Residuals by Cause')
    ax8.set_ylabel('Residuals (units/hour)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='x', rotation=45)
    
    # Salva plot
    plt.savefig(get_plots_path('manufacturing_forecast.png'), dpi=300, bbox_inches='tight')
    logger.info("📁 Plot salvato in outputs/plots/manufacturing_forecast.png")
    
    # plt.show()  # Disabled for Windows compatibility
    print("Plot saved as 'outputs/plots/manufacturing_forecast.png'")
    
    # Manufacturing Operational Insights
    print(f"\n🏭 Manufacturing Operational Insights:")
    
    # Forecast futuro con insights
    future_production_total = future_forecast['forecast'].sum()
    current_week_total = production_data[-168:].sum()  # Last week
    
    print(f"  📊 Produzione prevista prossimi 7 giorni: {future_production_total:,.0f} unità")
    print(f"  📈 vs settimana precedente: {((future_production_total - current_week_total) / current_week_total * 100):+.1f}%")
    
    # Capacity utilization forecast
    theoretical_capacity = 100 * future_steps  # Max theoretical production
    forecast_utilization = future_production_total / theoretical_capacity
    print(f"  🎯 Utilizzo capacità previsto: {forecast_utilization:.1%}")
    
    # Production planning insights
    daily_forecasts = []
    for day in range(7):
        day_start = day * 24
        day_end = (day + 1) * 24
        daily_production = future_forecast['forecast'][day_start:day_end].sum()
        daily_forecasts.append(daily_production)
    
    best_day = np.argmax(daily_forecasts)
    worst_day = np.argmin(daily_forecasts)
    
    print(f"  📊 Migliore giorno previsto: Giorno {best_day + 1} ({daily_forecasts[best_day]:,.0f} unità)")
    print(f"  📊 Giorno più critico: Giorno {worst_day + 1} ({daily_forecasts[worst_day]:,.0f} unità)")
    
    # Maintenance recommendations
    current_efficiency = metadata['line_efficiency'].iloc[-168:].mean()  # Last week
    efficiency_trend = metadata['line_efficiency'].iloc[-168:].mean() - metadata['line_efficiency'].iloc[-336:-168].mean()
    
    print(f"  🔧 Efficienza attuale: {current_efficiency:.1%}")
    print(f"  📈 Trend efficienza: {efficiency_trend:+.1%} (settimana vs precedente)")
    
    if current_efficiency < 0.85:
        print("  🚨 RACCOMANDAZIONE: Pianificare manutenzione straordinaria")
    elif efficiency_trend < -0.02:
        print("  ⚠️  ALERT: Degrado efficienza rilevato - monitorare")
    else:
        print("  ✅ Efficienza linea nominale")
    
    # Supply chain insights
    material_issues = (metadata['material_availability'] < 0.9).sum()
    if material_issues > 0:
        print(f"  📦 Eventi shortage materiali: {material_issues} ore nell'ultimo periodo")
        print("  🔄 RACCOMANDAZIONE: Rivedere strategia inventory management")
    
    # Quality insights  
    quality_issues = (metadata['quality_factor'] < 0.9).sum()
    if quality_issues > 0:
        print(f"  🎯 Eventi problemi qualità: {quality_issues} ore")
        print("  🔍 RACCOMANDAZIONE: Analisi root cause qualità")
    
    # Salva modello
    model_path = get_models_path('manufacturing_arima_model.joblib')
    model.save(model_path)
    logger.info(f"💾 Modello salvato in {model_path}")

    # Genera report Quarto
    logger.info("Generazione report Quarto...")
    try:
        # Get the plot filename if it exists
        plot_files = {}
        # Try to find the most recent plot file
        if 'plot_path' in locals():
            plot_files['main_plot'] = str(plot_path)
        elif 'plt' in locals():
            # If we have a matplotlib figure, save it temporarily
            temp_plot = get_plots_path('temp_report_plot.png')
            plt.savefig(temp_plot, dpi=300, bbox_inches='tight')
            plot_files['analysis_plot'] = str(temp_plot)
        
        report_path = model.generate_report(
            plots_data=plot_files if plot_files else None,
            report_title="Manufacturing Forecasting Analysis",
            output_filename="manufacturing_forecasting_report",
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
    
    print(f"\n✅ Analisi produzione completata!")
    print(f"📁 Risultati e KPI salvati in outputs/")
    print(f"🏭 Modello pronto per integrazione con MES/ERP systems")

if __name__ == "__main__":
    main()