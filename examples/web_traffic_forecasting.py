#!/usr/bin/env python3
"""
Web Traffic Forecasting Example

Questo esempio dimostra l'applicazione di ARIMA per il forecasting del traffico web
con pattern complessi: stagionalità multiple (giornaliera, settimanale, mensile),
eventi speciali, campagne marketing, e trend stagionali tipici dei siti web.
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

warnings.filterwarnings('ignore')

def generate_web_traffic_data():
    """Genera dati traffico web con pattern realistici e stagionalità multiple"""
    np.random.seed(42)
    
    # Periodo: 1 anno di dati orari (8760 punti)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31, 23, 0)
    timestamps = pd.date_range(start_date, end_date, freq='h')
    n_points = len(timestamps)
    
    # Traffico base
    base_traffic = 1000
    
    # 1. Pattern giornaliero - traffico più alto durante ore lavorative
    daily_pattern = np.zeros(n_points)
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if 8 <= hour <= 12:  # Mattina picco
            daily_pattern[i] = 1.8
        elif 13 <= hour <= 17:  # Pomeriggio picco
            daily_pattern[i] = 2.0
        elif 18 <= hour <= 22:  # Sera moderato
            daily_pattern[i] = 1.5
        elif 23 <= hour <= 24 or 0 <= hour <= 7:  # Notte basso
            daily_pattern[i] = 0.4
        else:
            daily_pattern[i] = 1.0
    
    # 2. Pattern settimanale - più traffico durante weekdays
    weekly_pattern = np.array([
        1.2 if ts.weekday() < 5 else 0.7  # Weekday vs Weekend
        for ts in timestamps
    ])
    
    # 3. Pattern mensile - stagionalità business
    monthly_seasonal = np.array([
        1.3 if ts.month in [1, 9, 10] else  # New Year, Back to school, Q4
        1.1 if ts.month in [3, 4, 11] else  # Q1 end, Q2 start, Black Friday
        0.8 if ts.month in [7, 8, 12] else  # Summer, holidays
        1.0  # Altri mesi
        for ts in timestamps
    ])
    
    # 4. Trend annuale (crescita organica)
    annual_trend = 1 + 0.3 * np.arange(n_points) / n_points  # 30% crescita annuale
    
    # 5. Eventi speciali e campagne marketing
    special_events = np.ones(n_points)
    
    # Black Friday (ultimo venerdì novembre)
    black_friday = datetime(2023, 11, 24)  # Esempio
    bf_start = (black_friday - start_date).days * 24 + (black_friday - start_date).seconds // 3600
    if bf_start < n_points:
        bf_end = min(bf_start + 72, n_points)  # 3 giorni evento
        special_events[bf_start:bf_end] = np.random.uniform(3.0, 5.0, bf_end - bf_start)
    
    # Campagne marketing mensili (primo lunedì di ogni mese)
    for month in range(1, 13):
        # Trova primo lunedì del mese
        first_day = datetime(2023, month, 1)
        days_until_monday = (7 - first_day.weekday()) % 7
        if first_day.weekday() != 0:  # Se non è lunedì
            days_until_monday = (7 - first_day.weekday()) % 7
        campaign_date = first_day + timedelta(days=days_until_monday)
        
        if campaign_date.year == 2023:
            campaign_start = (campaign_date - start_date).days * 24
            if 0 <= campaign_start < n_points:
                campaign_end = min(campaign_start + 168, n_points)  # 1 settimana
                boost = np.random.uniform(1.5, 2.5)
                special_events[campaign_start:campaign_end] *= boost
    
    # Viral content spikes (casuali)
    viral_events = np.random.choice(n_points, size=8, replace=False)
    for viral_start in viral_events:
        viral_duration = np.random.randint(6, 48)  # 6-48 ore
        viral_end = min(viral_start + viral_duration, n_points)
        viral_boost = np.random.uniform(2.0, 8.0)
        special_events[viral_start:viral_end] *= viral_boost
    
    # 6. Effetti SEO/Algorithm changes
    algo_changes = np.ones(n_points)
    
    # Simulazione Google algorithm update (impatto negativo)
    algo_update_dates = [
        datetime(2023, 3, 15),  # Core update marzo
        datetime(2023, 8, 22),  # Core update agosto
        datetime(2023, 11, 2)   # Core update novembre
    ]
    
    for update_date in algo_update_dates:
        update_start = (update_date - start_date).days * 24
        if 0 <= update_start < n_points:
            # Impatto graduale che si stabilizza
            recovery_period = 30 * 24  # 30 giorni per recovery
            update_end = min(update_start + recovery_period, n_points)
            
            initial_drop = np.random.uniform(0.6, 0.8)
            recovery_curve = np.linspace(initial_drop, np.random.uniform(0.9, 1.1), update_end - update_start)
            algo_changes[update_start:update_end] *= recovery_curve
    
    # 7. Device/Browser trends 
    mobile_trend = 1 + 0.1 * np.arange(n_points) / n_points  # Mobile traffic increasing
    
    # 8. Weather effects (esempio: più traffico quando piove - indoor activities)
    weather_effects = np.ones(n_points)
    # Simulazione giorni di pioggia (aumenta traffico)
    rainy_days = np.random.choice(365, size=50, replace=False)
    for day in rainy_days:
        day_start = day * 24
        if day_start < n_points:
            day_end = min(day_start + 24, n_points)
            weather_effects[day_start:day_end] *= np.random.uniform(1.1, 1.3)
    
    # 9. Referral traffic spikes
    referral_spikes = np.ones(n_points)
    referral_events = np.random.choice(n_points, size=20, replace=False)
    for ref_start in referral_events:
        ref_duration = np.random.randint(2, 12)  # 2-12 ore
        ref_end = min(ref_start + ref_duration, n_points)
        referral_boost = np.random.uniform(1.2, 2.0)
        referral_spikes[ref_start:ref_end] *= referral_boost
    
    # 10. Server downtime/technical issues
    downtime_events = np.ones(n_points)
    downtime_occurrences = np.random.choice(n_points, size=5, replace=False)
    for downtime_start in downtime_occurrences:
        downtime_duration = np.random.randint(1, 6)  # 1-6 ore
        downtime_end = min(downtime_start + downtime_duration, n_points)
        downtime_events[downtime_start:downtime_end] = np.random.uniform(0.1, 0.3)
    
    # 11. Rumore random
    noise_factor = np.random.lognormal(0, 0.1, n_points)  # Log-normal per positive skew
    
    # Combina tutti i fattori
    traffic = (base_traffic * daily_pattern * weekly_pattern * monthly_seasonal * 
              annual_trend * special_events * algo_changes * mobile_trend * 
              weather_effects * referral_spikes * downtime_events * noise_factor)
    
    # Assicurati che non sia negativo
    traffic = np.maximum(traffic, 50)  # Minimo 50 visitatori/ora anche durante downtime
    
    # Crea metadata per analisi
    metadata = pd.DataFrame({
        'daily_factor': daily_pattern,
        'weekly_factor': weekly_pattern,
        'monthly_factor': monthly_seasonal,
        'trend': annual_trend,
        'special_events': special_events,
        'algo_impact': algo_changes,
        'mobile_trend': mobile_trend,
        'weather_boost': weather_effects,
        'referral_boost': referral_spikes,
        'downtime_factor': downtime_events,
        'is_weekend': weekly_pattern < 1.0,
        'is_business_hours': (timestamps.hour >= 8) & (timestamps.hour <= 17)
    }, index=timestamps)
    
    traffic_series = pd.Series(traffic, index=timestamps, name='hourly_visitors')
    
    return traffic_series, metadata

def calculate_web_metrics(traffic, metadata):
    """Calcola metriche web analytics standard"""
    
    # Traffico totale
    total_visitors = traffic.sum()
    daily_avg = traffic.resample('D').sum().mean()
    peak_hour_avg = traffic.groupby(traffic.index.hour).mean().max()
    
    # Pattern analysis
    weekday_traffic = traffic[~metadata['is_weekend']].mean()
    weekend_traffic = traffic[metadata['is_weekend']].mean()
    business_hours_traffic = traffic[metadata['is_business_hours']].mean()
    
    # Growth metrics
    first_month = traffic.resample('M').sum().iloc[0]
    last_month = traffic.resample('M').sum().iloc[-1]
    monthly_growth = ((last_month - first_month) / first_month) * 100 / 11  # Per month
    
    # Volatility
    daily_totals = traffic.resample('D').sum()
    traffic_volatility = daily_totals.std() / daily_totals.mean()
    
    # Event impact analysis
    special_event_impact = metadata['special_events'][metadata['special_events'] > 1.5].mean()
    downtime_impact = (1 - metadata['downtime_factor'][metadata['downtime_factor'] < 1]).mean()
    
    metrics = {
        'total_visitors': total_visitors,
        'daily_average': daily_avg,
        'peak_hour_average': peak_hour_avg,
        'weekday_vs_weekend_ratio': weekday_traffic / weekend_traffic,
        'business_hours_lift': business_hours_traffic / traffic.mean(),
        'monthly_growth_rate': monthly_growth,
        'traffic_volatility': traffic_volatility,
        'special_event_boost': special_event_impact,
        'downtime_loss': downtime_impact
    }
    
    return metrics

def main():
    logger = setup_logger('web_traffic_forecasting', level='INFO')
    logger.info("🌐 Avvio analisi forecasting traffico web")
    
    # Genera dati
    logger.info("📊 Generazione dati traffico web con stagionalità multiple...")
    traffic_data, metadata = generate_web_traffic_data()
    
    # Calcola metriche web
    web_metrics = calculate_web_metrics(traffic_data, metadata)
    
    print(f"📊 Dataset generato: {len(traffic_data):,} ore di traffico")
    print(f"📅 Periodo: {traffic_data.index[0].strftime('%Y-%m-%d')} - {traffic_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"👥 Visitatori totali: {web_metrics['total_visitors']:,.0f}")
    print(f"📈 Media giornaliera: {web_metrics['daily_average']:,.0f} visitatori/giorno")
    print(f"⭐ Picco orario medio: {web_metrics['peak_hour_average']:,.0f} visitatori/ora")
    
    print(f"\n📊 Metriche Web Analytics:")
    print(f"  📅 Weekday vs Weekend ratio: {web_metrics['weekday_vs_weekend_ratio']:.1f}x")
    print(f"  🏢 Business hours lift: {web_metrics['business_hours_lift']:.1f}x")
    print(f"  📈 Crescita mensile: {web_metrics['monthly_growth_rate']:+.1f}%")
    print(f"  📊 Volatilità traffico: {web_metrics['traffic_volatility']:.1%}")
    print(f"  🎯 Boost medio eventi: {web_metrics['special_event_boost']:.1f}x")
    print(f"  🚨 Perdita media downtime: {web_metrics['downtime_loss']:.1%}")
    
    # Split train/test (80/20)
    train_size = int(len(traffic_data) * 0.8)
    train_data = traffic_data[:train_size]
    test_data = traffic_data[train_size:]
    
    print(f"\n🔄 Split dataset:")
    print(f"  📚 Training: {len(train_data):,} ore ({train_data.index[0].strftime('%Y-%m-%d')} - {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"  🧪 Test: {len(test_data):,} ore ({test_data.index[0].strftime('%Y-%m-%d')} - {test_data.index[-1].strftime('%Y-%m-%d')})")
    
    # Preprocessing per web traffic
    logger.info("🔧 Preprocessing dati traffico web...")
    preprocessor = TimeSeriesPreprocessor()
    
    # Log transform per stabilizzare varianza (traffico ha spikes)
    log_train = np.log1p(train_data)  # log(1+x) per evitare log(0)
    
    # Check stazionarietà
    stationarity_result = preprocessor.check_stationarity(log_train)
    is_stationary = stationarity_result['is_stationary']
    if not is_stationary:
        print("📈 Serie non stazionaria - il modello userà differenziazione")
    
    # Selezione automatica modello per web traffic
    logger.info("🔍 Selezione automatica modello ARIMA per traffico web...")
    
    # Use simple ARIMA model for web traffic data
    print("Utilizzo modello ARIMA(2,1,2) per dati traffico web...")
    best_order = (2, 1, 2)
    seasonal_order = None
    use_seasonal = False
    
    print(f"\nModello selezionato:")
    print(f"  ARIMA{best_order}")
    
    # Training modello
    logger.info("🎯 Training modello ARIMA per traffico web...")
    model = ARIMAForecaster(order=best_order)
    model.fit(log_train)
    
    # Forecast
    forecast_steps = len(test_data)
    logger.info(f"🔮 Generazione forecast per {forecast_steps} ore ({forecast_steps/24:.1f} giorni)...")
    log_forecast_result = model.forecast(
        steps=forecast_steps, 
        confidence_intervals=True,
        alpha=0.05
    )
    
    # Trasforma back da log-scale
    forecast_traffic = np.expm1(log_forecast_result['forecast'])  # exp(x) - 1
    forecast_lower = np.expm1(log_forecast_result['confidence_intervals']['lower'])
    forecast_upper = np.expm1(log_forecast_result['confidence_intervals']['upper'])
    
    forecast_result = {
        'forecast': pd.Series(forecast_traffic, index=test_data.index),
        'confidence_intervals': {
            'lower': pd.Series(forecast_lower, index=test_data.index),
            'upper': pd.Series(forecast_upper, index=test_data.index)
        }
    }
    
    # Valutazione
    logger.info("📊 Valutazione performance modello...")
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_forecast_metrics(test_data, forecast_result['forecast'])
    
    print(f"\n📊 Metriche Performance:")
    print(f"  📈 MAPE: {metrics['mape']:.2f}%")
    print(f"  📉 MAE: {metrics['mae']:.0f} visitatori/ora")
    print(f"  🎯 RMSE: {metrics['rmse']:.0f} visitatori/ora")
    
    # Check for R² score with different possible key names
    if 'r2_score' in metrics:
        print(f"  📊 R²: {metrics['r2_score']:.3f}")
    elif 'r_squared' in metrics:
        print(f"  📊 R²: {metrics['r_squared']:.3f}")
    else:
        print(f"  📊 R²: N/A")
    
    # Web-specific metrics
    test_daily_total = test_data.resample('D').sum().sum()
    forecast_daily_total = forecast_result['forecast'].resample('D').sum().sum()
    daily_volume_accuracy = 1 - abs(forecast_daily_total - test_daily_total) / test_daily_total
    
    print(f"  📊 Daily Volume Accuracy: {daily_volume_accuracy:.1%}")
    
    # Peak hour accuracy
    test_peak_hours = test_data.groupby(test_data.index.hour).mean()
    forecast_peak_hours = forecast_result['forecast'].groupby(forecast_result['forecast'].index.hour).mean()
    peak_hour_correlation = np.corrcoef(test_peak_hours, forecast_peak_hours)[0, 1]
    print(f"  ⏰ Peak Hour Pattern Correlation: {peak_hour_correlation:.3f}")
    
    # Future forecast (prossime 2 settimane)
    future_steps = 24 * 14  # 2 settimane
    logger.info("🚀 Forecast traffico prossime 2 settimane...")
    future_log_forecast = model.forecast(steps=future_steps, confidence_intervals=True)
    future_traffic = np.expm1(future_log_forecast['forecast'])
    future_lower = np.expm1(future_log_forecast['confidence_intervals']['lower'])
    future_upper = np.expm1(future_log_forecast['confidence_intervals']['upper'])
    
    # Visualizzazione specializzata web traffic
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Plot 1: Overview traffico completo
    ax1 = fig.add_subplot(gs[0, :])
    
    # Visualizza trend mensile
    monthly_train = train_data.resample('D').sum()
    monthly_test = test_data.resample('D').sum()
    
    ax1.plot(monthly_train.index, monthly_train, 
             label='Training (Daily)', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(monthly_test.index, monthly_test, 
             label='Test Actual (Daily)', color='green', alpha=0.8, linewidth=1.5)
    
    forecast_daily = forecast_result['forecast'].resample('D').sum()
    ax1.plot(forecast_daily.index, forecast_daily, 
             label='Forecast (Daily)', color='red', linestyle='--', linewidth=2)
    
    ax1.set_title('🌐 Web Traffic Forecast - Daily Overview')
    ax1.set_ylabel('Daily Visitors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dettaglio pattern orario
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Ultimi 7 giorni di training + test period dettagliato
    recent_train = train_data[-168:]  # Ultima settimana training
    
    ax2.plot(recent_train.index, recent_train, 
             label='Recent Training', color='blue', alpha=0.7, linewidth=0.8)
    ax2.plot(test_data.index, test_data, 
             label='Test Actual', color='green', alpha=0.8, linewidth=1)
    ax2.plot(test_data.index, forecast_result['forecast'], 
             label='Forecast', color='red', linestyle='--', linewidth=1.5)
    
    ax2.fill_between(test_data.index,
                     forecast_result['confidence_intervals']['lower'],
                     forecast_result['confidence_intervals']['upper'],
                     color='red', alpha=0.2, label='95% CI')
    
    ax2.set_title('⏰ Hourly Traffic Detail')
    ax2.set_ylabel('Hourly Visitors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Pattern giornaliero medio
    ax3 = fig.add_subplot(gs[1, 1])
    
    hourly_pattern_train = train_data.groupby(train_data.index.hour).mean()
    hourly_pattern_test = test_data.groupby(test_data.index.hour).mean()
    hourly_pattern_forecast = forecast_result['forecast'].groupby(forecast_result['forecast'].index.hour).mean()
    
    ax3.plot(hourly_pattern_train.index, hourly_pattern_train.values, 
             'o-', label='Training Avg', color='blue', linewidth=2)
    ax3.plot(hourly_pattern_test.index, hourly_pattern_test.values, 
             's-', label='Test Actual', color='green', linewidth=2)
    ax3.plot(hourly_pattern_forecast.index, hourly_pattern_forecast.values, 
             '^-', label='Forecast', color='red', linewidth=2)
    
    # Evidenzia business hours
    ax3.axvspan(8, 17, alpha=0.2, color='yellow', label='Business Hours')
    
    ax3.set_title('⏰ Daily Traffic Pattern')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Avg Visitors/Hour')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)
    
    # Plot 4: Pattern settimanale
    ax4 = fig.add_subplot(gs[1, 2])
    
    weekly_pattern_train = train_data.groupby(train_data.index.dayofweek).mean()
    weekly_pattern_test = test_data.groupby(test_data.index.dayofweek).mean()
    weekly_pattern_forecast = forecast_result['forecast'].groupby(forecast_result['forecast'].index.dayofweek).mean()
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x_pos = range(7)
    
    width = 0.25
    ax4.bar([x - width for x in x_pos], weekly_pattern_train.values, 
            width, label='Training', alpha=0.7, color='blue')
    ax4.bar(x_pos, weekly_pattern_test.values, 
            width, label='Test Actual', alpha=0.7, color='green')
    ax4.bar([x + width for x in x_pos], weekly_pattern_forecast.values, 
            width, label='Forecast', alpha=0.7, color='red')
    
    ax4.set_title('📅 Weekly Traffic Pattern')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Avg Visitors/Hour')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(days)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Eventi speciali impact
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Trova eventi nel test period
    test_metadata = metadata.iloc[train_size:train_size+len(test_data)]
    special_events_mask = test_metadata['special_events'] > 1.5
    
    ax5.plot(test_data.index, test_data, color='lightblue', alpha=0.6, linewidth=0.8, label='Actual Traffic')
    ax5.scatter(test_data.index[special_events_mask], 
                test_data[special_events_mask],
                color='red', s=30, alpha=0.8, 
                label=f'Special Events ({special_events_mask.sum()})')
    
    # Mostra forecast per questi eventi
    ax5.plot(test_data.index, forecast_result['forecast'], 
             linestyle='--', color='purple', linewidth=1.5, label='Forecast')
    
    ax5.set_title('🎯 Special Events Impact')
    ax5.set_ylabel('Visitors/Hour')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Algorithm impact analysis
    ax6 = fig.add_subplot(gs[2, 1])
    
    algo_impact_mask = test_metadata['algo_impact'] < 0.95
    
    # Rolling mean per smooth trend
    actual_smooth = test_data.rolling(24).mean()  # 24h rolling
    forecast_smooth = forecast_result['forecast'].rolling(24).mean()
    
    ax6.plot(actual_smooth.index, actual_smooth, 
             label='Actual (24h avg)', color='green', linewidth=2)
    ax6.plot(forecast_smooth.index, forecast_smooth, 
             label='Forecast (24h avg)', color='red', linestyle='--', linewidth=2)
    
    if algo_impact_mask.any():
        ax6.fill_between(test_data.index, 0, test_data.max(),
                         where=algo_impact_mask, alpha=0.3, color='orange',
                         label='Algorithm Impact Period')
    
    ax6.set_title('🔍 Algorithm Impact Analysis')
    ax6.set_ylabel('Visitors/Hour')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    # Plot 7: Mobile trend
    ax7 = fig.add_subplot(gs[2, 2])
    
    mobile_trend = test_metadata['mobile_trend']
    
    ax7_twin = ax7.twinx()
    
    # Traffic on primary axis
    ax7.plot(test_data.index, test_data, color='blue', alpha=0.7, linewidth=1)
    ax7.set_ylabel('Visitors/Hour', color='blue')
    ax7.tick_params(axis='y', labelcolor='blue')
    
    # Mobile trend on secondary axis
    ax7_twin.plot(mobile_trend.index, mobile_trend, 
                  color='red', linewidth=2, label='Mobile Trend')
    ax7_twin.set_ylabel('Mobile Factor', color='red')
    ax7_twin.tick_params(axis='y', labelcolor='red')
    
    ax7.set_title('📱 Mobile Traffic Trend')
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', rotation=45)
    
    # Plot 8: Future forecast 2 settimane
    ax8 = fig.add_subplot(gs[3, 0])
    
    # Ultime 2 settimane per contesto
    recent_context = traffic_data[-336:]  # 14 giorni * 24 ore
    ax8.plot(recent_context.index, recent_context, 
             label='Recent Data', color='blue', alpha=0.7, linewidth=0.8)
    
    # Future dates
    future_dates = pd.date_range(
        start=traffic_data.index[-1] + pd.Timedelta(hours=1),
        periods=future_steps, 
        freq='h'
    )
    
    ax8.plot(future_dates, future_traffic, 
             's-', label='2-Week Forecast', color='purple', 
             linewidth=1.5, markersize=1, alpha=0.8)
    
    ax8.fill_between(future_dates, future_lower, future_upper,
                     color='purple', alpha=0.2, label='95% CI')
    
    ax8.set_title('🚀 Next 2 Weeks Traffic Forecast')
    ax8.set_ylabel('Visitors/Hour')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='x', rotation=45)
    
    # Plot 9: Residuals analysis
    ax9 = fig.add_subplot(gs[3, 1])
    
    residuals = test_data - forecast_result['forecast']
    
    # Time series residuals
    ax9.plot(test_data.index, residuals, 'o-', 
             color='orange', alpha=0.7, markersize=1.5, linewidth=0.8)
    ax9.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax9.axhline(y=residuals.std(), color='red', linestyle='--', alpha=0.5, label='+1σ')
    ax9.axhline(y=-residuals.std(), color='red', linestyle='--', alpha=0.5, label='-1σ')
    
    # Evidenzia outlier residuals
    outlier_threshold = 2 * residuals.std()
    outliers = np.abs(residuals) > outlier_threshold
    ax9.scatter(test_data.index[outliers], residuals[outliers],
                color='red', s=20, alpha=0.8, 
                label=f'Outliers ({outliers.sum()})')
    
    ax9.set_title('📉 Forecast Residuals')
    ax9.set_ylabel('Residuals (visitors/hour)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='x', rotation=45)
    
    # Plot 10: Performance by time of day
    ax10 = fig.add_subplot(gs[3, 2])
    
    # MAPE by hour of day
    hourly_mape = []
    for hour in range(24):
        hour_mask = test_data.index.hour == hour
        if hour_mask.any():
            hour_actual = test_data[hour_mask]
            hour_forecast = forecast_result['forecast'][hour_mask]
            hour_mape = np.mean(np.abs((hour_actual - hour_forecast) / hour_actual)) * 100
            hourly_mape.append(hour_mape)
        else:
            hourly_mape.append(0)
    
    colors = ['red' if mape > 20 else 'orange' if mape > 10 else 'green' for mape in hourly_mape]
    ax10.bar(range(24), hourly_mape, color=colors, alpha=0.7)
    ax10.axhline(y=np.mean(hourly_mape), color='blue', linestyle='--', 
                 label=f'Avg MAPE: {np.mean(hourly_mape):.1f}%')
    
    ax10.set_title('📊 Forecast Accuracy by Hour')
    ax10.set_xlabel('Hour of Day')
    ax10.set_ylabel('MAPE (%)')
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis='y')
    ax10.set_xlim(-0.5, 23.5)
    
    # Salva plot
    plt.savefig('outputs/plots/web_traffic_forecast.png', dpi=300, bbox_inches='tight')
    logger.info("📁 Plot salvato in outputs/plots/web_traffic_forecast.png")
    
    # plt.show()  # Disabled for Windows compatibility
    print("Plot saved as 'outputs/plots/web_traffic_forecast.png'")
    
    # Web Analytics Insights
    print(f"\n🌐 Web Analytics Insights:")
    
    # Future traffic projections
    future_total = future_traffic.sum()
    current_2weeks = traffic_data[-336:].sum()  # Last 2 weeks
    growth_projection = ((future_total - current_2weeks) / current_2weeks) * 100
    
    print(f"  📊 Traffico previsto prossime 2 settimane: {future_total:,.0f} visitatori")
    print(f"  📈 vs ultime 2 settimane: {growth_projection:+.1f}%")
    
    # Peak traffic predictions
    future_series = pd.Series(future_traffic, index=future_dates)
    future_daily = future_series.resample('D').sum()
    best_day = future_daily.idxmax()
    worst_day = future_daily.idxmin()
    
    print(f"  🏆 Miglior giorno previsto: {best_day.strftime('%A %Y-%m-%d')} ({future_daily.max():,.0f} visitatori)")
    print(f"  📉 Giorno più basso: {worst_day.strftime('%A %Y-%m-%d')} ({future_daily.min():,.0f} visitatori)")
    
    # Peak hours analysis
    future_hourly_avg = future_series.groupby(future_series.index.hour).mean()
    peak_hour = future_hourly_avg.idxmax()
    peak_traffic = future_hourly_avg.max()
    
    if pd.isna(peak_hour) or pd.isna(peak_traffic):
        print(f"  ⏰ Ora di picco media: N/A (dati insufficienti)")
    else:
        print(f"  ⏰ Ora di picco media: {int(peak_hour):02d}:00 ({peak_traffic:,.0f} visitatori/ora)")
    
    # Capacity planning
    current_max_hourly = traffic_data.max()
    future_max_hourly = future_traffic.max()
    capacity_increase_needed = (future_max_hourly - current_max_hourly) / current_max_hourly * 100
    
    if capacity_increase_needed > 10:
        print(f"  🚨 CAPACITY ALERT: Picco previsto +{capacity_increase_needed:.1f}% vs storico")
        print(f"    Raccomandazione: Scale server capacity")
    else:
        print(f"  ✅ Capacità attuale sufficiente per traffico previsto")
    
    # SEO/Content recommendations
    best_performing_hours = np.argsort(hourly_mape)[:3]  # Top 3 accurate hours
    worst_performing_hours = np.argsort(hourly_mape)[-3:]  # Top 3 inaccurate hours
    
    print(f"  📈 Ore con forecast più accurato: {[f'{h:02d}:00' for h in best_performing_hours]}")
    print(f"  📉 Ore con forecast meno accurato: {[f'{h:02d}:00' for h in worst_performing_hours]}")
    print(f"    Raccomandazione: Analizzare user behavior in ore difficili da prevedere")
    
    # Weekend vs weekday insights
    future_weekday_avg = future_series[future_series.index.weekday < 5].mean()
    future_weekend_avg = future_series[future_series.index.weekday >= 5].mean()
    future_weekday_weekend_ratio = future_weekday_avg / future_weekend_avg
    
    print(f"  📅 Ratio weekday/weekend previsto: {future_weekday_weekend_ratio:.1f}x")
    
    if future_weekday_weekend_ratio > 2.0:
        print(f"    💼 Business-focused content consigliato per weekdays")
    elif future_weekday_weekend_ratio < 1.2:
        print(f"    🎯 Consumer-focused content opportunità per weekend")
    
    # Salva modello
    model_path = 'outputs/models/web_traffic_arima_model.joblib'
    model.save(model_path)
    logger.info(f"💾 Modello salvato in {model_path}")
    
    print(f"\n✅ Analisi web traffic completata!")
    print(f"📁 Risultati e insights salvati in outputs/")
    print(f"🌐 Modello pronto per integrazione con Google Analytics/web analytics platforms")

if __name__ == "__main__":
    main()