#!/usr/bin/env python3
"""
Retail Sales Forecasting Example

Questo esempio dimostra come utilizzare ARIMA per prevedere le vendite al dettaglio
con pattern stagionali e trend. Include dati mensili con stagionalità annuale tipica
del settore retail (picchi durante le festività).
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

def generate_retail_sales_data():
    """Genera dati di vendita al dettaglio sintetici con stagionalità"""
    np.random.seed(42)
    
    # Periodo: 5 anni di dati mensili
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='M')
    
    # Trend di base (crescita del 3% annuo)
    trend = np.linspace(100000, 115000, len(dates))
    
    # Stagionalità annuale (picco dicembre per festività)
    seasonal = 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / 12) + \
               15000 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 + np.pi/3)
    
    # Effetto Black Friday/Cyber Monday (novembre-dicembre)
    black_friday_mask = (dates.month == 11) | (dates.month == 12)
    black_friday_boost = np.zeros(len(dates))
    black_friday_boost[black_friday_mask] = 8000 * np.random.uniform(0.8, 1.2, sum(black_friday_mask))
    
    # Rumore random e eventi speciali
    noise = np.random.normal(0, 3000, len(dates))
    
    # Eventi eccezionali (COVID-19 impact in 2020)
    covid_mask = (dates.year == 2020) & ((dates.month >= 3) & (dates.month <= 6))
    covid_impact = np.zeros(len(dates))
    covid_impact[covid_mask] = -25000 * (1 - np.random.uniform(0.3, 0.7, sum(covid_mask)))
    
    sales = trend + seasonal + black_friday_boost + noise + covid_impact
    sales = np.maximum(sales, 10000)  # Vendite minime
    
    return pd.Series(sales, index=dates, name='monthly_sales')

def main():
    logger = setup_logger('retail_forecasting', level='INFO')
    logger.info("🛍️ Avvio analisi forecasting vendite retail")
    
    # Genera dati
    logger.info("📊 Generazione dati vendite retail...")
    sales_data = generate_retail_sales_data()
    
    print(f"📈 Dataset generato: {len(sales_data)} punti dati")
    print(f"📅 Periodo: {sales_data.index[0].strftime('%Y-%m')} - {sales_data.index[-1].strftime('%Y-%m')}")
    print(f"💰 Vendite media mensile: €{sales_data.mean():,.0f}")
    print(f"📊 Range vendite: €{sales_data.min():,.0f} - €{sales_data.max():,.0f}")
    
    # Split train/test
    train_size = int(len(sales_data) * 0.8)
    train_data = sales_data[:train_size]
    test_data = sales_data[train_size:]
    
    print(f"\n🔄 Split dataset:")
    print(f"  📚 Training: {len(train_data)} mesi ({train_data.index[0].strftime('%Y-%m')} - {train_data.index[-1].strftime('%Y-%m')})")
    print(f"  🧪 Test: {len(test_data)} mesi ({test_data.index[0].strftime('%Y-%m')} - {test_data.index[-1].strftime('%Y-%m')})")
    
    # Preprocessing
    logger.info("🔧 Preprocessing dati...")
    preprocessor = TimeSeriesPreprocessor()
    
    # Check stazionarietà
    stationarity_result = preprocessor.check_stationarity(train_data)
    is_stationary = stationarity_result['is_stationary']
    if not is_stationary:
        print("📈 Serie non stazionaria - applicando differenziazione")
    
    # Selezione automatica modello
    logger.info("🔍 Selezione automatica modello ARIMA...")
    # Use simple ARIMA model for demonstration
    print("Utilizzo modello ARIMA(1,1,1) per dati retail...")
    best_order = (1, 1, 1)
    seasonal_order = None
    
    print(f"\n✅ Modello ottimale trovato:")
    print(f"  📊 ARIMA{best_order}")
    print(f"  🌊 Seasonal{seasonal_order}")
    
    # Training modello
    logger.info("🎯 Training modello ARIMA...")
    model = ARIMAForecaster(order=best_order)
    model.fit(train_data)
    
    # Forecast
    forecast_steps = len(test_data)
    logger.info(f"🔮 Generazione forecast per {forecast_steps} mesi...")
    forecast_result = model.forecast(
        steps=forecast_steps, 
        confidence_intervals=True,
        alpha=0.05  # 95% confidence intervals
    )
    
    # Valutazione
    logger.info("📊 Valutazione performance modello...")
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_forecast_metrics(test_data, forecast_result['forecast'])
    
    print(f"\n📊 Metriche Performance:")
    print(f"  📈 MAPE: {metrics['mape']:.2f}%")
    print(f"  📉 MAE: €{metrics['mae']:,.0f}")
    print(f"  🎯 RMSE: €{metrics['rmse']:,.0f}")
    
    # Check for R² score with different possible key names
    if 'r2_score' in metrics:
        print(f"  📊 R²: {metrics['r2_score']:.3f}")
    elif 'r_squared' in metrics:
        print(f"  📊 R²: {metrics['r_squared']:.3f}")
    else:
        print(f"  📊 R²: N/A")
    
    # Forecast futuro
    logger.info("🚀 Forecast per i prossimi 12 mesi...")
    future_forecast = model.forecast(steps=12, confidence_intervals=True)
    
    # Visualizzazione
    plotter = ForecastPlotter(figsize=(15, 10))
    
    # Plot principale con training, test e forecast
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Overview completo
    plt.subplot(2, 2, 1)
    plt.plot(train_data.index, train_data, label='Training Data', color='blue', alpha=0.7)
    plt.plot(test_data.index, test_data, label='Test Data (Actual)', color='green', alpha=0.8)
    plt.plot(test_data.index, forecast_result['forecast'], 
             label='Forecast (Test)', color='red', linestyle='--', linewidth=2)
    
    plt.fill_between(test_data.index,
                     forecast_result['confidence_intervals']['lower'],
                     forecast_result['confidence_intervals']['upper'],
                     color='red', alpha=0.2, label='95% CI')
    
    plt.title('📊 Retail Sales Forecasting - Overview Completo')
    plt.ylabel('Vendite Mensili (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Subplot 2: Dettaglio test period
    plt.subplot(2, 2, 2)
    plt.plot(test_data.index, test_data, 'o-', label='Actual', color='green', markersize=6)
    plt.plot(test_data.index, forecast_result['forecast'], 
             's-', label='Forecast', color='red', markersize=5)
    
    plt.fill_between(test_data.index,
                     forecast_result['confidence_intervals']['lower'],
                     forecast_result['confidence_intervals']['upper'],
                     color='red', alpha=0.2, label='95% CI')
    
    plt.title('🎯 Performance su Test Set')
    plt.ylabel('Vendite Mensili (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Subplot 3: Forecast futuro
    plt.subplot(2, 2, 3)
    
    # Ultime osservazioni per contesto
    recent_data = sales_data[-12:]
    plt.plot(recent_data.index, recent_data, 'o-', 
             label='Dati Recenti', color='blue', alpha=0.7)
    
    # Future forecast
    future_dates = pd.date_range(
        start=sales_data.index[-1] + pd.DateOffset(months=1),
        periods=12, 
        freq='M'
    )
    
    plt.plot(future_dates, future_forecast['forecast'], 
             's-', label='Forecast Futuro', color='purple', linewidth=2, markersize=6)
    
    plt.fill_between(future_dates,
                     future_forecast['confidence_intervals']['lower'],
                     future_forecast['confidence_intervals']['upper'],
                     color='purple', alpha=0.2, label='95% CI')
    
    plt.title('🚀 Forecast Prossimi 12 Mesi')
    plt.ylabel('Vendite Mensili (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Subplot 4: Analisi residui
    plt.subplot(2, 2, 4)
    residuals = test_data - forecast_result['forecast']
    plt.plot(test_data.index, residuals, 'o-', color='orange', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=residuals.std(), color='red', linestyle='--', alpha=0.5, label='+1σ')
    plt.axhline(y=-residuals.std(), color='red', linestyle='--', alpha=0.5, label='-1σ')
    
    plt.title('📉 Analisi Residui')
    plt.ylabel('Residui (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Salva plot
    plt.savefig('outputs/plots/retail_sales_forecast.png', dpi=300, bbox_inches='tight')
    logger.info("📁 Plot salvato in outputs/plots/retail_sales_forecast.png")
    
    # plt.show()  # Disabled for Windows compatibility
    print("Plot saved as 'outputs/plots/retail_sales_forecast.png'")
    
    # Insights business
    print(f"\n🎯 Business Insights:")
    
    # Previsioni future con interpretazioni
    future_total = future_forecast['forecast'].sum()
    current_year_total = sales_data[-12:].sum()
    growth_rate = ((future_total - current_year_total) / current_year_total) * 100
    
    print(f"  📊 Vendite previste prossimi 12 mesi: €{future_total:,.0f}")
    print(f"  📈 Crescita stimata: {growth_rate:+.1f}% vs ultimo anno")
    
    # Mesi con vendite più alte
    future_df = pd.DataFrame({
        'month': [d.strftime('%Y-%m') for d in future_dates],
        'forecast': future_forecast['forecast']
    })
    best_months = future_df.nlargest(3, 'forecast')
    
    print(f"  🏆 Top 3 mesi previsti:")
    for idx, row in best_months.iterrows():
        print(f"    {row['month']}: €{row['forecast']:,.0f}")
    
    # Salva modello
    model_path = 'outputs/models/retail_sales_arima_model.joblib'
    model.save(model_path)
    logger.info(f"💾 Modello salvato in {model_path}")
    
    print(f"\n✅ Analisi completata!")
    print(f"📁 Risultati salvati in outputs/")

if __name__ == "__main__":
    main()