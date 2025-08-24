"""
Test minimo funzionalità per Moretti S.p.A.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test import base
try:
    from arima_forecaster import ARIMAForecaster
    print("[OK] Import ARIMAForecaster")
except ImportError as e:
    print(f"[ERRORE] Import fallito: {e}")
    exit(1)

# Genera dati semplici
print("\n1. Generazione dati test...")
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = 100 + np.cumsum(np.random.randn(365))  # Random walk con drift
series = pd.Series(values, index=dates, name='vendite')
print(f"   Shape: {series.shape}")
print(f"   Range: {series.min():.1f} - {series.max():.1f}")

# Split train/test
train = series[:-30]
test = series[-30:]
print(f"\n2. Train: {len(train)}, Test: {len(test)}")

# Addestra modello
print("\n3. Training ARIMA(1,1,1)...")
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(train)
print("   [OK] Training completato")

# Forecast semplice
print("\n4. Forecast 30 giorni...")
try:
    forecast = model.forecast(steps=30)
    print(f"   Tipo risultato: {type(forecast)}")
    
    if isinstance(forecast, pd.Series):
        print(f"   Shape forecast: {forecast.shape}")
        print(f"   Valori validi: {forecast.notna().sum()}/{len(forecast)}")
        
        if forecast.notna().any():
            print(f"   Media: {forecast.mean():.2f}")
            print(f"   Primi 5 valori: {forecast.head().values}")
            
            # Confronto con test
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            print(f"\n5. Performance:")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
        else:
            print("   [PROBLEMA] Tutti i valori sono NaN!")
            
            # Debug: prova forecast a 1 passo
            print("\n   Test forecast 1 passo...")
            f1 = model.forecast(steps=1)
            print(f"   Valore: {f1}")
            
    elif isinstance(forecast, dict):
        print("   Risultato è un dizionario")
        print(f"   Chiavi: {forecast.keys()}")
        if 'forecast' in forecast:
            print(f"   Shape forecast: {forecast['forecast'].shape}")
            print(f"   Primi valori: {forecast['forecast'].head()}")
            
except Exception as e:
    print(f"   [ERRORE] Forecast fallito: {e}")
    import traceback
    traceback.print_exc()

print("\n[COMPLETATO]")