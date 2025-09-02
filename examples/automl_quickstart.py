"""
AutoML Forecasting Quickstart
One-click forecasting per utenti business

Autore: Claude Code
Data: 2025-09-02  
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from arima_forecaster import AutoForecastSelector


def main():
    """
    Quickstart AutoML in 5 righe di codice
    """
    print("="*60)
    print(" AutoML FORECASTING - One Click Solution")
    print("="*60)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    
    print("\n1. Genera dati di esempio...")
    # Crea serie stagionale realistica (es. vendite retail)
    days = 300
    dates = pd.date_range('2023-01-01', periods=days, freq='D') 
    
    # Base trend + stagionalità + noise
    base = 100
    trend = 0.1 * np.arange(days)  # Crescita graduale
    weekly = 15 * np.sin(2 * np.pi * np.arange(days) / 7)  # Pattern settimanale
    monthly = 10 * np.sin(2 * np.pi * np.arange(days) / 30)  # Pattern mensile
    noise = np.random.normal(0, 8, days)
    
    values = base + trend + weekly + monthly + noise
    data = pd.Series(values, index=dates)
    
    print(f"   Dati generati: {len(data)} giorni")
    print(f"   Media: {data.mean():.1f}")
    print(f"   Range: {data.min():.1f} - {data.max():.1f}")
    
    print("\n2. AutoML Magic...")
    # ✨ QUESTA È LA MAGIA - UNA RIGA!
    automl = AutoForecastSelector(verbose=True)
    best_model, explanation = automl.fit(data)
    
    print("\n3. Risultati:")
    print("-" * 30)
    print(f"[MODEL] {explanation.recommended_model}")
    print(f"[CONF] {explanation.confidence_score:.1%}")
    print(f"[PATTERN] {explanation.pattern_detected}")
    print(f"[WHY] {explanation.why_chosen}")
    print(f"[BUSINESS] {explanation.business_recommendation}")
    
    print("\n4. Forecast esempio (prossimi 30 giorni):")
    forecast = best_model.forecast(steps=30)
    print(f"   Media forecast: {np.mean(forecast):.1f}")
    print(f"   Range forecast: {np.min(forecast):.1f} - {np.max(forecast):.1f}")
    print(f"   Crescita prevista: {(np.mean(forecast) - data.tail(30).mean()):.1f}")
    
    print("\n5. Confronto modelli:")
    comparison = automl.get_model_comparison()
    if not comparison.empty:
        print(comparison[['Model', 'Accuracy_Score', 'Confidence', 'Status']].to_string(index=False))
    
    print(f"\n6. Risk Assessment:")
    print(f"   {explanation.risk_assessment}")
    
    if explanation.alternative_models:
        print(f"\n7. Modelli alternativi:")
        for i, alt in enumerate(explanation.alternative_models[:3], 1):
            print(f"   {i}. {alt['model']} (Score: {alt['score']:.3f})")
    
    print("\n" + "="*60)
    print(" [SUCCESS] AutoML COMPLETATO - Ready for Production!")
    print("="*60)
    
    # Bonus: simple usage example
    print("\n[CODE] CODICE PER PRODUZIONE:")
    print("-" * 30)
    print("from arima_forecaster import AutoForecastSelector")
    print("automl = AutoForecastSelector()")  
    print("model, explanation = automl.fit(your_data)")
    print("forecast = model.forecast(30)")
    print(f"# Confidence: {explanation.confidence_score:.1%} | Model: {explanation.recommended_model}")
    
    return best_model, explanation, automl


def business_example():
    """
    Esempio business: portfolio di 3 prodotti
    """
    print("\n" + "="*60)  
    print(" BUSINESS CASE: Multi-Product Portfolio")
    print("="*60)
    
    products = {
        'Prodotto A (Regular)': 'regular',
        'Prodotto B (Seasonal)': 'seasonal', 
        'Ricambio C (Intermittent)': 'intermittent'
    }
    
    results = {}
    total_time = 0
    
    for product, pattern in products.items():
        print(f"\n[ANALYZE] {product}:")
        
        # Generate realistic data for each pattern
        if pattern == 'regular':
            data = pd.Series(100 + np.cumsum(np.random.normal(0.1, 3, 200)))
        elif pattern == 'seasonal':
            x = np.arange(200)
            data = pd.Series(100 + 20 * np.sin(2 * np.pi * x / 30) + np.random.normal(0, 5, 200))
        else:  # intermittent
            values = np.random.choice([0, 0, 0, 1, 2, 3], size=200, p=[0.7, 0.1, 0.05, 0.1, 0.03, 0.02])
            data = pd.Series(values)
        
        # AutoML
        import time
        start = time.time()
        automl = AutoForecastSelector(verbose=False)
        model, explanation = automl.fit(data)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"   [OK] {explanation.recommended_model} ({explanation.confidence_score:.1%} conf, {elapsed:.1f}s)")
        
        results[product] = {
            'model': explanation.recommended_model,
            'confidence': explanation.confidence_score,
            'time': elapsed
        }
    
    print(f"\n[SUMMARY] PORTFOLIO RESULTS:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg time per product: {total_time/len(products):.1f}s") 
    print(f"   All models selected with >70% confidence: {'[OK]' if all(r['confidence'] > 0.7 for r in results.values()) else '[WARN]'}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--business':
        business_example()
    else:
        main()