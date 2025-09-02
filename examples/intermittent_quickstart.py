"""
Quickstart: Intermittent Demand Forecasting
Esempio rapido per iniziare con spare parts forecasting

Autore: Claude Code
Data: 2025-09-02
"""

import numpy as np
import pandas as pd
from arima_forecaster import IntermittentForecaster, IntermittentConfig, IntermittentMethod
from arima_forecaster.evaluation import IntermittentEvaluator
import warnings
warnings.filterwarnings('ignore')


def quickstart_example():
    """
    Esempio quickstart completo per Intermittent Demand
    """
    print("="*60)
    print(" QUICKSTART: INTERMITTENT DEMAND FORECASTING")
    print("="*60)
    
    # ========================================
    # 1. GENERA DATI ESEMPIO
    # ========================================
    print("\n1. GENERAZIONE DATI SPARE PARTS")
    print("-" * 40)
    
    # Simula domanda ricambio: 90% giorni senza domanda, 10% con domanda 1-5 pezzi
    np.random.seed(42)
    days = 365
    demand_probability = 0.1  # 10% probabilità domanda giornaliera
    
    demand = []
    for _ in range(days):
        if np.random.random() < demand_probability:
            # Quando c'è domanda, quantità tra 1 e 5 pezzi
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            demand.append(quantity)
        else:
            demand.append(0)
    
    demand = np.array(demand)
    print(f"Giorni totali: {days}")
    print(f"Giorni con domanda: {np.sum(demand > 0)} ({100*np.sum(demand > 0)/days:.1f}%)")
    print(f"Domanda totale: {np.sum(demand)} pezzi")
    print(f"Domanda media (quando presente): {np.mean(demand[demand > 0]):.1f} pezzi")
    
    # ========================================
    # 2. ANALISI PATTERN
    # ========================================
    print("\n2. ANALISI PATTERN DOMANDA")
    print("-" * 40)
    
    # Crea forecaster e analizza pattern
    forecaster = IntermittentForecaster()
    pattern = forecaster.analyze_pattern(demand)
    
    print(f"Classificazione: {pattern.classification}")
    print(f"ADI (Average Demand Interval): {pattern.adi:.1f} giorni")
    print(f"CV² (Coefficient of Variation²): {pattern.cv2:.2f}")
    print(f"Intermittenza: {pattern.intermittence:.1%}")
    print(f"Lumpiness: {pattern.lumpiness:.2f}")
    
    # ========================================
    # 3. CONFRONTO METODI
    # ========================================
    print("\n3. CONFRONTO METODI FORECASTING")
    print("-" * 40)
    
    # Split train/test
    train_size = int(0.8 * len(demand))
    train_data = demand[:train_size]
    test_data = demand[train_size:]
    
    methods = {
        'Croston': IntermittentMethod.CROSTON,
        'SBA': IntermittentMethod.SBA,
        'TSB': IntermittentMethod.TSB
    }
    
    results = {}
    
    for name, method in methods.items():
        # Configura e addestra
        config = IntermittentConfig(
            method=method,
            alpha=0.1,
            optimize_alpha=True
        )
        
        model = IntermittentForecaster(config)
        model.fit(train_data)
        
        # Forecast
        forecast = model.forecast(len(test_data))
        
        # Valuta
        evaluator = IntermittentEvaluator(
            holding_cost=1,    # €1/pezzo/giorno stoccaggio
            stockout_cost=10   # €10/pezzo mancante
        )
        
        metrics = evaluator.evaluate(test_data, forecast)
        
        results[name] = {
            'forecast_mean': np.mean(forecast),
            'mase': metrics.mase,
            'fill_rate': metrics.fill_rate,
            'service_level': metrics.achieved_service_level
        }
        
        print(f"\n{name}:")
        print(f"  Forecast medio: {np.mean(forecast):.3f} pezzi/giorno")
        print(f"  MASE: {metrics.mase:.3f}")
        print(f"  Fill Rate: {metrics.fill_rate:.1f}%")
        print(f"  Service Level: {metrics.achieved_service_level:.1f}%")
    
    # ========================================
    # 4. CALCOLO INVENTORY OTTIMALE
    # ========================================
    print("\n4. CALCOLO PARAMETRI INVENTORY")
    print("-" * 40)
    
    # Usa il metodo migliore (SBA solitamente)
    best_config = IntermittentConfig(
        method=IntermittentMethod.SBA,
        optimize_alpha=True
    )
    
    best_model = IntermittentForecaster(best_config)
    best_model.fit(train_data)
    
    # Parametri inventory
    lead_time = 15  # giorni
    service_level = 0.95  # 95%
    
    safety_stock = best_model.calculate_safety_stock(lead_time, service_level)
    reorder_point = best_model.calculate_reorder_point(lead_time, service_level)
    
    print(f"Lead Time: {lead_time} giorni")
    print(f"Service Level Target: {service_level:.0%}")
    print(f"Safety Stock: {safety_stock:.0f} pezzi")
    print(f"Reorder Point: {reorder_point:.0f} pezzi")
    
    # Calcolo investimento (esempio)
    cost_per_piece = 50  # €50/pezzo
    investment = reorder_point * cost_per_piece
    
    print(f"\nCosto unitario: €{cost_per_piece}")
    print(f"Investimento scorta: €{investment:,.2f}")
    
    # ========================================
    # 5. RACCOMANDAZIONI
    # ========================================
    print("\n5. RACCOMANDAZIONI FINALI")
    print("-" * 40)
    
    print(f"[OK] Pattern identificato: {pattern.classification}")
    print(f"[OK] Metodo consigliato: SBA (Syntetos-Boylan)")
    print(f"[OK] Quantita riordino: {reorder_point:.0f} pezzi")
    print(f"[OK] Riordinare quando stock < {reorder_point:.0f} pezzi")
    
    # EOQ semplificato
    annual_demand = np.sum(train_data) * 365 / len(train_data)
    ordering_cost = 100  # €100 per ordine
    holding_cost_annual = cost_per_piece * 0.2  # 20% del valore
    
    if annual_demand > 0:
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost_annual)
        print(f"[OK] EOQ (lotto economico): {eoq:.0f} pezzi")
        print(f"[OK] Ordini/anno: {annual_demand/eoq:.1f}")
    
    print("\n" + "="*60)
    print(" QUICKSTART COMPLETATO CON SUCCESSO!")
    print("="*60)
    
    return best_model, results


def minimal_example():
    """
    Esempio minimo in 10 righe di codice
    """
    print("\nESEMPIO MINIMO (10 righe):")
    print("-" * 40)
    
    # 1. Genera dati
    demand = np.random.choice([0, 0, 0, 1, 2], size=100)
    
    # 2. Crea e addestra modello
    model = IntermittentForecaster()
    model.fit(demand)
    
    # 3. Forecast
    forecast = model.forecast(30)
    
    # 4. Calcola reorder point
    rop = model.calculate_reorder_point(lead_time=7, service_level=0.95)
    
    print(f"Forecast: {forecast[0]:.3f} unità/giorno")
    print(f"Reorder Point: {rop:.0f} unità")
    print(f"Pattern: {model.pattern_.classification}")


if __name__ == "__main__":
    # Esegui quickstart completo
    model, results = quickstart_example()
    
    # Mostra anche esempio minimo
    minimal_example()