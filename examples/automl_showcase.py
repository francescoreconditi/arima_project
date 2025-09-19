"""
AutoML Forecasting Showcase
Dimostrazione completa dell'AutoML Engine con diversi tipi di serie temporali

Autore: Claude Code
Data: 2025-09-02
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.arima_forecaster.automl.auto_selector import AutoForecastSelector
from src.arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


def generate_test_series(series_type: str, length: int = 365, seed: int = 42) -> pd.Series:
    """
    Genera serie temporali test per diversi pattern
    """
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=length, freq="D")

    if series_type == "regular":
        # Serie regolare con noise
        trend = np.linspace(100, 120, length)
        noise = np.random.normal(0, 5, length)
        values = trend + noise

    elif series_type == "seasonal":
        # Serie con stagionalità annuale e settimanale
        trend = np.linspace(100, 150, length)
        annual = 20 * np.sin(2 * np.pi * np.arange(length) / 365.25)
        weekly = 10 * np.sin(2 * np.pi * np.arange(length) / 7)
        noise = np.random.normal(0, 8, length)
        values = trend + annual + weekly + noise

    elif series_type == "trending":
        # Serie con trend forte
        trend = 50 + 0.3 * np.arange(length) + 0.0001 * np.arange(length) ** 2
        noise = np.random.normal(0, 10, length)
        values = trend + noise

    elif series_type == "intermittent":
        # Domanda intermittente (spare parts)
        values = np.zeros(length)
        demand_prob = 0.08  # 8% giorni con domanda
        for i in range(length):
            if np.random.random() < demand_prob:
                values[i] = np.random.poisson(3) + 1  # 1-10 pezzi

    elif series_type == "volatile":
        # Serie volatile con outlier
        base = 100 + np.cumsum(np.random.normal(0, 2, length))
        outlier_prob = 0.05  # 5% outlier
        for i in range(length):
            if np.random.random() < outlier_prob:
                base[i] += np.random.choice([-50, 50])
        values = base

    elif series_type == "short":
        # Serie breve
        dates = dates[:50]  # Solo 50 osservazioni
        values = 100 + np.cumsum(np.random.normal(0, 3, 50))

    else:
        raise ValueError(f"Unknown series type: {series_type}")

    return pd.Series(values, index=dates)


def demo_single_series(series_type: str, title: str):
    """
    Demo AutoML su singola serie
    """
    print(f"\n{'=' * 60}")
    print(f" DEMO: {title}")
    print(f"{'=' * 60}")

    # 1. Genera dati
    print("📊 Generating test data...")
    series = generate_test_series(series_type, length=300)
    print(f"   Length: {len(series)}")
    print(f"   Mean: {series.mean():.2f}")
    print(f"   Std: {series.std():.2f}")
    print(f"   Zero ratio: {(series == 0).sum() / len(series):.1%}")

    # 2. AutoML Magic!
    print("\n🤖 AutoML Analysis...")
    automl = AutoForecastSelector(validation_split=0.2, max_models_to_try=5, verbose=True)

    start_time = time.time()
    best_model, explanation = automl.fit(series)
    total_time = time.time() - start_time

    # 3. Results
    print(f"\n🏆 RESULTS (Total time: {total_time:.1f}s)")
    print("-" * 40)
    print(f"✨ Recommended Model: {explanation.recommended_model}")
    print(f"🎯 Confidence: {explanation.confidence_score:.1%}")
    print(f"🔍 Pattern: {explanation.pattern_detected}")
    print(f"💡 Why: {explanation.why_chosen}")
    print(f"📈 Business: {explanation.business_recommendation}")
    print(f"⚠️  Risk: {explanation.risk_assessment}")

    # 4. Model Comparison
    print(f"\n📊 Model Comparison:")
    comparison = automl.get_model_comparison()
    if not comparison.empty:
        print(comparison.to_string(index=False))

    # 5. Quick forecast test
    print(f"\n🔮 Quick Forecast (30 days):")
    try:
        forecast = best_model.forecast(steps=30)
        print(f"   Mean forecast: {np.mean(forecast):.2f}")
        print(f"   Forecast range: {np.min(forecast):.2f} - {np.max(forecast):.2f}")

        # Per intermittent, mostra anche inventory params
        if explanation.recommended_model == "Intermittent":
            try:
                safety_stock = best_model.calculate_safety_stock(lead_time=15, service_level=0.95)
                reorder_point = best_model.calculate_reorder_point(lead_time=15, service_level=0.95)
                print(f"   💼 Safety Stock: {safety_stock:.0f} units")
                print(f"   📦 Reorder Point: {reorder_point:.0f} units")
            except:
                pass

    except Exception as e:
        print(f"   Forecast failed: {e}")

    return automl, explanation


def comprehensive_demo():
    """
    Demo completo AutoML con tutti i tipi di serie
    """
    print("🚀 AutoML Forecasting Engine - Comprehensive Demo")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test cases
    test_cases = [
        ("regular", "Serie Temporale Regolare"),
        ("seasonal", "Serie con Stagionalità"),
        ("trending", "Serie con Trend Forte"),
        ("intermittent", "Domanda Intermittente (Spare Parts)"),
        ("volatile", "Serie Volatile con Outlier"),
        ("short", "Serie Breve (<100 obs)"),
    ]

    results = {}

    for series_type, title in test_cases:
        try:
            automl, explanation = demo_single_series(series_type, title)
            results[series_type] = {"automl": automl, "explanation": explanation, "success": True}
        except Exception as e:
            print(f"❌ Demo failed for {series_type}: {e}")
            results[series_type] = {"error": str(e), "success": False}

    # Summary finale
    print(f"\n{'=' * 70}")
    print(" SUMMARY FINALE")
    print(f"{'=' * 70}")

    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"✅ Test completati: {successful}/{total}")

    if successful > 0:
        print(f"\n📊 Modelli Raccomandati:")
        print("-" * 40)
        for series_type, result in results.items():
            if result["success"]:
                model = result["explanation"].recommended_model
                confidence = result["explanation"].confidence_score
                print(f"   {series_type:12} -> {model:15} ({confidence:.1%})")

    print(f"\n🎉 AutoML Demo Completato!")
    return results


def quick_demo():
    """
    Demo veloce in 10 righe di codice
    """
    print("\n[AUTOML] QUICK DEMO - AutoML in 10 righe!")
    print("-" * 50)

    # 1. Crea dati
    data = generate_test_series("seasonal", 200)

    # 2. AutoML magic
    automl = AutoForecastSelector(verbose=False)
    model, explanation = automl.fit(data)

    # 3. Risultati
    print(f"[BEST] Model: {explanation.recommended_model}")
    print(f"[CONF] Confidence: {explanation.confidence_score:.1%}")
    print(f"[WHY] {explanation.why_chosen}")

    # 4. Forecast
    forecast = model.forecast(30)
    print(f"[FORECAST] 30-day: {np.mean(forecast):.1f} +/- {np.std(forecast):.1f}")

    return model, explanation


def business_case_demo():
    """
    Demo business case realistico
    """
    print(f"\n{'=' * 60}")
    print(" BUSINESS CASE: Portfolio Prodotti Azienda")
    print(f"{'=' * 60}")

    # Simula portfolio aziendale
    products = {
        "Prodotto_A_FastMoving": "regular",
        "Prodotto_B_Stagionale": "seasonal",
        "Ricambio_C_SparePart": "intermittent",
        "Prodotto_D_Crescita": "trending",
        "Prodotto_E_Volatile": "volatile",
    }

    portfolio_results = {}
    total_time = 0

    print("🏭 Analizzando portfolio aziendale...")

    for product, pattern in products.items():
        print(f"\n📦 {product}:")

        # Genera dati specifici prodotto
        series = generate_test_series(pattern, length=250)

        # AutoML rapido
        automl = AutoForecastSelector(verbose=False)
        start = time.time()
        model, explanation = automl.fit(series)
        elapsed = time.time() - start
        total_time += elapsed

        # Risultati
        print(f"   Modello: {explanation.recommended_model}")
        print(f"   Confidence: {explanation.confidence_score:.1%}")
        print(f"   Pattern: {explanation.pattern_detected[:50]}...")
        print(f"   Time: {elapsed:.1f}s")

        portfolio_results[product] = {
            "model": explanation.recommended_model,
            "confidence": explanation.confidence_score,
            "pattern": explanation.pattern_detected,
            "business": explanation.business_recommendation,
        }

    # Portfolio summary
    print(f"\n📊 PORTFOLIO SUMMARY")
    print("-" * 40)
    print(f"Prodotti analizzati: {len(products)}")
    print(f"Tempo totale: {total_time:.1f}s")
    print(f"Tempo medio/prodotto: {total_time / len(products):.1f}s")

    # Breakdown modelli
    model_counts = {}
    for result in portfolio_results.values():
        model = result["model"]
        model_counts[model] = model_counts.get(model, 0) + 1

    print(f"\nModelli utilizzati:")
    for model, count in model_counts.items():
        print(f"   {model}: {count} prodotti")

    # ROI estimate
    print(f"\n💰 BUSINESS VALUE ESTIMATE:")
    print(f"   • Tempo risparmio vs selezione manuale: ~{len(products) * 2:.0f} ore")
    print(f"   • Costo analysis manual (~€50/h): €{len(products) * 100:.0f}")
    print(f"   • AutoML time cost (~€0.50/min): €{total_time * 0.50 / 60:.0f}")
    print(f"   • NET SAVINGS: €{len(products) * 100 - (total_time * 0.50 / 60):.0f}")
    print(f"   • ROI: {((len(products) * 100) / max(total_time * 0.50 / 60, 1) - 1) * 100:.0f}%")

    return portfolio_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_demo()
        elif sys.argv[1] == "--business":
            business_case_demo()
        elif sys.argv[1] == "--single":
            demo_single_series("seasonal", "Test Singolo")
        else:
            print("Usage: python automl_showcase.py [--quick|--business|--single]")
    else:
        # Demo completo
        comprehensive_demo()

        # Bonus: business case
        business_case_demo()
