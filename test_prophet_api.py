#!/usr/bin/env python3
"""
Test script per verificare gli endpoints Prophet della API FastAPI.

Testa tutti i nuovi endpoints Prophet implementati:
1. Training Prophet base
2. Auto-selection Prophet  
3. Lista modelli Prophet
4. Forecasting standard
5. Forecasting Prophet avanzato
6. Comparazione modelli

Esegui con: uv run python test_prophet_api.py
"""

import requests
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configurazione API
API_BASE = "http://localhost:8000"
TIMEOUT = 30  # secondi

def generate_test_data(days=200):
    """Genera dati demo per testing."""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Trend + stagionalità + rumore
    trend = np.linspace(100, 150, days)
    weekly = 10 * np.sin(np.arange(days) * 2 * np.pi / 7)
    noise = np.random.normal(0, 3, days)
    values = trend + weekly + noise
    
    return {
        "timestamps": [d.strftime('%Y-%m-%d') for d in dates],
        "values": values.tolist()
    }

def test_api_health():
    """Test health endpoint."""
    print("\n[TEST] 🏥 Health Check")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure the API is running with: uv run python scripts/run_api.py")
        return False

def test_prophet_training():
    """Test Prophet training endpoint."""
    print("\n[TEST] 🎨 Prophet Training")
    
    data = generate_test_data(150)
    
    request_data = {
        "data": data,
        "growth": "linear",
        "yearly_seasonality": False,
        "weekly_seasonality": True, 
        "daily_seasonality": False,
        "seasonality_mode": "additive",
        "country_holidays": "IT"
    }
    
    try:
        response = requests.post(f"{API_BASE}/models/train/prophet", 
                               json=request_data, timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            model_id = result["model_id"]
            print(f"✅ Prophet training started: {model_id}")
            print(f"   Estimated time: {result['estimated_time_seconds']}s")
            return model_id
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Training request failed: {e}")
        return None

def test_prophet_auto_selection():
    """Test Prophet auto-selection endpoint."""
    print("\n[TEST] 🧠 Prophet Auto-Selection")
    
    data = generate_test_data(180)
    
    request_data = {
        "data": data,
        "growth_types": ["linear"],  # Limitato per test veloce
        "seasonality_modes": ["additive"],
        "country_holidays": ["IT", None],
        "max_models": 4,  # Test veloce
        "cv_horizon": "30 days"
    }
    
    try:
        response = requests.post(f"{API_BASE}/models/train/prophet/auto-select",
                               json=request_data, timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            model_id = result["model_id"]
            print(f"✅ Prophet auto-selection started: {model_id}")
            print(f"   Max models: {result['search_space']['max_models_tested']}")
            print(f"   Estimated time: {result['estimated_time_seconds']}s")
            return model_id
        else:
            print(f"❌ Auto-selection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Auto-selection request failed: {e}")
        return None

def wait_for_completion(model_id, max_wait=300):
    """Attende il completamento del training."""
    print(f"⏳ Waiting for model {model_id} completion...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{API_BASE}/models/{model_id}", timeout=10)
            if response.status_code == 200:
                status = response.json().get("status", "unknown")
                print(f"   Status: {status}")
                
                if status == "completed":
                    print("✅ Model training completed!")
                    return True
                elif status == "failed":
                    print("❌ Model training failed!")
                    return False
                    
            time.sleep(10)  # Wait 10 seconds before next check
            
        except Exception as e:
            print(f"   Error checking status: {e}")
            time.sleep(5)
    
    print(f"⏰ Timeout waiting for model completion ({max_wait}s)")
    return False

def test_prophet_forecasting(model_id):
    """Test Prophet forecasting endpoints."""
    print(f"\n[TEST] 📈 Prophet Forecasting: {model_id}")
    
    # Test standard forecasting
    forecast_request = {
        "steps": 15,
        "confidence_level": 0.95,
        "return_intervals": True
    }
    
    try:
        # Standard forecast
        response = requests.post(f"{API_BASE}/models/{model_id}/forecast",
                               json=forecast_request, timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Standard forecast successful")
            print(f"   Forecast length: {len(result['forecast'])}")
            print(f"   Has confidence intervals: {result['confidence_intervals'] is not None}")
        else:
            print(f"❌ Standard forecast failed: {response.status_code}")
            return False
            
        # Prophet advanced forecast (only if Prophet model)
        response = requests.post(f"{API_BASE}/models/{model_id}/forecast/prophet",
                               json=forecast_request, timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prophet advanced forecast successful")
            print(f"   Components available: {list(result['prophet_components'].keys())}")
            print(f"   Changepoints: {len(result['changepoints'].get('dates', []))}")
            print(f"   Decomposition info: {result['decomposition_info']}")
        elif response.status_code == 400:
            print("ℹ️  Advanced forecast skipped (not a Prophet model)")
        else:
            print(f"❌ Prophet advanced forecast failed: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Forecasting test failed: {e}")
        return False

def test_prophet_models_list():
    """Test Prophet models listing."""
    print("\n[TEST] 📋 Prophet Models List")
    
    try:
        response = requests.get(f"{API_BASE}/models/train/prophet/models", timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prophet models list successful")
            print(f"   Total Prophet models: {result['total_count']}")
            print(f"   By status: {result['by_status']}")
            print(f"   By type: {result['model_types']}")
            return result.get("models", [])
        else:
            print(f"❌ Models list failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Models list test failed: {e}")
        return []

def test_models_comparison(model_ids):
    """Test models comparison endpoint."""
    print("\n[TEST] 🆚 Models Comparison")
    
    if len(model_ids) < 2:
        print("⚠️  Need at least 2 models for comparison test")
        return False
    
    comparison_request = {
        "model_ids": model_ids[:3]  # Test max 3 modelli
    }
    
    try:
        response = requests.post(f"{API_BASE}/models/compare",
                               json=comparison_request, timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Models comparison successful")
            
            best = result["comparison_summary"]["best_model"]
            print(f"   Best model: {best['model_type']} ({best['model_id'][:12]}...)")
            print(f"   Overall score: {best['overall_score']}")
            
            recs = result["recommendations"]
            print(f"   Best for accuracy: {recs.get('best_for_accuracy', 'N/A')[:12]}...")
            print(f"   Best for speed: {recs.get('best_for_speed', 'N/A')[:12]}...")
            print(f"   Best for seasonality: {recs.get('best_for_seasonality', 'N/A')[:12]}...")
            
            return True
        else:
            print(f"❌ Comparison failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Esegue tutti i test Prophet API."""
    print("🚀 PROPHET API TESTING")
    print("=" * 50)
    
    # Test risultati
    results = {
        "health": False,
        "training": False,
        "auto_selection": False,
        "forecasting": False,
        "models_list": False,
        "comparison": False
    }
    
    model_ids = []
    
    # 1. Health check
    results["health"] = test_api_health()
    if not results["health"]:
        print("\n❌ Cannot proceed without API connection")
        return
    
    # 2. Prophet training
    training_model_id = test_prophet_training()
    if training_model_id:
        results["training"] = True
        model_ids.append(training_model_id)
    
    # 3. Prophet auto-selection
    auto_model_id = test_prophet_auto_selection()
    if auto_model_id:
        results["auto_selection"] = True
        model_ids.append(auto_model_id)
    
    # 4. Wait for at least one model to complete
    if model_ids:
        print(f"\n⏳ Waiting for model completion (max 2 minutes)...")
        completed_models = []
        
        for model_id in model_ids:
            if wait_for_completion(model_id, max_wait=120):
                completed_models.append(model_id)
        
        # 5. Test forecasting on completed models
        if completed_models:
            results["forecasting"] = test_prophet_forecasting(completed_models[0])
        
        # 6. Test models listing
        prophet_models = test_prophet_models_list()
        if prophet_models:
            results["models_list"] = True
            
            # Add any existing models to comparison
            existing_model_ids = [m["model_id"] for m in prophet_models]
            all_model_ids = list(set(completed_models + existing_model_ids))
            
            # 7. Test comparison
            if len(all_model_ids) >= 2:
                results["comparison"] = test_models_comparison(all_model_ids)
            else:
                print("\n⚠️  Skipping comparison test (need ≥2 models)")
    
    # Summary risultati
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Prophet API is fully functional.")
    elif passed_tests >= total_tests * 0.7:
        print("✅ Most tests passed. Prophet API is mostly functional.")
    else:
        print("⚠️  Several tests failed. Check API implementation.")
    
    print(f"\n💡 Check full API documentation at: {API_BASE}/docs")

if __name__ == "__main__":
    main()