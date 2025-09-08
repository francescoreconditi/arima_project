#!/usr/bin/env python3
"""
Test script per verificare gli health endpoints.

Avvia l'API e testa tutti gli endpoint di health monitoring.
"""

import asyncio
import json
import requests
import time
from datetime import datetime


async def test_health_endpoints():
    """Testa tutti gli health endpoints."""
    
    print(f"🏥 TEST HEALTH ENDPOINTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Base URL (assumendo API in ascolto su porta 8000)
    base_url = "http://localhost:8000"
    
    # Lista endpoints da testare
    endpoints = [
        "/health/simple",
        "/health",
        "/health/ready", 
        "/health/metrics",
        "/health/performance",
        "/health/all"
    ]
    
    print("📡 Testando connessione API...")
    
    try:
        # Test connessione base
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ API connessa e funzionante")
        else:
            print(f"⚠️  API risponde con status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API non raggiungibile: {e}")
        print("\n💡 Per avviare l'API:")
        print("   uv run python scripts/run_api.py")
        print("   oppure:")
        print("   uv run uvicorn src.arima_forecaster.api.main:app --reload")
        return
    
    print()
    
    # Test ogni endpoint
    for endpoint in endpoints:
        print(f"🔍 Testing {endpoint}")
        
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ⏱️  Response Time: {response_time*1000:.1f}ms")
                
                # Informazioni chiave per ogni endpoint
                if endpoint == "/health/simple":
                    print(f"   📊 Status: {data.get('status', 'N/A')}")
                    
                elif endpoint == "/health":
                    print(f"   📊 Status: {data.get('status', 'N/A')}")
                    print(f"   📦 Version: {data.get('version', 'N/A')}")
                    deps = data.get('dependencies', {})
                    deps_ok = sum(1 for v in deps.values() if v)
                    print(f"   🔗 Dependencies: {deps_ok}/{len(deps)} OK")
                    
                elif endpoint == "/health/ready":
                    print(f"   ✅ Ready: {data.get('ready', 'N/A')}")
                    
                elif endpoint == "/health/metrics":
                    system = data.get('system', {})
                    if 'memory' in system:
                        print(f"   💾 Memory: {system['memory'].get('percent', 0):.1f}%")
                    if 'cpu_percent' in system:
                        print(f"   🖥️  CPU: {system.get('cpu_percent', 0):.1f}%")
                        
                elif endpoint == "/health/performance":
                    perf = data.get('performance', {})
                    if 'total_requests' in perf:
                        print(f"   📈 Requests: {perf.get('total_requests', 0)}")
                        print(f"   ⚡ Avg Response: {perf.get('average_response_time_ms', 0):.1f}ms")
                        print(f"   ❌ Error Rate: {perf.get('error_rate', 0)*100:.1f}%")
                        
                elif endpoint == "/health/all":
                    print(f"   🎯 Overall Status: {data.get('overall_status', 'N/A')}")
                    alerts = data.get('alerts', {})
                    active_alerts = sum(1 for v in alerts.values() if v)
                    print(f"   🚨 Active Alerts: {active_alerts}/{len(alerts)}")
                
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   💬 Response: {response.text[:100]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ Health endpoints test completato!")
    
    # Suggerimenti per monitoring
    print("\n💡 SUGGERIMENTI MONITORING:")
    print("• Load Balancer: usa /health/simple (più veloce)")
    print("• Kubernetes: usa /health/ready per readiness probe")
    print("• Monitoring System: usa /health/all per dashboard complete")
    print("• Performance Tracking: usa /health/performance")
    print("• System Metrics: usa /health/metrics per dettagli sistema")


if __name__ == "__main__":
    asyncio.run(test_health_endpoints())