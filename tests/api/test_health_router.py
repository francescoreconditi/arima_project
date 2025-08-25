"""
Test per Health Router.

Testa endpoint di monitoraggio stato servizio.
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test endpoint root (/.)"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert data["message"] == "ARIMA Forecaster API"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"


def test_health_check_endpoint(client: TestClient):
    """Test endpoint health check (/health)."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "timestamp" in data
    assert "service" in data
    assert data["status"] == "healthy"
    assert data["service"] == "arima-forecaster-api"
    
    # Verifica formato timestamp ISO
    import datetime
    datetime.datetime.fromisoformat(data["timestamp"])


def test_health_endpoints_response_format(client: TestClient):
    """Test formato risposte endpoint health."""
    # Test root
    root_response = client.get("/")
    assert root_response.headers["content-type"].startswith("application/json")
    
    # Test health
    health_response = client.get("/health")
    assert health_response.headers["content-type"].startswith("application/json")


def test_health_endpoints_performance(client: TestClient):
    """Test performance endpoint health (devono essere veloci)."""
    import time
    
    # Test root
    start = time.time()
    response = client.get("/")
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 0.1  # Deve rispondere in meno di 100ms
    
    # Test health
    start = time.time()
    response = client.get("/health")
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 0.1  # Deve rispondere in meno di 100ms


def test_multiple_health_checks(client: TestClient):
    """Test chiamate multiple per verificare stabilità."""
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"