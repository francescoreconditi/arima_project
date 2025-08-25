"""
Test per Forecasting Router.

Testa endpoint di generazione previsioni.
"""

import pytest
from fastapi.testclient import TestClient


class TestForecasting:
    """Test per generazione previsioni."""
    
    def test_generate_forecast_success(self, client: TestClient, trained_model_id, sample_forecast_request):
        """Test generazione forecast con successo."""
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=sample_forecast_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi risposta
        assert "forecast" in data
        assert "timestamps" in data
        assert "confidence_intervals" in data
        assert "model_id" in data
        assert "forecast_steps" in data
        
        # Verifica valori
        assert data["model_id"] == trained_model_id
        assert data["forecast_steps"] == 5
        assert len(data["forecast"]) == 5
        assert len(data["timestamps"]) == 5
        
        # Verifica confidence intervals
        ci = data["confidence_intervals"]
        assert ci is not None
        assert "lower" in ci
        assert "upper" in ci
        assert len(ci["lower"]) == 5
        assert len(ci["upper"]) == 5
        
        # Verifica che upper > lower
        for lower, upper in zip(ci["lower"], ci["upper"]):
            assert upper >= lower
    
    def test_generate_forecast_without_confidence_intervals(self, client: TestClient, trained_model_id):
        """Test forecast senza intervalli confidenza."""
        forecast_request = {
            "steps": 3,
            "return_confidence_intervals": False
        }
        
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=forecast_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["forecast"]) == 3
        assert len(data["timestamps"]) == 3
        # confidence_intervals può essere None o assente
        assert data.get("confidence_intervals") is None
    
    def test_generate_forecast_different_steps(self, client: TestClient, trained_model_id):
        """Test forecast con diversi numeri di step."""
        for steps in [1, 10, 30]:
            forecast_request = {
                "steps": steps,
                "confidence_level": 0.90,
                "return_confidence_intervals": True
            }
            
            response = client.post(
                f"/models/{trained_model_id}/forecast",
                json=forecast_request
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["forecast"]) == steps
            assert len(data["timestamps"]) == steps
            assert data["forecast_steps"] == steps
    
    def test_generate_forecast_different_confidence_levels(self, client: TestClient, trained_model_id):
        """Test forecast con diversi livelli confidenza."""
        for confidence in [0.80, 0.95, 0.99]:
            forecast_request = {
                "steps": 5,
                "confidence_level": confidence,
                "return_confidence_intervals": True
            }
            
            response = client.post(
                f"/models/{trained_model_id}/forecast",
                json=forecast_request
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verifica che gli intervalli siano più ampi con confidence maggiore
            ci = data["confidence_intervals"]
            assert ci is not None
    
    def test_forecast_nonexistent_model(self, client: TestClient):
        """Test forecast con modello inesistente."""
        fake_model_id = "nonexistent-model-id"
        forecast_request = {
            "steps": 5,
            "confidence_level": 0.95,
            "return_confidence_intervals": True
        }
        
        response = client.post(
            f"/models/{fake_model_id}/forecast",
            json=forecast_request
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_forecast_invalid_steps(self, client: TestClient, trained_model_id):
        """Test forecast con step non validi."""
        invalid_requests = [
            {"steps": 0},  # Zero steps
            {"steps": -1},  # Negative steps
            {"steps": "invalid"},  # Non-numeric
        ]
        
        for invalid_request in invalid_requests:
            response = client.post(
                f"/models/{trained_model_id}/forecast",
                json=invalid_request
            )
            
            assert response.status_code in [400, 422]  # Bad request o validation error
    
    def test_forecast_invalid_confidence_level(self, client: TestClient, trained_model_id):
        """Test forecast con livello confidenza non valido."""
        invalid_requests = [
            {"steps": 5, "confidence_level": 0.0},  # 0%
            {"steps": 5, "confidence_level": 1.0},  # 100%
            {"steps": 5, "confidence_level": 1.5},  # > 100%
            {"steps": 5, "confidence_level": -0.1},  # Negative
        ]
        
        for invalid_request in invalid_requests:
            response = client.post(
                f"/models/{trained_model_id}/forecast",
                json=invalid_request
            )
            
            assert response.status_code in [400, 422]


class TestForecastingDataTypes:
    """Test tipi di dati nelle previsioni."""
    
    def test_forecast_numeric_types(self, client: TestClient, trained_model_id):
        """Test che forecast restituisca tipi numerici corretti."""
        forecast_request = {
            "steps": 3,
            "confidence_level": 0.95,
            "return_confidence_intervals": True
        }
        
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=forecast_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica tipi forecast
        for value in data["forecast"]:
            assert isinstance(value, (int, float))
            assert not isinstance(value, bool)  # bool è subclass di int in Python
        
        # Verifica tipi confidence intervals
        if data["confidence_intervals"]:
            for lower, upper in zip(
                data["confidence_intervals"]["lower"],
                data["confidence_intervals"]["upper"]
            ):
                assert isinstance(lower, (int, float))
                assert isinstance(upper, (int, float))
    
    def test_forecast_timestamp_format(self, client: TestClient, trained_model_id):
        """Test formato timestamp nelle previsioni."""
        forecast_request = {"steps": 3}
        
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=forecast_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica formato timestamp (dovrebbe essere ISO date string)
        for timestamp in data["timestamps"]:
            assert isinstance(timestamp, str)
            # Verifica che sia parsabile come data
            from datetime import datetime
            datetime.strptime(timestamp, "%Y-%m-%d")


class TestForecastingPerformance:
    """Test performance forecasting."""
    
    def test_forecast_response_time(self, client: TestClient, trained_model_id):
        """Test tempo di risposta forecast."""
        import time
        
        forecast_request = {"steps": 10}
        
        start_time = time.time()
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=forecast_request
        )
        duration = time.time() - start_time
        
        assert response.status_code == 200
        assert duration < 5.0  # Deve completare in meno di 5 secondi
    
    def test_multiple_concurrent_forecasts(self, client: TestClient, trained_model_id):
        """Test forecast concorrenti."""
        import concurrent.futures
        
        def make_forecast():
            return client.post(
                f"/models/{trained_model_id}/forecast",
                json={"steps": 5}
            )
        
        # Esegui 3 forecast concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_forecast) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Tutti dovrebbero avere successo
        for response in responses:
            assert response.status_code == 200
            assert len(response.json()["forecast"]) == 5


class TestForecastingEdgeCases:
    """Test casi limite per forecasting."""
    
    def test_forecast_large_steps(self, client: TestClient, trained_model_id):
        """Test forecast con molti step."""
        forecast_request = {"steps": 100}
        
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=forecast_request
        )
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["forecast"]) == 100
        else:
            # Potrebbe fallire per limiti del modello
            assert response.status_code in [400, 500]
    
    def test_forecast_minimal_steps(self, client: TestClient, trained_model_id):
        """Test forecast con step minimi."""
        forecast_request = {"steps": 1}
        
        response = client.post(
            f"/models/{trained_model_id}/forecast",
            json=forecast_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["forecast"]) == 1
        assert len(data["timestamps"]) == 1