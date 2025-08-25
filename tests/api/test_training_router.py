"""
Test per Training Router.

Testa endpoint di addestramento modelli.
"""

import pytest
from fastapi.testclient import TestClient


class TestModelTraining:
    """Test per training modelli ARIMA/SARIMA."""
    
    def test_train_arima_model_success(self, client: TestClient, sample_arima_request):
        """Test training ARIMA con successo."""
        response = client.post("/models/train", json=sample_arima_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi risposta
        assert "model_id" in data
        assert "model_type" in data
        assert "status" in data
        assert "created_at" in data
        assert "training_observations" in data
        assert "parameters" in data
        assert "metrics" in data
        
        # Verifica valori
        assert data["model_type"] == "arima"
        assert data["status"] == "training"
        assert data["training_observations"] == 20
        assert isinstance(data["parameters"], dict)
        assert isinstance(data["metrics"], dict)
    
    def test_train_sarima_model_success(self, client: TestClient, sample_sarima_request):
        """Test training SARIMA con successo."""
        response = client.post("/models/train", json=sample_sarima_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_type"] == "sarima"
        assert data["status"] == "training"
        assert data["training_observations"] == 20
    
    def test_train_model_invalid_data(self, client: TestClient):
        """Test training con dati non validi."""
        invalid_request = {
            "model_type": "arima",
            "data": {
                "timestamps": ["2023-01-01"],
                "values": []  # Array vuoto non valido
            },
            "order": {"p": 1, "d": 1, "q": 1}
        }
        
        response = client.post("/models/train", json=invalid_request)
        assert response.status_code == 400
    
    def test_train_model_missing_fields(self, client: TestClient):
        """Test training con campi mancanti."""
        incomplete_request = {
            "model_type": "arima",
            # Manca 'data' e 'order'
        }
        
        response = client.post("/models/train", json=incomplete_request)
        assert response.status_code == 422  # Validation error
    
    def test_train_model_invalid_model_type(self, client: TestClient, sample_time_series_data):
        """Test training con tipo modello non valido."""
        invalid_request = {
            "model_type": "invalid_model",
            "data": sample_time_series_data,
            "order": {"p": 1, "d": 1, "q": 1}
        }
        
        response = client.post("/models/train", json=invalid_request)
        assert response.status_code == 400


class TestVARTraining:
    """Test per training modelli VAR."""
    
    def test_train_var_model_success(self, client: TestClient, sample_var_request):
        """Test training VAR con successo."""
        response = client.post("/models/train/var", json=sample_var_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi VAR specifici
        assert "model_id" in data
        assert "model_type" in data
        assert "variables" in data
        assert "max_lags" in data
        assert "selected_lag_order" in data
        assert "causality_tests" in data
        
        assert data["model_type"] == "var"
        assert data["max_lags"] == 2
        assert "sales" in data["variables"]
        assert "temperature" in data["variables"]
    
    def test_train_var_model_single_variable(self, client: TestClient):
        """Test VAR con una sola variabile (dovrebbe fallire)."""
        single_var_request = {
            "data": {
                "series": [
                    {
                        "name": "sales",
                        "timestamps": ["2023-01-01", "2023-01-02"],
                        "values": [100, 102]
                    }
                ]
            },
            "max_lags": 2
        }
        
        response = client.post("/models/train/var", json=single_var_request)
        assert response.status_code == 400


class TestAutoSelection:
    """Test per selezione automatica parametri."""
    
    def test_auto_select_arima_success(self, client: TestClient, sample_auto_select_request):
        """Test auto-selezione ARIMA con successo."""
        response = client.post("/models/auto-select", json=sample_auto_select_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi risposta
        assert "best_model" in data
        assert "all_models" in data
        assert "search_time_seconds" in data
        
        # Verifica best_model
        best_model = data["best_model"]
        assert "order" in best_model
        assert "aic" in best_model or "bic" in best_model
        
        # Verifica all_models
        assert isinstance(data["all_models"], list)
        assert len(data["all_models"]) > 0
        
        # Verifica tempo di ricerca
        assert isinstance(data["search_time_seconds"], (int, float))
        assert data["search_time_seconds"] > 0
    
    def test_auto_select_sarima_success(self, client: TestClient, sample_time_series_data):
        """Test auto-selezione SARIMA con successo."""
        sarima_request = {
            "data": sample_time_series_data,
            "max_p": 2,
            "max_d": 1,
            "max_q": 2,
            "seasonal": True,
            "seasonal_period": 12,
            "criterion": "aic"
        }
        
        response = client.post("/models/auto-select", json=sarima_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Per SARIMA dovrebbe avere seasonal_order
        best_model = data["best_model"]
        assert "seasonal_order" in best_model
    
    def test_auto_select_invalid_criterion(self, client: TestClient, sample_time_series_data):
        """Test auto-selezione con criterio non valido."""
        invalid_request = {
            "data": sample_time_series_data,
            "max_p": 2,
            "max_d": 1,
            "max_q": 2,
            "seasonal": False,
            "criterion": "invalid_criterion"
        }
        
        response = client.post("/models/auto-select", json=invalid_request)
        assert response.status_code == 422  # Validation error


class TestTrainingEdgeCases:
    """Test casi limite per training."""
    
    def test_train_with_minimal_data(self, client: TestClient):
        """Test training con dati minimi."""
        minimal_request = {
            "model_type": "arima",
            "data": {
                "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "values": [100.0, 102.0, 101.0]
            },
            "order": {"p": 1, "d": 0, "q": 0}  # Modello semplice
        }
        
        response = client.post("/models/train", json=minimal_request)
        assert response.status_code in [200, 400]  # Può essere valido o troppo pochi dati
    
    def test_train_with_missing_values(self, client: TestClient):
        """Test training con valori mancanti."""
        # Note: questo potrebbe essere gestito dal preprocessing
        nan_request = {
            "model_type": "arima",
            "data": {
                "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "values": [100.0, None, 102.0]  # Valore mancante
            },
            "order": {"p": 1, "d": 1, "q": 1}
        }
        
        response = client.post("/models/train", json=nan_request)
        # Il comportamento dipende dalla validazione - potrebbe essere 400 o 422
        assert response.status_code in [400, 422]
    
    def test_concurrent_training_requests(self, client: TestClient, sample_arima_request):
        """Test richieste training concorrenti."""
        import concurrent.futures
        
        def make_training_request():
            return client.post("/models/train", json=sample_arima_request)
        
        # Esegui 3 richieste concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_training_request) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Tutte dovrebbero avere successo
        for response in responses:
            assert response.status_code == 200
            
        # Ogni modello dovrebbe avere ID unico
        model_ids = [resp.json()["model_id"] for resp in responses]
        assert len(set(model_ids)) == 3  # Tutti ID diversi