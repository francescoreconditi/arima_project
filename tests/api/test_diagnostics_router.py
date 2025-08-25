"""
Test per Diagnostics Router.

Testa endpoint di diagnostica modelli.
"""

import pytest
from fastapi.testclient import TestClient


class TestModelDiagnostics:
    """Test per diagnostica modelli."""
    
    def test_get_diagnostics_success(self, client: TestClient, trained_model_id):
        """Test diagnostica modello con successo."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi principali
        required_fields = [
            "residuals_stats",
            "ljung_box_test", 
            "jarque_bera_test",
            "acf_values",
            "pacf_values",
            "performance_metrics"
        ]
        
        for field in required_fields:
            assert field in data
    
    def test_residuals_statistics(self, client: TestClient, trained_model_id):
        """Test statistiche residui."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        residuals_stats = data["residuals_stats"]
        
        # Verifica campi statistiche
        stat_fields = ["mean", "std", "skewness", "kurtosis"]
        for field in stat_fields:
            assert field in residuals_stats
            assert isinstance(residuals_stats[field], (int, float))
        
        # Verifica valori ragionevoli
        assert residuals_stats["std"] >= 0  # Deviazione standard non negativa
        # Media dovrebbe essere vicina a zero per buoni residui
        assert abs(residuals_stats["mean"]) < 10  # Tolleranza ragionevole
    
    def test_ljung_box_test(self, client: TestClient, trained_model_id):
        """Test Ljung-Box per autocorrelazione."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        ljung_box = data["ljung_box_test"]
        
        # Verifica campi test
        assert "statistic" in ljung_box
        assert "p_value" in ljung_box
        assert "result" in ljung_box
        
        # Verifica tipi
        assert isinstance(ljung_box["statistic"], (int, float))
        assert isinstance(ljung_box["p_value"], (int, float))
        assert isinstance(ljung_box["result"], str)
        
        # Verifica valori
        assert ljung_box["statistic"] >= 0
        assert 0 <= ljung_box["p_value"] <= 1
        
        # Verifica interpretazione
        if ljung_box["p_value"] > 0.05:
            assert "no autocorrelation" in ljung_box["result"].lower()
        else:
            assert "autocorrelation" in ljung_box["result"].lower()
    
    def test_jarque_bera_test(self, client: TestClient, trained_model_id):
        """Test Jarque-Bera per normalità."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        jarque_bera = data["jarque_bera_test"]
        
        # Verifica campi test
        assert "statistic" in jarque_bera
        assert "p_value" in jarque_bera
        assert "result" in jarque_bera
        
        # Verifica tipi e valori
        assert isinstance(jarque_bera["statistic"], (int, float))
        assert isinstance(jarque_bera["p_value"], (int, float))
        assert isinstance(jarque_bera["result"], str)
        
        assert jarque_bera["statistic"] >= 0
        assert 0 <= jarque_bera["p_value"] <= 1
        
        # Verifica interpretazione
        if jarque_bera["p_value"] > 0.05:
            assert "normally distributed" in jarque_bera["result"].lower()
        else:
            assert "not normally distributed" in jarque_bera["result"].lower()
    
    def test_acf_pacf_values(self, client: TestClient, trained_model_id):
        """Test valori ACF e PACF."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        acf_values = data["acf_values"]
        pacf_values = data["pacf_values"]
        
        # Verifica che siano liste
        assert isinstance(acf_values, list)
        assert isinstance(pacf_values, list)
        
        # Dovrebbero avere almeno alcuni valori
        assert len(acf_values) > 0
        assert len(pacf_values) > 0
        
        # Primo valore ACF dovrebbe essere 1 (autocorrelazione con se stesso)
        assert abs(acf_values[0] - 1.0) < 0.001
        
        # Tutti i valori dovrebbero essere numerici e tra -1 e 1
        for acf_val in acf_values:
            assert isinstance(acf_val, (int, float))
            assert -1 <= acf_val <= 1
        
        for pacf_val in pacf_values:
            assert isinstance(pacf_val, (int, float))
            assert -1 <= pacf_val <= 1
    
    def test_performance_metrics(self, client: TestClient, trained_model_id):
        """Test metriche di performance."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        metrics = data["performance_metrics"]
        assert isinstance(metrics, dict)
        
        # Verifica che tutti i valori siano numerici o None
        for key, value in metrics.items():
            if value is not None:
                assert isinstance(value, (int, float))
                assert not (isinstance(value, float) and (value != value))  # No NaN
        
        # Se ci sono metriche, verifica valori ragionevoli
        if "mae" in metrics and metrics["mae"] is not None:
            assert metrics["mae"] >= 0  # MAE non può essere negativo
        
        if "rmse" in metrics and metrics["rmse"] is not None:
            assert metrics["rmse"] >= 0  # RMSE non può essere negativo
        
        if "r2" in metrics and metrics["r2"] is not None:
            # R² può essere negativo per modelli molto cattivi, ma non > 1
            assert metrics["r2"] <= 1


class TestDiagnosticsErrorCases:
    """Test casi di errore per diagnostica."""
    
    def test_diagnostics_nonexistent_model(self, client: TestClient):
        """Test diagnostica con modello inesistente."""
        fake_model_id = "nonexistent-model-789"
        response = client.post(f"/models/{fake_model_id}/diagnostics")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["detail"].lower()
    
    def test_diagnostics_invalid_model_id(self, client: TestClient):
        """Test diagnostica con ID modello non valido."""
        invalid_ids = [
            "",  # ID vuoto
            "invalid-id-format",
            "   ",  # Solo spazi
        ]
        
        for invalid_id in invalid_ids:
            if invalid_id.strip():  # Skip ID vuoti
                response = client.post(f"/models/{invalid_id}/diagnostics")
                assert response.status_code in [400, 404, 422]
    
    def test_diagnostics_model_without_residuals(self, client: TestClient, sample_arima_request):
        """Test diagnostica con modello senza residui (appena creato)."""
        # Crea modello ma non aspettare che completi il training
        train_response = client.post("/models/train", json=sample_arima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Prova diagnostica immediata (potrebbe non avere residui)
        diag_response = client.post(f"/models/{model_id}/diagnostics")
        
        # Potrebbe essere 400 se non ha residui, o 200 se il training è veloce
        if diag_response.status_code == 400:
            error_data = diag_response.json()
            assert "residuals" in error_data["detail"].lower()
        else:
            assert diag_response.status_code == 200


class TestDiagnosticsDataConsistency:
    """Test consistenza dati diagnostici."""
    
    def test_diagnostics_data_types(self, client: TestClient, trained_model_id):
        """Test che tutti i tipi di dati siano corretti."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica tipi strutture principali
        assert isinstance(data["residuals_stats"], dict)
        assert isinstance(data["ljung_box_test"], dict)
        assert isinstance(data["jarque_bera_test"], dict)
        assert isinstance(data["acf_values"], list)
        assert isinstance(data["pacf_values"], list)
        assert isinstance(data["performance_metrics"], dict)
    
    def test_diagnostics_value_ranges(self, client: TestClient, trained_model_id):
        """Test che i valori siano in range ragionevoli."""
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Test range p-values (devono essere 0-1)
        assert 0 <= data["ljung_box_test"]["p_value"] <= 1
        assert 0 <= data["jarque_bera_test"]["p_value"] <= 1
        
        # Test range ACF/PACF (-1 to 1)
        for acf_val in data["acf_values"]:
            assert -1 <= acf_val <= 1
        
        for pacf_val in data["pacf_values"]:
            assert -1 <= pacf_val <= 1
        
        # Test statistiche residui ragionevoli
        stats = data["residuals_stats"]
        assert stats["std"] >= 0  # Deviazione standard non negativa
        assert isinstance(stats["skewness"], (int, float))
        assert isinstance(stats["kurtosis"], (int, float))
    
    def test_diagnostics_consistency_across_calls(self, client: TestClient, trained_model_id):
        """Test che la diagnostica sia consistente tra chiamate multiple."""
        # Prima chiamata
        response1 = client.post(f"/models/{trained_model_id}/diagnostics")
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Seconda chiamata
        response2 = client.post(f"/models/{trained_model_id}/diagnostics")
        assert response2.status_code == 200
        data2 = response2.json()
        
        # I risultati dovrebbero essere identici (stesso modello)
        assert data1["residuals_stats"] == data2["residuals_stats"]
        assert data1["ljung_box_test"] == data2["ljung_box_test"]
        assert data1["jarque_bera_test"] == data2["jarque_bera_test"]
        assert data1["acf_values"] == data2["acf_values"]
        assert data1["pacf_values"] == data2["pacf_values"]


class TestDiagnosticsPerformance:
    """Test performance diagnostica."""
    
    def test_diagnostics_response_time(self, client: TestClient, trained_model_id):
        """Test tempo di risposta diagnostica."""
        import time
        
        start_time = time.time()
        response = client.post(f"/models/{trained_model_id}/diagnostics")
        duration = time.time() - start_time
        
        assert response.status_code == 200
        assert duration < 5.0  # Deve completare in meno di 5 secondi
    
    def test_concurrent_diagnostics(self, client: TestClient, trained_model_id):
        """Test diagnostica concorrente."""
        import concurrent.futures
        
        def run_diagnostics():
            return client.post(f"/models/{trained_model_id}/diagnostics")
        
        # Esegui 3 diagnostica concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_diagnostics) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Tutte dovrebbero avere successo
        for response in responses:
            assert response.status_code == 200
            
        # I risultati dovrebbero essere identici
        data_sets = [resp.json() for resp in responses]
        for i in range(1, len(data_sets)):
            assert data_sets[0]["residuals_stats"] == data_sets[i]["residuals_stats"]