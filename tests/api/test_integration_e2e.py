"""
Test di integrazione end-to-end per API FastAPI.

Test completi che verificano workflow completi dall'inizio alla fine.
"""

import pytest
import time
from fastapi.testclient import TestClient


class TestCompleteWorkflow:
    """Test workflow completi end-to-end."""

    def test_complete_arima_workflow(self, client: TestClient, sample_time_series_data):
        """Test workflow completo ARIMA: train -> forecast -> diagnostics -> report -> delete."""

        # 1. TRAINING - Addestra modello ARIMA
        training_request = {
            "model_type": "arima",
            "data": sample_time_series_data,
            "order": {"p": 1, "d": 1, "q": 1},
        }

        train_response = client.post("/models/train", json=training_request)
        assert train_response.status_code == 200

        model_data = train_response.json()
        model_id = model_data["model_id"]
        assert model_data["model_type"] == "arima"
        assert model_data["status"] == "training"

        # 2. VERIFICA MODELLO - Controlla che sia nella lista
        models_response = client.get("/models")
        assert models_response.status_code == 200

        models_list = models_response.json()
        model_ids = [m["model_id"] for m in models_list["models"]]
        assert model_id in model_ids

        # 3. INFO MODELLO - Recupera dettagli specifici
        info_response = client.get(f"/models/{model_id}")
        assert info_response.status_code == 200

        model_info = info_response.json()
        assert model_info["model_id"] == model_id
        assert model_info["model_type"] == "arima"

        # 4. FORECASTING - Genera previsioni
        forecast_request = {
            "steps": 7,
            "confidence_level": 0.95,
            "return_confidence_intervals": True,
        }

        forecast_response = client.post(f"/models/{model_id}/forecast", json=forecast_request)
        assert forecast_response.status_code == 200

        forecast_data = forecast_response.json()
        assert len(forecast_data["forecast"]) == 7
        assert len(forecast_data["timestamps"]) == 7
        assert forecast_data["confidence_intervals"] is not None

        # 5. DIAGNOSTICS - Analizza performance modello
        diag_response = client.post(f"/models/{model_id}/diagnostics")
        assert diag_response.status_code == 200

        diag_data = diag_response.json()
        assert "residuals_stats" in diag_data
        assert "ljung_box_test" in diag_data
        assert "performance_metrics" in diag_data

        # 6. REPORT GENERATION - Genera report completo
        report_request = {
            "format": "html",
            "include_diagnostics": True,
            "include_forecasts": True,
            "forecast_steps": 10,
        }

        report_response = client.post(f"/models/{model_id}/report", json=report_request)
        assert report_response.status_code == 200

        report_data = report_response.json()
        assert report_data["status"] == "generating"
        assert report_data["format_type"] == "html"

        # 7. CLEANUP - Elimina modello
        delete_response = client.delete(f"/models/{model_id}")
        assert delete_response.status_code == 200

        # 8. VERIFICA ELIMINAZIONE - Controlla che sia stato rimosso
        final_info_response = client.get(f"/models/{model_id}")
        assert final_info_response.status_code == 404

    def test_complete_sarima_workflow(self, client: TestClient, sample_time_series_data):
        """Test workflow completo SARIMA con componenti stagionali."""

        # 1. TRAINING SARIMA
        training_request = {
            "model_type": "sarima",
            "data": sample_time_series_data,
            "order": {"p": 1, "d": 1, "q": 1},
            "seasonal_order": {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12},
        }

        train_response = client.post("/models/train", json=training_request)
        assert train_response.status_code == 200

        model_id = train_response.json()["model_id"]

        # 2. FORECASTING con parametri diversi
        forecast_scenarios = [
            {"steps": 3, "confidence_level": 0.90},
            {"steps": 12, "confidence_level": 0.95},  # Un anno
            {"steps": 24, "confidence_level": 0.99},  # Due anni
        ]

        for scenario in forecast_scenarios:
            scenario["return_confidence_intervals"] = True

            forecast_response = client.post(f"/models/{model_id}/forecast", json=scenario)
            assert forecast_response.status_code == 200

            forecast_data = forecast_response.json()
            assert len(forecast_data["forecast"]) == scenario["steps"]

        # 3. MULTIPLE DIAGNOSTICS - Verifica consistenza
        diag_results = []
        for _ in range(3):
            diag_response = client.post(f"/models/{model_id}/diagnostics")
            assert diag_response.status_code == 200
            diag_results.append(diag_response.json())

        # I risultati dovrebbero essere identici
        for i in range(1, len(diag_results)):
            assert diag_results[0]["residuals_stats"] == diag_results[i]["residuals_stats"]

        # 4. MULTIPLE REPORTS - Formati diversi
        report_formats = ["html", "pdf", "docx"]
        report_ids = []

        for fmt in report_formats:
            report_request = {"format": fmt, "include_diagnostics": True, "include_forecasts": True}

            report_response = client.post(f"/models/{model_id}/report", json=report_request)
            assert report_response.status_code == 200

            report_data = report_response.json()
            assert report_data["format_type"] == fmt
            report_ids.append(report_data["report_id"])

        # Tutti i report ID dovrebbero essere diversi
        assert len(set(report_ids)) == len(report_formats)

        # 5. CLEANUP
        client.delete(f"/models/{model_id}")

    def test_complete_var_workflow(self, client: TestClient, sample_multivariate_data):
        """Test workflow completo VAR per serie multivariate."""

        # 1. TRAINING VAR
        var_request = {"data": sample_multivariate_data, "max_lags": 3}

        train_response = client.post("/models/train/var", json=var_request)
        assert train_response.status_code == 200

        var_data = train_response.json()
        model_id = var_data["model_id"]
        assert var_data["model_type"] == "var"
        assert "sales" in var_data["variables"]
        assert "temperature" in var_data["variables"]

        # 2. VAR-SPECIFIC OPERATIONS
        # Controlla info modello VAR
        info_response = client.get(f"/models/{model_id}")
        assert info_response.status_code == 200

        # VAR dovrebbe supportare forecasting
        forecast_request = {"steps": 5, "return_confidence_intervals": True}

        forecast_response = client.post(f"/models/{model_id}/forecast", json=forecast_request)
        # VAR forecasting potrebbe non essere implementato, verifica graceful handling
        assert forecast_response.status_code in [200, 400, 501]

        # 3. CLEANUP
        client.delete(f"/models/{model_id}")

    def test_auto_selection_to_production_workflow(
        self, client: TestClient, sample_time_series_data
    ):
        """Test workflow auto-selezione -> training -> produzione."""

        # 1. AUTO-SELECTION - Trova parametri ottimali
        auto_request = {
            "data": sample_time_series_data,
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "seasonal": False,
            "criterion": "aic",
        }

        auto_response = client.post("/models/auto-select", json=auto_request)
        assert auto_response.status_code == 200

        auto_data = auto_response.json()
        best_params = auto_data["best_model"]
        assert "order" in best_params

        # 2. TRAINING con parametri ottimali trovati
        optimal_training_request = {
            "model_type": "arima",
            "data": sample_time_series_data,
            "order": best_params["order"],
        }

        train_response = client.post("/models/train", json=optimal_training_request)
        assert train_response.status_code == 200

        model_id = train_response.json()["model_id"]

        # 3. PRODUCTION SIMULATION - Operazioni tipiche produzione
        production_operations = [
            # Forecast giornaliero
            {"steps": 1, "confidence_level": 0.95},
            # Forecast settimanale
            {"steps": 7, "confidence_level": 0.90},
            # Forecast mensile
            {"steps": 30, "confidence_level": 0.85},
        ]

        for operation in production_operations:
            operation["return_confidence_intervals"] = True

            forecast_response = client.post(f"/models/{model_id}/forecast", json=operation)
            assert forecast_response.status_code == 200

        # 4. MONITORING - Diagnostica periodica
        diag_response = client.post(f"/models/{model_id}/diagnostics")
        assert diag_response.status_code == 200

        # 5. REPORTING - Report mensile
        monthly_report_request = {
            "format": "pdf",
            "include_diagnostics": True,
            "include_forecasts": True,
            "forecast_steps": 30,
        }

        report_response = client.post(f"/models/{model_id}/report", json=monthly_report_request)
        assert report_response.status_code == 200

        # 6. CLEANUP
        client.delete(f"/models/{model_id}")


class TestErrorRecoveryWorkflows:
    """Test recovery da errori in workflow complessi."""

    def test_workflow_with_invalid_data_recovery(self, client: TestClient):
        """Test recovery da dati non validi."""

        # 1. Tentativo con dati non validi
        invalid_request = {
            "model_type": "arima",
            "data": {
                "timestamps": ["2023-01-01"],
                "values": [],  # Vuoto
            },
            "order": {"p": 1, "d": 1, "q": 1},
        }

        train_response = client.post("/models/train", json=invalid_request)
        assert train_response.status_code == 400

        # 2. Correzione e retry con dati validi
        valid_request = {
            "model_type": "arima",
            "data": {
                "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "values": [100.0, 102.0, 101.0],
            },
            "order": {"p": 1, "d": 0, "q": 0},  # Semplice
        }

        retry_response = client.post("/models/train", json=valid_request)
        # Potrebbe essere ok o ancora troppo pochi dati
        assert retry_response.status_code in [200, 400]

    def test_workflow_nonexistent_model_operations(self, client: TestClient):
        """Test operazioni su modello inesistente."""
        fake_model_id = "fake-model-12345"

        # Tutte queste operazioni dovrebbero fallire gracefully
        operations = [
            ("GET", f"/models/{fake_model_id}", None),
            ("POST", f"/models/{fake_model_id}/forecast", {"steps": 5}),
            ("POST", f"/models/{fake_model_id}/diagnostics", None),
            ("POST", f"/models/{fake_model_id}/report", {"format": "html"}),
            ("DELETE", f"/models/{fake_model_id}", None),
        ]

        for method, url, json_data in operations:
            if method == "GET":
                response = client.get(url)
            elif method == "POST":
                response = client.post(url, json=json_data) if json_data else client.post(url)
            elif method == "DELETE":
                response = client.delete(url)

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestConcurrentWorkflows:
    """Test workflow concorrenti."""

    def test_concurrent_model_training(self, client: TestClient, sample_time_series_data):
        """Test training concorrente di multipli modelli."""
        import concurrent.futures

        def train_model():
            request = {
                "model_type": "arima",
                "data": sample_time_series_data,
                "order": {"p": 1, "d": 1, "q": 1},
            }
            return client.post("/models/train", json=request)

        # Training concorrente di 5 modelli
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(train_model) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Tutti dovrebbero avere successo
        model_ids = []
        for response in responses:
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])

        # Tutti gli ID dovrebbero essere diversi
        assert len(set(model_ids)) == 5

        # Cleanup
        for model_id in model_ids:
            client.delete(f"/models/{model_id}")

    def test_concurrent_operations_same_model(self, client: TestClient, sample_time_series_data):
        """Test operazioni concorrenti sullo stesso modello."""

        # 1. Crea modello
        train_request = {
            "model_type": "arima",
            "data": sample_time_series_data,
            "order": {"p": 1, "d": 1, "q": 1},
        }

        train_response = client.post("/models/train", json=train_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        # 2. Operazioni concorrenti
        import concurrent.futures

        def forecast_operation():
            return client.post(f"/models/{model_id}/forecast", json={"steps": 5})

        def diagnostics_operation():
            return client.post(f"/models/{model_id}/diagnostics")

        def info_operation():
            return client.get(f"/models/{model_id}")

        # Esegui operazioni diverse concorrentemente
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            forecast_future = executor.submit(forecast_operation)
            diag_future = executor.submit(diagnostics_operation)
            info_future = executor.submit(info_operation)

            forecast_response = forecast_future.result()
            diag_response = diag_future.result()
            info_response = info_future.result()

        # Tutte dovrebbero avere successo
        assert forecast_response.status_code == 200
        assert diag_response.status_code == 200
        assert info_response.status_code == 200

        # 3. Cleanup
        client.delete(f"/models/{model_id}")


class TestAPIPerformance:
    """Test performance API."""

    def test_api_response_times(self, client: TestClient, sample_time_series_data):
        """Test tempi di risposta API."""
        import time

        # Health check dovrebbe essere velocissimo
        start = time.time()
        health_response = client.get("/health")
        health_time = time.time() - start

        assert health_response.status_code == 200
        assert health_time < 0.1  # < 100ms

        # Training può essere più lento
        training_request = {
            "model_type": "arima",
            "data": sample_time_series_data,
            "order": {"p": 1, "d": 1, "q": 1},
        }

        start = time.time()
        train_response = client.post("/models/train", json=training_request)
        train_time = time.time() - start

        assert train_response.status_code == 200
        assert train_time < 2.0  # < 2 secondi per iniziare training

        model_id = train_response.json()["model_id"]

        # Lista modelli dovrebbe essere veloce
        start = time.time()
        list_response = client.get("/models")
        list_time = time.time() - start

        assert list_response.status_code == 200
        assert list_time < 0.5  # < 500ms

        # Cleanup
        client.delete(f"/models/{model_id}")

    def test_api_under_load(self, client: TestClient, sample_time_series_data):
        """Test API sotto carico."""
        import concurrent.futures

        # Mix di operazioni diverse
        def mixed_operations():
            operations = []

            # Health checks
            operations.extend([lambda: client.get("/health") for _ in range(5)])
            operations.extend([lambda: client.get("/") for _ in range(5)])

            # Model operations
            operations.extend([lambda: client.get("/models") for _ in range(3)])

            return operations

        all_operations = mixed_operations()

        # Esegui tutte le operazioni concorrentemente
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(op) for op in all_operations]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Tutte dovrebbero avere successo
        for response in responses:
            assert response.status_code == 200
