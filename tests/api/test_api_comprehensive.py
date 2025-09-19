"""
Test comprensivi per API FastAPI.

Test aggiuntivi per coverage completa e casi edge.
"""

import pytest
from fastapi.testclient import TestClient
import json


class TestAPIDocumentation:
    """Test per documentazione API."""

    def test_openapi_schema(self, client: TestClient):
        """Test schema OpenAPI."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        # Verifica struttura base OpenAPI
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

        # Verifica info API
        info = schema["info"]
        assert "title" in info
        assert "version" in info
        assert info["title"] == "ARIMA Forecaster API"

        # Verifica che tutti i path siano presenti
        expected_paths = [
            "/",
            "/health",
            "/models",
            "/models/train",
            "/models/train/var",
            "/models/auto-select",
            "/models/{model_id}",
            "/models/{model_id}/forecast",
            "/models/{model_id}/diagnostics",
            "/models/{model_id}/report",
            "/reports/{filename}",
        ]

        paths = schema["paths"]
        for expected_path in expected_paths:
            assert expected_path in paths

    def test_swagger_ui_access(self, client: TestClient):
        """Test accesso Swagger UI."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Verifica che contenga riferimenti Swagger
        content = response.text
        assert "swagger" in content.lower() or "openapi" in content.lower()

    def test_redoc_access(self, client: TestClient):
        """Test accesso ReDoc."""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Verifica che contenga riferimenti ReDoc
        content = response.text
        assert "redoc" in content.lower()

    def test_scalar_access(self, client: TestClient):
        """Test accesso Scalar UI."""
        response = client.get("/scalar")

        # Scalar potrebbe non essere abilitato in test
        assert response.status_code in [200, 404]


class TestAPIValidation:
    """Test validazione richieste API."""

    def test_json_content_type_validation(self, client: TestClient, sample_arima_request):
        """Test validazione Content-Type JSON."""

        # Request con Content-Type corretto
        response = client.post(
            "/models/train", json=sample_arima_request, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200

        # Request senza Content-Type (dovrebbe essere gestito da FastAPI)
        response = client.post("/models/train", json=sample_arima_request)
        assert response.status_code == 200

    def test_malformed_json_handling(self, client: TestClient):
        """Test gestione JSON malformato."""
        malformed_json = '{"model_type": "arima", "data": {'  # JSON incompleto

        response = client.post(
            "/models/train", data=malformed_json, headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_extra_fields_handling(self, client: TestClient, sample_arima_request):
        """Test gestione campi extra nelle richieste."""
        request_with_extra = sample_arima_request.copy()
        request_with_extra["extra_field"] = "should_be_ignored"
        request_with_extra["another_extra"] = 123

        response = client.post("/models/train", json=request_with_extra)

        # FastAPI/Pydantic dovrebbe ignorare campi extra
        assert response.status_code == 200

    def test_required_fields_validation(self, client: TestClient):
        """Test validazione campi richiesti."""
        incomplete_requests = [
            {},  # Completamente vuoto
            {"model_type": "arima"},  # Manca data e order
            {"data": {"timestamps": [], "values": []}},  # Manca model_type e order
            {"model_type": "arima", "data": {"timestamps": [], "values": []}},  # Manca order
        ]

        for incomplete_request in incomplete_requests:
            response = client.post("/models/train", json=incomplete_request)
            assert response.status_code == 422

            error_detail = response.json()["detail"]
            assert isinstance(error_detail, list)
            assert len(error_detail) > 0


class TestAPIHeaders:
    """Test headers HTTP."""

    def test_cors_headers(self, client: TestClient):
        """Test headers CORS."""
        response = client.get("/health")

        # Verifica presenza headers CORS (se configurati)
        headers = response.headers

        # In test environment potrebbero non essere tutti presenti
        cors_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers",
        ]

        # Almeno alcuni headers CORS dovrebbero essere presenti
        cors_present = any(header.lower() in headers for header in cors_headers)

        if cors_present:
            # Se CORS è configurato, verifica valori
            if "access-control-allow-origin" in headers:
                assert headers["access-control-allow-origin"] == "*"

    def test_content_type_headers(self, client: TestClient):
        """Test headers Content-Type."""

        # JSON endpoints
        json_endpoints = ["/health", "/models", "/openapi.json"]

        for endpoint in json_endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                assert "application/json" in response.headers["content-type"]

        # HTML endpoints
        html_endpoints = ["/docs", "/redoc"]

        for endpoint in html_endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                assert "text/html" in response.headers["content-type"]

    def test_security_headers(self, client: TestClient):
        """Test headers di sicurezza."""
        response = client.get("/health")

        # Verifica che non ci siano headers sensibili esposti
        sensitive_headers = ["server", "x-powered-by"]

        for header in sensitive_headers:
            assert header.lower() not in response.headers


class TestAPIErrorHandling:
    """Test gestione errori API."""

    def test_404_handling(self, client: TestClient):
        """Test gestione errori 404."""
        non_existent_endpoints = [
            "/nonexistent",
            "/models/nonexistent/endpoint",
            "/api/v1/something",
            "/admin",
        ]

        for endpoint in non_existent_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 404

    def test_405_method_not_allowed(self, client: TestClient):
        """Test errori 405 Method Not Allowed."""

        # GET su endpoint che accetta solo POST
        post_only_endpoints = ["/models/train", "/models/train/var", "/models/auto-select"]

        for endpoint in post_only_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 405

        # POST su endpoint che accetta solo GET
        get_only_endpoints = ["/health", "/models"]

        for endpoint in get_only_endpoints:
            response = client.post(endpoint, json={})
            assert response.status_code == 405

    def test_error_response_format(self, client: TestClient):
        """Test formato risposte di errore."""

        # Errore 404
        response = client.get("/nonexistent")
        assert response.status_code == 404

        error_data = response.json()
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)

        # Errore 422 (validation)
        response = client.post("/models/train", json={})
        assert response.status_code == 422

        validation_error = response.json()
        assert "detail" in validation_error
        assert isinstance(validation_error["detail"], list)

    def test_internal_server_error_handling(self, client: TestClient):
        """Test gestione errori server interni."""

        # Test con dati che potrebbero causare errori interni
        problematic_requests = [
            {
                "model_type": "arima",
                "data": {"timestamps": ["invalid-date"], "values": [100]},
                "order": {"p": 1, "d": 1, "q": 1},
            }
        ]

        for request in problematic_requests:
            response = client.post("/models/train", json=request)

            # Dovrebbe essere gestito gracefully
            assert response.status_code in [400, 422, 500]

            if response.status_code == 500:
                # Anche gli errori 500 dovrebbero avere formato consistente
                error_data = response.json()
                assert "detail" in error_data


class TestAPILimits:
    """Test limiti e edge cases API."""

    def test_large_data_handling(self, client: TestClient):
        """Test gestione dati grandi."""

        # Genera dataset relativamente grande
        large_timestamps = [f"2023-{i // 30 + 1:02d}-{i % 30 + 1:02d}" for i in range(1000)]
        large_values = [100 + i * 0.1 for i in range(1000)]

        large_request = {
            "model_type": "arima",
            "data": {"timestamps": large_timestamps, "values": large_values},
            "order": {"p": 1, "d": 1, "q": 1},
        }

        response = client.post("/models/train", json=large_request)

        # Dovrebbe gestire dati grandi o dare errore graceful
        assert response.status_code in [200, 400, 413, 422]  # 413 = Payload Too Large

    def test_empty_data_handling(self, client: TestClient):
        """Test gestione dati vuoti."""
        empty_requests = [
            {
                "model_type": "arima",
                "data": {"timestamps": [], "values": []},
                "order": {"p": 1, "d": 1, "q": 1},
            },
            {
                "model_type": "sarima",
                "data": {"timestamps": ["2023-01-01"], "values": [100]},  # Un solo punto
                "order": {"p": 1, "d": 1, "q": 1},
                "seasonal_order": {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12},
            },
        ]

        for empty_request in empty_requests:
            response = client.post("/models/train", json=empty_request)
            assert response.status_code in [400, 422]  # Dovrebbe rifiutare dati insufficienti

    def test_extreme_parameter_values(self, client: TestClient, sample_time_series_data):
        """Test valori estremi nei parametri."""
        extreme_requests = [
            {
                "model_type": "arima",
                "data": sample_time_series_data,
                "order": {"p": 0, "d": 0, "q": 0},  # Tutti zero
            },
            {
                "model_type": "arima",
                "data": sample_time_series_data,
                "order": {"p": 10, "d": 5, "q": 10},  # Valori molto alti
            },
            {
                "model_type": "sarima",
                "data": sample_time_series_data,
                "order": {"p": 1, "d": 1, "q": 1},
                "seasonal_order": {
                    "p": 1,
                    "d": 1,
                    "q": 1,
                    "P": 1,
                    "D": 1,
                    "Q": 1,
                    "s": 1,
                },  # Periodo stagionale molto piccolo
            },
        ]

        for extreme_request in extreme_requests:
            response = client.post("/models/train", json=extreme_request)

            # Dovrebbe gestire o rifiutare gracefully
            assert response.status_code in [200, 400, 422]


class TestAPICaching:
    """Test comportamento caching (se implementato)."""

    def test_model_list_consistency(self, client: TestClient, sample_arima_request):
        """Test consistenza lista modelli."""

        # Lista iniziale
        initial_response = client.get("/models")
        assert initial_response.status_code == 200
        initial_count = initial_response.json()["total_count"]

        # Aggiungi modello
        train_response = client.post("/models/train", json=sample_arima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        # Lista aggiornata dovrebbe riflettere il cambiamento
        updated_response = client.get("/models")
        assert updated_response.status_code == 200
        updated_count = updated_response.json()["total_count"]

        assert updated_count == initial_count + 1

        # Cleanup
        delete_response = client.delete(f"/models/{model_id}")
        assert delete_response.status_code == 200

        # Lista finale dovrebbe tornare al valore iniziale
        final_response = client.get("/models")
        assert final_response.status_code == 200
        final_count = final_response.json()["total_count"]

        assert final_count == initial_count


class TestAPIMetrics:
    """Test metriche e monitoring (se implementato)."""

    def test_response_time_consistency(self, client: TestClient):
        """Test consistenza tempi di risposta."""
        import time

        # Misura tempi multipli per endpoint veloci
        health_times = []

        for _ in range(5):
            start = time.time()
            response = client.get("/health")
            duration = time.time() - start

            assert response.status_code == 200
            health_times.append(duration)

        # I tempi dovrebbero essere tutti sotto una soglia
        for duration in health_times:
            assert duration < 0.5  # 500ms

        # La varianza dovrebbe essere bassa per endpoint semplici
        import statistics

        if len(health_times) > 1:
            std_dev = statistics.stdev(health_times)
            assert std_dev < 0.1  # Bassa variabilità

    def test_memory_usage_stability(self, client: TestClient):
        """Test stabilità uso memoria durante operazioni."""

        # Esegui molte operazioni leggere
        for _ in range(50):
            response = client.get("/health")
            assert response.status_code == 200

            response = client.get("/models")
            assert response.status_code == 200

        # L'API dovrebbe rimanere stabile
        # (In un test reale si monitorerebbe l'uso memoria)
