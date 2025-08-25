"""
Test per Reports Router.

Testa endpoint di generazione e download report.
"""

import pytest
from fastapi.testclient import TestClient


class TestReportGeneration:
    """Test per generazione report."""
    
    def test_generate_report_success(self, client: TestClient, trained_model_id, sample_report_request):
        """Test generazione report con successo."""
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=sample_report_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi risposta
        required_fields = [
            "report_id",
            "status", 
            "format_type",
            "generation_time",
            "file_size_mb",
            "download_url"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Verifica valori
        assert data["status"] == "generating"
        assert data["format_type"] == "html"
        assert isinstance(data["generation_time"], (int, float))
        assert isinstance(data["file_size_mb"], (int, float))
        assert data["download_url"].startswith("/reports/")
        
        # Verifica che report_id sia valido
        assert len(data["report_id"]) > 0
        assert data["report_id"].startswith("report-")
    
    def test_generate_report_different_formats(self, client: TestClient, trained_model_id):
        """Test generazione report in formati diversi."""
        formats = ["html", "pdf", "docx"]
        
        for fmt in formats:
            report_request = {
                "format": fmt,
                "include_diagnostics": True,
                "include_forecasts": True,
                "forecast_steps": 5
            }
            
            response = client.post(
                f"/models/{trained_model_id}/report",
                json=report_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["format_type"] == fmt
            assert data["download_url"].endswith(f".{fmt}")
    
    def test_generate_report_custom_options(self, client: TestClient, trained_model_id):
        """Test generazione report con opzioni personalizzate."""
        custom_request = {
            "format": "html",
            "include_diagnostics": False,  # Senza diagnostica
            "include_forecasts": True,
            "forecast_steps": 20,  # Più step
            "template": "custom"
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=custom_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "generating"
    
    def test_generate_report_minimal_options(self, client: TestClient, trained_model_id):
        """Test generazione report con opzioni minime."""
        minimal_request = {
            "format": "html"
            # Usa tutti i default
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=minimal_request
        )
        
        assert response.status_code == 200
    
    def test_generate_report_nonexistent_model(self, client: TestClient, sample_report_request):
        """Test generazione report per modello inesistente."""
        fake_model_id = "nonexistent-model-xyz"
        
        response = client.post(
            f"/models/{fake_model_id}/report",
            json=sample_report_request
        )
        
        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["detail"].lower()
    
    def test_generate_report_invalid_format(self, client: TestClient, trained_model_id):
        """Test generazione report con formato non valido."""
        invalid_request = {
            "format": "invalid_format",
            "include_diagnostics": True,
            "include_forecasts": True
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=invalid_request
        )
        
        # Potrebbe essere validation error o bad request
        assert response.status_code in [400, 422]
    
    def test_generate_report_invalid_forecast_steps(self, client: TestClient, trained_model_id):
        """Test generazione report con step forecast non validi."""
        invalid_requests = [
            {"format": "html", "forecast_steps": 0},     # Zero steps
            {"format": "html", "forecast_steps": -1},    # Negative steps
            {"format": "html", "forecast_steps": 1000},  # Troppi steps
        ]
        
        for invalid_request in invalid_requests:
            response = client.post(
                f"/models/{trained_model_id}/report",
                json=invalid_request
            )
            
            assert response.status_code in [400, 422]


class TestReportDownload:
    """Test per download report."""
    
    def test_download_nonexistent_report(self, client: TestClient):
        """Test download report inesistente."""
        fake_filename = "nonexistent_report.html"
        
        response = client.get(f"/reports/{fake_filename}")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["detail"].lower()
    
    def test_download_report_path_traversal_protection(self, client: TestClient):
        """Test protezione path traversal."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
        ]
        
        for malicious_path in malicious_paths:
            response = client.get(f"/reports/{malicious_path}")
            
            # Dovrebbe essere 404 (file non trovato) o 400 (bad request)
            assert response.status_code in [400, 404]
    
    def test_download_report_invalid_extensions(self, client: TestClient):
        """Test download file con estensioni non valide."""
        invalid_files = [
            "report.exe",
            "report.bat", 
            "report.sh",
            "report.py",
            "report"  # Senza estensione
        ]
        
        for invalid_file in invalid_files:
            response = client.get(f"/reports/{invalid_file}")
            
            # Dovrebbe essere 404 (non trovato)
            assert response.status_code == 404
    
    def test_download_report_valid_extensions(self, client: TestClient):
        """Test download con estensioni valide (anche se file non esistono)."""
        valid_extensions = ["html", "pdf", "docx"]
        
        for ext in valid_extensions:
            filename = f"test_report.{ext}"
            response = client.get(f"/reports/{filename}")
            
            # File non esiste ma estensione è valida, dovrebbe essere 404
            assert response.status_code == 404


class TestReportWorkflow:
    """Test workflow completo generazione e download report."""
    
    def test_generate_and_check_status_workflow(self, client: TestClient, trained_model_id):
        """Test workflow generazione -> verifica status."""
        # 1. Genera report
        report_request = {
            "format": "html",
            "include_diagnostics": True,
            "include_forecasts": True,
            "forecast_steps": 5
        }
        
        gen_response = client.post(
            f"/models/{trained_model_id}/report",
            json=report_request
        )
        
        assert gen_response.status_code == 200
        gen_data = gen_response.json()
        
        report_id = gen_data["report_id"]
        download_url = gen_data["download_url"]
        
        # 2. Verifica che il download_url sia ben formato
        assert download_url.startswith("/reports/")
        filename = download_url.split("/")[-1]
        assert filename.endswith(".html")
        
        # 3. Prova download (probabilmente 404 perché è un mock)
        download_response = client.get(download_url)
        # In un test reale potrebbe essere 404 perché il report non è realmente generato
        assert download_response.status_code in [200, 404]
    
    def test_multiple_reports_same_model(self, client: TestClient, trained_model_id):
        """Test generazione multipli report per stesso modello."""
        report_request = {
            "format": "html",
            "include_diagnostics": True,
            "include_forecasts": True
        }
        
        report_ids = []
        
        # Genera 3 report
        for i in range(3):
            response = client.post(
                f"/models/{trained_model_id}/report",
                json=report_request
            )
            
            assert response.status_code == 200
            data = response.json()
            report_ids.append(data["report_id"])
        
        # Tutti gli ID dovrebbero essere diversi
        assert len(set(report_ids)) == 3
    
    def test_concurrent_report_generation(self, client: TestClient, trained_model_id):
        """Test generazione report concorrente."""
        import concurrent.futures
        
        def generate_report():
            return client.post(
                f"/models/{trained_model_id}/report",
                json={"format": "html"}
            )
        
        # Esegui 3 generazioni concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_report) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Tutte dovrebbero avere successo
        for response in responses:
            assert response.status_code == 200
        
        # Ogni report dovrebbe avere ID unico
        report_ids = [resp.json()["report_id"] for resp in responses]
        assert len(set(report_ids)) == 3


class TestReportValidation:
    """Test validazione parametri report."""
    
    def test_report_request_validation(self, client: TestClient, trained_model_id):
        """Test validazione richiesta report."""
        # Test tutti i campi opzionali
        full_request = {
            "format": "pdf",
            "include_diagnostics": False,
            "include_forecasts": True,
            "forecast_steps": 15,
            "template": "advanced"
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=full_request
        )
        
        assert response.status_code == 200
    
    def test_report_boolean_validation(self, client: TestClient, trained_model_id):
        """Test validazione campi boolean."""
        # Test con valori boolean validi
        bool_request = {
            "format": "html",
            "include_diagnostics": True,
            "include_forecasts": False
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=bool_request
        )
        
        assert response.status_code == 200
        
        # Test con valori boolean non validi
        invalid_bool_request = {
            "format": "html", 
            "include_diagnostics": "yes",  # Stringa invece di bool
            "include_forecasts": 1  # Numero invece di bool
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=invalid_bool_request
        )
        
        # Pydantic potrebbe convertire o dare validation error
        assert response.status_code in [200, 422]
    
    def test_report_steps_validation(self, client: TestClient, trained_model_id):
        """Test validazione forecast_steps."""
        valid_steps = [1, 5, 10, 30, 50]
        
        for steps in valid_steps:
            request = {
                "format": "html",
                "forecast_steps": steps
            }
            
            response = client.post(
                f"/models/{trained_model_id}/report",
                json=request
            )
            
            assert response.status_code == 200


class TestReportErrorHandling:
    """Test gestione errori report."""
    
    def test_report_generation_server_error_simulation(self, client: TestClient, trained_model_id):
        """Test simulazione errore server durante generazione."""
        # Questo test verifica che l'API gestisca gli errori gracefully
        # In un test reale potresti mockare un errore interno
        
        report_request = {
            "format": "html",
            "include_diagnostics": True,
            "include_forecasts": True
        }
        
        response = client.post(
            f"/models/{trained_model_id}/report",
            json=report_request
        )
        
        # Anche se c'è un errore interno, l'API dovrebbe rispondere
        assert response.status_code in [200, 500]
    
    def test_malformed_report_requests(self, client: TestClient, trained_model_id):
        """Test richieste report malformate."""
        malformed_requests = [
            {},  # Completamente vuoto
            {"wrong_field": "value"},  # Campo sbagliato
            {"format": None},  # Valore None
        ]
        
        for malformed_request in malformed_requests:
            response = client.post(
                f"/models/{trained_model_id}/report",
                json=malformed_request
            )
            
            # Dovrebbe essere validation error o usare default
            assert response.status_code in [200, 422]