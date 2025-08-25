"""
Test per Models Router.

Testa endpoint di gestione modelli.
"""

import pytest
from fastapi.testclient import TestClient


class TestModelListing:
    """Test per listing modelli."""
    
    def test_list_models_empty(self, client: TestClient):
        """Test lista modelli quando non ci sono modelli."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "total_count" in data
        assert data["total_count"] == 0
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 0
    
    def test_list_models_with_trained_model(self, client: TestClient, trained_model_id):
        """Test lista modelli con modello addestrato."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_count"] >= 1
        assert len(data["models"]) >= 1
        
        # Verifica struttura modelli
        for model in data["models"]:
            assert "model_id" in model
            assert "model_type" in model
            assert "status" in model
            assert "created_at" in model
            assert "training_observations" in model
            assert "parameters" in model
            assert "metrics" in model
    
    def test_list_models_response_format(self, client: TestClient):
        """Test formato risposta lista modelli."""
        response = client.get("/models")
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        
        data = response.json()
        assert isinstance(data, dict)
        assert isinstance(data["models"], list)
        assert isinstance(data["total_count"], int)


class TestModelRetrieval:
    """Test per recupero informazioni modello specifico."""
    
    def test_get_model_info_success(self, client: TestClient, trained_model_id):
        """Test recupero info modello con successo."""
        response = client.get(f"/models/{trained_model_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica campi richiesti
        required_fields = [
            "model_id", "model_type", "status", "created_at",
            "training_observations", "parameters", "metrics"
        ]
        for field in required_fields:
            assert field in data
        
        # Verifica valori
        assert data["model_id"] == trained_model_id
        assert data["model_type"] in ["arima", "sarima", "var"]
        assert data["status"] in ["training", "completed", "failed"]
        assert isinstance(data["training_observations"], int)
        assert isinstance(data["parameters"], dict)
        assert isinstance(data["metrics"], dict)
    
    def test_get_model_info_nonexistent(self, client: TestClient):
        """Test recupero info modello inesistente."""
        fake_model_id = "nonexistent-model-123"
        response = client.get(f"/models/{fake_model_id}")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["detail"].lower()
    
    def test_get_model_info_invalid_id_format(self, client: TestClient):
        """Test recupero con ID formato non valido."""
        invalid_ids = [
            "",  # ID vuoto
            "   ",  # Solo spazi
            "invalid/id/with/slashes",  # Caratteri non validi
        ]
        
        for invalid_id in invalid_ids:
            if invalid_id.strip():  # Skip ID vuoti che causerebbero 404 su route diversa
                response = client.get(f"/models/{invalid_id}")
                assert response.status_code in [400, 404, 422]


class TestModelDeletion:
    """Test per eliminazione modelli."""
    
    def test_delete_model_success(self, client: TestClient, sample_arima_request):
        """Test eliminazione modello con successo."""
        # Prima crea un modello
        train_response = client.post("/models/train", json=sample_arima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Poi eliminalo
        delete_response = client.delete(f"/models/{model_id}")
        
        assert delete_response.status_code == 200
        data = delete_response.json()
        
        assert "message" in data
        assert model_id in data["message"]
        assert "deleted successfully" in data["message"]
        
        # Verifica che sia stato eliminato
        get_response = client.get(f"/models/{model_id}")
        assert get_response.status_code == 404
    
    def test_delete_nonexistent_model(self, client: TestClient):
        """Test eliminazione modello inesistente."""
        fake_model_id = "nonexistent-model-456"
        response = client.delete(f"/models/{fake_model_id}")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["detail"].lower()
    
    def test_delete_model_twice(self, client: TestClient, sample_arima_request):
        """Test eliminazione stesso modello due volte."""
        # Crea modello
        train_response = client.post("/models/train", json=sample_arima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Prima eliminazione
        first_delete = client.delete(f"/models/{model_id}")
        assert first_delete.status_code == 200
        
        # Seconda eliminazione
        second_delete = client.delete(f"/models/{model_id}")
        assert second_delete.status_code == 404


class TestModelCRUDWorkflow:
    """Test workflow completo CRUD per modelli."""
    
    def test_complete_model_lifecycle(self, client: TestClient, sample_arima_request):
        """Test ciclo di vita completo del modello."""
        # 1. Lista iniziale (dovrebbe essere vuota o avere modelli esistenti)
        initial_list = client.get("/models")
        assert initial_list.status_code == 200
        initial_count = initial_list.json()["total_count"]
        
        # 2. Crea nuovo modello
        train_response = client.post("/models/train", json=sample_arima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # 3. Verifica che sia nella lista
        updated_list = client.get("/models")
        assert updated_list.status_code == 200
        assert updated_list.json()["total_count"] == initial_count + 1
        
        # 4. Recupera info specifiche
        model_info = client.get(f"/models/{model_id}")
        assert model_info.status_code == 200
        assert model_info.json()["model_id"] == model_id
        
        # 5. Elimina modello
        delete_response = client.delete(f"/models/{model_id}")
        assert delete_response.status_code == 200
        
        # 6. Verifica eliminazione
        final_list = client.get("/models")
        assert final_list.status_code == 200
        assert final_list.json()["total_count"] == initial_count
        
        # 7. Verifica che non sia più accessibile
        get_deleted = client.get(f"/models/{model_id}")
        assert get_deleted.status_code == 404
    
    def test_multiple_models_management(self, client: TestClient, sample_arima_request):
        """Test gestione multipli modelli."""
        model_ids = []
        
        # Crea 3 modelli
        for i in range(3):
            response = client.post("/models/train", json=sample_arima_request)
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])
        
        # Verifica che tutti siano nella lista
        models_list = client.get("/models")
        assert models_list.status_code == 200
        list_data = models_list.json()
        listed_ids = {model["model_id"] for model in list_data["models"]}
        
        for model_id in model_ids:
            assert model_id in listed_ids
        
        # Elimina un modello in mezzo
        middle_id = model_ids[1]
        delete_response = client.delete(f"/models/{middle_id}")
        assert delete_response.status_code == 200
        
        # Verifica che gli altri due siano ancora presenti
        updated_list = client.get("/models")
        assert updated_list.status_code == 200
        updated_ids = {model["model_id"] for model in updated_list.json()["models"]}
        
        assert model_ids[0] in updated_ids
        assert model_ids[2] in updated_ids
        assert middle_id not in updated_ids
        
        # Pulisci i modelli rimanenti
        for model_id in [model_ids[0], model_ids[2]]:
            client.delete(f"/models/{model_id}")


class TestModelMetadata:
    """Test metadati modelli."""
    
    def test_model_metadata_persistence(self, client: TestClient, sample_sarima_request):
        """Test persistenza metadati modello."""
        # Crea modello SARIMA con parametri specifici
        train_response = client.post("/models/train", json=sample_sarima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        # Recupera info e verifica metadati
        info_response = client.get(f"/models/{model_id}")
        assert info_response.status_code == 200
        
        data = info_response.json()
        assert data["model_type"] == "sarima"
        assert data["training_observations"] == 20  # Dalla fixture
        
        # I parametri potrebbero essere vuoti durante training asincrono
        assert isinstance(data["parameters"], dict)
        assert isinstance(data["metrics"], dict)
    
    def test_model_timestamps(self, client: TestClient, sample_arima_request):
        """Test timestamp creazione modello."""
        import datetime
        
        before_creation = datetime.datetime.now()
        
        # Crea modello
        train_response = client.post("/models/train", json=sample_arima_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]
        
        after_creation = datetime.datetime.now()
        
        # Verifica timestamp
        info_response = client.get(f"/models/{model_id}")
        assert info_response.status_code == 200
        
        created_at_str = info_response.json()["created_at"]
        created_at = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
        
        # Il timestamp dovrebbe essere tra prima e dopo la creazione (con un po' di tolleranza)
        tolerance = datetime.timedelta(seconds=10)
        assert before_creation - tolerance <= created_at <= after_creation + tolerance