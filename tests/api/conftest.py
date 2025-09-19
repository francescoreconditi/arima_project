"""
Configurazione fixture per test API FastAPI.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from arima_forecaster.api.main import create_app


@pytest.fixture
def temp_model_dir():
    """Crea directory temporanea per test modelli."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_app(temp_model_dir):
    """Crea istanza app FastAPI per test."""
    app = create_app(model_storage_path=temp_model_dir, enable_scalar=False, production_mode=False)
    return app


@pytest.fixture
def client(test_app):
    """Client di test FastAPI."""
    return TestClient(test_app)


@pytest.fixture
def sample_time_series_data():
    """Dati di esempio per serie temporale."""
    return {
        "timestamps": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
            "2023-01-07",
            "2023-01-08",
            "2023-01-09",
            "2023-01-10",
            "2023-01-11",
            "2023-01-12",
            "2023-01-13",
            "2023-01-14",
            "2023-01-15",
            "2023-01-16",
            "2023-01-17",
            "2023-01-18",
            "2023-01-19",
            "2023-01-20",
        ],
        "values": [
            100.0,
            102.0,
            105.0,
            103.0,
            108.0,
            110.0,
            107.0,
            112.0,
            115.0,
            118.0,
            116.0,
            120.0,
            125.0,
            122.0,
            128.0,
            130.0,
            127.0,
            132.0,
            135.0,
            138.0,
        ],
    }


@pytest.fixture
def sample_multivariate_data():
    """Dati di esempio per serie multivariate (VAR)."""
    return {
        "series": [
            {
                "name": "sales",
                "timestamps": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ],
                "values": [100, 102, 105, 103, 108, 110, 107, 112, 115, 118],
            },
            {
                "name": "temperature",
                "timestamps": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ],
                "values": [20, 22, 18, 25, 23, 26, 21, 24, 27, 25],
            },
        ]
    }


@pytest.fixture
def sample_arima_request(sample_time_series_data):
    """Request di esempio per training ARIMA."""
    return {
        "model_type": "arima",
        "data": sample_time_series_data,
        "order": {"p": 1, "d": 1, "q": 1},
    }


@pytest.fixture
def sample_sarima_request(sample_time_series_data):
    """Request di esempio per training SARIMA."""
    return {
        "model_type": "sarima",
        "data": sample_time_series_data,
        "order": {"p": 1, "d": 1, "q": 1},
        "seasonal_order": {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12},
    }


@pytest.fixture
def sample_var_request(sample_multivariate_data):
    """Request di esempio per training VAR."""
    return {"data": sample_multivariate_data, "max_lags": 2}


@pytest.fixture
def sample_forecast_request():
    """Request di esempio per forecasting."""
    return {"steps": 5, "confidence_level": 0.95, "return_confidence_intervals": True}


@pytest.fixture
def sample_auto_select_request(sample_time_series_data):
    """Request di esempio per auto-selezione."""
    return {
        "data": sample_time_series_data,
        "max_p": 2,
        "max_d": 1,
        "max_q": 2,
        "seasonal": False,
        "criterion": "aic",
    }


@pytest.fixture
def sample_report_request():
    """Request di esempio per generazione report."""
    return {
        "format": "html",
        "include_diagnostics": True,
        "include_forecasts": True,
        "forecast_steps": 10,
        "template": "default",
    }


@pytest.fixture
def trained_model_id(client, sample_arima_request):
    """Modello già addestrato per test che richiedono model_id."""
    # Avvia training
    response = client.post("/models/train", json=sample_arima_request)
    assert response.status_code == 200

    model_info = response.json()
    model_id = model_info["model_id"]

    # Simula completamento training (per test)
    # In produzione il training sarebbe asincrono
    import time

    time.sleep(0.1)  # Breve pausa per simulare training

    return model_id
