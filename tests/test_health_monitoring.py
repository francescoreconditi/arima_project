"""
Test per sistema di health monitoring.
"""

import pytest
from unittest.mock import Mock, patch
import asyncio


def test_health_monitoring_imports():
    """Test import moduli health monitoring."""
    from arima_forecaster.api.routers.health import (
        _check_dependencies,
        _check_optional_services,
        _get_system_info,
        _get_version
    )
    from arima_forecaster.api.middleware import (
        PerformanceMiddleware,
        SecurityMiddleware,
        get_performance_stats
    )
    
    # Verifica che le funzioni siano importabili
    assert _check_dependencies is not None
    assert _check_optional_services is not None
    assert _get_system_info is not None
    assert _get_version is not None
    assert PerformanceMiddleware is not None
    assert SecurityMiddleware is not None
    assert get_performance_stats is not None


def test_dependency_check():
    """Test controllo dipendenze critiche."""
    from arima_forecaster.api.routers.health import _check_dependencies
    
    deps = _check_dependencies()
    
    assert isinstance(deps, dict)
    assert len(deps) > 0
    
    # Dipendenze critiche che dovrebbero essere presenti
    critical_deps = ['pandas', 'numpy', 'statsmodels', 'fastapi']
    for dep in critical_deps:
        assert dep in deps
        assert deps[dep] == True  # Dovrebbero essere installate


def test_version_check():
    """Test recupero versione."""
    from arima_forecaster.api.routers.health import _get_version
    
    version = _get_version()
    
    assert isinstance(version, str)
    assert len(version) > 0
    # Dovrebbe essere 0.4.0 o simile
    assert version.count('.') >= 1  # Formato x.y o x.y.z


def test_optional_services_check():
    """Test controllo servizi opzionali."""
    from arima_forecaster.api.routers.health import _check_optional_services
    
    services = _check_optional_services()
    
    assert isinstance(services, dict)
    assert 'kafka' in services
    assert 'redis' in services
    
    # Kafka dovrebbe essere "down" o "unavailable" (normale senza server)
    assert services['kafka'] in ['up', 'down', 'unavailable']
    # Redis dovrebbe essere "not_configured" per ora
    assert services['redis'] == 'not_configured'


@patch('psutil.virtual_memory')
@patch('psutil.cpu_percent')
def test_system_info(mock_cpu, mock_memory):
    """Test informazioni sistema con mock."""
    from arima_forecaster.api.routers.health import _get_system_info
    
    # Mock dati sistema
    mock_memory.return_value = Mock(
        total=8589934592,  # 8GB
        available=4294967296,  # 4GB 
        percent=50.0
    )
    mock_cpu.return_value = 25.5
    
    with patch('psutil.disk_usage') as mock_disk:
        mock_disk.return_value = Mock(
            total=1000000000,  # 1GB
            free=500000000,    # 500MB
            used=500000000     # 500MB
        )
        
        system_info = _get_system_info()
    
    assert isinstance(system_info, dict)
    assert 'cpu_percent' in system_info
    assert 'memory' in system_info
    assert 'disk' in system_info
    assert 'python_version' in system_info
    assert 'platform' in system_info
    
    assert system_info['cpu_percent'] == 25.5
    assert system_info['memory']['percent'] == 50.0


def test_performance_middleware_creation():
    """Test creazione middleware performance."""
    from arima_forecaster.api.middleware import PerformanceMiddleware
    from starlette.applications import Starlette
    
    app = Starlette()
    middleware = PerformanceMiddleware(app)
    
    assert middleware is not None
    assert hasattr(middleware, 'request_count')
    assert hasattr(middleware, 'error_count')
    assert hasattr(middleware, 'get_stats')
    
    # Statistiche iniziali
    stats = middleware.get_stats()
    assert stats['total_requests'] == 0
    assert stats['total_errors'] == 0
    assert stats['error_rate'] == 0


def test_security_middleware_creation():
    """Test creazione middleware security."""
    from arima_forecaster.api.middleware import SecurityMiddleware
    from starlette.applications import Starlette
    
    app = Starlette()
    middleware = SecurityMiddleware(app)
    
    assert middleware is not None
    assert hasattr(middleware, 'blocked_ips')
    assert hasattr(middleware, 'request_counts')
    assert isinstance(middleware.blocked_ips, set)
    assert isinstance(middleware.request_counts, dict)


def test_performance_stats_without_middleware():
    """Test performance stats quando middleware non è inizializzato."""
    from arima_forecaster.api.middleware import get_performance_stats
    
    # Resetta globale per test
    import arima_forecaster.api.middleware as middleware_module
    middleware_module.performance_middleware_instance = None
    
    stats = get_performance_stats()
    
    assert isinstance(stats, dict)
    assert 'error' in stats


async def test_health_endpoint_logic():
    """Test logica health endpoint (senza FastAPI)."""
    from arima_forecaster.api.routers.health import (
        _check_dependencies,
        _check_optional_services,
        _get_system_info,
        _get_version
    )
    
    # Test componenti individuali
    deps = _check_dependencies()
    services = _check_optional_services() 
    system_info = _get_system_info()
    version = _get_version()
    
    # Simula logica endpoint /health
    critical_deps_ok = all(deps.values())
    
    if not critical_deps_ok:
        status = "unhealthy"
    elif "error" in system_info:
        status = "degraded"
    elif system_info.get("memory", {}).get("percent", 0) > 90:
        status = "degraded"
    else:
        status = "healthy"
    
    # Verifica risultato
    assert status in ["healthy", "degraded", "unhealthy"]
    assert isinstance(version, str)


def test_health_thresholds():
    """Test soglie per health monitoring."""
    
    # Soglie configurate nel sistema
    thresholds = {
        "max_memory_percent": 90,
        "max_disk_percent": 95,
        "max_error_rate": 0.05,  # 5%
        "max_response_time_ms": 3000
    }
    
    # Test logica soglie
    assert thresholds["max_memory_percent"] == 90
    assert thresholds["max_error_rate"] == 0.05
    assert thresholds["max_response_time_ms"] == 3000
    
    # Test valori sopra/sotto soglia
    assert 95 > thresholds["max_memory_percent"]  # Trigger degraded
    assert 85 < thresholds["max_memory_percent"]  # OK
    assert 0.06 > thresholds["max_error_rate"]    # Trigger degraded
    assert 0.02 < thresholds["max_error_rate"]    # OK