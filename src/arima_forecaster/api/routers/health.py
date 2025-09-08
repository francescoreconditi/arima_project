"""
Router per endpoint di health check e status.

Gestisce tutti gli endpoint relativi allo stato del servizio.
"""

import psutil
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import importlib.metadata

from fastapi import APIRouter, HTTPException
from ..middleware import get_performance_stats

# Crea router con prefix e tags
router = APIRouter(
    tags=["Health"],
    responses={404: {"description": "Not found"}}
)

"""
🏥 HEALTH ROUTER

Gestisce gli endpoint per il monitoraggio dello stato del servizio:

• GET /           - Endpoint root con informazioni base API
• GET /health     - Health check per monitoring e load balancer

Utilizzato per:
- Verificare che il servizio sia attivo
- Monitoring automatico da sistemi esterni
- Status check per deployment e CI/CD
"""


@router.get("/")
async def root() -> Dict[str, str]:
    """
    Endpoint root dell'API che fornisce informazioni di base sul servizio.
    
    <h4>Valore di Ritorno:</h4>
    <table >
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>message</td><td>string</td><td>Messaggio di benvenuto</td></tr>
        <tr><td>version</td><td>string</td><td>Versione dell'API</td></tr>
        <tr><td>docs</td><td>string</td><td>URL della documentazione Swagger</td></tr>
    </table>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "message": "ARIMA Forecaster API",
        "version": "1.0.0",
        "docs": "/docs"
    }
    </code></pre>
    """
    return {
        "message": "ARIMA Forecaster API",
        "version": "1.0.0",
        "docs": "/docs"
    }


def _get_version() -> str:
    """Ottiene versione del package."""
    try:
        return importlib.metadata.version("arima-forecaster")
    except:
        return "0.4.0"  # Fallback


def _check_dependencies() -> Dict[str, bool]:
    """Controlla dipendenze critiche."""
    deps = {}
    critical_deps = [
        'pandas', 'numpy', 'statsmodels', 'sklearn',
        'fastapi', 'uvicorn'
    ]
    
    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            deps[dep] = True
        except ImportError:
            deps[dep] = False
    
    return deps


def _check_optional_services() -> Dict[str, str]:
    """Controlla servizi opzionali."""
    services = {}
    
    # Kafka check
    try:
        from arima_forecaster.streaming import KafkaForecastProducer, StreamingConfig
        config = StreamingConfig()
        producer = KafkaForecastProducer(config)
        stats = producer.get_stats()
        services["kafka"] = "up" if stats["is_connected"] else "down"
        producer.close()
    except Exception:
        services["kafka"] = "unavailable"
    
    # Redis check (opzionale)
    services["redis"] = "not_configured"  # Per ora
    
    return services


def _get_system_info() -> Dict[str, Any]:
    """Informazioni sistema."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
    except Exception as e:
        return {"error": f"System info unavailable: {str(e)}"}


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check avanzato per production monitoring.
    
    Controlla:
    - Stato generale del servizio
    - Dipendenze critiche
    - Servizi opzionali (Kafka, Redis)
    - Metriche sistema (CPU, memoria, disco)
    - Informazioni versione
    
    <h4>Valore di Ritorno:</h4>
    <table>
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>status</td><td>string</td><td>healthy/degraded/unhealthy</td></tr>
        <tr><td>timestamp</td><td>string</td><td>Timestamp ISO 8601</td></tr>
        <tr><td>version</td><td>string</td><td>Versione del servizio</td></tr>
        <tr><td>dependencies</td><td>object</td><td>Stato dipendenze critiche</td></tr>
        <tr><td>services</td><td>object</td><td>Stato servizi opzionali</td></tr>
        <tr><td>system</td><td>object</td><td>Metriche sistema</td></tr>
    </table>
    
    <h4>Esempio Risposta:</h4>
    <pre><code>
    {
        "status": "healthy",
        "timestamp": "2025-01-08T23:55:00.000Z",
        "version": "0.4.0",
        "dependencies": {
            "pandas": true,
            "numpy": true,
            "statsmodels": true
        },
        "services": {
            "kafka": "down",
            "redis": "not_configured"
        },
        "system": {
            "cpu_percent": 25.5,
            "memory": {"percent": 65.2},
            "disk": {"percent": 45.8}
        }
    }
    </code></pre>
    """
    
    # Check dipendenze
    deps = _check_dependencies()
    critical_deps_ok = all(deps.values())
    
    # Check servizi opzionali
    services = _check_optional_services()
    
    # Informazioni sistema
    system_info = _get_system_info()
    
    # Determina stato generale
    if not critical_deps_ok:
        status = "unhealthy"
    elif "error" in system_info:
        status = "degraded"
    elif system_info.get("memory", {}).get("percent", 0) > 90:
        status = "degraded"  # Memoria alta
    elif system_info.get("disk", {}).get("percent", 0) > 95:
        status = "degraded"  # Disco pieno
    else:
        status = "healthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "version": _get_version(),
        "service": "arima-forecaster-api",
        "dependencies": deps,
        "services": services,
        "system": system_info,
        "uptime_seconds": os.times().elapsed if hasattr(os.times(), 'elapsed') else None
    }


@router.get("/health/simple")
async def simple_health_check() -> Dict[str, str]:
    """
    Health check semplificato per load balancer.
    
    Restituisce solo lo status essenziale senza controlli approfonditi.
    Ottimizzato per performance con latenza <10ms.
    
    <h4>Valore di Ritorno:</h4>
    <table>
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>status</td><td>string</td><td>Sempre "ok" se il servizio risponde</td></tr>
        <tr><td>timestamp</td><td>string</td><td>Timestamp ISO 8601</td></tr>
    </table>
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check per Kubernetes e orchestratori.
    
    Verifica se il servizio è pronto ad accettare traffico.
    Controlla dipendenze critiche e risorse essenziali.
    
    Restituisce HTTP 200 se ready, HTTP 503 se not ready.
    """
    
    # Check dipendenze critiche
    deps = _check_dependencies()
    critical_deps_ok = all(deps.values())
    
    if not critical_deps_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "reason": "Critical dependencies not available",
                "dependencies": deps
            }
        )
    
    # Check memoria disponibile (almeno 100MB)
    try:
        memory = psutil.virtual_memory()
        if memory.available < 100 * 1024 * 1024:  # 100MB
            raise HTTPException(
                status_code=503,
                detail={
                    "ready": False,
                    "reason": "Insufficient memory available",
                    "memory_available_mb": memory.available / (1024 * 1024)
                }
            )
    except Exception:
        pass  # Se psutil non funziona, assume OK
    
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat(),
        "dependencies": deps
    }


@router.get("/health/metrics")
async def metrics() -> Dict[str, Any]:
    """
    Metriche dettagliate per monitoring systems.
    
    Fornisce metriche complete per sistemi di monitoring come
    Prometheus, Grafana, DataDog, etc.
    
    <h4>Metriche Incluse:</h4>
    - Performance sistema (CPU, RAM, Disco)
    - Stato dipendenze e servizi
    - Informazioni processo Python
    - Timestamp e versioning
    """
    
    system_info = _get_system_info()
    deps = _check_dependencies()
    services = _check_optional_services()
    
    # Informazioni processo Python
    process = psutil.Process()
    process_info = {
        "pid": process.pid,
        "memory_mb": process.memory_info().rss / (1024 * 1024),
        "cpu_percent": process.cpu_percent(),
        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
        "num_threads": process.num_threads()
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "version": _get_version(),
        "service": "arima-forecaster-api",
        "system": system_info,
        "process": process_info,
        "dependencies": deps,
        "services": services,
        "dependencies_healthy": all(deps.values()),
        "services_count": len([s for s in services.values() if s == "up"])
    }


@router.get("/health/performance")
async def performance_metrics() -> Dict[str, Any]:
    """
    Metriche di performance delle API in tempo reale.
    
    Fornisce statistiche sulle richieste API elaborate dal middleware
    di performance monitoring.
    
    <h4>Metriche Performance:</h4>
    - Totale richieste elaborate
    - Tasso di errore (error rate)
    - Tempo medio di risposta
    - Richieste per minuto
    - Uptime del servizio
    
    <h4>Esempio Risposta:</h4>
    <pre><code>
    {
        "timestamp": "2025-01-08T23:55:00.000Z",
        "performance": {
            "total_requests": 1250,
            "total_errors": 15,
            "error_rate": 0.012,
            "average_response_time_ms": 145.6,
            "requests_per_minute": 12.3,
            "uptime_seconds": 7200
        },
        "status": "healthy"
    }
    </code></pre>
    """
    
    # Ottieni metriche performance
    perf_stats = get_performance_stats()
    
    # Determina status basato su metriche
    status = "healthy"
    if "error" in perf_stats:
        status = "degraded"
    elif perf_stats.get("error_rate", 0) > 0.05:  # >5% errori
        status = "degraded" 
    elif perf_stats.get("average_response_time_ms", 0) > 3000:  # >3s risposta
        status = "degraded"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "performance": perf_stats,
        "status": status,
        "thresholds": {
            "max_error_rate": 0.05,
            "max_response_time_ms": 3000,
            "alert_if_exceeded": True
        }
    }


@router.get("/health/all")
async def comprehensive_health_check() -> Dict[str, Any]:
    """
    Health check completo che combina tutte le metriche.
    
    Endpoint master che combina:
    - Sistema e dipendenze (/health)
    - Metriche dettagliate (/health/metrics)  
    - Performance API (/health/performance)
    
    Ideale per dashboard di monitoring centralizzate.
    
    <h4>Sezioni Incluse:</h4>
    - general: Status generale e dipendenze
    - system: Metriche sistema (CPU, RAM, disco)
    - performance: Performance API real-time
    - services: Status servizi esterni (Kafka, Redis)
    """
    
    # Health generale
    deps = _check_dependencies()
    services = _check_optional_services()
    system_info = _get_system_info()
    
    # Performance metrics
    perf_stats = get_performance_stats()
    
    # Determina status complessivo
    status = "healthy"
    
    # Check dipendenze critiche
    if not all(deps.values()):
        status = "unhealthy"
    # Check sistema
    elif "error" in system_info:
        status = "degraded"
    elif system_info.get("memory", {}).get("percent", 0) > 90:
        status = "degraded"
    # Check performance
    elif perf_stats.get("error_rate", 0) > 0.05:
        status = "degraded"
    elif perf_stats.get("average_response_time_ms", 0) > 3000:
        status = "degraded"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": status,
        "version": _get_version(),
        "service": "arima-forecaster-api",
        
        "general": {
            "dependencies": deps,
            "dependencies_healthy": all(deps.values()),
            "services": services
        },
        
        "system": system_info,
        
        "performance": perf_stats,
        
        "alerts": {
            "critical_dependencies_down": not all(deps.values()),
            "high_memory_usage": system_info.get("memory", {}).get("percent", 0) > 90,
            "high_error_rate": perf_stats.get("error_rate", 0) > 0.05,
            "slow_responses": perf_stats.get("average_response_time_ms", 0) > 3000
        }
    }