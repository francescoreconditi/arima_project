"""
Router per endpoint di health check e status.

Gestisce tutti gli endpoint relativi allo stato del servizio.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

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
    <table class="table table-striped">
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


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint per verificare che il servizio sia attivo.
    
    Utilizzato dai sistemi di monitoring e load balancer per verificare
    lo stato del servizio in produzione.
    
    <h4>Valore di Ritorno:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>status</td><td>string</td><td>Stato del servizio (sempre "healthy")</td></tr>
        <tr><td>timestamp</td><td>string</td><td>Timestamp corrente ISO 8601</td></tr>
        <tr><td>service</td><td>string</td><td>Nome del servizio</td></tr>
    </table>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "status": "healthy",
        "timestamp": "2024-08-23T14:30:00.000Z",
        "service": "arima-forecaster-api"
    }
    </code></pre>
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "arima-forecaster-api"
    }