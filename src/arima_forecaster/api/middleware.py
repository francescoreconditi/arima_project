"""
Middleware per monitoring e performance tracking.

Fornisce middleware per tracciare performance, errori e metriche delle API.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any
import asyncio

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..utils.logger import setup_logger

# Logger per metriche
metrics_logger = setup_logger("api.metrics")


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware per tracking performance delle API.
    
    Traccia:
    - Tempo di risposta per ogni endpoint
    - Status code delle risposte
    - Dimensione payload
    - Errori e eccezioni
    - IP client e user agent
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.start_time = datetime.now()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Processa ogni richiesta HTTP."""
        
        # Incrementa contatore richieste
        self.request_count += 1
        
        # Informazioni richiesta
        start_time = time.time()
        method = request.method
        url = str(request.url)
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Headers utili per debugging
        headers_info = {
            "content-type": request.headers.get("content-type"),
            "content-length": request.headers.get("content-length"),
            "accept": request.headers.get("accept")
        }
        
        try:
            # Esegui richiesta
            response = await call_next(request)
            
            # Calcola tempo risposta
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            # Informazioni risposta
            status_code = response.status_code
            response_size = response.headers.get("content-length", "unknown")
            
            # Log metriche
            metrics_logger.info(
                f"API_REQUEST",
                extra={
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "response_size": response_size,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": self.request_count
                }
            )
            
            # Log performance warnings
            if response_time > 5.0:  # Richieste >5 secondi
                metrics_logger.warning(
                    f"SLOW_REQUEST: {method} {path} took {response_time:.2f}s",
                    extra={
                        "response_time_ms": round(response_time * 1000, 2),
                        "path": path,
                        "status_code": status_code
                    }
                )
            
            # Log errori HTTP
            if status_code >= 400:
                self.error_count += 1
                level = "ERROR" if status_code >= 500 else "WARNING"
                metrics_logger.log(
                    logging.ERROR if status_code >= 500 else logging.WARNING,
                    f"HTTP_{level}: {method} {path} returned {status_code}",
                    extra={
                        "status_code": status_code,
                        "path": path,
                        "client_ip": client_ip,
                        "response_time_ms": round(response_time * 1000, 2)
                    }
                )
            
            # Aggiungi headers per debugging
            response.headers["X-Response-Time"] = str(round(response_time * 1000, 2))
            response.headers["X-Request-ID"] = str(self.request_count)
            
            return response
            
        except Exception as e:
            # Log eccezioni
            response_time = time.time() - start_time
            self.error_count += 1
            
            metrics_logger.error(
                f"API_EXCEPTION: {method} {path} - {str(e)}",
                extra={
                    "method": method,
                    "path": path,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "response_time_ms": round(response_time * 1000, 2),
                    "client_ip": client_ip,
                    "request_id": self.request_count
                },
                exc_info=True
            )
            
            # Re-raise per gestione normale FastAPI
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche del middleware."""
        uptime = datetime.now() - self.start_time
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_response_time_ms": round(avg_response_time * 1000, 2),
            "uptime_seconds": uptime.total_seconds(),
            "requests_per_minute": (
                self.request_count / max(uptime.total_seconds() / 60, 1)
            )
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware per security headers e rate limiting basico.
    
    Aggiunge security headers e controlla richieste sospette.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.blocked_ips = set()  # IP bloccati
        self.request_counts = {}  # Rate limiting per IP
        self.security_logger = setup_logger("api.security")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Applica controlli di sicurezza."""
        
        client_ip = request.client.host if request.client else "unknown"
        
        # Check IP bloccati
        if client_ip in self.blocked_ips:
            self.security_logger.warning(f"BLOCKED_IP_REQUEST: {client_ip}")
            return Response("Forbidden", status_code=403)
        
        # Rate limiting basico (100 req/min per IP)
        current_time = int(time.time() / 60)  # Minuto corrente
        ip_key = f"{client_ip}_{current_time}"
        
        if ip_key in self.request_counts:
            self.request_counts[ip_key] += 1
            if self.request_counts[ip_key] > 100:  # Limite per minuto
                self.security_logger.warning(
                    f"RATE_LIMIT_EXCEEDED: {client_ip} exceeded 100 req/min"
                )
                return Response(
                    "Rate limit exceeded", 
                    status_code=429,
                    headers={"Retry-After": "60"}
                )
        else:
            self.request_counts[ip_key] = 1
        
        # Cleanup contatori vecchi (mantieni solo ultimo minuto)
        old_keys = [k for k in self.request_counts.keys() 
                   if not k.endswith(f"_{current_time}")]
        for key in old_keys:
            del self.request_counts[key]
        
        # Esegui richiesta
        response = await call_next(request)
        
        # Aggiungi security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache"
        })
        
        return response


# Istanza globale per tracking metriche
performance_middleware_instance = None


def get_performance_stats() -> Dict[str, Any]:
    """Ottiene statistiche performance correnti."""
    if performance_middleware_instance:
        return performance_middleware_instance.get_stats()
    else:
        return {"error": "Performance middleware not initialized"}