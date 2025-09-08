"""
WebSocket Server per Real-Time Dashboard Updates

Fornisce connessioni WebSocket per aggiornamenti live del dashboard.
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Set, List, Optional, Any, Callable
from dataclasses import dataclass
from pydantic import BaseModel
import threading
import queue

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    _has_websockets = True
except ImportError:
    _has_websockets = False

try:
    import redis
    _has_redis = True
except ImportError:
    _has_redis = False

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WebSocketConfig:
    """Configurazione WebSocket server"""
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 100
    heartbeat_interval: int = 30
    redis_url: Optional[str] = "redis://localhost:6379/0"
    allowed_origins: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]


class ClientSubscription(BaseModel):
    """Schema sottoscrizione client"""
    client_id: str
    model_ids: List[str] = []
    update_types: List[str] = ["new_forecast", "model_retrained", "anomaly_detected"]
    data_format: str = "json"
    filters: Dict[str, Any] = {}


class WebSocketMessage(BaseModel):
    """Schema messaggi WebSocket"""
    type: str  # "forecast_update", "model_update", "anomaly_alert", "heartbeat"
    timestamp: datetime
    data: Dict[str, Any]
    client_id: Optional[str] = None


class WebSocketServer:
    """
    Server WebSocket per aggiornamenti dashboard real-time
    
    Features:
    - Connessioni multiple simultanee
    - Sottoscrizioni selettive per modello
    - Heartbeat automatico
    - Integrazione Redis per scalabilità
    - Filtraggio messaggi per client
    """
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, ClientSubscription] = {}
        self.message_queue = queue.Queue()
        self.is_running = False
        self.server = None
        self.redis_client = None
        
        if not _has_websockets:
            logger.warning("WebSockets non disponibile. Installare: pip install websockets")
            return
        
        # Inizializza Redis se disponibile
        if _has_redis and self.config.redis_url:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
                self.redis_client.ping()
                logger.info("Redis connesso per WebSocket scaling")
            except Exception as e:
                logger.warning(f"Redis non disponibile: {e}")
                self.redis_client = None
    
    async def start_server(self):
        """Avvia server WebSocket"""
        if not _has_websockets:
            logger.error("WebSockets non disponibile")
            return
        
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.config.host,
                self.config.port,
                max_size=10**6,  # 1MB max message size
                max_queue=32,
                origins=self.config.allowed_origins
            )
            
            self.is_running = True
            logger.info(f"WebSocket server avviato su ws://{self.config.host}:{self.config.port}")
            
            # Avvia task background
            await asyncio.gather(
                self.heartbeat_task(),
                self.message_processor_task()
            )
            
        except Exception as e:
            logger.error(f"Errore avvio WebSocket server: {e}")
            self.is_running = False
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Gestisce connessione client"""
        client_id = f"client_{datetime.now().timestamp()}"
        self.clients[client_id] = websocket
        
        logger.info(f"Client connesso: {client_id} da {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.process_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnesso: {client_id}")
        except Exception as e:
            logger.error(f"Errore gestione client {client_id}: {e}")
        finally:
            await self.disconnect_client(client_id)
    
    async def process_client_message(self, client_id: str, raw_message: str):
        """Processa messaggio da client"""
        try:
            message = json.loads(raw_message)
            message_type = message.get("type")
            
            if message_type == "subscribe":
                # Gestisce sottoscrizione
                subscription = ClientSubscription(
                    client_id=client_id,
                    model_ids=message.get("model_ids", []),
                    update_types=message.get("update_types", ["new_forecast"]),
                    data_format=message.get("data_format", "json"),
                    filters=message.get("filters", {})
                )
                
                self.subscriptions[client_id] = subscription
                logger.info(f"Client {client_id} sottoscritto a modelli: {subscription.model_ids}")
                
                # Invia conferma
                await self.send_to_client(client_id, {
                    "type": "subscription_confirmed",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"subscribed_models": subscription.model_ids}
                })
            
            elif message_type == "unsubscribe":
                # Rimuove sottoscrizione
                if client_id in self.subscriptions:
                    del self.subscriptions[client_id]
                    logger.info(f"Client {client_id} non più sottoscritto")
            
            elif message_type == "ping":
                # Risponde a ping
                await self.send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                    "data": {}
                })
            
        except json.JSONDecodeError:
            logger.error(f"Messaggio JSON non valido da client {client_id}")
        except Exception as e:
            logger.error(f"Errore processo messaggio client {client_id}: {e}")
    
    async def disconnect_client(self, client_id: str):
        """Disconnette client e pulisce risorse"""
        if client_id in self.clients:
            try:
                await self.clients[client_id].close()
            except:
                pass
            del self.clients[client_id]
        
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        logger.info(f"Client {client_id} rimosso")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Invia messaggio a client specifico"""
        if client_id not in self.clients:
            return False
        
        try:
            websocket = self.clients[client_id]
            message_json = json.dumps(message, default=str)
            await websocket.send(message_json)
            return True
            
        except websockets.exceptions.ConnectionClosed:
            await self.disconnect_client(client_id)
            return False
        except Exception as e:
            logger.error(f"Errore invio messaggio a {client_id}: {e}")
            return False
    
    async def broadcast_forecast_update(self, model_id: str, forecast_data: Dict[str, Any]):
        """Broadcast aggiornamento forecast a client sottoscritti"""
        message = {
            "type": "forecast_update",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "model_id": model_id,
                "forecast": forecast_data
            }
        }
        
        # Filtra client sottoscritti a questo modello
        target_clients = []
        for client_id, subscription in self.subscriptions.items():
            if (not subscription.model_ids or model_id in subscription.model_ids) and \
               "new_forecast" in subscription.update_types:
                target_clients.append(client_id)
        
        # Invia a client target
        success_count = 0
        for client_id in target_clients:
            if await self.send_to_client(client_id, message):
                success_count += 1
        
        logger.info(f"Forecast update inviato a {success_count}/{len(target_clients)} client")
        
        # Salva in Redis se disponibile
        if self.redis_client:
            try:
                self.redis_client.lpush(f"forecast_updates:{model_id}", 
                                      json.dumps(message, default=str))
                self.redis_client.expire(f"forecast_updates:{model_id}", 3600)  # 1h TTL
            except Exception as e:
                logger.warning(f"Errore salvataggio Redis: {e}")
    
    async def broadcast_anomaly_alert(self, model_id: str, anomaly_data: Dict[str, Any]):
        """Broadcast alert anomalia"""
        message = {
            "type": "anomaly_alert",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "model_id": model_id,
                "anomaly": anomaly_data,
                "severity": anomaly_data.get("severity", "medium")
            }
        }
        
        # Invia a tutti i client sottoscritti agli alert
        target_clients = []
        for client_id, subscription in self.subscriptions.items():
            if (not subscription.model_ids or model_id in subscription.model_ids) and \
               "anomaly_detected" in subscription.update_types:
                target_clients.append(client_id)
        
        success_count = 0
        for client_id in target_clients:
            if await self.send_to_client(client_id, message):
                success_count += 1
        
        logger.warning(f"Anomaly alert inviato a {success_count} client per modello {model_id}")
    
    async def heartbeat_task(self):
        """Task heartbeat per mantenere connessioni attive"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "server_time": datetime.now().isoformat(),
                        "active_connections": len(self.clients)
                    }
                }
                
                # Invia heartbeat a tutti i client
                disconnected_clients = []
                for client_id in list(self.clients.keys()):
                    if not await self.send_to_client(client_id, heartbeat_message):
                        disconnected_clients.append(client_id)
                
                # Rimuovi client disconnessi
                for client_id in disconnected_clients:
                    await self.disconnect_client(client_id)
                
                if disconnected_clients:
                    logger.info(f"Rimossi {len(disconnected_clients)} client disconnessi")
                
            except Exception as e:
                logger.error(f"Errore heartbeat task: {e}")
    
    async def message_processor_task(self):
        """Task processamento coda messaggi"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)
                
                # Processa messaggi in coda
                while not self.message_queue.empty():
                    try:
                        message = self.message_queue.get_nowait()
                        await self._process_queued_message(message)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Errore processo messaggio: {e}")
                
            except Exception as e:
                logger.error(f"Errore message processor: {e}")
    
    async def _process_queued_message(self, message: Dict[str, Any]):
        """Processa messaggio dalla coda"""
        message_type = message.get("type")
        
        if message_type == "forecast_update":
            await self.broadcast_forecast_update(
                message["model_id"], 
                message["data"]
            )
        elif message_type == "anomaly_alert":
            await self.broadcast_anomaly_alert(
                message["model_id"],
                message["data"]
            )
    
    def queue_message(self, message_type: str, model_id: str, data: Dict[str, Any]):
        """Accoda messaggio per invio asincrono"""
        message = {
            "type": message_type,
            "model_id": model_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Coda messaggi WebSocket piena")
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche server"""
        return {
            "is_running": self.is_running,
            "active_connections": len(self.clients),
            "subscriptions": len(self.subscriptions),
            "host": self.config.host,
            "port": self.config.port,
            "redis_available": self.redis_client is not None,
            "queued_messages": self.message_queue.qsize()
        }
    
    async def shutdown(self):
        """Chiude server WebSocket"""
        self.is_running = False
        
        # Disconnetti tutti i client
        for client_id in list(self.clients.keys()):
            await self.disconnect_client(client_id)
        
        # Chiudi server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Chiudi Redis
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("WebSocket server chiuso")


def run_websocket_server(config: WebSocketConfig = None):
    """Utility per avviare server WebSocket"""
    if config is None:
        config = WebSocketConfig()
    
    server = WebSocketServer(config)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server interrotto da utente")
    except Exception as e:
        logger.error(f"Errore server WebSocket: {e}")


if __name__ == "__main__":
    # Test server WebSocket
    config = WebSocketConfig(port=8765)
    run_websocket_server(config)