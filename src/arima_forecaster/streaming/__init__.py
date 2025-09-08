"""
Real-Time Streaming Module

Fornisce funzionalità per streaming real-time di forecast e aggiornamenti live.
"""

from .kafka_producer import KafkaForecastProducer, StreamingConfig, ForecastMessage
from .realtime_forecaster import RealtimeForecastService
from .websocket_server import WebSocketServer, WebSocketConfig
from .event_processor import EventProcessor, ForecastEvent, EventType, EventPriority

__all__ = [
    "KafkaForecastProducer",
    "StreamingConfig", 
    "ForecastMessage",
    "RealtimeForecastService",
    "WebSocketServer",
    "WebSocketConfig",
    "EventProcessor",
    "ForecastEvent",
    "EventType",
    "EventPriority"
]