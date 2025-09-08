"""
Test per moduli streaming v0.4.0.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime


def test_streaming_imports():
    """Test import moduli streaming."""
    from arima_forecaster.streaming import (
        KafkaForecastProducer,
        StreamingConfig,
        ForecastMessage,
        WebSocketServer,
        WebSocketConfig,
        RealtimeForecastService,
        EventProcessor,
        ForecastEvent,
        EventType,
        EventPriority
    )
    
    # Verifica che le classi siano importabili
    assert KafkaForecastProducer is not None
    assert StreamingConfig is not None
    assert ForecastMessage is not None
    assert WebSocketServer is not None
    assert WebSocketConfig is not None
    assert RealtimeForecastService is not None
    assert EventProcessor is not None
    assert ForecastEvent is not None
    assert EventType is not None
    assert EventPriority is not None


def test_streaming_config_creation():
    """Test creazione configurazione streaming."""
    from arima_forecaster.streaming import StreamingConfig
    
    config = StreamingConfig()
    assert config.bootstrap_servers == ["localhost:9092"]
    assert config.topic == "arima-forecasts"
    
    # Con parametri custom
    custom_config = StreamingConfig(
        bootstrap_servers=["server1:9092", "server2:9092"],
        topic="custom-topic"
    )
    assert len(custom_config.bootstrap_servers) == 2
    assert custom_config.topic == "custom-topic"


def test_forecast_message_creation():
    """Test creazione messaggio forecast."""
    from arima_forecaster.streaming import ForecastMessage
    
    message = ForecastMessage(
        model_id="test_model",
        timestamp=datetime.now(),
        predicted_value=123.45,
        confidence_interval=[100.0, 150.0],
        forecast_horizon=7,
        model_type="ARIMA"
    )
    
    assert message.model_id == "test_model"
    assert message.predicted_value == 123.45
    assert message.confidence_interval == [100.0, 150.0]
    assert message.forecast_horizon == 7
    assert message.model_type == "ARIMA"


def test_kafka_producer_fallback():
    """Test fallback locale per Kafka producer."""
    from arima_forecaster.streaming import KafkaForecastProducer, StreamingConfig
    
    config = StreamingConfig()
    producer = KafkaForecastProducer(config)
    
    # Producer dovrebbe essere creato anche se Kafka non è disponibile
    assert producer is not None
    assert hasattr(producer, 'send_forecast')
    
    # Statistiche dovrebbero mostrare che Kafka non è connesso
    stats = producer.get_stats()
    assert stats['is_connected'] == False
    assert stats['topic'] == "arima-forecasts"


def test_websocket_config():
    """Test configurazione WebSocket."""
    from arima_forecaster.streaming import WebSocketConfig
    
    config = WebSocketConfig()
    assert config.host == "localhost"
    assert config.port == 8765
    assert config.max_connections == 100
    
    # Configurazione custom
    custom_config = WebSocketConfig(
        host="0.0.0.0",
        port=9999,
        max_connections=50
    )
    assert custom_config.host == "0.0.0.0"
    assert custom_config.port == 9999
    assert custom_config.max_connections == 50


def test_event_processor_creation():
    """Test creazione event processor."""
    from arima_forecaster.streaming import EventProcessor
    
    processor = EventProcessor()
    assert processor is not None
    assert hasattr(processor, 'create_event')
    assert hasattr(processor, 'submit_event')
    
    # Test configurazione custom
    custom_processor = EventProcessor(max_queue_size=50, worker_threads=1)
    assert custom_processor is not None


def test_event_creation():
    """Test creazione eventi."""
    from arima_forecaster.streaming import EventProcessor, EventType, EventPriority
    
    processor = EventProcessor()
    
    # Evento base
    event = processor.create_event(
        event_type=EventType.FORECAST_GENERATED,
        model_id="test_model",
        data={"predicted_value": 123.45}
    )
    
    assert event.event_type == EventType.FORECAST_GENERATED
    assert event.model_id == "test_model"
    assert event.data["predicted_value"] == 123.45
    assert event.priority == EventPriority.NORMAL  # Default
    
    # Evento con priorità alta
    high_priority_event = processor.create_event(
        event_type=EventType.ANOMALY_DETECTED,
        model_id="test_model",
        data={"anomaly_score": 0.95},
        priority=EventPriority.HIGH
    )
    
    assert high_priority_event.priority == EventPriority.HIGH


def test_realtime_forecaster_service():
    """Test servizio forecasting real-time."""
    from arima_forecaster.streaming import RealtimeForecastService
    
    service = RealtimeForecastService()
    assert service is not None
    assert hasattr(service, 'add_model')
    assert hasattr(service, 'get_stats')
    
    # Statistiche iniziali
    stats = service.get_stats()
    assert 'models_registered' in stats
    assert 'forecasts_generated' in stats
    assert 'anomalies_detected' in stats