"""
Kafka Producer per Real-Time Forecast Streaming

Gestisce la pubblicazione di forecast in real-time via Apache Kafka.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    _has_kafka = True
except ImportError:
    _has_kafka = False

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StreamingConfig:
    """Configurazione per streaming Kafka"""
    bootstrap_servers: List[str] = None
    topic: str = "arima-forecasts"
    key_serializer: str = "string"
    value_serializer: str = "json"
    batch_size: int = 16
    flush_interval_ms: int = 1000
    max_request_size: int = 1048576
    compression_type: Optional[str] = "gzip"
    
    def __post_init__(self):
        if self.bootstrap_servers is None:
            self.bootstrap_servers = ["localhost:9092"]


class ForecastMessage(BaseModel):
    """Schema messaggi forecast per Kafka"""
    model_id: str = Field(..., description="ID del modello")
    timestamp: datetime = Field(..., description="Timestamp generazione forecast")
    predicted_value: float = Field(..., description="Valore predetto")
    confidence_interval: List[float] = Field(..., description="Intervallo confidenza [lower, upper]")
    forecast_horizon: int = Field(..., description="Orizzonte previsione")
    model_type: str = Field(..., description="Tipo modello (ARIMA, SARIMA, etc)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadati aggiuntivi")


class KafkaForecastProducer:
    """
    Producer Kafka per streaming forecast real-time
    
    Features:
    - Pubblicazione asincrona forecast
    - Serializzazione JSON automatica
    - Gestione errori e retry
    - Metriche performance
    - Fallback locale in caso di problemi Kafka
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.producer = None
        self.is_connected = False
        self.message_count = 0
        self.error_count = 0
        
        if not _has_kafka:
            logger.warning("Kafka non disponibile. Installare: pip install kafka-python")
            return
            
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Errore inizializzazione Kafka producer: {e}")
    
    def _connect(self) -> bool:
        """Connette al cluster Kafka"""
        if not _has_kafka:
            return False
            
        try:
            producer_config = {
                'bootstrap_servers': self.config.bootstrap_servers,
                'value_serializer': lambda v: json.dumps(v, default=str).encode('utf-8'),
                'key_serializer': lambda k: str(k).encode('utf-8') if k else None,
                'batch_size': self.config.batch_size,
                'linger_ms': self.config.flush_interval_ms,
                'max_request_size': self.config.max_request_size,
                'compression_type': self.config.compression_type,
                'acks': 'all',
                'retries': 3
            }
            
            self.producer = KafkaProducer(**producer_config)
            self.is_connected = True
            logger.info(f"Kafka producer connesso: {self.config.bootstrap_servers}")
            return True
            
        except Exception as e:
            logger.error(f"Connessione Kafka fallita: {e}")
            self.is_connected = False
            return False
    
    def send_forecast(self, forecast: ForecastMessage) -> bool:
        """
        Invia forecast al topic Kafka
        
        Args:
            forecast: Messaggio forecast da inviare
            
        Returns:
            bool: True se invio riuscito, False altrimenti
        """
        if not self.is_connected or not self.producer:
            logger.warning("Kafka producer non connesso. Forecast non inviato.")
            self._store_locally(forecast)
            return False
        
        try:
            # Prepara messaggio
            message_data = forecast.dict()
            key = f"{forecast.model_id}_{forecast.timestamp.isoformat()}"
            
            # Invia asincrono
            future = self.producer.send(
                topic=self.config.topic,
                key=key,
                value=message_data
            )
            
            # Callback per gestione risultato
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            self.message_count += 1
            logger.debug(f"Forecast inviato: {forecast.model_id} -> {forecast.predicted_value}")
            return True
            
        except Exception as e:
            logger.error(f"Errore invio forecast: {e}")
            self.error_count += 1
            self._store_locally(forecast)
            return False
    
    def send_batch_forecasts(self, forecasts: List[ForecastMessage]) -> Dict[str, int]:
        """
        Invia batch di forecast
        
        Args:
            forecasts: Lista forecast da inviare
            
        Returns:
            dict: Statistiche invio {success: int, errors: int}
        """
        stats = {"success": 0, "errors": 0}
        
        for forecast in forecasts:
            if self.send_forecast(forecast):
                stats["success"] += 1
            else:
                stats["errors"] += 1
        
        # Flush per garantire invio
        if self.producer:
            self.producer.flush(timeout=5)
        
        logger.info(f"Batch forecast inviato: {stats}")
        return stats
    
    def flush(self, timeout: int = 10) -> bool:
        """Forza invio messaggi in coda"""
        if self.producer:
            try:
                self.producer.flush(timeout=timeout)
                return True
            except Exception as e:
                logger.error(f"Errore flush Kafka: {e}")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche producer"""
        return {
            "is_connected": self.is_connected,
            "messages_sent": self.message_count,
            "errors": self.error_count,
            "topic": self.config.topic,
            "bootstrap_servers": self.config.bootstrap_servers
        }
    
    def close(self):
        """Chiude connessione Kafka"""
        if self.producer:
            try:
                self.producer.flush(timeout=5)
                self.producer.close(timeout=5)
                logger.info("Kafka producer chiuso correttamente")
            except Exception as e:
                logger.error(f"Errore chiusura Kafka producer: {e}")
        
        self.is_connected = False
        self.producer = None
    
    def _on_send_success(self, record_metadata):
        """Callback successo invio"""
        logger.debug(f"Messaggio inviato: topic={record_metadata.topic}, "
                    f"partition={record_metadata.partition}, offset={record_metadata.offset}")
    
    def _on_send_error(self, exception):
        """Callback errore invio"""
        logger.error(f"Errore invio messaggio: {exception}")
        self.error_count += 1
    
    def _store_locally(self, forecast: ForecastMessage):
        """Fallback: salva forecast localmente se Kafka non disponibile"""
        try:
            import os
            from pathlib import Path
            
            # Crea directory fallback
            fallback_dir = Path("outputs/streaming_fallback")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            
            # Salva come JSON
            filename = f"forecast_{forecast.model_id}_{forecast.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = fallback_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(forecast.dict(), f, default=str, indent=2)
            
            logger.info(f"Forecast salvato localmente: {filepath}")
            
        except Exception as e:
            logger.error(f"Errore salvataggio locale: {e}")


# Utility functions
def create_forecast_message(
    model_id: str,
    predicted_value: float,
    confidence_interval: List[float],
    forecast_horizon: int = 1,
    model_type: str = "ARIMA",
    metadata: Optional[Dict[str, Any]] = None
) -> ForecastMessage:
    """Factory function per creare ForecastMessage"""
    return ForecastMessage(
        model_id=model_id,
        timestamp=datetime.now(),
        predicted_value=predicted_value,
        confidence_interval=confidence_interval,
        forecast_horizon=forecast_horizon,
        model_type=model_type,
        metadata=metadata or {}
    )


def get_default_streaming_config() -> StreamingConfig:
    """Restituisce configurazione streaming default"""
    return StreamingConfig(
        bootstrap_servers=["localhost:9092"],
        topic="arima-forecasts",
        batch_size=10,
        flush_interval_ms=1000
    )