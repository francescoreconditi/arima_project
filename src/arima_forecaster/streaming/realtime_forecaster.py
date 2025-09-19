"""
Real-Time Forecast Service

Servizio per generazione forecast in tempo reale con streaming.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import threading
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np

from arima_forecaster import ARIMAForecaster, SARIMAForecaster
from arima_forecaster.utils.logger import get_logger
from .kafka_producer import KafkaForecastProducer, ForecastMessage, StreamingConfig
from .websocket_server import WebSocketServer, WebSocketConfig

logger = get_logger(__name__)


@dataclass
class RealtimeConfig:
    """Configurazione servizio real-time"""

    update_frequency_seconds: int = 60
    models_directory: str = "outputs/models"
    streaming_enabled: bool = True
    websocket_enabled: bool = True
    kafka_config: Optional[StreamingConfig] = None
    websocket_config: Optional[WebSocketConfig] = None
    anomaly_threshold: float = 0.8
    max_concurrent_forecasts: int = 10

    def __post_init__(self):
        if self.kafka_config is None:
            self.kafka_config = StreamingConfig()
        if self.websocket_config is None:
            self.websocket_config = WebSocketConfig()


class ModelRegistry:
    """Registry per modelli attivi in real-time"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.last_predictions: Dict[str, Dict] = {}
        self.model_metadata: Dict[str, Dict] = {}

    def register_model(self, model_id: str, model: Any, metadata: Optional[Dict] = None):
        """Registra modello per forecast real-time"""
        self.models[model_id] = model
        self.model_metadata[model_id] = metadata or {}
        logger.info(f"Modello registrato: {model_id}")

    def unregister_model(self, model_id: str):
        """Rimuove modello dal registry"""
        self.models.pop(model_id, None)
        self.last_predictions.pop(model_id, None)
        self.model_metadata.pop(model_id, None)
        logger.info(f"Modello rimosso: {model_id}")

    def get_model(self, model_id: str):
        """Ottiene modello per ID"""
        return self.models.get(model_id)

    def list_models(self) -> List[str]:
        """Lista modelli registrati"""
        return list(self.models.keys())

    def get_last_prediction(self, model_id: str) -> Optional[Dict]:
        """Ottiene ultima predizione per modello"""
        return self.last_predictions.get(model_id)

    def update_prediction(self, model_id: str, prediction_data: Dict):
        """Aggiorna ultima predizione"""
        self.last_predictions[model_id] = {
            **prediction_data,
            "timestamp": datetime.now(),
            "model_id": model_id,
        }


class AnomalyDetector:
    """Rilevatore anomalie per forecast real-time"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.history: Dict[str, List[float]] = {}
        self.window_size = 50

    def check_anomaly(self, model_id: str, predicted_value: float) -> Dict[str, Any]:
        """
        Controlla se la predizione è anomala

        Returns:
            dict: {is_anomaly: bool, score: float, details: dict}
        """
        if model_id not in self.history:
            self.history[model_id] = []

        history = self.history[model_id]

        # Non abbastanza dati per rilevare anomalie
        if len(history) < 10:
            history.append(predicted_value)
            return {
                "is_anomaly": False,
                "score": 0.0,
                "details": {"reason": "insufficient_history", "history_length": len(history)},
            }

        # Calcola statistiche storiche
        mean_val = np.mean(history)
        std_val = np.std(history)

        if std_val == 0:
            score = 0.0
        else:
            # Z-score normalizzato
            z_score = abs(predicted_value - mean_val) / std_val
            score = min(z_score / 3.0, 1.0)  # Normalizza a [0,1]

        is_anomaly = score >= self.threshold

        # Aggiorna history (mantieni finestra scorrevole)
        history.append(predicted_value)
        if len(history) > self.window_size:
            history.pop(0)

        details = {
            "z_score": z_score if std_val > 0 else 0,
            "mean_historical": float(mean_val),
            "std_historical": float(std_val),
            "deviation": float(abs(predicted_value - mean_val)),
            "history_length": len(history),
        }

        if is_anomaly:
            logger.warning(
                f"Anomalia rilevata per {model_id}: valore={predicted_value}, "
                f"score={score:.3f}, z_score={z_score:.3f}"
            )

        return {"is_anomaly": is_anomaly, "score": float(score), "details": details}


class RealtimeForecastService:
    """
    Servizio principale per forecast real-time

    Features:
    - Forecast automatici schedulati
    - Streaming via Kafka e WebSocket
    - Rilevamento anomalie
    - Registry modelli dinamico
    - Gestione errori robusta
    """

    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.registry = ModelRegistry()
        self.anomaly_detector = AnomalyDetector(config.anomaly_threshold)
        self.is_running = False
        self.executor = None

        # Inizializza componenti streaming
        self.kafka_producer = None
        self.websocket_server = None

        if config.streaming_enabled:
            self.kafka_producer = KafkaForecastProducer(config.kafka_config)

        if config.websocket_enabled:
            self.websocket_server = WebSocketServer(config.websocket_config)

        self.forecast_count = 0
        self.error_count = 0
        self.start_time = None

    async def start_service(self):
        """Avvia servizio real-time"""
        if self.is_running:
            logger.warning("Servizio già in esecuzione")
            return

        self.is_running = True
        self.start_time = datetime.now()
        logger.info("Avvio servizio forecast real-time")

        tasks = []

        # Avvia WebSocket server se abilitato
        if self.websocket_server:
            tasks.append(self.websocket_server.start_server())

        # Avvia forecast scheduler
        tasks.append(self._forecast_scheduler())

        # Avvia task monitoraggio
        tasks.append(self._monitoring_task())

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Errore servizio real-time: {e}")
        finally:
            await self.shutdown()

    async def _forecast_scheduler(self):
        """Task principale per scheduling forecast"""
        logger.info(
            f"Forecast scheduler avviato (frequenza: {self.config.update_frequency_seconds}s)"
        )

        while self.is_running:
            try:
                start_time = time.time()

                # Genera forecast per tutti i modelli registrati
                model_ids = self.registry.list_models()
                if model_ids:
                    await self._generate_batch_forecasts(model_ids)

                elapsed = time.time() - start_time
                logger.debug(
                    f"Batch forecast completato in {elapsed:.2f}s per {len(model_ids)} modelli"
                )

                # Attendi prossimo ciclo
                await asyncio.sleep(self.config.update_frequency_seconds)

            except Exception as e:
                logger.error(f"Errore forecast scheduler: {e}")
                self.error_count += 1
                await asyncio.sleep(5)  # Pausa in caso di errore

    async def _generate_batch_forecasts(self, model_ids: List[str]):
        """Genera forecast per batch di modelli"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_forecasts)

        tasks = []
        for model_id in model_ids:
            task = self._generate_single_forecast(model_id, semaphore)
            tasks.append(task)

        # Esegui in parallelo con limite concorrenza
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count

        if error_count > 0:
            logger.warning(f"Forecast batch: {success_count} successi, {error_count} errori")

    async def _generate_single_forecast(self, model_id: str, semaphore: asyncio.Semaphore):
        """Genera forecast per singolo modello"""
        async with semaphore:
            try:
                model = self.registry.get_model(model_id)
                if not model:
                    logger.error(f"Modello non trovato: {model_id}")
                    return

                # Genera forecast (1 step ahead)
                prediction = await self._run_forecast(model, steps=1)

                if prediction is None:
                    return

                predicted_value = prediction["values"][0] if prediction["values"] else 0.0
                confidence_interval = [
                    prediction["lower_ci"][0] if prediction["lower_ci"] else predicted_value * 0.9,
                    prediction["upper_ci"][0] if prediction["upper_ci"] else predicted_value * 1.1,
                ]

                # Controllo anomalie
                anomaly_result = self.anomaly_detector.check_anomaly(model_id, predicted_value)

                # Prepara dati forecast
                forecast_data = {
                    "model_id": model_id,
                    "predicted_value": predicted_value,
                    "confidence_interval": confidence_interval,
                    "timestamp": datetime.now(),
                    "anomaly_detected": anomaly_result["is_anomaly"],
                    "anomaly_score": anomaly_result["score"],
                    "metadata": self.registry.model_metadata.get(model_id, {}),
                }

                # Aggiorna registry
                self.registry.update_prediction(model_id, forecast_data)

                # Invia via streaming
                await self._stream_forecast(forecast_data)

                # Alert anomalie se necessario
                if anomaly_result["is_anomaly"]:
                    await self._send_anomaly_alert(model_id, forecast_data, anomaly_result)

                self.forecast_count += 1
                logger.debug(f"Forecast generato per {model_id}: {predicted_value:.2f}")

            except Exception as e:
                logger.error(f"Errore forecast {model_id}: {e}")
                self.error_count += 1

    async def _run_forecast(self, model, steps: int = 1) -> Optional[Dict]:
        """Esegue forecast su modello"""
        try:
            # Esegui in thread separato per non bloccare async loop
            loop = asyncio.get_event_loop()

            def do_forecast():
                if hasattr(model, "predict"):
                    # Nuovo API
                    predictions = model.predict(steps)
                    return {
                        "values": predictions.tolist()
                        if hasattr(predictions, "tolist")
                        else [predictions],
                        "lower_ci": [(p * 0.9) for p in predictions.tolist()]
                        if hasattr(predictions, "tolist")
                        else [predictions * 0.9],
                        "upper_ci": [(p * 1.1) for p in predictions.tolist()]
                        if hasattr(predictions, "tolist")
                        else [predictions * 1.1],
                    }
                elif hasattr(model, "forecast"):
                    # API legacy
                    result = model.forecast(steps=steps, confidence_intervals=True)
                    return result
                else:
                    logger.error(f"Modello senza metodo predict/forecast")
                    return None

            result = await loop.run_in_executor(None, do_forecast)
            return result

        except Exception as e:
            logger.error(f"Errore esecuzione forecast: {e}")
            return None

    async def _stream_forecast(self, forecast_data: Dict[str, Any]):
        """Invia forecast via streaming"""
        try:
            # Kafka streaming
            if self.kafka_producer:
                forecast_msg = ForecastMessage(
                    model_id=forecast_data["model_id"],
                    timestamp=forecast_data["timestamp"],
                    predicted_value=forecast_data["predicted_value"],
                    confidence_interval=forecast_data["confidence_interval"],
                    forecast_horizon=1,
                    model_type=forecast_data["metadata"].get("model_type", "ARIMA"),
                    metadata=forecast_data["metadata"],
                )
                self.kafka_producer.send_forecast(forecast_msg)

            # WebSocket streaming
            if self.websocket_server:
                self.websocket_server.queue_message(
                    "forecast_update", forecast_data["model_id"], forecast_data
                )

        except Exception as e:
            logger.error(f"Errore streaming forecast: {e}")

    async def _send_anomaly_alert(self, model_id: str, forecast_data: Dict, anomaly_result: Dict):
        """Invia alert per anomalia rilevata"""
        try:
            alert_data = {
                "model_id": model_id,
                "anomaly_score": anomaly_result["score"],
                "predicted_value": forecast_data["predicted_value"],
                "anomaly_details": anomaly_result["details"],
                "timestamp": datetime.now(),
                "severity": "HIGH" if anomaly_result["score"] > 0.9 else "MEDIUM",
            }

            # WebSocket alert
            if self.websocket_server:
                self.websocket_server.queue_message("anomaly_alert", model_id, alert_data)

            logger.warning(f"Alert anomalia inviato per {model_id}")

        except Exception as e:
            logger.error(f"Errore invio alert anomalia: {e}")

    async def _monitoring_task(self):
        """Task monitoraggio servizio"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Log stats ogni 5 minuti

                uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                stats = {
                    "uptime_seconds": uptime.total_seconds(),
                    "registered_models": len(self.registry.models),
                    "forecast_count": self.forecast_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(self.forecast_count, 1),
                    "kafka_stats": self.kafka_producer.get_stats() if self.kafka_producer else None,
                    "websocket_stats": self.websocket_server.get_stats()
                    if self.websocket_server
                    else None,
                }

                logger.info(f"Stats servizio real-time: {json.dumps(stats, indent=2)}")

            except Exception as e:
                logger.error(f"Errore monitoring task: {e}")

    def add_model(self, model_id: str, model: Any, metadata: Optional[Dict] = None):
        """Aggiunge modello al servizio real-time"""
        self.registry.register_model(model_id, model, metadata)

    def remove_model(self, model_id: str):
        """Rimuove modello dal servizio"""
        self.registry.unregister_model(model_id)

    def get_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche servizio"""
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)

        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime.total_seconds(),
            "registered_models": len(self.registry.models),
            "forecast_count": self.forecast_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.forecast_count, 1) * 100,
            "last_forecast_time": max(
                [pred.get("timestamp") for pred in self.registry.last_predictions.values()],
                default=None,
            ),
        }

    async def shutdown(self):
        """Chiude servizio real-time"""
        logger.info("Chiusura servizio real-time")
        self.is_running = False

        if self.kafka_producer:
            self.kafka_producer.close()

        if self.websocket_server:
            await self.websocket_server.shutdown()

        logger.info("Servizio real-time chiuso")


# Utility functions
def create_realtime_service(
    update_frequency: int = 60, streaming_enabled: bool = True, websocket_enabled: bool = True
) -> RealtimeForecastService:
    """Factory per creare servizio real-time"""
    config = RealtimeConfig(
        update_frequency_seconds=update_frequency,
        streaming_enabled=streaming_enabled,
        websocket_enabled=websocket_enabled,
    )
    return RealtimeForecastService(config)


async def run_realtime_demo():
    """Demo servizio real-time"""
    from arima_forecaster import ARIMAForecaster

    # Crea servizio
    service = create_realtime_service(update_frequency=10)

    # Crea modello demo
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.random.normal(1000, 100, 100) + np.sin(np.arange(100) * 2 * np.pi / 7) * 50
    data = pd.Series(values, index=dates)

    model = ARIMAForecaster(order=(2, 1, 1))
    model.fit(data)

    # Registra modello
    service.add_model("demo_sales", model, {"product": "Widget A", "model_type": "ARIMA"})

    # Avvia servizio
    try:
        await service.start_service()
    except KeyboardInterrupt:
        logger.info("Demo interrotto da utente")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(run_realtime_demo())
