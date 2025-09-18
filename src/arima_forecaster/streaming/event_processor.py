"""
Event Processor per Real-Time Streaming

Gestisce eventi e trigger per aggiornamenti automatici.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel
import threading
import queue
from pathlib import Path

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Tipi di eventi supportati"""

    FORECAST_GENERATED = "forecast_generated"
    MODEL_RETRAINED = "model_retrained"
    ANOMALY_DETECTED = "anomaly_detected"
    DATA_UPDATED = "data_updated"
    ALERT_TRIGGERED = "alert_triggered"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADED = "performance_degraded"


class EventPriority(Enum):
    """Priorità eventi"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ForecastEvent(BaseModel):
    """Schema evento forecast"""

    id: str
    type: EventType
    priority: EventPriority = EventPriority.MEDIUM
    timestamp: datetime
    model_id: str
    data: Dict[str, Any]
    source: str = "forecast_system"
    processed: bool = False
    retry_count: int = 0
    max_retries: int = 3


class EventRule(BaseModel):
    """Regola per processamento eventi"""

    name: str
    event_type: EventType
    condition: str  # Python expression da valutare
    action: str  # Nome dell'azione da eseguire
    enabled: bool = True
    priority_filter: Optional[EventPriority] = None


class EventAction:
    """Azione da eseguire per evento"""

    def __init__(self, name: str, handler: Callable[[ForecastEvent], Any]):
        self.name = name
        self.handler = handler
        self.execution_count = 0
        self.error_count = 0
        self.last_execution = None

    async def execute(self, event: ForecastEvent) -> bool:
        """Esegue azione per evento"""
        try:
            self.last_execution = datetime.now()

            if asyncio.iscoroutinefunction(self.handler):
                await self.handler(event)
            else:
                self.handler(event)

            self.execution_count += 1
            logger.debug(f"Azione {self.name} eseguita per evento {event.id}")
            return True

        except Exception as e:
            self.error_count += 1
            logger.error(f"Errore esecuzione azione {self.name}: {e}")
            return False


class EventProcessor:
    """
    Processore eventi per sistema real-time

    Features:
    - Coda eventi prioritaria
    - Regole condizionali
    - Azioni personalizzabili
    - Retry automatico
    - Persistenza eventi
    - Metriche performance
    """

    def __init__(self, max_queue_size: int = 1000, worker_threads: int = 4):
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads

        # Code eventi per priorità
        self.event_queues = {
            EventPriority.CRITICAL: queue.PriorityQueue(),
            EventPriority.HIGH: queue.PriorityQueue(),
            EventPriority.MEDIUM: queue.PriorityQueue(),
            EventPriority.LOW: queue.PriorityQueue(),
        }

        self.rules: List[EventRule] = []
        self.actions: Dict[str, EventAction] = {}
        self.is_running = False
        self.worker_tasks = []

        # Statistiche
        self.processed_events = 0
        self.failed_events = 0
        self.start_time = None

        # Default actions
        self._register_default_actions()

    def _register_default_actions(self):
        """Registra azioni predefinite"""

        def log_event(event: ForecastEvent):
            """Log evento"""
            logger.info(f"Evento {event.type.value}: {event.model_id} - {event.data}")

        def save_to_file(event: ForecastEvent):
            """Salva evento su file"""
            try:
                events_dir = Path("outputs/events")
                events_dir.mkdir(parents=True, exist_ok=True)

                filename = (
                    f"event_{event.type.value}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                )
                filepath = events_dir / filename

                with open(filepath, "w") as f:
                    json.dump(
                        {
                            "id": event.id,
                            "type": event.type.value,
                            "priority": event.priority.value,
                            "timestamp": event.timestamp.isoformat(),
                            "model_id": event.model_id,
                            "data": event.data,
                            "source": event.source,
                        },
                        f,
                        indent=2,
                    )

                logger.debug(f"Evento salvato: {filepath}")

            except Exception as e:
                logger.error(f"Errore salvataggio evento: {e}")

        async def send_alert_email(event: ForecastEvent):
            """Invia alert email (placeholder)"""
            if event.priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
                logger.warning(f"ALERT EMAIL: {event.type.value} per {event.model_id}")
                # TODO: Integrazione SMTP reale

        def update_metrics(event: ForecastEvent):
            """Aggiorna metriche sistema"""
            if event.type == EventType.PERFORMANCE_DEGRADED:
                logger.warning(f"Performance degradata per {event.model_id}: {event.data}")
            elif event.type == EventType.SYSTEM_ERROR:
                logger.error(f"Errore sistema per {event.model_id}: {event.data}")

        # Registra azioni
        self.register_action("log_event", log_event)
        self.register_action("save_to_file", save_to_file)
        self.register_action("send_alert", send_alert_email)
        self.register_action("update_metrics", update_metrics)

    def register_action(self, name: str, handler: Callable[[ForecastEvent], Any]):
        """Registra nuova azione"""
        self.actions[name] = EventAction(name, handler)
        logger.info(f"Azione registrata: {name}")

    def add_rule(self, rule: EventRule):
        """Aggiunge regola processamento"""
        self.rules.append(rule)
        logger.info(f"Regola aggiunta: {rule.name} per {rule.event_type.value}")

    def remove_rule(self, rule_name: str):
        """Rimuove regola"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Regola rimossa: {rule_name}")

    def create_event(
        self,
        event_type: EventType,
        model_id: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.MEDIUM,
        source: str = "forecast_system",
    ) -> ForecastEvent:
        """Crea nuovo evento"""
        event = ForecastEvent(
            id=f"{event_type.value}_{model_id}_{datetime.now().timestamp()}",
            type=event_type,
            priority=priority,
            timestamp=datetime.now(),
            model_id=model_id,
            data=data,
            source=source,
        )

        return event

    def submit_event(self, event: ForecastEvent) -> bool:
        """Sottomette evento alla coda"""
        try:
            queue_obj = self.event_queues[event.priority]

            if queue_obj.qsize() >= self.max_queue_size:
                logger.warning(f"Coda eventi piena per priorità {event.priority.name}")
                return False

            # Usa timestamp negativo per ordinamento FIFO nella PriorityQueue
            queue_obj.put((-event.timestamp.timestamp(), event))
            logger.debug(f"Evento sottomesso: {event.id}")
            return True

        except Exception as e:
            logger.error(f"Errore sottomissione evento: {e}")
            return False

    def submit_forecast_event(
        self,
        model_id: str,
        predicted_value: float,
        confidence_interval: List[float],
        anomaly_detected: bool = False,
    ) -> bool:
        """Utility per sottomettere evento forecast"""
        event_type = (
            EventType.ANOMALY_DETECTED if anomaly_detected else EventType.FORECAST_GENERATED
        )
        priority = EventPriority.HIGH if anomaly_detected else EventPriority.MEDIUM

        event = self.create_event(
            event_type=event_type,
            model_id=model_id,
            data={
                "predicted_value": predicted_value,
                "confidence_interval": confidence_interval,
                "anomaly_detected": anomaly_detected,
            },
            priority=priority,
        )

        return self.submit_event(event)

    def submit_error_event(
        self, model_id: str, error_message: str, error_type: str = "general"
    ) -> bool:
        """Utility per sottomettere evento errore"""
        event = self.create_event(
            event_type=EventType.SYSTEM_ERROR,
            model_id=model_id,
            data={"error_message": error_message, "error_type": error_type},
            priority=EventPriority.HIGH,
        )

        return self.submit_event(event)

    async def start_processing(self):
        """Avvia processamento eventi"""
        if self.is_running:
            logger.warning("Event processor già in esecuzione")
            return

        self.is_running = True
        self.start_time = datetime.now()
        logger.info(f"Event processor avviato con {self.worker_threads} worker")

        # Avvia worker tasks
        self.worker_tasks = []
        for i in range(self.worker_threads):
            task = asyncio.create_task(self._worker_loop(i))
            self.worker_tasks.append(task)

        # Avvia task monitoraggio
        monitor_task = asyncio.create_task(self._monitor_loop())
        self.worker_tasks.append(monitor_task)

        # Attendi completamento
        try:
            await asyncio.gather(*self.worker_tasks)
        except Exception as e:
            logger.error(f"Errore event processor: {e}")
        finally:
            self.is_running = False

    async def _worker_loop(self, worker_id: int):
        """Loop principale worker"""
        logger.info(f"Worker {worker_id} avviato")

        while self.is_running:
            try:
                event = await self._get_next_event()

                if event:
                    await self._process_event(event)
                    self.processed_events += 1
                else:
                    # Nessun evento, pausa breve
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Errore worker {worker_id}: {e}")
                self.failed_events += 1
                await asyncio.sleep(1)

        logger.info(f"Worker {worker_id} terminato")

    async def _get_next_event(self) -> Optional[ForecastEvent]:
        """Ottiene prossimo evento dalla coda (per priorità)"""
        # Controlla code in ordine di priorità
        for priority in [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.MEDIUM,
            EventPriority.LOW,
        ]:
            queue_obj = self.event_queues[priority]

            try:
                # Non bloccante
                _, event = queue_obj.get_nowait()
                return event
            except queue.Empty:
                continue

        return None

    async def _process_event(self, event: ForecastEvent):
        """Processa singolo evento"""
        try:
            logger.debug(f"Processando evento {event.id}")

            # Trova regole applicabili
            applicable_rules = self._find_applicable_rules(event)

            if not applicable_rules:
                logger.debug(f"Nessuna regola per evento {event.id}")
                return

            # Esegui azioni per ogni regola
            for rule in applicable_rules:
                if rule.action in self.actions:
                    action = self.actions[rule.action]
                    success = await action.execute(event)

                    if not success and event.retry_count < event.max_retries:
                        # Riprova evento fallito
                        event.retry_count += 1
                        await asyncio.sleep(2**event.retry_count)  # Backoff exponential
                        self.submit_event(event)
                        logger.info(
                            f"Evento {event.id} riprovato ({event.retry_count}/{event.max_retries})"
                        )
                else:
                    logger.warning(f"Azione non trovata: {rule.action}")

            event.processed = True

        except Exception as e:
            logger.error(f"Errore processamento evento {event.id}: {e}")
            self.failed_events += 1

    def _find_applicable_rules(self, event: ForecastEvent) -> List[EventRule]:
        """Trova regole applicabili per evento"""
        applicable = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Filtra per tipo evento
            if rule.event_type != event.type:
                continue

            # Filtra per priorità se specificato
            if rule.priority_filter and rule.priority_filter != event.priority:
                continue

            # Valuta condizione
            if self._evaluate_condition(rule.condition, event):
                applicable.append(rule)

        return applicable

    def _evaluate_condition(self, condition: str, event: ForecastEvent) -> bool:
        """Valuta condizione Python per evento"""
        try:
            # Crea context per valutazione
            context = {
                "event": event,
                "data": event.data,
                "model_id": event.model_id,
                "priority": event.priority,
                "type": event.type,
                "timestamp": event.timestamp,
                "datetime": datetime,
                "timedelta": timedelta,
            }

            # Valuta condizione (sicura)
            result = eval(condition, {"__builtins__": {}}, context)
            return bool(result)

        except Exception as e:
            logger.error(f"Errore valutazione condizione '{condition}': {e}")
            return False

    async def _monitor_loop(self):
        """Loop monitoraggio statistiche"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Stats ogni 5 minuti

                uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)

                queue_sizes = {
                    priority.name: queue_obj.qsize()
                    for priority, queue_obj in self.event_queues.items()
                }

                action_stats = {
                    name: {
                        "executions": action.execution_count,
                        "errors": action.error_count,
                        "last_execution": action.last_execution.isoformat()
                        if action.last_execution
                        else None,
                    }
                    for name, action in self.actions.items()
                }

                stats = {
                    "uptime_seconds": uptime.total_seconds(),
                    "processed_events": self.processed_events,
                    "failed_events": self.failed_events,
                    "success_rate": (
                        self.processed_events / max(self.processed_events + self.failed_events, 1)
                    )
                    * 100,
                    "queue_sizes": queue_sizes,
                    "active_rules": len([r for r in self.rules if r.enabled]),
                    "registered_actions": len(self.actions),
                    "action_stats": action_stats,
                }

                logger.info(f"Event processor stats: {json.dumps(stats, indent=2)}")

            except Exception as e:
                logger.error(f"Errore monitor loop: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche event processor"""
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)

        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime.total_seconds(),
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "success_rate": (
                self.processed_events / max(self.processed_events + self.failed_events, 1)
            )
            * 100,
            "queue_sizes": {
                priority.name: queue_obj.qsize()
                for priority, queue_obj in self.event_queues.items()
            },
            "active_rules": len([r for r in self.rules if r.enabled]),
            "registered_actions": len(self.actions),
        }

    async def shutdown(self):
        """Chiude event processor"""
        logger.info("Chiusura event processor")
        self.is_running = False

        # Attendi completamento worker
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        logger.info("Event processor chiuso")


def create_default_rules() -> List[EventRule]:
    """Crea regole predefinite"""
    rules = [
        EventRule(
            name="log_all_events",
            event_type=EventType.FORECAST_GENERATED,
            condition="True",
            action="log_event",
        ),
        EventRule(
            name="save_anomalies",
            event_type=EventType.ANOMALY_DETECTED,
            condition="True",
            action="save_to_file",
        ),
        EventRule(
            name="alert_critical_anomalies",
            event_type=EventType.ANOMALY_DETECTED,
            condition="priority == EventPriority.CRITICAL or data.get('anomaly_score', 0) > 0.9",
            action="send_alert",
            priority_filter=EventPriority.HIGH,
        ),
        EventRule(
            name="log_errors",
            event_type=EventType.SYSTEM_ERROR,
            condition="True",
            action="log_event",
        ),
        EventRule(
            name="save_errors",
            event_type=EventType.SYSTEM_ERROR,
            condition="True",
            action="save_to_file",
        ),
        EventRule(
            name="update_performance_metrics",
            event_type=EventType.PERFORMANCE_DEGRADED,
            condition="True",
            action="update_metrics",
        ),
    ]

    return rules


# Demo function
async def run_event_processor_demo():
    """Demo event processor"""
    processor = EventProcessor(max_queue_size=100, worker_threads=2)

    # Aggiungi regole predefinite
    for rule in create_default_rules():
        processor.add_rule(rule)

    # Crea task per generare eventi demo
    async def generate_demo_events():
        await asyncio.sleep(2)  # Attendi avvio processor

        for i in range(10):
            # Eventi forecast normali
            processor.submit_forecast_event(
                model_id=f"demo_model_{i % 3}",
                predicted_value=1000 + i * 10,
                confidence_interval=[900 + i * 10, 1100 + i * 10],
            )

            # Alcuni eventi anomalia
            if i % 4 == 0:
                processor.submit_forecast_event(
                    model_id=f"demo_model_{i % 3}",
                    predicted_value=2000,  # Valore anomalo
                    confidence_interval=[1800, 2200],
                    anomaly_detected=True,
                )

            # Eventi errore occasionali
            if i % 6 == 0:
                processor.submit_error_event(
                    model_id=f"demo_model_{i % 3}",
                    error_message=f"Demo error {i}",
                    error_type="demo",
                )

            await asyncio.sleep(1)

    # Avvia processor e generatore eventi
    try:
        await asyncio.gather(processor.start_processing(), generate_demo_events())
    except KeyboardInterrupt:
        logger.info("Demo interrotto")
    finally:
        await processor.shutdown()


if __name__ == "__main__":
    asyncio.run(run_event_processor_demo())
