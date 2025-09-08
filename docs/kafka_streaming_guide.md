# 🌊 Guida Completa al Real-Time Streaming con Apache Kafka

## Panoramica

Il modulo **Real-Time Streaming** di ARIMA Forecaster permette di creare architetture event-driven per forecast in tempo reale, utilizzando Apache Kafka come message broker e WebSocket per aggiornamenti dashboard live.

---

## 📋 Indice

1. [Introduzione al Real-Time Streaming](#introduzione)
2. [Installazione e Setup](#installazione)
3. [Configurazione Kafka](#configurazione-kafka)
4. [Kafka Producer per Forecast](#kafka-producer)
5. [WebSocket Server per Dashboard](#websocket-server)
6. [Servizio Forecast Real-Time](#servizio-real-time)
7. [Event Processing](#event-processing)
8. [Esempi Pratici](#esempi-pratici)
9. [Monitoraggio e Debugging](#monitoraggio)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduzione al Real-Time Streaming {#introduzione}

### Perché Real-Time Streaming?

Il forecasting tradizionale segue un pattern batch:
- **Batch**: Genera forecast ogni X ore/giorni
- **Latenza**: Ritardi tra eventi e decisioni
- **Reattività**: Risposta lenta a cambiamenti

Il **Real-Time Streaming** abilita:
- **Forecast continui**: Aggiornamenti ogni minuto/secondo
- **Bassa latenza**: <100ms dal dato al forecast
- **Event-driven**: Reazione immediata a eventi business
- **Scalabilità**: Gestione migliaia di modelli simultanei

### Architettura Event-Driven

```
[Dati Input] -> [Kafka Topic] -> [Forecast Service] -> [WebSocket] -> [Dashboard]
     |              |                    |                |             |
  Real-time      Message            Model Pool       Live Updates   Business
   Events         Queue             Processing        Streaming      Decisions
```

### Componenti Principali

1. **KafkaForecastProducer**: Invia forecast a topic Kafka
2. **RealtimeForecastService**: Servizio principale con scheduling
3. **WebSocketServer**: Server per dashboard real-time
4. **EventProcessor**: Processore eventi con priorità e regole

---

## 🛠️ Installazione e Setup {#installazione}

### Prerequisiti

```bash
# Python 3.9+
python --version

# Docker per Kafka
docker --version

# ARIMA Forecaster v0.4.0+
pip install arima-forecaster
```

### Installazione Dipendenze Streaming

```bash
# Opzione 1: Con UV (raccomandato)
uv add kafka-python websockets redis

# Opzione 2: Con pip
pip install kafka-python websockets redis

# Verifica installazione
python -c "
from arima_forecaster.streaming import KafkaForecastProducer
from arima_forecaster.streaming import WebSocketServer
print('✅ Streaming modules installati correttamente')
"
```

### Setup Kafka con Docker

```bash
# Opzione 1: Kafka Standalone
docker run -d \
  --name kafka \
  -p 9092:9092 \
  -e KAFKA_ZOOKEEPER_CONNECT=localhost:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  apache/kafka:latest

# Opzione 2: Docker Compose (raccomandato)
# Crea file docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"
      
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
EOF

# Avvia stack completo
docker-compose up -d

# Verifica servizi
docker-compose ps
```

### Test Connettività

```bash
# Test Kafka connection
python -c "
from kafka import KafkaProducer
try:
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    producer.close()
    print('✅ Kafka connessione OK')
except Exception as e:
    print(f'❌ Kafka errore: {e}')
"

# Test Redis connection
python -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print('✅ Redis connessione OK')
except Exception as e:
    print(f'❌ Redis errore: {e}')
"
```

---

## ⚙️ Configurazione Kafka {#configurazione-kafka}

### StreamingConfig

```python
from arima_forecaster.streaming import StreamingConfig

# Configurazione base
config = StreamingConfig(
    bootstrap_servers=["localhost:9092"],
    topic="arima-forecasts",
    key_serializer="string",
    value_serializer="json",
    batch_size=16,
    flush_interval_ms=1000,
    max_request_size=1048576,
    compression_type="gzip"
)

print(f"Topic: {config.topic}")
print(f"Servers: {config.bootstrap_servers}")
```

### Configurazione Avanzata

```python
# Configurazione production con retry e failover
production_config = StreamingConfig(
    bootstrap_servers=["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
    topic="prod-forecasts",
    compression_type="snappy",
    batch_size=100,        # Batch più grandi per performance
    flush_interval_ms=5000, # Flush meno frequente
    max_request_size=10485760  # 10MB max message
)

# Configurazione development per debugging
dev_config = StreamingConfig(
    bootstrap_servers=["localhost:9092"],
    topic="dev-forecasts",
    compression_type=None,  # No compression per debug
    batch_size=1,          # Invio immediato
    flush_interval_ms=100  # Flush frequente per test
)
```

### Variabili Ambiente

Crea file `.env` per configurazione esterna:

```bash
# .env file
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=arima-forecasts
KAFKA_COMPRESSION=gzip
REDIS_URL=redis://localhost:6379/0
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8765
```

```python
# Caricamento configurazione da .env
import os
from dotenv import load_dotenv

load_dotenv()

config = StreamingConfig(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(","),
    topic=os.getenv("KAFKA_TOPIC", "arima-forecasts"),
    compression_type=os.getenv("KAFKA_COMPRESSION", "gzip")
)
```

---

## 📡 Kafka Producer per Forecast {#kafka-producer}

### KafkaForecastProducer Basics

```python
from arima_forecaster.streaming import KafkaForecastProducer, StreamingConfig, ForecastMessage
from datetime import datetime

# Inizializza producer
config = StreamingConfig(topic="my-forecasts")
producer = KafkaForecastProducer(config)

# Verifica connessione
stats = producer.get_stats()
print(f"Connesso: {stats['is_connected']}")
print(f"Topic: {stats['topic']}")
```

### Invio Singolo Forecast

```python
# Crea messaggio forecast
forecast_msg = ForecastMessage(
    model_id="sales_model_001",
    timestamp=datetime.now(),
    predicted_value=1285.4,
    confidence_interval=[1180.5, 1390.3],
    forecast_horizon=7,
    model_type="SARIMA",
    metadata={
        "product_id": "PROD-001",
        "region": "Nord",
        "accuracy": 0.92
    }
)

# Invia messaggio
success = producer.send_forecast(forecast_msg)
print(f"Forecast inviato: {success}")

# Forza invio
producer.flush()
```

### Invio Batch di Forecast

```python
# Crea batch di forecast
batch_forecasts = []

for i in range(10):
    forecast = ForecastMessage(
        model_id=f"model_{i}",
        timestamp=datetime.now(),
        predicted_value=1000 + i * 50,
        confidence_interval=[900 + i * 45, 1100 + i * 55],
        forecast_horizon=1,
        model_type="ARIMA"
    )
    batch_forecasts.append(forecast)

# Invio batch
results = producer.send_batch_forecasts(batch_forecasts)
print(f"Batch results: {results}")
print(f"Successi: {results['success']}")
print(f"Errori: {results['errors']}")
```

### Gestione Errori e Fallback

```python
import logging

# Setup logging per debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Producer con gestione errori
try:
    producer = KafkaForecastProducer(config)
    
    # Il producer ha fallback automatico:
    # - Se Kafka non disponibile -> salva in locale
    # - Se topic non esiste -> viene creato automaticamente
    # - Se messaggio troppo grande -> viene diviso
    
    for i in range(100):
        forecast = ForecastMessage(
            model_id=f"robust_model_{i}",
            timestamp=datetime.now(),
            predicted_value=1000 + i,
            confidence_interval=[950 + i, 1050 + i],
            forecast_horizon=1,
            model_type="ARIMA"
        )
        
        sent = producer.send_forecast(forecast)
        if not sent:
            logger.warning(f"Forecast {i} salvato localmente (Kafka non disponibile)")
        
except Exception as e:
    logger.error(f"Errore producer: {e}")
    
finally:
    # Cleanup
    producer.close()
```

### Monitoraggio Producer

```python
import time

# Monitor producer performance
start_time = time.time()
messages_sent = 0

for i in range(1000):
    forecast = ForecastMessage(
        model_id="performance_test",
        timestamp=datetime.now(),
        predicted_value=1000,
        confidence_interval=[900, 1100],
        forecast_horizon=1,
        model_type="ARIMA"
    )
    
    if producer.send_forecast(forecast):
        messages_sent += 1

# Statistiche
elapsed = time.time() - start_time
stats = producer.get_stats()

print(f"Performance Test Results:")
print(f"Messaggi inviati: {messages_sent}/1000")
print(f"Tempo elapsed: {elapsed:.2f}s")
print(f"Throughput: {messages_sent/elapsed:.1f} msg/s")
print(f"Errori totali: {stats['errors']}")
print(f"Connessione: {stats['is_connected']}")
```

---

## 🌐 WebSocket Server per Dashboard {#websocket-server}

### WebSocketServer Basics

```python
import asyncio
from arima_forecaster.streaming import WebSocketServer, WebSocketConfig

# Configurazione WebSocket
ws_config = WebSocketConfig(
    host="localhost",
    port=8765,
    max_connections=100,
    heartbeat_interval=30,
    redis_url="redis://localhost:6379/0"
)

# Crea server
server = WebSocketServer(ws_config)

# Avvia server (async)
async def run_server():
    await server.start_server()

# In production
# asyncio.run(run_server())
```

### Client WebSocket Subscription

```python
# Esempio client JavaScript per dashboard
javascript_client_example = """
// Connessione WebSocket
const ws = new WebSocket('ws://localhost:8765/forecast-updates');

// Sottoscrizione a modelli specifici
ws.onopen = function(event) {
    const subscription = {
        type: 'subscribe',
        model_ids: ['sales_model_001', 'inventory_model_002'],
        update_types: ['new_forecast', 'anomaly_detected'],
        data_format: 'json'
    };
    ws.send(JSON.stringify(subscription));
};

// Gestione aggiornamenti real-time
ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    
    if (update.type === 'forecast_update') {
        updateDashboardChart(update.data);
    } else if (update.type === 'anomaly_alert') {
        showAlert(update.data);
    } else if (update.type === 'heartbeat') {
        console.log('Server alive:', update.data.server_time);
    }
};

// Riconnessione automatica
ws.onclose = function(event) {
    console.log('WebSocket closed, reconnecting...');
    setTimeout(() => {
        // Riconnetti dopo 5 secondi
        location.reload();
    }, 5000);
};
"""

print("Esempio client JavaScript salvato")
```

### Broadcasting Forecast Updates

```python
# Server WebSocket con broadcast
async def demo_websocket_broadcast():
    server = WebSocketServer(WebSocketConfig(port=8765))
    
    # Avvia server in background
    server_task = asyncio.create_task(server.start_server())
    
    # Simula broadcast periodici
    await asyncio.sleep(2)  # Attendi connessioni client
    
    for i in range(10):
        # Broadcast forecast update
        forecast_data = {
            "model_id": "demo_model",
            "predicted_value": 1000 + i * 10,
            "confidence_interval": [900 + i * 9, 1100 + i * 11],
            "timestamp": datetime.now().isoformat(),
            "iteration": i
        }
        
        await server.broadcast_forecast_update("demo_model", forecast_data)
        print(f"Broadcast {i}: forecast={forecast_data['predicted_value']}")
        
        await asyncio.sleep(1)  # Attendi 1 secondo
    
    # Cleanup
    await server.shutdown()

# Esegui demo
# asyncio.run(demo_websocket_broadcast())
```

### Anomaly Alerts

```python
async def demo_anomaly_alerts():
    server = WebSocketServer(WebSocketConfig())
    
    # Simula anomalia critica
    anomaly_data = {
        "anomaly_score": 0.95,
        "predicted_value": 2500,  # Valore anomalo
        "expected_range": [800, 1200],
        "severity": "CRITICAL",
        "message": "Forecast 150% superiore al normale",
        "recommended_actions": [
            "Verificare dati input",
            "Controllare processo produttivo",
            "Escalation a supervisore"
        ]
    }
    
    # Broadcast alert
    await server.broadcast_anomaly_alert("critical_model", anomaly_data)
    print("Alert anomalia inviato a tutti i client sottoscritti")

# In contesto real-time
async def monitor_anomalies(server, models):
    """Monitor continuo per anomalie"""
    while True:
        for model_id, model in models.items():
            # Genera forecast
            prediction = model.predict(1)[0]
            
            # Controlla se anomalo (semplificato)
            if prediction > model.historical_mean * 1.5:
                await server.broadcast_anomaly_alert(
                    model_id, 
                    {"severity": "HIGH", "value": prediction}
                )
        
        await asyncio.sleep(60)  # Check ogni minuto
```

---

## ⚡ Servizio Forecast Real-Time {#servizio-real-time}

### RealtimeForecastService Setup

```python
from arima_forecaster.streaming import RealtimeForecastService, RealtimeConfig
from arima_forecaster import ARIMAForecaster, SARIMAForecaster

# Configurazione servizio
config = RealtimeConfig(
    update_frequency_seconds=60,    # Forecast ogni minuto
    streaming_enabled=True,         # Abilita Kafka
    websocket_enabled=True,         # Abilita WebSocket
    anomaly_threshold=0.8,         # Soglia anomalie
    max_concurrent_forecasts=10    # Max modelli paralleli
)

# Crea servizio
service = RealtimeForecastService(config)
```

### Registrazione Modelli

```python
import pandas as pd
import numpy as np

# Crea dati demo
dates = pd.date_range('2024-01-01', periods=100, freq='D')
sales_data = pd.Series(
    1000 + np.cumsum(np.random.randn(100) * 10) + 
    100 * np.sin(2 * np.pi * np.arange(100) / 7),  # Pattern settimanale
    index=dates
)

# Addestra modelli
arima_model = ARIMAForecaster(order=(2, 1, 1))
arima_model.fit(sales_data)

sarima_model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,7))
sarima_model.fit(sales_data)

# Registra modelli nel servizio
service.add_model("sales_arima", arima_model, {
    "product": "Widget A",
    "region": "North",
    "model_type": "ARIMA"
})

service.add_model("sales_sarima", sarima_model, {
    "product": "Widget A", 
    "region": "North",
    "model_type": "SARIMA",
    "seasonality": "weekly"
})

print(f"Modelli registrati: {len(service.registry.models)}")
```

### Avvio Servizio Real-Time

```python
async def run_realtime_service():
    """Avvia servizio forecast real-time completo"""
    
    # Il servizio automaticamente:
    # 1. Genera forecast per tutti i modelli ogni 60 secondi
    # 2. Rileva anomalie con threshold 0.8
    # 3. Invia forecast via Kafka
    # 4. Aggiorna dashboard via WebSocket
    # 5. Gestisce errori con retry automatico
    
    try:
        print("🚀 Avvio servizio real-time...")
        await service.start_service()
        
    except KeyboardInterrupt:
        print("\n⏹️  Arresto servizio richiesto dall'utente")
    except Exception as e:
        print(f"❌ Errore servizio: {e}")
    finally:
        await service.shutdown()
        print("✅ Servizio fermato correttamente")

# Esegui servizio
# asyncio.run(run_realtime_service())
```

### Monitoring e Statistiche

```python
# Monitor servizio in esecuzione
async def monitor_service():
    """Monitor continuo performance servizio"""
    
    while service.is_running:
        stats = service.get_stats()
        
        print(f"\n📊 Statistiche Servizio Real-Time:")
        print(f"Uptime: {stats['uptime_seconds']:.0f} secondi")
        print(f"Modelli registrati: {stats['registered_models']}")
        print(f"Forecast generati: {stats['forecast_count']}")
        print(f"Errori: {stats['error_count']}")
        print(f"Error rate: {stats['error_rate']:.1f}%")
        print(f"Ultima predizione: {stats['last_forecast_time']}")
        
        # Statistiche Kafka
        if service.kafka_producer:
            kafka_stats = service.kafka_producer.get_stats()
            print(f"Kafka messaggi: {kafka_stats['messages_sent']}")
            print(f"Kafka errori: {kafka_stats['errors']}")
        
        # Statistiche WebSocket
        if service.websocket_server:
            ws_stats = service.websocket_server.get_stats()
            print(f"WebSocket connessioni: {ws_stats['active_connections']}")
            print(f"WebSocket messaggi in coda: {ws_stats['queued_messages']}")
        
        await asyncio.sleep(30)  # Monitor ogni 30 secondi

# Avvia monitor in background
# monitor_task = asyncio.create_task(monitor_service())
```

### Gestione Dinamica Modelli

```python
# Aggiunta/rimozione modelli runtime
async def dynamic_model_management():
    """Gestione dinamica modelli durante esecuzione"""
    
    # Simula aggiunta modelli runtime
    await asyncio.sleep(5)
    
    # Nuovo modello per diverso prodotto
    new_model = ARIMAForecaster(order=(1,1,1))
    new_data = sales_data * 1.2 + np.random.randn(100) * 20
    new_model.fit(new_data)
    
    service.add_model("sales_product_b", new_model, {
        "product": "Widget B",
        "region": "South"
    })
    print("➕ Nuovo modello aggiunto: sales_product_b")
    
    # Rimozione modello dopo 60 secondi
    await asyncio.sleep(60)
    service.remove_model("sales_arima")
    print("➖ Modello rimosso: sales_arima")
    
    # Lista modelli attivi
    active_models = service.registry.list_models()
    print(f"🔄 Modelli attivi: {active_models}")

# In contesto async main
# asyncio.create_task(dynamic_model_management())
```

---

## 📋 Event Processing {#event-processing}

### EventProcessor Basics

```python
from arima_forecaster.streaming import EventProcessor, EventType, EventPriority

# Crea event processor
processor = EventProcessor(
    max_queue_size=1000,
    worker_threads=4
)

# Event processor ha regole predefinite:
# - log_all_events: Logga tutti i forecast
# - save_anomalies: Salva anomalie su file
# - alert_critical_anomalies: Alert email per anomalie critiche
# - update_performance_metrics: Aggiorna metriche sistema

print(f"Regole attive: {len(processor.rules)}")
print(f"Azioni registrate: {len(processor.actions)}")
```

### Creazione Eventi Custom

```python
# Eventi forecast standard
forecast_event = processor.create_event(
    event_type=EventType.FORECAST_GENERATED,
    model_id="sales_model",
    data={
        "predicted_value": 1285.4,
        "confidence_interval": [1180.5, 1390.3],
        "model_accuracy": 0.92
    }
)

# Eventi anomalia con priorità alta
anomaly_event = processor.create_event(
    event_type=EventType.ANOMALY_DETECTED,
    model_id="critical_model",
    data={
        "anomaly_score": 0.95,
        "predicted_value": 2500,
        "expected_range": [800, 1200],
        "deviation": "150% above normal"
    },
    priority=EventPriority.CRITICAL
)

# Eventi errore sistema
error_event = processor.create_event(
    event_type=EventType.SYSTEM_ERROR,
    model_id="failing_model",
    data={
        "error_message": "Training failed: insufficient data",
        "error_code": "INSUFFICIENT_DATA",
        "retry_count": 3
    },
    priority=EventPriority.HIGH
)

print("Eventi creati per processamento")
```

### Sottomissione e Processamento Eventi

```python
# Sottometti eventi alla coda
submitted_forecast = processor.submit_event(forecast_event)
submitted_anomaly = processor.submit_event(anomaly_event)
submitted_error = processor.submit_event(error_event)

print(f"Eventi sottomessi: {submitted_forecast + submitted_anomaly + submitted_error}")

# Avvia processamento asincrono
async def run_event_processor():
    """Avvia event processor in background"""
    try:
        await processor.start_processing()
    except KeyboardInterrupt:
        print("Event processor fermato")
    finally:
        await processor.shutdown()

# Il processor automaticamente:
# 1. Processa eventi per priorità (CRITICAL > HIGH > MEDIUM > LOW)
# 2. Applica regole condizionali
# 3. Esegue azioni (log, save, email, etc.)
# 4. Gestisce retry per eventi falliti
# 5. Mantiene statistiche performance

# asyncio.run(run_event_processor())
```

### Regole Custom

```python
from arima_forecaster.streaming import EventRule, RuleType, RuleAction

# Regola custom per high-value predictions
high_value_rule = EventRule(
    name="high_value_forecast_alert",
    event_type=EventType.FORECAST_GENERATED,
    condition="data.get('predicted_value', 0) > 5000",  # Forecast > 5000
    action="send_alert",
    priority_filter=None,
    enabled=True
)

# Regola per modelli con accuracy bassa
low_accuracy_rule = EventRule(
    name="low_accuracy_warning",
    event_type=EventType.FORECAST_GENERATED,
    condition="data.get('model_accuracy', 1) < 0.7",  # Accuracy < 70%
    action="save_to_file",
    priority_filter=None,
    enabled=True
)

# Regola per errori ripetuti
repeated_errors_rule = EventRule(
    name="repeated_errors_escalation",
    event_type=EventType.SYSTEM_ERROR,
    condition="data.get('retry_count', 0) > 2",  # Più di 2 retry
    action="send_alert",
    priority_filter=EventPriority.HIGH,
    enabled=True
)

# Aggiungi regole al processor
processor.add_rule(high_value_rule)
processor.add_rule(low_accuracy_rule)
processor.add_rule(repeated_errors_rule)

print(f"Regole totali: {len(processor.rules)}")
```

### Azioni Custom

```python
# Azione custom per Slack notifications
async def send_slack_notification(event):
    """Invia notifica a canale Slack"""
    import aiohttp
    
    webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    
    message = {
        "text": f"🚨 ARIMA Forecast Alert",
        "attachments": [{
            "color": "danger" if event.type == EventType.ANOMALY_DETECTED else "good",
            "fields": [
                {"title": "Model", "value": event.model_id, "short": True},
                {"title": "Event", "value": event.type.value, "short": True},
                {"title": "Data", "value": str(event.data), "short": False}
            ]
        }]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=message) as resp:
            if resp.status == 200:
                print(f"Slack notification sent for {event.id}")
            else:
                print(f"Slack notification failed: {resp.status}")

# Registra azione custom
processor.register_action("slack_notify", send_slack_notification)

# Regola che usa azione custom
slack_rule = EventRule(
    name="slack_critical_alerts",
    event_type=EventType.ANOMALY_DETECTED,
    condition="priority == EventPriority.CRITICAL",
    action="slack_notify",
    priority_filter=EventPriority.CRITICAL
)

processor.add_rule(slack_rule)
print("Azione Slack registrata e regola aggiunta")
```

---

## 💡 Esempi Pratici {#esempi-pratici}

### Esempio 1: E-commerce Real-Time Forecasting

```python
"""
Scenario: E-commerce con forecast real-time per inventory management
- 100+ prodotti
- Aggiornamenti ogni 5 minuti
- Alert per stockout prediction
- Dashboard manager real-time
"""

import asyncio
from arima_forecaster import ARIMAForecaster
from arima_forecaster.streaming import *

async def ecommerce_realtime_example():
    # 1. Setup servizio real-time
    config = RealtimeConfig(
        update_frequency_seconds=300,  # 5 minuti
        anomaly_threshold=0.85,
        max_concurrent_forecasts=20
    )
    service = RealtimeForecastService(config)
    
    # 2. Registra modelli prodotti
    products = ['PROD-001', 'PROD-002', 'PROD-003']  # Simulato
    
    for product_id in products:
        # Simula dati storici prodotto
        historical_sales = generate_product_sales_data(product_id)
        
        # Addestra modello
        model = ARIMAForecaster(order=(2,1,1))
        model.fit(historical_sales)
        
        # Registra nel servizio
        service.add_model(f"sales_{product_id}", model, {
            "product_id": product_id,
            "category": "Electronics",
            "last_updated": datetime.now().isoformat()
        })
    
    # 3. Setup event processing per stockout alerts
    processor = service.event_processor if hasattr(service, 'event_processor') else EventProcessor()
    
    # Regola stockout prediction
    stockout_rule = EventRule(
        name="stockout_prediction_alert",
        event_type=EventType.FORECAST_GENERATED,
        condition="data.get('predicted_value', 0) < 10",  # Stock < 10 unità
        action="send_alert"
    )
    processor.add_rule(stockout_rule)
    
    # 4. Avvia servizio
    print("🛒 Avvio e-commerce real-time forecasting...")
    await service.start_service()

def generate_product_sales_data(product_id):
    """Genera dati vendite simulati"""
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    base_sales = hash(product_id) % 100 + 50  # Sales base per prodotto
    trend = np.linspace(0, 20, 90)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(90) / 7)  # Weekly pattern
    noise = np.random.randn(90) * 5
    
    sales = base_sales + trend + seasonal + noise
    return pd.Series(np.maximum(sales, 0), index=dates)  # No negative sales

# asyncio.run(ecommerce_realtime_example())
```

### Esempio 2: Manufacturing Demand Sensing

```python
"""
Scenario: Produzione manifatturiera con demand sensing
- Sensori IoT real-time
- Correlazione produzione-domanda
- Predictive maintenance integration
"""

async def manufacturing_demand_sensing():
    # 1. Setup streaming per sensori IoT
    iot_config = StreamingConfig(
        topic="iot-sensors",
        batch_size=50  # Batch per alta frequenza IoT
    )
    
    sensor_producer = KafkaForecastProducer(iot_config)
    
    # 2. Setup demand forecasting
    demand_config = RealtimeConfig(
        update_frequency_seconds=120,  # 2 minuti
        streaming_enabled=True,
        websocket_enabled=True
    )
    
    demand_service = RealtimeForecastService(demand_config)
    
    # 3. Correlazione sensori-domanda
    async def process_iot_data():
        """Processa dati IoT e aggiorna forecast"""
        while True:
            # Simula lettura sensori
            sensor_data = {
                "machine_efficiency": np.random.uniform(0.8, 0.98),
                "temperature": np.random.uniform(18, 25),
                "vibration": np.random.uniform(0, 0.1),
                "power_consumption": np.random.uniform(80, 120)
            }
            
            # Correlazione con domanda (semplificata)
            if sensor_data["machine_efficiency"] < 0.85:
                # Efficienza bassa -> possibile aumento domanda
                demand_impact = 1.15
            else:
                demand_impact = 1.0
            
            # Invia via Kafka per processing downstream
            iot_message = ForecastMessage(
                model_id="manufacturing_demand",
                timestamp=datetime.now(),
                predicted_value=sensor_data["power_consumption"] * demand_impact,
                confidence_interval=[80, 140],
                forecast_horizon=1,
                model_type="IoT_Correlated",
                metadata=sensor_data
            )
            
            sensor_producer.send_forecast(iot_message)
            await asyncio.sleep(10)  # Ogni 10 secondi
    
    # 4. Avvia processing parallelo
    iot_task = asyncio.create_task(process_iot_data())
    service_task = asyncio.create_task(demand_service.start_service())
    
    await asyncio.gather(iot_task, service_task)

# asyncio.run(manufacturing_demand_sensing())
```

### Esempio 3: Financial Real-Time Risk Management

```python
"""
Scenario: Risk management finanziario
- Forecast volatilità mercato
- Alert threshold risk
- Integration con trading systems
"""

async def financial_risk_management():
    # 1. Setup per dati finanziari ad alta frequenza
    financial_config = StreamingConfig(
        topic="market-risk",
        compression_type="snappy",  # Compressione veloce
        batch_size=1,               # Latenza minima
        flush_interval_ms=100
    )
    
    risk_producer = KafkaForecastProducer(financial_config)
    
    # 2. Event processor per risk alerts
    risk_processor = EventProcessor(max_queue_size=5000)
    
    # Regola risk threshold
    risk_rule = EventRule(
        name="high_volatility_alert",
        event_type=EventType.FORECAST_GENERATED,
        condition="data.get('predicted_value', 0) > 0.05",  # Volatilità > 5%
        action="send_alert",
        priority_filter=EventPriority.CRITICAL
    )
    risk_processor.add_rule(risk_rule)
    
    # 3. Risk calculation engine
    async def calculate_portfolio_risk():
        """Calcola rischio portfolio real-time"""
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']  # Simulato
        
        while True:
            portfolio_risk = 0
            
            for asset in assets:
                # Simula volatilità asset
                volatility = np.random.uniform(0.01, 0.08)
                weight = 0.25  # Equal weight
                
                asset_risk = volatility * weight
                portfolio_risk += asset_risk
                
                # Crea evento risk
                risk_event = risk_processor.create_event(
                    event_type=EventType.FORECAST_GENERATED,
                    model_id=f"risk_{asset}",
                    data={
                        "predicted_value": volatility,
                        "asset": asset,
                        "portfolio_weight": weight,
                        "contribution_to_risk": asset_risk
                    },
                    priority=EventPriority.CRITICAL if volatility > 0.05 else EventPriority.MEDIUM
                )
                
                # Sottometti evento
                risk_processor.submit_event(risk_event)
                
                # Invia via Kafka per trading system
                risk_message = ForecastMessage(
                    model_id=f"volatility_{asset}",
                    timestamp=datetime.now(),
                    predicted_value=volatility,
                    confidence_interval=[volatility * 0.8, volatility * 1.2],
                    forecast_horizon=1,
                    model_type="GARCH",
                    metadata={
                        "asset": asset,
                        "portfolio_risk": portfolio_risk,
                        "risk_level": "HIGH" if volatility > 0.05 else "NORMAL"
                    }
                )
                
                risk_producer.send_forecast(risk_message)
            
            await asyncio.sleep(5)  # Ogni 5 secondi
    
    # 4. Avvia risk management
    risk_task = asyncio.create_task(calculate_portfolio_risk())
    processor_task = asyncio.create_task(risk_processor.start_processing())
    
    await asyncio.gather(risk_task, processor_task)

# asyncio.run(financial_risk_management())
```

---

## 📊 Monitoraggio e Debugging {#monitoraggio}

### Logging Configurabile

```python
import logging
from arima_forecaster.utils.logger import get_logger

# Setup logging per debugging streaming
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Logger specifici per componenti
kafka_logger = get_logger("arima_forecaster.streaming.kafka_producer")
websocket_logger = get_logger("arima_forecaster.streaming.websocket_server")
realtime_logger = get_logger("arima_forecaster.streaming.realtime_forecaster")

# Configura livelli differenziati
kafka_logger.setLevel(logging.INFO)      # Meno verbose per Kafka
websocket_logger.setLevel(logging.DEBUG) # Debug completo per WebSocket
realtime_logger.setLevel(logging.INFO)   # Info per servizio principale
```

### Metriche Custom

```python
from collections import defaultdict
import time

class StreamingMetrics:
    """Collector metriche streaming custom"""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.timings = defaultdict(list)
        self.start_time = time.time()
    
    def increment(self, metric_name):
        self.metrics[metric_name] += 1
    
    def time_operation(self, operation_name):
        """Context manager per timing operations"""
        class TimingContext:
            def __init__(self, metrics, name):
                self.metrics = metrics
                self.name = name
                self.start = None
            
            def __enter__(self):
                self.start = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start
                self.metrics.timings[self.name].append(elapsed)
        
        return TimingContext(self, operation_name)
    
    def get_summary(self):
        """Riassunto metriche"""
        uptime = time.time() - self.start_time
        
        summary = {
            "uptime_seconds": uptime,
            "counters": dict(self.metrics),
            "average_timings": {}
        }
        
        for operation, times in self.timings.items():
            if times:
                summary["average_timings"][operation] = {
                    "avg_ms": np.mean(times) * 1000,
                    "max_ms": np.max(times) * 1000,
                    "min_ms": np.min(times) * 1000,
                    "count": len(times)
                }
        
        return summary

# Usage
metrics = StreamingMetrics()

# In producer
with metrics.time_operation("kafka_send"):
    producer.send_forecast(forecast_msg)

metrics.increment("forecasts_sent")
metrics.increment("kafka_messages")

# Report ogni 60 secondi
async def metrics_reporter():
    while True:
        summary = metrics.get_summary()
        print(f"\n📈 Streaming Metrics Report:")
        print(f"Uptime: {summary['uptime_seconds']:.1f}s")
        
        for counter, value in summary['counters'].items():
            print(f"{counter}: {value}")
        
        for operation, timing in summary['average_timings'].items():
            print(f"{operation}: {timing['avg_ms']:.2f}ms avg ({timing['count']} ops)")
        
        await asyncio.sleep(60)

# asyncio.create_task(metrics_reporter())
```

### Health Check Endpoints

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Health check API per monitoraggio
app = FastAPI(title="ARIMA Streaming Health Check")

@app.get("/health")
async def health_check():
    """Health check generale"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health/kafka")
async def kafka_health():
    """Health check Kafka connection"""
    try:
        # Test connessione Kafka
        test_config = StreamingConfig()
        test_producer = KafkaForecastProducer(test_config)
        stats = test_producer.get_stats()
        test_producer.close()
        
        return {
            "status": "healthy" if stats["is_connected"] else "unhealthy",
            "kafka_connected": stats["is_connected"],
            "messages_sent": stats["messages_sent"],
            "errors": stats["errors"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/health/websocket")
async def websocket_health():
    """Health check WebSocket server"""
    try:
        # Test WebSocket (simplified)
        ws_config = WebSocketConfig()
        return {
            "status": "configured",
            "host": ws_config.host,
            "port": ws_config.port,
            "max_connections": ws_config.max_connections
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/metrics")
async def get_metrics():
    """Endpoint metriche per monitoring"""
    return metrics.get_summary()

# Avvia health check server
# uvicorn health_check:app --host 0.0.0.0 --port 8080
```

### Debugging Tools

```python
# Debug Kafka messages
def debug_kafka_message(message: ForecastMessage):
    """Debug helper per messaggi Kafka"""
    print(f"\n🔍 Debug Kafka Message:")
    print(f"Model ID: {message.model_id}")
    print(f"Timestamp: {message.timestamp}")
    print(f"Value: {message.predicted_value}")
    print(f"Confidence: {message.confidence_interval}")
    print(f"Metadata: {message.metadata}")
    print(f"Serialized size: {len(json.dumps(message.dict()))} bytes")

# Debug WebSocket connections
async def debug_websocket_connections(server: WebSocketServer):
    """Monitor connessioni WebSocket"""
    while True:
        stats = server.get_stats()
        print(f"\n🌐 WebSocket Debug:")
        print(f"Active connections: {stats['active_connections']}")
        print(f"Queued messages: {stats['queued_messages']}")
        
        for client_id, subscription in server.subscriptions.items():
            print(f"Client {client_id}: models={subscription.model_ids}")
        
        await asyncio.sleep(30)

# Debug event processing
def debug_event_processing(processor: EventProcessor):
    """Debug event processor"""
    stats = processor.get_stats()
    
    print(f"\n⚡ Event Processor Debug:")
    print(f"Processed events: {stats['processed_events']}")
    print(f"Failed events: {stats['failed_events']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    for priority, queue in processor.event_queues.items():
        print(f"Queue {priority.name}: {queue.qsize()} eventi")

# Performance profiling
import cProfile
import pstats

def profile_streaming_performance():
    """Profila performance streaming"""
    
    def run_streaming_test():
        # Test performance con 1000 messaggi
        config = StreamingConfig()
        producer = KafkaForecastProducer(config)
        
        for i in range(1000):
            message = ForecastMessage(
                model_id=f"perf_test_{i}",
                timestamp=datetime.now(),
                predicted_value=1000 + i,
                confidence_interval=[900 + i, 1100 + i],
                forecast_horizon=1,
                model_type="ARIMA"
            )
            producer.send_forecast(message)
        
        producer.close()
    
    # Profile execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_streaming_test()
    
    profiler.disable()
    
    # Risultati
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# profile_streaming_performance()
```

---

## ✅ Best Practices {#best-practices}

### 1. Performance Optimization

```python
# ✅ DO: Batch processing per alta throughput
batch_messages = []
for i in range(100):
    message = create_forecast_message(...)
    batch_messages.append(message)

# Invia batch invece di singoli messaggi
results = producer.send_batch_forecasts(batch_messages)

# ✅ DO: Compressione per messaggi grandi
config = StreamingConfig(compression_type="gzip")  # o "snappy" per velocità

# ❌ DON'T: Flush dopo ogni messaggio (lento)
producer.send_forecast(message)
producer.flush()  # ❌ Troppo frequente

# ✅ DO: Flush periodico o automatico
for message in messages:
    producer.send_forecast(message)
producer.flush()  # ✅ Una volta alla fine
```

### 2. Error Handling

```python
# ✅ DO: Gestione errori robusta
try:
    success = producer.send_forecast(message)
    if not success:
        # Fallback strategy
        store_message_locally(message)
        schedule_retry(message)
except KafkaTimeoutError:
    logger.warning("Kafka timeout, using fallback")
    use_local_storage(message)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    alert_operations_team(e)

# ✅ DO: Retry con backoff exponential
import time

def send_with_retry(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            return producer.send_forecast(message)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
            else:
                raise e
```

### 3. Resource Management

```python
# ✅ DO: Context managers per cleanup
async def streaming_context():
    producer = None
    server = None
    try:
        producer = KafkaForecastProducer(config)
        server = WebSocketServer(ws_config)
        
        yield producer, server
        
    finally:
        if producer:
            producer.close()
        if server:
            await server.shutdown()

# Usage
async with streaming_context() as (producer, server):
    # Use producer and server
    pass  # Cleanup automatico

# ✅ DO: Connection pooling per WebSocket
class WebSocketPool:
    def __init__(self, max_connections=100):
        self.max_connections = max_connections
        self.active_connections = {}
    
    def add_connection(self, client_id, websocket):
        if len(self.active_connections) >= self.max_connections:
            # Remove oldest connection
            oldest = min(self.active_connections.keys())
            self.remove_connection(oldest)
        
        self.active_connections[client_id] = websocket
```

### 4. Monitoring e Alerting

```python
# ✅ DO: Monitoring proattivo
class StreamingMonitor:
    def __init__(self):
        self.thresholds = {
            "error_rate": 0.05,        # 5% error rate
            "latency_p95": 100,        # 100ms p95 latency
            "queue_size": 1000,        # Max queue size
            "memory_usage": 0.8        # 80% memory usage
        }
    
    def check_health(self, metrics):
        alerts = []
        
        if metrics["error_rate"] > self.thresholds["error_rate"]:
            alerts.append("HIGH_ERROR_RATE")
        
        if metrics["latency_p95"] > self.thresholds["latency_p95"]:
            alerts.append("HIGH_LATENCY")
        
        return alerts
    
    def send_alerts(self, alerts):
        for alert in alerts:
            # Send to monitoring system
            send_to_datadog(alert)
            send_to_slack(alert)

# ✅ DO: Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
```

### 5. Security

```python
# ✅ DO: Configurazione sicura Kafka
secure_config = StreamingConfig(
    bootstrap_servers=["kafka-ssl:9093"],
    # SSL configuration
    security_protocol="SSL",
    ssl_cafile="/path/to/ca-cert",
    ssl_certfile="/path/to/client-cert", 
    ssl_keyfile="/path/to/client-key",
    # SASL authentication
    sasl_mechanism="PLAIN",
    sasl_username="forecaster",
    sasl_password="secure_password"
)

# ✅ DO: Validazione input
def validate_forecast_message(message: dict) -> bool:
    required_fields = ["model_id", "predicted_value", "timestamp"]
    
    # Check required fields
    for field in required_fields:
        if field not in message:
            return False
    
    # Validate types
    if not isinstance(message["predicted_value"], (int, float)):
        return False
    
    # Range validation
    if message["predicted_value"] < 0 or message["predicted_value"] > 1e6:
        return False
    
    return True

# ✅ DO: Rate limiting
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True
```

---

## 🔧 Troubleshooting {#troubleshooting}

### Problemi Comuni Kafka

#### Errore: "No brokers available"

```bash
# Diagnosi
netstat -an | grep 9092  # Verifica porta aperta
telnet localhost 9092    # Test connessione TCP

# Soluzioni
# 1. Verifica Kafka sia avviato
docker ps | grep kafka

# 2. Verifica configurazione
docker logs kafka

# 3. Restart Kafka
docker restart kafka
```

#### Errore: "Topic does not exist"

```python
# Creazione topic automatica
from kafka.admin import KafkaAdminClient, NewTopic

admin = KafkaAdminClient(bootstrap_servers=['localhost:9092'])

# Crea topic se non esiste
topic = NewTopic(
    name="arima-forecasts",
    num_partitions=3,
    replication_factor=1
)

try:
    admin.create_topics([topic])
    print("Topic creato")
except Exception as e:
    print(f"Topic già esistente o errore: {e}")
```

#### Performance Problemi

```python
# Diagnosis performance
def diagnose_kafka_performance(producer):
    stats = producer.get_stats()
    
    print("🔍 Kafka Performance Diagnosis:")
    print(f"Messages sent: {stats['messages_sent']}")
    print(f"Errors: {stats['errors']}")
    print(f"Error rate: {stats['errors'] / max(stats['messages_sent'], 1) * 100:.2f}%")
    
    if stats['errors'] > stats['messages_sent'] * 0.05:
        print("❌ High error rate detected")
        print("Solutions:")
        print("- Check Kafka broker health")
        print("- Increase producer timeout")
        print("- Reduce batch size")
        print("- Check network connectivity")

# Ottimizzazioni performance
optimized_config = StreamingConfig(
    batch_size=100,           # Batch più grandi
    flush_interval_ms=5000,   # Flush meno frequente
    compression_type="snappy", # Compressione veloce
    max_request_size=10485760  # 10MB max
)
```

### Problemi WebSocket

#### Errore: "Connection refused"

```python
# Test WebSocket connection
import websockets
import asyncio

async def test_websocket():
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ WebSocket connection OK")
            
            # Test ping
            await websocket.ping()
            print("✅ WebSocket ping OK")
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        print("Solutions:")
        print("- Check WebSocket server is running")
        print("- Verify port 8765 is not in use")
        print("- Check firewall settings")

# asyncio.run(test_websocket())
```

#### Memory leaks WebSocket

```python
# Monitor WebSocket memory
import psutil
import gc

def monitor_websocket_memory(server):
    """Monitor memory usage WebSocket server"""
    process = psutil.Process()
    
    while True:
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
        
        stats = server.get_stats()
        print(f"Active connections: {stats['active_connections']}")
        print(f"Queued messages: {stats['queued_messages']}")
        
        # Force garbage collection se memoria alta
        if memory_info.rss > 500 * 1024 * 1024:  # >500MB
            gc.collect()
            print("🗑️  Garbage collection forced")
        
        time.sleep(30)
```

### Problemi Real-Time Service

#### Errore: "Model not found"

```python
# Debug model registry
def debug_model_registry(service):
    registry = service.registry
    
    print("🔍 Model Registry Debug:")
    print(f"Registered models: {len(registry.models)}")
    
    for model_id, model in registry.models.items():
        metadata = registry.model_metadata.get(model_id, {})
        last_pred = registry.get_last_prediction(model_id)
        
        print(f"\nModel: {model_id}")
        print(f"Type: {type(model).__name__}")
        print(f"Metadata: {metadata}")
        print(f"Last prediction: {last_pred}")

# Risoluzione
if "missing_model" not in service.registry.models:
    # Re-registra modello
    model = ARIMAForecaster(order=(2,1,1))
    model.fit(data)
    service.add_model("missing_model", model)
```

#### Performance degradation

```python
# Monitor performance servizio
async def monitor_service_performance(service):
    previous_stats = None
    
    while True:
        current_stats = service.get_stats()
        
        if previous_stats:
            # Calcola delta
            forecast_delta = current_stats['forecast_count'] - previous_stats['forecast_count']
            error_delta = current_stats['error_count'] - previous_stats['error_count']
            
            print(f"📊 Performance Delta (last 60s):")
            print(f"Forecasts: +{forecast_delta}")
            print(f"Errors: +{error_delta}")
            
            # Alert se performance degrada
            if forecast_delta < 5:  # Meno di 5 forecast/min
                print("⚠️  Low forecasting rate detected")
                print("Check:")
                print("- Model training time")
                print("- Data availability")
                print("- System resources")
            
            if error_delta > forecast_delta * 0.1:  # >10% error rate
                print("⚠️  High error rate detected")
                debug_model_registry(service)
        
        previous_stats = current_stats
        await asyncio.sleep(60)
```

### Logs Debugging

```python
# Setup logging dettagliato per debugging
import logging

# Logger per ogni componente
loggers = [
    "arima_forecaster.streaming.kafka_producer",
    "arima_forecaster.streaming.websocket_server", 
    "arima_forecaster.streaming.realtime_forecaster",
    "arima_forecaster.streaming.event_processor"
]

for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Handler con formato dettagliato
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# File logging per troubleshooting
file_handler = logging.FileHandler('streaming_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

for logger_name in loggers:
    logging.getLogger(logger_name).addHandler(file_handler)

print("🔍 Debug logging abilitato")
print("Check file: streaming_debug.log")
```

---

## 📞 Supporto e Risorse

### Documentazione Correlata

- [API Reference](api_reference.md) - Documentazione API completa
- [GPU Setup](GPU_SETUP.md) - Configurazione accelerazione GPU
- [AutoML Guide](AUTOML_GUIDE.md) - Guida Auto-ML avanzato
- [Batch Processing](BATCH_PROCESSING_GUIDE.md) - Processamento batch portfolio

### Community e Supporto

- **GitHub Issues**: Segnalazione bug e richieste feature
- **Discussions**: Q&A e discussioni tecniche
- **Wiki**: Guide community e esempi avanzati

### Performance Benchmarks

```python
# Benchmark streaming performance
async def benchmark_streaming():
    """Benchmark completo streaming performance"""
    
    # Test Kafka throughput
    kafka_start = time.time()
    producer = KafkaForecastProducer(StreamingConfig())
    
    for i in range(1000):
        message = ForecastMessage(
            model_id=f"bench_{i}",
            timestamp=datetime.now(),
            predicted_value=1000,
            confidence_interval=[900, 1100],
            forecast_horizon=1,
            model_type="ARIMA"
        )
        producer.send_forecast(message)
    
    producer.flush()
    kafka_time = time.time() - kafka_start
    
    print(f"🚀 Kafka Benchmark:")
    print(f"1000 messages in {kafka_time:.2f}s")
    print(f"Throughput: {1000/kafka_time:.1f} msg/s")
    
    producer.close()

# asyncio.run(benchmark_streaming())
```

**Target Performance:**
- **Kafka Throughput**: >1000 msg/s su hardware standard
- **WebSocket Latency**: <50ms per update
- **Real-time Service**: <5s forecast generation per modello
- **Event Processing**: >500 eventi/s con regole complesse

---

## 🎉 Conclusioni

Apache Kafka integration in ARIMA Forecaster abilita architetture event-driven moderne per forecasting enterprise. La combinazione di Kafka, WebSocket e Event Processing fornisce:

- **Real-time responsiveness** per decisioni business critiche
- **Scalabilità orizzontale** per gestire centinaia di modelli
- **Reliability** con fallback automatici e retry
- **Observability** completa per monitoring production

Il modulo è production-ready e testato per carichi di lavoro enterprise con SLA rigorosi.

---

*Documentazione aggiornata per ARIMA Forecaster v0.4.0 - Dicembre 2024*