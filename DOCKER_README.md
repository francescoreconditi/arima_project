# 🐳 Docker Deployment - ARIMA Forecaster FastAPI

Guida completa per deployare la FastAPI REST API in Docker container.

## 📋 Prerequisiti

- Docker Desktop installato (Windows/Mac/Linux)
- Docker Compose v2.0+
- Almeno 2GB RAM disponibili per il container

## 🚀 Quick Start

### 1️⃣ Build dell'immagine Docker

```bash
# Dalla root del progetto
docker build -t arima-forecaster-api:0.4.0 .
```

### 2️⃣ Avvio con Docker Compose (CONSIGLIATO)

```bash
# Avvia tutti i servizi
docker-compose up -d

# Visualizza logs
docker-compose logs -f arima-api

# Verifica status
docker-compose ps
```

### 3️⃣ Verifica funzionamento

```bash
# Health check
curl http://localhost:8000/health

# Documentazione API
# Browser: http://localhost:8000/docs
# Browser: http://localhost:8000/scalar
```

## 📦 Comandi Docker Utili

### Gestione Container

```bash
# Stop servizi
docker-compose down

# Stop e rimuovi volumi
docker-compose down -v

# Rebuild immagine
docker-compose up -d --build

# Restart singolo servizio
docker-compose restart arima-api

# View logs real-time
docker-compose logs -f arima-api

# Accedi al container
docker-compose exec arima-api bash
```

### Gestione Immagini

```bash
# Lista immagini
docker images | grep arima

# Rimuovi immagine
docker rmi arima-forecaster-api:0.4.0

# Pulisci immagini non utilizzate
docker image prune
```

## ⚙️ Configurazione

### Variabili d'Ambiente

Modifica [docker-compose.yml](docker-compose.yml:1) per personalizzare:

```yaml
environment:
  - API_HOST=0.0.0.0          # Host binding
  - API_PORT=8000             # Porta interna
  - API_WORKERS=4             # Worker Uvicorn (CPU cores)
  - PRODUCTION_MODE=true      # Modalità produzione
  - LOG_LEVEL=info            # Livello logging (debug/info/warning/error)
  - CORS_ORIGINS=*            # CORS origins (modifica in produzione!)
```

### Porte Esposte

- **8000**: FastAPI REST API
- **6379**: Redis (opzionale, decommentare in compose)
- **9092**: Kafka (opzionale, decommentare in compose)

### Volumi Persistenti

I seguenti volumi sono montati per persistere i dati:

```yaml
volumes:
  - ./outputs/models:/app/outputs/models      # Modelli addestrati
  - ./outputs/plots:/app/outputs/plots        # Grafici generati
  - ./outputs/reports:/app/outputs/reports    # Report PDF/HTML
  - ./logs:/app/logs                          # Application logs
```

## 🔧 Servizi Opzionali

### Redis (Caching)

Decommenta nel [docker-compose.yml](docker-compose.yml:1):

```yaml
redis:
  image: redis:7-alpine
  container_name: arima-redis
  restart: unless-stopped
  ports:
    - "6379:6379"
  volumes:
    - redis-data:/data
  networks:
    - arima-network
```

### Kafka (Real-Time Streaming)

Decommenta nel [docker-compose.yml](docker-compose.yml:1):

```yaml
kafka:
  image: apache/kafka:latest
  container_name: arima-kafka
  restart: unless-stopped
  ports:
    - "9092:9092"
  environment:
    - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
  volumes:
    - kafka-data:/var/lib/kafka/data
  networks:
    - arima-network
```

## 📊 Monitoraggio

### Health Checks

Il container include health check automatici:

```bash
# Verifica health status
docker inspect --format='{{.State.Health.Status}}' arima-forecaster-api

# Visualizza health log
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' arima-forecaster-api
```

### Metriche Container

```bash
# Stats real-time
docker stats arima-forecaster-api

# Utilizzo risorse
docker-compose top
```

## 🔒 Sicurezza

### Best Practices Implementate

✅ **Utente non-root**: Container esegue come `apiuser` (UID 1000)
✅ **Minimal image**: Base `python:3.11-slim` riduce superficie attacco
✅ **Health checks**: Monitoring automatico stato servizio
✅ **CORS configurabile**: Limita origins in produzione
✅ **No secrets in env**: Usa Docker secrets o config files

### Produzione - CORS Origins

⚠️ **IMPORTANTE**: In produzione, modifica CORS origins:

```yaml
environment:
  - CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### Docker Secrets (Opzionale)

Per credenziali sensibili:

```bash
# Crea secret
echo "your-api-key" | docker secret create api_key -

# Usa nel compose
secrets:
  - api_key
```

## 🌐 Deployment Produzione

### Docker Swarm

```bash
# Inizializza swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml arima-stack

# Verifica servizi
docker service ls

# Scale workers
docker service scale arima-stack_arima-api=3
```

### Kubernetes

Converti compose in K8s manifests:

```bash
# Installa kompose
curl -L https://github.com/kubernetes/kompose/releases/download/v1.31.2/kompose-linux-amd64 -o kompose

# Converti
kompose convert -f docker-compose.yml
```

## 🐛 Troubleshooting

### Container non si avvia

```bash
# Visualizza logs dettagliati
docker-compose logs arima-api

# Verifica health status
docker ps -a | grep arima

# Rebuild forzato
docker-compose up -d --build --force-recreate
```

### Errore "port already in use"

```bash
# Trova processo su porta 8000
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000

# Modifica porta in docker-compose.yml
ports:
  - "8001:8000"  # Usa 8001 invece di 8000
```

### Performance lente

```bash
# Aumenta worker Uvicorn
environment:
  - API_WORKERS=8  # Numero core CPU

# Alloca più memoria
docker-compose up -d --memory="4g"
```

### Volumi non montati

```bash
# Crea directory se non esistono
mkdir -p outputs/models outputs/plots outputs/reports logs

# Verifica permessi
ls -la outputs/
```

## 📚 Esempi API Requests

### Training Modello

```bash
curl -X POST "http://localhost:8000/models/train/arima" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [100, 120, 132, 144, 155, 165, 175, 185, 195, 205],
    "order": [1, 1, 1],
    "model_name": "test_arima"
  }'
```

### Forecasting

```bash
curl -X POST "http://localhost:8000/models/test_arima/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "steps": 10,
    "confidence_level": 0.95
  }'
```

### List Modelli

```bash
curl http://localhost:8000/models
```

## 🔄 Update & Rollback

### Update Immagine

```bash
# Pull nuova versione
docker pull arima-forecaster-api:0.5.0

# Update servizio
docker-compose up -d arima-api

# Verifica versione
docker inspect arima-forecaster-api | grep -i version
```

### Rollback

```bash
# Torna alla versione precedente
docker tag arima-forecaster-api:0.4.0 arima-forecaster-api:latest
docker-compose up -d arima-api
```

## 📊 Performance Tuning

### Ottimizzazione Workers

```python
# Formula: workers = (2 x CPU_cores) + 1
# 4 core CPU = 9 workers ottimali
environment:
  - API_WORKERS=9
```

### Memory Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## 📞 Support

- **Issues**: https://github.com/tuonome/arima-forecaster/issues
- **Docs**: http://localhost:8000/docs
- **Email**: support@arima-forecaster.com

---

**🎉 Setup completato!** La tua FastAPI è ora containerizzata e pronta per il deployment.
