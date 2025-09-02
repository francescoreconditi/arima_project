# 🔧 Demand Sensing - Guida Tecnica Avanzata

Documentazione tecnica approfondita per sviluppatori e data scientist che implementano demand sensing in produzione.

## 📋 Indice

1. [Architettura Interna](#-architettura-interna)
2. [Algoritmi di Combinazione](#-algoritmi-di-combinazione)
3. [Gestione API Esterne](#-gestione-api-esterne)
4. [Performance Optimization](#-performance-optimization)
5. [Deployment Produzione](#-deployment-produzione)
6. [Monitoraggio e Alerting](#-monitoraggio-e-alerting)
7. [Estensibilità](#-estensibilità)
8. [Machine Learning Avanzato](#-machine-learning-avanzato)

## 🏗 Architettura Interna

### Design Pattern

Il sistema utilizza il **Strategy Pattern** per la combinazione dei fattori e il **Factory Pattern** per la creazione dinamica dei sensori.

```python
# Strategy Pattern per combinazione fattori
class CombinationStrategy:
    def combine(self, factors: List[ExternalFactor]) -> float:
        raise NotImplementedError

class WeightedAverageStrategy(CombinationStrategy):
    def combine(self, factors: List[ExternalFactor]) -> float:
        weights = [f.confidence for f in factors]
        impacts = [f.impact for f in factors]
        return np.average(impacts, weights=weights)

# Factory Pattern per sensori
class SensorFactory:
    @staticmethod
    def create_sensor(sensor_type: str, **kwargs):
        if sensor_type == "weather":
            return WeatherIntegration(**kwargs)
        elif sensor_type == "trends":
            return GoogleTrendsIntegration(**kwargs)
        # ...
```

### Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   External      │    │    Raw Data      │    │   Processed     │
│   APIs          │───▶│   Collection     │───▶│   Factors       │
│   (Weather,     │    │   (with cache)   │    │   (normalized)  │
│   Trends, etc.) │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Learning      │    │   Ensemble       │    │   Factor        │
│   Engine        │◀───│   Combination    │◀───│   Validation    │
│   (feedback)    │    │   Engine         │    │   & Filtering   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│   Model         │    │   Adjusted       │
│   Updates       │    │   Forecast       │
│   (weights)     │    │   Output         │
└─────────────────┘    └──────────────────┘
```

### Modulo Core: DemandSensor

```python
class DemandSensor:
    def __init__(self, config: SensingConfig):
        self.config = config
        self.factors: List[ExternalFactor] = []
        self.impact_history: List[FactorImpact] = []
        self.learning_cache: Dict[str, float] = {}
        
        # Strategy pattern per combinazione
        self.combination_strategy = self._get_combination_strategy()
    
    def _get_combination_strategy(self) -> CombinationStrategy:
        strategies = {
            'weighted_average': WeightedAverageStrategy(),
            'multiplicative': MultiplicativeStrategy(),
            'hybrid': HybridStrategy(),
            'ml': MachineLearningStrategy()
        }
        return strategies[self.config.strategy.value]
    
    def add_factor(self, factor: ExternalFactor) -> None:
        # Validazione fattore
        if not self._validate_factor(factor):
            raise ValueError(f"Invalid factor: {factor}")
        
        # Normalizzazione
        normalized_factor = self._normalize_factor(factor)
        
        # Aggiungi con timestamp
        normalized_factor.timestamp = datetime.now()
        self.factors.append(normalized_factor)
    
    def _validate_factor(self, factor: ExternalFactor) -> bool:
        \"\"\"Valida fattore prima dell'inserimento.\"\"\"
        return (
            -1 <= factor.impact <= 1 and
            0 <= factor.confidence <= 1 and
            factor.timestamp is not None
        )
    
    def _normalize_factor(self, factor: ExternalFactor) -> ExternalFactor:
        \"\"\"Normalizza fattore per consistenza.\"\"\"
        # Z-score normalization per valore
        if factor.metadata.get('mean') and factor.metadata.get('std'):
            mean = factor.metadata['mean']
            std = factor.metadata['std']
            normalized_value = (factor.value - mean) / std
        else:
            normalized_value = factor.value
        
        return ExternalFactor(
            name=factor.name,
            type=factor.type,
            value=normalized_value,
            impact=factor.impact,
            confidence=factor.confidence,
            timestamp=factor.timestamp,
            metadata={**factor.metadata, 'normalized': True}
        )
```

## 🧮 Algoritmi di Combinazione

### 1. Weighted Average (Default)

Combina fattori usando confidenza come peso:

```python
def weighted_average_combination(factors: List[ExternalFactor]) -> float:
    if not factors:
        return 0.0
    
    total_weight = sum(f.confidence for f in factors)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(f.impact * f.confidence for f in factors)
    return weighted_sum / total_weight
```

### 2. Multiplicative Strategy

Applica fattori moltiplicativi in sequenza:

```python
def multiplicative_combination(factors: List[ExternalFactor]) -> float:
    adjustment_factor = 1.0
    
    for factor in factors:
        # Converti impact in moltiplicatore
        multiplier = 1.0 + (factor.impact * factor.confidence)
        adjustment_factor *= multiplier
    
    # Ritorna variazione percentuale
    return adjustment_factor - 1.0
```

### 3. Hybrid Strategy

Combina additivo e moltiplicativo:

```python
def hybrid_combination(factors: List[ExternalFactor]) -> float:
    # Separa fattori per tipo
    additive_factors = [f for f in factors if f.type in ADDITIVE_TYPES]
    multiplicative_factors = [f for f in factors if f.type in MULTIPLICATIVE_TYPES]
    
    # Calcola contributi separati
    additive_impact = weighted_average_combination(additive_factors)
    multiplicative_impact = multiplicative_combination(multiplicative_factors)
    
    # Combina con pesi
    return additive_impact * 0.6 + multiplicative_impact * 0.4
```

### 4. Machine Learning Strategy

Usa modello ML per ottimizzare combinazione:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class MLCombinationStrategy:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, factor_history: List[List[ExternalFactor]], actual_adjustments: List[float]):
        \"\"\"Addestra modello su storia fattori -> aggiustamenti.\"\"\"
        # Converti fattori in feature matrix
        X = self._factors_to_features(factor_history)
        y = np.array(actual_adjustments)
        
        # Standardizza features
        X_scaled = self.scaler.fit_transform(X)
        
        # Addestra
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def combine(self, factors: List[ExternalFactor]) -> float:
        if not self.is_trained:
            # Fallback a weighted average
            return weighted_average_combination(factors)
        
        # Converti a features
        X = self._factors_to_features([factors])
        X_scaled = self.scaler.transform(X)
        
        # Predici aggiustamento
        prediction = self.model.predict(X_scaled)[0]
        
        # Limita output
        return np.clip(prediction, -0.5, 0.5)
    
    def _factors_to_features(self, factor_lists: List[List[ExternalFactor]]) -> np.ndarray:
        \"\"\"Converte liste di fattori in matrice features.\"\"\"
        features = []
        
        for factors in factor_lists:
            # Feature per ogni tipo di fattore
            weather_impact = sum(f.impact for f in factors if f.type == FactorType.WEATHER)
            trends_impact = sum(f.impact for f in factors if f.type == FactorType.TRENDS)
            social_impact = sum(f.impact for f in factors if f.type == FactorType.SOCIAL)
            economic_impact = sum(f.impact for f in factors if f.type == FactorType.ECONOMIC)
            calendar_impact = sum(f.impact for f in factors if f.type == FactorType.CALENDAR)
            
            # Confidenze medie
            avg_confidence = np.mean([f.confidence for f in factors]) if factors else 0
            
            # Numero fattori per tipo
            n_weather = len([f for f in factors if f.type == FactorType.WEATHER])
            n_trends = len([f for f in factors if f.type == FactorType.TRENDS])
            n_social = len([f for f in factors if f.type == FactorType.SOCIAL])
            n_economic = len([f for f in factors if f.type == FactorType.ECONOMIC])
            n_calendar = len([f for f in factors if f.type == FactorType.CALENDAR])
            
            features.append([
                weather_impact, trends_impact, social_impact, economic_impact, calendar_impact,
                avg_confidence,
                n_weather, n_trends, n_social, n_economic, n_calendar
            ])
        
        return np.array(features)
```

## 🌐 Gestione API Esterne

### Rate Limiting e Retry Logic

```python
import time
from functools import wraps
from typing import Callable

def rate_limit(calls_per_minute: int = 60):
    \"\"\"Decorator per rate limiting API calls.\"\"\"
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    \"\"\"Decorator per retry con exponential backoff.\"\"\"
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            
            return None
        
        return wrapper
    return decorator

# Uso nei moduli
class WeatherIntegration:
    @rate_limit(calls_per_minute=60)
    @retry_with_backoff(max_retries=3)
    def fetch_forecast(self, days_ahead: int) -> List[WeatherCondition]:
        # Implementazione API call
        pass
```

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normale funzionamento
    OPEN = "open"          # Circuit aperto, usa fallback
    HALF_OPEN = "half_open"  # Test se API è tornata up

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        \"\"\"Esegue chiamata attraverso circuit breaker.\"\"\"
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        \"\"\"Verifica se provare a resettare circuit.\"\"\"
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        \"\"\"Reset dopo successo.\"\"\"
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        \"\"\"Gestisce fallimento.\"\"\"
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Integrazione nei sensori
class WeatherIntegration:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300  # 5 minuti
        )
    
    def fetch_forecast(self, days_ahead: int):
        try:
            return self.circuit_breaker.call(self._api_call, days_ahead)
        except Exception:
            logger.warning("Weather API failed, using demo data")
            return self._generate_demo_forecast(days_ahead)
```

### Cache Management Avanzato

```python
import redis
import pickle
from typing import Optional, Any
import hashlib

class RedisCache:
    \"\"\"Cache Redis per API calls.\"\"\"
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 ora
    
    def get(self, key: str) -> Optional[Any]:
        \"\"\"Recupera valore da cache.\"\"\"
        try:
            data = self.redis_client.get(key)
            return pickle.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        \"\"\"Salva valore in cache.\"\"\"
        try:
            ttl = ttl or self.default_ttl
            data = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def generate_key(self, *args, **kwargs) -> str:
        \"\"\"Genera chiave cache da parametri.\"\"\"
        key_string = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_string.encode()).hexdigest()

def cached_api_call(cache: RedisCache, ttl: int = 3600):
    \"\"\"Decorator per cache API calls.\"\"\"
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Genera cache key
            cache_key = f"{func.__name__}_{cache.generate_key(*args, **kwargs)}"
            
            # Prova cache
            result = cache.get(cache_key)
            if result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return result
            
            # Esegui funzione
            result = func(*args, **kwargs)
            
            # Salva in cache
            cache.set(cache_key, result, ttl)
            logger.info(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        return wrapper
    return decorator

# Uso
cache = RedisCache()

class GoogleTrendsIntegration:
    @cached_api_call(cache, ttl=1800)  # Cache 30 minuti
    def fetch_trends(self, timeframe: str, keywords: List[str]):
        # Implementazione API call
        pass
```

## 🚀 Performance Optimization

### Parallel Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

class ParallelSensorOrchestrator:
    \"\"\"Orchestratore per esecuzione parallela sensori.\"\"\"
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def collect_all_factors_parallel(
        self,
        sensors: Dict[str, Any],
        forecast_horizon: int
    ) -> Dict[str, List[ExternalFactor]]:
        \"\"\"Raccoglie fattori da tutti i sensori in parallelo.\"\"\"
        
        # Prepara tasks
        futures = {}
        for sensor_name, sensor in sensors.items():
            if sensor is not None:
                future = self.executor.submit(
                    self._safe_sensor_call,
                    sensor_name,
                    sensor,
                    forecast_horizon
                )
                futures[future] = sensor_name
        
        # Raccogli risultati
        results = {}
        for future in as_completed(futures, timeout=30):
            sensor_name = futures[future]
            try:
                factors = future.result()
                results[sensor_name] = factors
            except Exception as e:
                logger.warning(f"Sensor {sensor_name} failed: {e}")
                results[sensor_name] = []
        
        return results
    
    def _safe_sensor_call(
        self,
        sensor_name: str,
        sensor: Any,
        forecast_horizon: int
    ) -> List[ExternalFactor]:
        \"\"\"Chiamata sicura a sensore con timeout.\"\"\"
        try:
            if sensor_name == 'weather':
                conditions = sensor.fetch_forecast(forecast_horizon)
                return sensor.calculate_weather_impact(conditions)
            elif sensor_name == 'trends':
                trend_data = sensor.fetch_trends()
                return sensor.calculate_trend_impact(trend_data, forecast_horizon)
            # ... altri sensori
            
        except Exception as e:
            logger.error(f"Error in {sensor_name} sensor: {e}")
            return []

# Async versione per I/O bound operations
class AsyncSensorOrchestrator:
    async def collect_factors_async(self, sensors: Dict[str, Any]) -> Dict[str, List[ExternalFactor]]:
        \"\"\"Versione asincrona per I/O intensive operations.\"\"\"
        tasks = []
        
        for sensor_name, sensor in sensors.items():
            if sensor is not None:
                task = asyncio.create_task(
                    self._async_sensor_call(sensor_name, sensor)
                )
                tasks.append((sensor_name, task))
        
        results = {}
        for sensor_name, task in tasks:
            try:
                factors = await asyncio.wait_for(task, timeout=20.0)
                results[sensor_name] = factors
            except asyncio.TimeoutError:
                logger.warning(f"Sensor {sensor_name} timed out")
                results[sensor_name] = []
            except Exception as e:
                logger.error(f"Sensor {sensor_name} error: {e}")
                results[sensor_name] = []
        
        return results
```

### Memory Optimization

```python
import gc
from typing import Generator
import pandas as pd

class MemoryEfficientProcessor:
    \"\"\"Processore ottimizzato per memoria.\"\"\"
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_large_forecast_batch(
        self,
        forecasts: List[pd.Series],
        sensor: EnsembleDemandSensor
    ) -> Generator[SensingResult, None, None]:
        \"\"\"Processa forecasts in batch per ottimizzare memoria.\"\"\"
        
        for i in range(0, len(forecasts), self.batch_size):
            batch = forecasts[i:i + self.batch_size]
            
            # Processa batch
            batch_results = []
            for forecast in batch:
                result = sensor.sense(forecast)
                batch_results.append(result)
            
            # Yield risultati
            for result in batch_results:
                yield result
            
            # Cleanup memoria
            del batch_results
            gc.collect()
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Ottimizza memoria DataFrame.\"\"\"
        for col in df.columns:
            if df[col].dtype == 'object':
                # Prova category per stringhe ripetitive
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            
            elif df[col].dtype == 'float64':
                # Riduci precisione se possibile
                if df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('float32')
            
            elif df[col].dtype == 'int64':
                # Ottimizza interi
                if df[col].min() >= 0:
                    if df[col].max() < 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
        
        return df
```

## 🏭 Deployment Produzione

### Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/
COPY config/ ./config/

# Environment variables
ENV PYTHONPATH=/app/src
ENV DEMAND_SENSING_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import src.arima_forecaster.demand_sensing; print('OK')"

CMD ["python", "-m", "src.arima_forecaster.demand_sensing.server"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demand-sensing-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: demand-sensing
  template:
    metadata:
      labels:
        app: demand-sensing
    spec:
      containers:
      - name: demand-sensing
        image: demand-sensing:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: OPENWEATHER_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openweather
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: demand-sensing-service
spec:
  selector:
    app: demand-sensing
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### Configuration Management

```python
# config/production.py
import os
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ProductionConfig:
    # Redis
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    redis_password: Optional[str] = os.getenv('REDIS_PASSWORD')
    
    # External APIs
    openweather_api_key: Optional[str] = os.getenv('OPENWEATHER_API_KEY')
    twitter_bearer_token: Optional[str] = os.getenv('TWITTER_BEARER_TOKEN')
    
    # Performance
    max_workers: int = int(os.getenv('MAX_WORKERS', '4'))
    api_timeout: int = int(os.getenv('API_TIMEOUT', '30'))
    cache_ttl: int = int(os.getenv('CACHE_TTL', '3600'))
    
    # Circuit breaker
    failure_threshold: int = int(os.getenv('FAILURE_THRESHOLD', '3'))
    recovery_timeout: int = int(os.getenv('RECOVERY_TIMEOUT', '300'))
    
    # Monitoring
    enable_metrics: bool = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    metrics_port: int = int(os.getenv('METRICS_PORT', '9090'))
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_format: str = os.getenv('LOG_FORMAT', 'json')

# Factory per configurazione environment-specific
class ConfigFactory:
    @staticmethod
    def get_config(env: str = None):
        env = env or os.getenv('DEMAND_SENSING_ENV', 'development')
        
        if env == 'production':
            return ProductionConfig()
        elif env == 'staging':
            return StagingConfig()
        else:
            return DevelopmentConfig()
```

## 📊 Monitoraggio e Alerting

### Metriche Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metriche custom
API_CALLS_TOTAL = Counter(
    'demand_sensing_api_calls_total',
    'Total API calls made',
    ['source', 'status']
)

API_RESPONSE_TIME = Histogram(
    'demand_sensing_api_response_seconds',
    'API response time',
    ['source']
)

ACTIVE_SENSORS = Gauge(
    'demand_sensing_active_sensors',
    'Number of active sensors'
)

FORECAST_ACCURACY = Gauge(
    'demand_sensing_forecast_accuracy',
    'Forecast accuracy (1 - MAPE)',
    ['category']
)

class MetricsCollector:
    \"\"\"Collector per metriche demand sensing.\"\"\"
    
    def __init__(self):
        self.start_time = time.time()
    
    def record_api_call(self, source: str, duration: float, success: bool):
        \"\"\"Registra chiamata API.\"\"\"
        status = 'success' if success else 'error'
        API_CALLS_TOTAL.labels(source=source, status=status).inc()
        API_RESPONSE_TIME.labels(source=source).observe(duration)
    
    def update_sensor_count(self, count: int):
        \"\"\"Aggiorna conteggio sensori attivi.\"\"\"
        ACTIVE_SENSORS.set(count)
    
    def record_forecast_accuracy(self, category: str, accuracy: float):
        \"\"\"Registra accuracy forecast.\"\"\"
        FORECAST_ACCURACY.labels(category=category).set(accuracy)

# Decoratore per metriche automatiche
def with_metrics(source: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                metrics.record_api_call(source, duration, success)
        
        return wrapper
    return decorator

# Integrazione nei sensori
class WeatherIntegration:
    @with_metrics('weather_api')
    def fetch_forecast(self, days_ahead: int):
        # Implementazione
        pass

# Avvio server metriche
def start_metrics_server(port: int = 9090):
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")
```

### Health Checks

```python
from typing import Dict, Any
import requests
from datetime import datetime

class HealthChecker:
    \"\"\"Health checker per sistema demand sensing.\"\"\"
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.checks = {
            'redis': self._check_redis,
            'weather_api': self._check_weather_api,
            'memory': self._check_memory,
            'disk': self._check_disk
        }
    
    def check_all(self) -> Dict[str, Any]:
        \"\"\"Esegue tutti i check di salute.\"\"\"
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results['checks'][check_name] = check_result
                
                if not check_result['healthy']:
                    overall_healthy = False
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'healthy': False,
                    'error': str(e)
                }
                overall_healthy = False
        
        results['status'] = 'healthy' if overall_healthy else 'unhealthy'
        return results
    
    def _check_redis(self) -> Dict[str, Any]:
        \"\"\"Verifica connessione Redis.\"\"\"
        try:
            import redis
            client = redis.from_url(self.config.redis_url)
            client.ping()
            
            return {
                'healthy': True,
                'message': 'Redis connection OK'
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def _check_weather_api(self) -> Dict[str, Any]:
        \"\"\"Verifica API meteo.\"\"\"
        if not self.config.openweather_api_key:
            return {
                'healthy': True,
                'message': 'Weather API key not configured (demo mode)'
            }
        
        try:
            # Test call rapida
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': 'London',
                'appid': self.config.openweather_api_key
            }
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            return {
                'healthy': True,
                'message': 'Weather API accessible'
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        \"\"\"Verifica uso memoria.\"\"\"
        import psutil
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        healthy = memory_percent < 90
        
        return {
            'healthy': healthy,
            'memory_percent': memory_percent,
            'message': f'Memory usage: {memory_percent:.1f}%'
        }
    
    def _check_disk(self) -> Dict[str, Any]:
        \"\"\"Verifica spazio disco.\"\"\"
        import psutil
        
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        healthy = disk_percent < 85
        
        return {
            'healthy': healthy,
            'disk_percent': disk_percent,
            'message': f'Disk usage: {disk_percent:.1f}%'
        }

# FastAPI endpoint
from fastapi import FastAPI, HTTPException

app = FastAPI()
health_checker = HealthChecker(config)

@app.get("/health")
async def health_check():
    results = health_checker.check_all()
    
    if results['status'] != 'healthy':
        raise HTTPException(status_code=503, detail=results)
    
    return results

@app.get("/ready")
async def readiness_check():
    # Check più veloce per readiness
    try:
        # Verifica componenti critici
        health_checker._check_redis()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not ready", "error": str(e)})
```

### Alerting con Slack

```python
import requests
import json
from typing import Optional

class SlackAlerter:
    \"\"\"Sistema alerting via Slack.\"\"\"
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        additional_fields: Optional[Dict] = None
    ):
        \"\"\"Invia alert a Slack.\"\"\"
        
        color_map = {
            "good": "#36a64f",
            "warning": "#ff9900", 
            "danger": "#ff0000"
        }
        
        attachment = {
            "color": color_map.get(severity, "#ff9900"),
            "title": title,
            "text": message,
            "ts": int(time.time())
        }
        
        if additional_fields:
            attachment["fields"] = [
                {"title": k, "value": v, "short": True}
                for k, v in additional_fields.items()
            ]
        
        payload = {
            "channel": self.channel,
            "username": "DemandSensing Bot",
            "attachments": [attachment]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

# Integrazione con monitoraggio
class AlertManager:
    \"\"\"Manager per gestione alert.\"\"\"
    
    def __init__(self, config: ProductionConfig):
        self.slack_alerter = SlackAlerter(config.slack_webhook_url)
        self.thresholds = {
            'api_error_rate': 0.1,  # 10% error rate
            'response_time_p99': 30.0,  # 30s p99
            'memory_usage': 0.9,  # 90% memoria
            'forecast_accuracy_drop': 0.2  # 20% drop in accuracy
        }
    
    def check_and_alert(self, metrics: Dict[str, float]):
        \"\"\"Controlla metriche e invia alert se necessario.\"\"\"
        
        # API error rate
        if metrics.get('api_error_rate', 0) > self.thresholds['api_error_rate']:
            self.slack_alerter.send_alert(
                title="High API Error Rate",
                message=f"API error rate: {metrics['api_error_rate']:.1%}",
                severity="danger",
                additional_fields={
                    "Threshold": f"{self.thresholds['api_error_rate']:.1%}",
                    "Current": f"{metrics['api_error_rate']:.1%}"
                }
            )
        
        # Response time
        if metrics.get('response_time_p99', 0) > self.thresholds['response_time_p99']:
            self.slack_alerter.send_alert(
                title="High Response Time",
                message=f"P99 response time: {metrics['response_time_p99']:.1f}s",
                severity="warning"
            )
        
        # Forecast accuracy drop
        if metrics.get('forecast_accuracy_drop', 0) > self.thresholds['forecast_accuracy_drop']:
            self.slack_alerter.send_alert(
                title="Forecast Accuracy Drop",
                message=f"Accuracy dropped by {metrics['forecast_accuracy_drop']:.1%}",
                severity="danger"
            )
```

## 🔧 Estensibilità

### Plugin System

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class DemandSensingPlugin(ABC):
    \"\"\"Interfaccia base per plugin.\"\"\"
    
    @abstractmethod
    def get_name(self) -> str:
        \"\"\"Nome del plugin.\"\"\"
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        \"\"\"Versione del plugin.\"\"\"
        pass
    
    @abstractmethod
    def collect_factors(
        self, 
        config: Dict[str, Any],
        forecast_horizon: int
    ) -> List[ExternalFactor]:
        \"\"\"Raccoglie fattori esterni.\"\"\"
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        \"\"\"Valida configurazione plugin.\"\"\"
        pass

# Esempio plugin custom
class CompetitorPricePlugin(DemandSensingPlugin):
    \"\"\"Plugin per monitoraggio prezzi competitor.\"\"\"
    
    def get_name(self) -> str:
        return "competitor_pricing"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def collect_factors(
        self, 
        config: Dict[str, Any],
        forecast_horizon: int
    ) -> List[ExternalFactor]:
        \"\"\"Raccoglie fattori basati su prezzi competitor.\"\"\"
        
        competitors = config.get('competitors', [])
        our_price = config.get('our_price', 100)
        
        factors = []
        
        for competitor in competitors:
            # Simula raccolta prezzo competitor
            competitor_price = self._get_competitor_price(competitor)
            
            # Calcola impatto differenza prezzo
            price_diff = (our_price - competitor_price) / competitor_price
            
            # Se siamo più cari, impatto negativo
            impact = -price_diff * 0.3 if price_diff > 0 else -price_diff * 0.2
            
            factor = ExternalFactor(
                name=f"price_vs_{competitor}",
                type=FactorType.CUSTOM,
                value=competitor_price,
                impact=impact,
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={
                    'competitor': competitor,
                    'our_price': our_price,
                    'competitor_price': competitor_price,
                    'price_difference_pct': price_diff * 100
                }
            )
            
            factors.append(factor)
        
        return factors
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ['competitors', 'our_price']
        return all(field in config for field in required_fields)
    
    def _get_competitor_price(self, competitor: str) -> float:
        # Implementazione raccolta prezzo
        # Potrebbe usare web scraping, API, database, etc.
        pass

# Plugin Registry
class PluginRegistry:
    \"\"\"Registry per gestione plugin.\"\"\"
    
    def __init__(self):
        self.plugins: Dict[str, DemandSensingPlugin] = {}
    
    def register_plugin(self, plugin: DemandSensingPlugin):
        \"\"\"Registra nuovo plugin.\"\"\"
        name = plugin.get_name()
        
        if name in self.plugins:
            raise ValueError(f"Plugin {name} already registered")
        
        self.plugins[name] = plugin
        logger.info(f"Registered plugin: {name} v{plugin.get_version()}")
    
    def get_plugin(self, name: str) -> Optional[DemandSensingPlugin]:
        \"\"\"Recupera plugin per nome.\"\"\"
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, str]]:
        \"\"\"Lista tutti i plugin registrati.\"\"\"
        return [
            {
                'name': plugin.get_name(),
                'version': plugin.get_version()
            }
            for plugin in self.plugins.values()
        ]

# Integrazione nell'EnsembleDemandSensor
class ExtensibleEnsembleDemandSensor(EnsembleDemandSensor):
    \"\"\"Versione estendibile con support plugin.\"\"\"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plugin_registry = PluginRegistry()
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, plugin: DemandSensingPlugin, config: Dict[str, Any]):
        \"\"\"Registra e configura plugin.\"\"\"
        if not plugin.validate_config(config):
            raise ValueError(f"Invalid config for plugin {plugin.get_name()}")
        
        self.plugin_registry.register_plugin(plugin)
        self.plugin_configs[plugin.get_name()] = config
    
    def collect_plugin_factors(self, forecast_horizon: int) -> Dict[str, List[ExternalFactor]]:
        \"\"\"Raccoglie fattori da tutti i plugin.\"\"\"
        plugin_factors = {}
        
        for plugin_name, plugin in self.plugin_registry.plugins.items():
            try:
                config = self.plugin_configs.get(plugin_name, {})
                factors = plugin.collect_factors(config, forecast_horizon)
                plugin_factors[plugin_name] = factors
                
            except Exception as e:
                logger.error(f"Plugin {plugin_name} error: {e}")
                plugin_factors[plugin_name] = []
        
        return plugin_factors
    
    def sense_with_plugins(self, base_forecast, **kwargs):
        \"\"\"Sensing con supporto plugin.\"\"\"
        # Raccoglie fattori standard
        result, details = super().sense(base_forecast, return_details=True, **kwargs)
        
        # Raccoglie fattori plugin
        plugin_factors = self.collect_plugin_factors(len(base_forecast))
        
        if plugin_factors:
            # Combina con fattori esistenti
            all_factors = {**details['factors_by_source'], **plugin_factors}
            
            # Ricalcola con tutti i fattori
            combined_factors = self.combine_factors(all_factors)
            
            # Applica nuovamente
            self.base_sensor.clear_factors()
            for factor in combined_factors:
                self.base_sensor.add_factor(factor)
            
            result = self.base_sensor.sense(base_forecast, fetch_external=False)
            details['factors_by_source'] = all_factors
        
        return result, details
```

### Custom Factor Types

```python
# Estensione enum FactorType
class ExtendedFactorType(str, Enum):
    # Tipi base esistenti
    WEATHER = "weather"
    TRENDS = "trends"
    SOCIAL = "social"
    ECONOMIC = "economic"
    CALENDAR = "calendar"
    
    # Nuovi tipi custom
    COMPETITOR = "competitor"
    SUPPLY_CHAIN = "supply_chain"
    REGULATORY = "regulatory"
    TECHNOLOGY = "technology"
    DEMOGRAPHIC = "demographic"

# Factory per fattori custom
class FactorFactory:
    \"\"\"Factory per creare fattori di diversi tipi.\"\"\"
    
    @staticmethod
    def create_competitor_factor(
        competitor_name: str,
        price_difference: float,
        market_share_change: float
    ) -> ExternalFactor:
        \"\"\"Crea fattore competitivo.\"\"\"
        
        # Impatto basato su differenza prezzo e market share
        price_impact = -price_difference * 0.5  # Se competitor più economico
        share_impact = -market_share_change * 0.3  # Se competitor guadagna share
        
        total_impact = price_impact + share_impact
        
        return ExternalFactor(
            name=f"competitor_{competitor_name}",
            type=ExtendedFactorType.COMPETITOR,
            value=price_difference,
            impact=np.clip(total_impact, -0.5, 0.5),
            confidence=0.8,
            timestamp=datetime.now(),
            metadata={
                'competitor': competitor_name,
                'price_difference': price_difference,
                'market_share_change': market_share_change,
                'price_impact': price_impact,
                'share_impact': share_impact
            }
        )
    
    @staticmethod
    def create_supply_chain_factor(
        disruption_level: float,
        affected_regions: List[str],
        duration_days: int
    ) -> ExternalFactor:
        \"\"\"Crea fattore supply chain.\"\"\"
        
        # Impatto negativo per disruption
        impact = -disruption_level * 0.4 * min(1.0, duration_days / 30)
        
        return ExternalFactor(
            name="supply_chain_disruption",
            type=ExtendedFactorType.SUPPLY_CHAIN,
            value=disruption_level,
            impact=np.clip(impact, -0.6, 0),
            confidence=0.9,  # Alta confidenza per disruption concrete
            timestamp=datetime.now(),
            metadata={
                'disruption_level': disruption_level,
                'affected_regions': affected_regions,
                'duration_days': duration_days
            }
        )
```

## 🤖 Machine Learning Avanzato

### Reinforcement Learning per Peso Ottimizzazione

```python
import numpy as np
from typing import Dict, List, Tuple
import random
from collections import deque

class QLearningWeightOptimizer:
    \"\"\"Q-Learning per ottimizzazione automatica pesi fonti.\"\"\"
    
    def __init__(
        self,
        sources: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995
    ):
        self.sources = sources
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Stati = combinazioni discrete di pesi (binned)
        self.weight_bins = np.linspace(0.05, 0.5, 10)  # 10 bins per peso
        self.state_size = len(self.weight_bins) ** len(sources)
        
        # Azioni = piccoli aggiustamenti ai pesi
        self.actions = ['increase', 'decrease', 'maintain']
        self.action_size = len(self.actions) * len(sources)  # Per ogni fonte
        
        # Q-table
        self.q_table = np.zeros((self.state_size, self.action_size))
        
        # Storia per learning
        self.history = deque(maxlen=1000)
    
    def state_to_index(self, weights: Dict[str, float]) -> int:
        \"\"\"Converte pesi in indice stato discreto.\"\"\"
        indices = []
        
        for source in self.sources:
            weight = weights.get(source, 0.2)
            bin_idx = np.digitize(weight, self.weight_bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(self.weight_bins) - 1)
            indices.append(bin_idx)
        
        # Converte in singolo indice
        state_idx = 0
        for i, idx in enumerate(indices):
            state_idx += idx * (len(self.weight_bins) ** i)
        
        return min(state_idx, self.state_size - 1)
    
    def choose_action(self, state_idx: int) -> int:
        \"\"\"Sceglie azione usando epsilon-greedy.\"\"\"
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def apply_action(
        self,
        action: int,
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        \"\"\"Applica azione ai pesi correnti.\"\"\"
        new_weights = current_weights.copy()
        
        # Decodifica azione
        source_idx = action // len(self.actions)
        action_type = action % len(self.actions)
        
        if source_idx < len(self.sources):
            source = self.sources[source_idx]
            current_weight = new_weights.get(source, 0.2)
            
            if action_type == 0:  # increase
                new_weights[source] = min(0.5, current_weight + 0.05)
            elif action_type == 1:  # decrease
                new_weights[source] = max(0.05, current_weight - 0.05)
            # maintain = no change
        
        # Normalizza pesi
        total = sum(new_weights.values())
        for source in new_weights:
            new_weights[source] /= total
        
        return new_weights
    
    def calculate_reward(
        self,
        old_accuracy: float,
        new_accuracy: float,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        \"\"\"Calcola reward basato su miglioramento accuracy.\"\"\"
        
        # Reward principale = miglioramento accuracy
        accuracy_reward = (new_accuracy - old_accuracy) * 10
        
        # Penalty per pesi estremi (incoraggia diversificazione)
        weight_values = list(new_weights.values())
        balance_penalty = -np.std(weight_values) * 2
        
        # Bonus per stabilità (pochi cambi drastici)
        weight_changes = [
            abs(new_weights[s] - old_weights[s])
            for s in self.sources
        ]
        stability_bonus = -sum(weight_changes) * 5
        
        return accuracy_reward + balance_penalty + stability_bonus
    
    def update_q_table(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool = False
    ):
        \"\"\"Aggiorna Q-table con Q-learning.\"\"\"
        
        current_q = self.q_table[state, action]
        
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * next_max_q
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)
    
    def train_episode(
        self,
        sensor: EnsembleDemandSensor,
        train_data: pd.Series,
        validation_data: pd.Series,
        steps: int = 10
    ):
        \"\"\"Allena per un episodio.\"\"\"
        
        current_weights = sensor.config.source_weights.copy()
        current_state = self.state_to_index(current_weights)
        
        # Baseline accuracy
        baseline_result = sensor.sense(train_data)
        baseline_accuracy = self._calculate_accuracy(
            baseline_result.adjusted_forecast,
            validation_data
        )
        
        for step in range(steps):
            # Scegli e applica azione
            action = self.choose_action(current_state)
            new_weights = self.apply_action(action, current_weights)
            
            # Testa nuovi pesi
            sensor.config.source_weights = new_weights
            result = sensor.sense(train_data)
            new_accuracy = self._calculate_accuracy(
                result.adjusted_forecast,
                validation_data
            )
            
            # Calcola reward
            reward = self.calculate_reward(
                baseline_accuracy,
                new_accuracy,
                current_weights,
                new_weights
            )
            
            # Prossimo stato
            next_state = self.state_to_index(new_weights)
            
            # Aggiorna Q-table
            self.update_q_table(
                current_state,
                action,
                reward,
                next_state,
                done=(step == steps - 1)
            )
            
            # Salva storia
            self.history.append({
                'weights': new_weights.copy(),
                'accuracy': new_accuracy,
                'reward': reward
            })
            
            # Aggiorna stato
            current_state = next_state
            current_weights = new_weights
            baseline_accuracy = new_accuracy
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
    def _calculate_accuracy(
        self,
        predictions: pd.Series,
        actuals: pd.Series
    ) -> float:
        \"\"\"Calcola accuracy (1 - MAPE).\"\"\"
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        return max(0, 100 - mape) / 100
    
    def get_best_weights(self) -> Dict[str, float]:
        \"\"\"Ottieni miglior configurazione pesi dalla storia.\"\"\"
        if not self.history:
            return {source: 1.0 / len(self.sources) for source in self.sources}
        
        best_episode = max(self.history, key=lambda x: x['accuracy'])
        return best_episode['weights']

# Integrazione nel sistema principale
class AdaptiveEnsembleDemandSensor(EnsembleDemandSensor):
    \"\"\"Versione adattiva con RL per ottimizzazione pesi.\"\"\"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sources = list(self.config.source_weights.keys())
        self.rl_optimizer = QLearningWeightOptimizer(self.sources)
        
        self.training_data = deque(maxlen=100)  # Ultimi 100 forecasts
        self.performance_history = deque(maxlen=50)
    
    def adaptive_learn(
        self,
        train_series: pd.Series,
        validation_series: pd.Series,
        episodes: int = 20
    ):
        \"\"\"Apprendimento adattivo dei pesi.\"\"\"
        
        logger.info(f"Starting adaptive learning for {episodes} episodes...")
        
        initial_weights = self.config.source_weights.copy()
        
        for episode in range(episodes):
            self.rl_optimizer.train_episode(
                sensor=self,
                train_data=train_series,
                validation_data=validation_series,
                steps=5
            )
            
            if episode % 5 == 0:
                # Log progresso
                best_weights = self.rl_optimizer.get_best_weights()
                logger.info(f"Episode {episode}: Best weights = {best_weights}")
        
        # Applica migliori pesi trovati
        best_weights = self.rl_optimizer.get_best_weights()
        self.config.source_weights = best_weights
        
        logger.info(f"Adaptive learning completed. New weights: {best_weights}")
        return best_weights
```

---

Questa guida tecnica copre tutti gli aspetti avanzati del sistema demand sensing per implementazioni production-grade. Per ulteriori dettagli su specifici componenti, consulta il codice sorgente o apri un issue su GitHub.