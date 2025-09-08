#!/usr/bin/env python3
"""
Demo Script - Nuove Funzionalita ARIMA Forecaster v0.4.0
Versione ASCII-safe per Windows

Dimostra Real-Time Streaming e Explainable AI implementati.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("[DEMO] ARIMA FORECASTER v0.4.0 - NUOVE FUNZIONALITA")
print("=" * 60)


def demo_explainable_ai():
    """Demo Explainable AI"""
    print("\n[AI] DEMO: EXPLAINABLE AI")
    print("-" * 30)
    
    try:
        # Import moduli explainability
        from arima_forecaster.explainability import (
            FeatureImportanceAnalyzer,
            AnomalyExplainer, 
            BusinessRulesEngine,
            BusinessContext,
            Rule,
            RuleType,
            RuleAction
        )
        
        print("[OK] Moduli Explainable AI caricati con successo")
        
        # 1. Feature Importance Analysis
        print("\n[1] Feature Importance Analysis")
        
        # Dati demo
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        X[:, 0] = X[:, 0] * 2 + 1  # Feature importante
        y = X[:, 0] * 3 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
        
        feature_names = ["lag_1", "lag_7", "trend", "seasonal"]
        
        analyzer = FeatureImportanceAnalyzer()
        results = analyzer.analyze_features(X, y, feature_names)
        
        top_features = results.get('top_features', {}).get('top_features_list', [])
        print(f"   Top 3 features: {top_features[:3]}")
        print(f"   Analisi completata: {len(results)} componenti")
        
        # 2. Anomaly Explainer
        print("\n[2] Anomaly Explainer")
        
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        normal_data = np.random.normal(1000, 50, 50)
        series = pd.Series(normal_data, index=dates)
        
        explainer = AnomalyExplainer()
        explainer.add_historical_data("demo_model", series)
        
        # Simula anomalia
        anomaly_value = 1600.0  # Valore anomalo
        explanation = explainer.explain_anomaly(
            model_id="demo_model",
            predicted_value=anomaly_value,
            anomaly_score=0.85
        )
        
        print(f"   Anomalia ID: {explanation.anomaly_id}")
        print(f"   Tipo: {explanation.anomaly_type.value}")
        print(f"   Severita: {explanation.severity.value}")
        print(f"   Confidenza: {explanation.confidence_level:.2f}")
        print(f"   Raccomandazioni: {len(explanation.recommended_actions)}")
        
        # 3. Business Rules Engine
        print("\n[3] Business Rules Engine")
        
        engine = BusinessRulesEngine()
        
        # Context di test
        context = BusinessContext(
            model_id="demo_product",
            product_id="PROD-001",
            forecast_date=datetime.now(),
            max_capacity=1000,
            is_weekend=False
        )
        
        # Test normale
        forecast_normal = 850.0
        final_value, results = engine.apply_rules(forecast_normal, context)
        print(f"   Forecast normale: {forecast_normal} -> {final_value}")
        print(f"   Regole applicate: {len([r for r in results if r.applied])}")
        
        # Test capacita superata
        forecast_high = 1200.0  
        final_value_high, results_high = engine.apply_rules(forecast_high, context)
        print(f"   Forecast alto: {forecast_high} -> {final_value_high}")
        print(f"   Modifiche per capacita: {final_value_high != forecast_high}")
        
        print("[OK] Explainable AI demo completato")
        
    except ImportError as e:
        print(f"[ERROR] Import Explainable AI: {e}")
        print("[INFO] Installare: pip install shap scikit-learn")
    except Exception as e:
        print(f"[ERROR] Explainable AI: {e}")


def demo_streaming():
    """Demo Real-Time Streaming"""
    print("\n[STREAM] DEMO: REAL-TIME STREAMING")
    print("-" * 30)
    
    try:
        # Import moduli streaming
        from arima_forecaster.streaming import (
            StreamingConfig,
            KafkaForecastProducer,
            ForecastMessage,
            WebSocketServer,
            WebSocketConfig,
            EventProcessor,
            ForecastEvent,
            EventType,
            EventPriority
        )
        
        print("[OK] Moduli Real-Time Streaming caricati con successo")
        
        # 1. Kafka Producer Demo
        print("\n[1] Kafka Producer")
        
        config = StreamingConfig(
            bootstrap_servers=["localhost:9092"],
            topic="demo-forecasts"
        )
        
        producer = KafkaForecastProducer(config)
        stats = producer.get_stats()
        print(f"   Kafka disponibile: {stats['is_connected']}")
        print(f"   Topic: {stats['topic']}")
        print(f"   Server: {stats['bootstrap_servers']}")
        
        # Crea messaggio demo
        forecast_msg = ForecastMessage(
            model_id="demo_model",
            timestamp=datetime.now(),
            predicted_value=1285.4,
            confidence_interval=[1180.5, 1390.3],
            forecast_horizon=1,
            model_type="ARIMA",
            metadata={"demo": True}
        )
        
        # Tenta invio (fallback locale se Kafka non disponibile)
        sent = producer.send_forecast(forecast_msg)
        print(f"   Messaggio inviato: {sent}")
        
        producer.close()
        
        # 2. WebSocket Server Demo (configurazione)
        print("\n[2] WebSocket Server")
        
        ws_config = WebSocketConfig(
            host="localhost",
            port=8765,
            max_connections=50
        )
        
        # Non avviamo server (richiede async event loop)
        print(f"   Host: {ws_config.host}:{ws_config.port}")
        print(f"   Max connessioni: {ws_config.max_connections}")
        print(f"   Configurato per real-time dashboard updates")
        
        # 3. Event Processor Demo
        print("\n[3] Event Processor")
        
        processor = EventProcessor(max_queue_size=100, worker_threads=2)
        
        # Crea eventi demo
        event1 = processor.create_event(
            event_type=EventType.FORECAST_GENERATED,
            model_id="demo_model",
            data={"predicted_value": 1285.4, "confidence": 0.95}
        )
        
        event2 = processor.create_event(
            event_type=EventType.ANOMALY_DETECTED,
            model_id="demo_model", 
            data={"anomaly_score": 0.85, "severity": "HIGH"},
            priority=EventPriority.HIGH
        )
        
        # Sottometti eventi
        submitted1 = processor.submit_event(event1)
        submitted2 = processor.submit_event(event2)
        
        print(f"   Eventi creati: 2")
        print(f"   Eventi sottomessi: {submitted1 + submitted2}")
        
        stats = processor.get_stats()
        print(f"   Regole attive: {stats['active_rules']}")
        print(f"   Azioni registrate: {stats['registered_actions']}")
        
        print("[OK] Real-Time Streaming demo completato")
        
    except ImportError as e:
        print(f"[ERROR] Import Streaming: {e}")
        print("[INFO] Installare: pip install kafka-python websockets redis")
    except Exception as e:
        print(f"[ERROR] Streaming: {e}")


def demo_integration_example():
    """Demo integrazione completa"""
    print("\n[INTEGRATION] DEMO: INTEGRAZIONE COMPLETA")
    print("-" * 30)
    
    try:
        # Import moduli base
        from arima_forecaster import ARIMAForecaster
        
        print("[OK] Preparazione demo integrazione")
        
        # Crea dati demo
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        trend = np.linspace(1000, 1200, 100)
        seasonal = 50 * np.sin(2 * np.pi * np.arange(100) / 7)  # Pattern settimanale
        noise = np.random.normal(0, 30, 100)
        values = trend + seasonal + noise
        
        series = pd.Series(values, index=dates)
        print(f"   Serie temporale creata: {len(series)} giorni")
        
        # 1. Addestra modello
        print("\n[1] Training Modello ARIMA")
        model = ARIMAForecaster(order=(2, 1, 1))
        model.fit(series)
        print("   [OK] Modello addestrato")
        
        # 2. Genera forecast
        forecast_result = model.predict(steps=7)
        forecast_values = forecast_result if isinstance(forecast_result, np.ndarray) else [forecast_result]
        avg_forecast = np.mean(forecast_values)
        print(f"   [FORECAST] Media 7 giorni: {avg_forecast:.2f}")
        
        # 3. Business Rules (se disponibile)
        try:
            from arima_forecaster.explainability import BusinessRulesEngine, BusinessContext
            
            engine = BusinessRulesEngine()
            context = BusinessContext(
                model_id="integration_demo",
                forecast_date=datetime.now(),
                max_capacity=1500,
                historical_average=np.mean(series.tail(30))
            )
            
            final_value, rules_results = engine.apply_rules(avg_forecast, context)
            print(f"   [RULES] Dopo regole business: {final_value:.2f}")
            print(f"   [RULES] Regole applicate: {len([r for r in rules_results if r.applied])}")
            
        except ImportError:
            print("   [WARN] Business Rules non disponibili")
        
        # 4. Streaming (se disponibile)
        try:
            from arima_forecaster.streaming import StreamingConfig, KafkaForecastProducer, ForecastMessage
            
            producer = KafkaForecastProducer(StreamingConfig())
            
            forecast_msg = ForecastMessage(
                model_id="integration_demo",
                timestamp=datetime.now(),
                predicted_value=final_value if 'final_value' in locals() else avg_forecast,
                confidence_interval=[avg_forecast * 0.9, avg_forecast * 1.1],
                forecast_horizon=7,
                model_type="ARIMA"
            )
            
            sent = producer.send_forecast(forecast_msg)
            status = "[OK]" if sent else "[WARN] (fallback locale)"
            print(f"   [STREAM] Streaming forecast: {status}")
            producer.close()
            
        except ImportError:
            print("   [WARN] Real-Time Streaming non disponibile")
        
        print("[OK] Integrazione completa demo completato")
        
    except Exception as e:
        print(f"[ERROR] Integrazione: {e}")


def main():
    """Funzione principale demo"""
    print("[TARGET] DIMOSTRAZIONI FUNZIONALITA ARIMA FORECASTER v0.4.0")
    print("[DATE]", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Demo 1: Explainable AI
    demo_explainable_ai()
    
    # Demo 2: Real-Time Streaming  
    demo_streaming()
    
    # Demo 3: Integrazione
    demo_integration_example()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] DEMO COMPLETATO!")
    print()
    print("[TODO] PROSSIMI STEP:")
    print("• Installare dipendenze opzionali per funzionalita complete")
    print("• Configurare Kafka per real-time streaming")
    print("• Integrare con sistema esistente usando nuove API")
    print()
    print("[INFO] Per supporto: https://github.com/tuonome/arima-forecaster")
    print("=" * 60)


if __name__ == "__main__":
    main()