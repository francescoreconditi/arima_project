"""
Esempi per documentazione OpenAPI.

Definisce esempi completi per request e response dei principali endpoint API.
"""

# Esempi dati di input
TIMESERIES_EXAMPLES = {
    "esempio_base": {
        "summary": "Serie temporale giornaliera semplice",
        "description": "Dati di vendite giornaliere per 30 giorni",
        "value": {
            "timestamps": [
                "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10",
                "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15",
                "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20",
                "2024-01-21", "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25",
                "2024-01-26", "2024-01-27", "2024-01-28", "2024-01-29", "2024-01-30"
            ],
            "values": [
                1250.5, 1300.2, 1180.7, 1400.1, 1350.8, 1200.3, 950.6, 
                1100.9, 1250.4, 1320.7, 1280.5, 1450.2, 1380.1, 1150.8,
                1050.3, 1200.7, 1300.1, 1420.5, 1380.9, 1250.4, 1100.2,
                1350.8, 1400.3, 1280.7, 1200.1, 1150.5, 1050.9, 1180.4,
                1250.7, 1320.1
            ]
        }
    },
    "esempio_stagionale": {
        "summary": "Serie temporale con pattern stagionale",
        "description": "Dati mensili con chiara stagionalità (12 mesi)",
        "value": {
            "timestamps": [
                "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01",
                "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01",
                "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"
            ],
            "values": [
                800, 750, 900, 1100, 1300, 1500, 1600, 1550, 1200, 1000, 850, 950,
                820, 780, 920, 1120, 1350, 1520
            ]
        }
    }
}

MULTIVARIATE_EXAMPLES = {
    "esempio_var": {
        "summary": "Dati multivariati per modello VAR",
        "description": "Serie temporali di vendite, marketing e temperatura correlate",
        "value": {
            "timestamps": [
                "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"
            ],
            "data": {
                "vendite": [1000, 1100, 950, 1200, 1050, 980, 1150, 1080, 1220, 1180],
                "marketing": [500, 600, 450, 700, 550, 480, 650, 580, 720, 680],
                "temperatura": [22.5, 24.1, 21.8, 25.3, 23.7, 20.9, 26.1, 24.8, 27.2, 25.9]
            }
        }
    }
}

# Esempi parametri training
ARIMA_TRAINING_EXAMPLES = {
    "arima_automatico": {
        "summary": "Training ARIMA con selezione automatica",
        "description": "Selezione automatica dei parametri ottimali",
        "value": {
            "model_id": "vendite_arima_auto",
            "data": TIMESERIES_EXAMPLES["esempio_base"]["value"],
            "model_params": {
                "order": None,
                "seasonal_order": None,
                "auto_select": True
            }
        }
    },
    "arima_manuale": {
        "summary": "Training ARIMA con parametri manuali",
        "description": "Specifica esatta dei parametri ARIMA(2,1,1)",
        "value": {
            "model_id": "vendite_arima_manual",
            "data": TIMESERIES_EXAMPLES["esempio_base"]["value"],
            "model_params": {
                "order": [2, 1, 1],
                "seasonal_order": None,
                "auto_select": False
            }
        }
    }
}

SARIMA_TRAINING_EXAMPLES = {
    "sarima_stagionale": {
        "summary": "Training SARIMA con stagionalità mensile",
        "description": "Modello SARIMA(1,1,1)(1,1,1,12) per dati mensili",
        "value": {
            "model_id": "vendite_sarima_monthly",
            "data": TIMESERIES_EXAMPLES["esempio_stagionale"]["value"],
            "model_params": {
                "order": [1, 1, 1],
                "seasonal_order": [1, 1, 1, 12],
                "auto_select": False
            }
        }
    }
}

# Esempi response training
TRAINING_RESPONSE_EXAMPLES = {
    "training_success": {
        "summary": "Training completato con successo",
        "description": "Response tipica per training ARIMA riuscito",
        "value": {
            "status": "success",
            "message": "Modello addestrato con successo",
            "model_id": "vendite_arima_auto",
            "model_type": "ARIMA",
            "parameters": {
                "order": [2, 1, 1],
                "seasonal_order": None
            },
            "performance_metrics": {
                "aic": 245.67,
                "bic": 255.32,
                "mae": 45.23,
                "rmse": 58.91,
                "mape": 3.87
            },
            "training_time": 2.34,
            "data_points": 30,
            "created_at": "2024-01-15T14:30:25Z"
        }
    },
    "training_error": {
        "summary": "Errore durante il training",
        "description": "Response in caso di errore nei dati o parametri",
        "value": {
            "status": "error",
            "message": "Training fallito: dati insufficienti",
            "error_code": "INSUFFICIENT_DATA",
            "details": "Sono necessari almeno 50 punti dati per SARIMA con stagionalità 12",
            "model_id": "vendite_sarima_monthly",
            "timestamp": "2024-01-15T14:30:25Z"
        }
    }
}

# Esempi forecast
FORECAST_REQUEST_EXAMPLES = {
    "forecast_base": {
        "summary": "Previsione standard 30 giorni",
        "description": "Forecast con intervalli di confidenza 95%",
        "value": {
            "model_id": "vendite_arima_auto",
            "steps": 30,
            "confidence_level": 0.95,
            "exog_variables": None
        }
    },
    "forecast_con_esogene": {
        "summary": "Forecast SARIMAX con variabili esogene",
        "description": "Previsione includendo dati meteo futuri",
        "value": {
            "model_id": "vendite_sarimax_weather",
            "steps": 15,
            "confidence_level": 0.90,
            "exog_variables": {
                "temperatura": [25.5, 26.1, 24.8, 23.9, 22.7, 24.3, 25.9, 27.2, 26.5, 25.1, 23.8, 22.9, 24.6, 25.7, 26.8],
                "pioggia": [0, 2.5, 0, 5.1, 0, 0, 3.2, 0, 1.8, 0, 0, 4.7, 0, 0, 2.1]
            }
        }
    }
}

FORECAST_RESPONSE_EXAMPLES = {
    "forecast_success": {
        "summary": "Previsione completata",
        "description": "Response con forecast e intervalli di confidenza",
        "value": {
            "status": "success",
            "model_id": "vendite_arima_auto",
            "forecast_horizon": 30,
            "confidence_level": 0.95,
            "predictions": {
                "timestamps": [
                    "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04",
                    "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09"
                ],
                "values": [1285.4, 1298.7, 1275.2, 1310.5, 1295.8, 1270.3, 1305.1, 1290.6, 1315.9, 1301.2],
                "lower_ci": [1180.5, 1190.2, 1165.8, 1200.1, 1185.4, 1160.9, 1195.2, 1180.7, 1205.0, 1190.3],
                "upper_ci": [1390.3, 1407.2, 1384.6, 1420.9, 1406.2, 1379.7, 1415.0, 1400.5, 1426.8, 1412.1]
            },
            "model_diagnostics": {
                "in_sample_mae": 45.23,
                "in_sample_rmse": 58.91,
                "residuals_normality_pvalue": 0.15,
                "ljung_box_pvalue": 0.23
            },
            "generated_at": "2024-01-15T14:35:12Z"
        }
    }
}

# Esempi inventory management
INVENTORY_EXAMPLES = {
    "classificazione_movimento": {
        "summary": "Classificazione movimento prodotti",
        "description": "Analisi ABC/XYZ per ottimizzazione inventory",
        "value": {
            "sales_data": [45, 52, 38, 61, 47, 55, 43, 58, 41, 49, 53, 46, 59, 42, 50],
            "product_info": {
                "sku": "PRD-001",
                "category": "Electronics",
                "unit_cost": 25.50,
                "holding_cost_rate": 0.20
            },
            "analysis_period_days": 90
        }
    },
    "ottimizzazione_scorte": {
        "summary": "Ottimizzazione parametri scorte",
        "description": "Calcolo reorder point e quantity ottimali",
        "value": {
            "product_id": "PRD-001",
            "demand_forecast": [48, 52, 45, 58, 51, 47, 55, 43, 59, 46],
            "lead_time_days": 7,
            "service_level": 0.95,
            "holding_cost_rate": 0.20,
            "ordering_cost": 50.0,
            "unit_cost": 25.50
        }
    }
}

# Esempi demand sensing
DEMAND_SENSING_EXAMPLES = {
    "weather_impact": {
        "summary": "Analisi impatto meteo",
        "description": "Correlazione tra condizioni meteo e domanda",
        "value": {
            "location": "Milano",
            "product_category": "Gelati",
            "weather_factors": ["temperatura", "precipitazioni", "umidita"],
            "date_range": {
                "start_date": "2024-06-01",
                "end_date": "2024-08-31"
            },
            "weather_sensitivity": {
                "temperatura_ottimale": 28.0,
                "rain_negative_impact": True,
                "humidity_threshold": 70
            }
        }
    },
    "trends_analysis": {
        "summary": "Analisi Google Trends",
        "description": "Correlazione tra ricerche online e domanda",
        "value": {
            "keywords": ["gelato", "ice cream", "gelateria"],
            "geo": "IT",
            "product_id": "GELATO-001",
            "timeframe": "2024-06-01 2024-08-31",
            "correlation_threshold": 0.3
        }
    }
}

# Esempi AutoML
AUTOML_EXAMPLES = {
    "hyperparameter_tuning": {
        "summary": "Ottimizzazione automatica iperparametri",
        "description": "Tuning Bayesian optimization con Optuna",
        "value": {
            "data": TIMESERIES_EXAMPLES["esempio_stagionale"]["value"],
            "model_types": ["ARIMA", "SARIMA", "Prophet"],
            "optimization_config": {
                "n_trials": 100,
                "timeout_minutes": 30,
                "optimization_metric": "mae",
                "cross_validation_folds": 5
            },
            "search_space": {
                "arima_p": {"min": 0, "max": 5},
                "arima_d": {"min": 0, "max": 2},
                "arima_q": {"min": 0, "max": 5},
                "seasonal_period": [7, 12, 24, 52]
            }
        }
    }
}

# Esempi diagnostics
DIAGNOSTICS_RESPONSE_EXAMPLES = {
    "diagnostics_complete": {
        "summary": "Diagnostica completa modello",
        "description": "Analisi residui e test statistici",
        "value": {
            "model_id": "vendite_arima_auto",
            "residuals_analysis": {
                "mean": 0.02,
                "std": 58.91,
                "skewness": -0.15,
                "kurtosis": 3.24,
                "jarque_bera": {
                    "statistic": 2.45,
                    "p_value": 0.29,
                    "normality_confirmed": True
                }
            },
            "autocorrelation_tests": {
                "ljung_box": {
                    "statistic": 12.34,
                    "p_value": 0.23,
                    "lags": 10,
                    "residuals_independent": True
                },
                "durbin_watson": 1.95
            },
            "information_criteria": {
                "aic": 245.67,
                "bic": 255.32,
                "hqic": 249.12
            },
            "performance_metrics": {
                "in_sample": {
                    "mae": 45.23,
                    "rmse": 58.91,
                    "mape": 3.87,
                    "r_squared": 0.87
                },
                "cross_validation": {
                    "cv_mae": 52.18,
                    "cv_rmse": 67.23,
                    "cv_mape": 4.52
                }
            },
            "model_stability": {
                "stationarity_confirmed": True,
                "parameters_significant": True,
                "invertibility_confirmed": True
            }
        }
    }
}

# Esempi Real-Time Streaming
STREAMING_EXAMPLES = {
    "kafka_producer": {
        "summary": "Configurazione producer Kafka",
        "description": "Setup producer per streaming real-time dei forecast",
        "value": {
            "kafka_config": {
                "bootstrap_servers": ["localhost:9092"],
                "topic": "arima-forecasts",
                "key_serializer": "string",
                "value_serializer": "json"
            },
            "streaming_config": {
                "batch_size": 10,
                "flush_interval_ms": 1000,
                "max_request_size": 1048576
            },
            "forecast_pipeline": {
                "model_ids": ["vendite_arima", "inventory_sarima"],
                "update_frequency_seconds": 60,
                "real_time_features": ["temperatura", "marketing_spend"]
            }
        }
    },
    "websocket_connection": {
        "summary": "Connessione WebSocket dashboard",
        "description": "Real-time updates per dashboard Streamlit",
        "value": {
            "websocket_url": "ws://localhost:8765/forecast-updates",
            "subscription": {
                "model_ids": ["vendite_arima"],
                "update_types": ["new_forecast", "model_retrained", "anomaly_detected"],
                "data_format": "json"
            },
            "real_time_data": {
                "timestamp": "2024-01-15T14:35:12Z",
                "model_id": "vendite_arima",
                "latest_prediction": 1285.4,
                "confidence_interval": [1180.5, 1390.3],
                "anomaly_score": 0.02,
                "feature_importance": {
                    "lag_1": 0.45,
                    "trend": 0.32,
                    "seasonal": 0.23
                }
            }
        }
    }
}

# Esempi Explainable AI
EXPLAINABILITY_EXAMPLES = {
    "shap_explanation": {
        "summary": "SHAP values per forecast explanation",
        "description": "Spiegazione contributi features alla predizione",
        "value": {
            "model_id": "vendite_sarima",
            "forecast_timestamp": "2024-01-31",
            "predicted_value": 1285.4,
            "baseline_value": 1200.0,
            "shap_values": {
                "lag_1": 45.2,
                "lag_2": 28.7,
                "lag_7": 12.1,
                "seasonal_component": 18.5,
                "trend_component": 15.3,
                "marketing_spend": 8.9,
                "temperatura": -5.2,
                "day_of_week": 3.1
            },
            "feature_importance_ranking": [
                {"feature": "lag_1", "importance": 0.42, "direction": "positive"},
                {"feature": "lag_2", "importance": 0.26, "direction": "positive"},
                {"feature": "seasonal_component", "importance": 0.17, "direction": "positive"},
                {"feature": "trend_component", "importance": 0.14, "direction": "positive"},
                {"feature": "lag_7", "importance": 0.11, "direction": "positive"}
            ],
            "confidence_factors": {
                "historical_similarity": 0.87,
                "model_stability": 0.92,
                "data_quality": 0.95,
                "seasonal_alignment": 0.83
            }
        }
    },
    "anomaly_explanation": {
        "summary": "Spiegazione anomalie rilevate",
        "description": "Analisi cause anomalie nei forecast",
        "value": {
            "anomaly_timestamp": "2024-01-15T10:30:00Z",
            "anomaly_score": 0.85,
            "severity": "HIGH",
            "predicted_value": 1850.2,
            "expected_range": [1200, 1400],
            "deviation": 450.2,
            "possible_causes": [
                {
                    "factor": "Marketing Campaign Launch",
                    "probability": 0.75,
                    "impact": "+25% demand spike",
                    "evidence": "Marketing spend increased 300%"
                },
                {
                    "factor": "Weekend Effect",
                    "probability": 0.45,
                    "impact": "+15% weekend surge",
                    "evidence": "Historical pattern Friday->Saturday"
                },
                {
                    "factor": "Weather Impact",
                    "probability": 0.30,
                    "impact": "+10% sunny weather boost",
                    "evidence": "Temperature 5°C above average"
                }
            ],
            "recommended_actions": [
                "Increase inventory by 40% for next 3 days",
                "Alert procurement team for potential stockout",
                "Monitor competitor pricing for market changes"
            ],
            "confidence_explanation": "High confidence (85%) based on historical marketing campaign patterns and strong feature correlations"
        }
    },
    "business_rules": {
        "summary": "Business rules e vincoli applicati",
        "description": "Logica business integrata nel forecast",
        "value": {
            "model_id": "vendite_inventory",
            "applied_rules": [
                {
                    "rule_id": "MIN_INVENTORY_LEVEL",
                    "description": "Scorta minima 30 giorni",
                    "condition": "inventory_level < 30_days_demand",
                    "action": "trigger_reorder",
                    "applied": True,
                    "impact": "Reorder point increased by 15%"
                },
                {
                    "rule_id": "PROMOTIONAL_LIFT",
                    "description": "Boost vendite durante promozioni",
                    "condition": "promotional_period = True",
                    "action": "apply_lift_factor",
                    "applied": False,
                    "impact": "No promotions scheduled"
                },
                {
                    "rule_id": "SEASONAL_CAP",
                    "description": "Cap massimo domanda stagionale",
                    "condition": "seasonal_demand > capacity_limit",
                    "action": "cap_at_capacity",
                    "applied": True,
                    "impact": "Forecast capped at 95% capacity (1520 units)"
                }
            ],
            "rule_engine_version": "1.2.0",
            "last_updated": "2024-01-10T09:00:00Z"
        }
    }
}