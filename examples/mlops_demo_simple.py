# ============================================
# FILE DEMO MLOPS SEMPLIFICATO
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Demo MLOps senza emoji (Windows-safe)
# ============================================

"""
Demo semplificata delle funzionalità MLOps integrate.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from arima_forecaster.core import ARIMAForecaster
from arima_forecaster.mlops import (
    ModelRegistry,
    ExperimentTracker,
    DeploymentManager,
    ModelStage,
    EnvironmentType,
    DeploymentConfig,
)


def create_sample_data():
    """Crea dati di esempio per demo."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Serie con trend e stagionalità
    trend = np.linspace(100, 120, 100)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 7)
    noise = np.random.normal(0, 2, 100)

    values = trend + seasonal + noise
    return pd.Series(values, index=dates)


def main():
    """Demo MLOps principale."""
    print("[DEMO] ARIMA Forecaster - MLOps Demo")
    print("=" * 40)

    # 1. Setup componenti MLOps
    print("\n[SETUP] Inizializzazione componenti MLOps...")

    registry = ModelRegistry("demo_registry")
    tracker = ExperimentTracker("demo_experiments")
    deployment_manager = DeploymentManager("demo_deployments", registry)

    # 2. Experiment Tracking
    print("\n[EXPERIMENT] Creazione esperimento...")

    experiment = tracker.create_experiment(
        name="ARIMA Demo Test", description="Test semplificato MLOps"
    )

    print(f"Esperimento creato: {experiment.experiment_id}")

    # 3. Training e run
    print("\n[TRAINING] Training modello ARIMA...")

    run = tracker.start_run(
        experiment_id=experiment.experiment_id, parameters={"order": [1, 1, 1]}, model_type="ARIMA"
    )

    # Dati di test
    data = create_sample_data()

    # Training modello
    model = ARIMAForecaster(order=(1, 1, 1))
    model.fit(data)

    # Metriche simulate
    metrics = {"mae": 2.1, "rmse": 2.8, "mape": 10.5}
    tracker.log_metrics(run.run_id, metrics)

    # Termina run
    completed_run = tracker.end_run(run.run_id)
    print(f"Run completato con MAE: {completed_run.metrics['mae']}")

    # 4. Model Registry
    print("\n[REGISTRY] Registrazione modello...")

    metadata = registry.register_model(
        model=model,
        name="demo_arima",
        model_type="ARIMA",
        parameters={"order": [1, 1, 1]},
        performance_metrics=metrics,
        description="Demo ARIMA model",
    )

    print(f"Modello registrato: {metadata.model_name} v{metadata.version}")

    # 5. Promozione modello
    print("\n[PROMOTION] Promozione a staging...")

    promoted = registry.promote_model("demo_arima", "1.0.0", ModelStage.STAGING)

    print(f"Modello promosso a: {promoted.stage}")

    # 6. Deployment
    print("\n[DEPLOYMENT] Configurazione deployment...")

    config = DeploymentConfig(
        environment=EnvironmentType.STAGING,
        model_name="demo_arima",
        model_version="1.0.0",
        replicas=1,
    )

    deployment = deployment_manager.create_deployment(config=config, auto_deploy=False)

    print(f"Deployment creato: {deployment.deployment_id}")
    print(f"Status: {deployment.status}")

    # 7. Summary finale
    print("\n[SUMMARY] Riepilogo demo:")
    print(f"- Esperimento: {experiment.name}")
    print(f"- Modello: {metadata.model_name} v{metadata.version}")
    print(f"- Performance: MAE {metrics['mae']}")
    print(f"- Stage: {promoted.stage}")
    print(f"- Deployment: {deployment.environment}")

    print("\n[SUCCESS] Demo MLOps completato con successo!")

    # 8. Cleanup opzionale
    print("\n[INFO] Artifacts salvati in:")
    print("- demo_registry/")
    print("- demo_experiments/")
    print("- demo_deployments/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Errore durante demo: {e}")
        import traceback

        traceback.print_exc()
