# -*- coding: utf-8 -*-
# ============================================
# FILE DEMO MLOPS COMPLETO
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Demo completa features MLOps
# ============================================

"""
Demo completa delle funzionalità MLOps integrate.

Questo esempio dimostra:
1. Experiment Tracking
2. Model Registry
3. Deployment Management
4. Workflow MLOps end-to-end
"""

import numpy as np
import pandas as pd
from pathlib import Path

from arima_forecaster.core import ARIMAForecaster, SARIMAForecaster
from arima_forecaster.mlops import (
    ModelRegistry,
    ExperimentTracker,
    DeploymentManager,
    ModelStage,
    EnvironmentType,
    DeploymentConfig,
    create_model_registry,
    create_experiment_tracker,
    create_deployment_manager
)


class SimpleModel:
    """Modello semplice serializzabile per demo."""

    def __init__(self, name="demo_model", order=(1,1,1)):
        self.name = name
        self.order = order
        self.fitted = False
        self.aic = None
        self.bic = None

    def fit(self, data):
        """Simula training."""
        self.fitted = True
        self.aic = np.random.uniform(500, 600)
        self.bic = np.random.uniform(510, 610)
        return self

    def predict(self, steps=5):
        """Simula predizione."""
        return np.random.randn(steps) + 100


def create_sample_data():
    """Crea dati di esempio per demo."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # Serie con trend e stagionalità
    trend = np.linspace(100, 120, 100)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 7)  # Stagionalità settimanale
    noise = np.random.normal(0, 2, 100)

    values = trend + seasonal + noise
    return pd.Series(values, index=dates)


def demo_experiment_tracking():
    """Demo Experiment Tracking."""
    print("\n[EXPERIMENT] === EXPERIMENT TRACKING DEMO ===")

    # Inizializza tracker
    tracker = create_experiment_tracker("demo_experiments")

    # Crea esperimento
    experiment = tracker.create_experiment(
        name="ARIMA Model Comparison",
        description="Confronto modelli ARIMA con ordini diversi",
        tags=["demo", "arima", "comparison"],
        created_by="demo_user"
    )

    print(f"[OK] Esperimento creato: {experiment.name} ({experiment.experiment_id})")

    # Dati di test
    data = create_sample_data()

    # Test diversi modelli
    models_config = [
        {"order": (1, 1, 1), "name": "ARIMA_111"},
        {"order": (2, 1, 1), "name": "ARIMA_211"},
        {"order": (1, 1, 2), "name": "ARIMA_112"}
    ]

    results = []

    for config in models_config:
        print(f"[TRAIN] Training {config['name']}...")

        # Inizia run
        run = tracker.start_run(
            experiment_id=experiment.experiment_id,
            name=config['name'],
            parameters={"order": config['order']},
            model_type="ARIMA",
            created_by="demo_user"
        )

        try:
            # Training modello
            model = SimpleModel(config['name'], config['order'])
            model.fit(data)

            # Simula calcolo metriche
            mae = np.random.uniform(1.5, 3.0)
            rmse = np.random.uniform(2.0, 4.0)
            mape = np.random.uniform(8.0, 15.0)

            # Logga metriche
            metrics = {"mae": mae, "rmse": rmse, "mape": mape}
            tracker.log_metrics(run.run_id, metrics)

            # Termina run con successo
            completed_run = tracker.end_run(run.run_id)

            results.append({
                "run_id": run.run_id,
                "config": config,
                "metrics": metrics,
                "model": model
            })

            print(f"   [OK] {config['name']}: MAE={mae:.2f}, RMSE={rmse:.2f}")

        except Exception as e:
            print(f"   [ERROR] {config['name']} failed: {e}")
            tracker.end_run(run.run_id, status="failed")

    # Trova best run
    best_run = tracker.get_best_run(experiment.experiment_id, "mae", maximize=False)
    if best_run:
        print(f"[BEST] Best model: {best_run.name} (MAE: {best_run.metrics['mae']:.2f})")

    # Comparazione runs
    run_ids = [r["run_id"] for r in results]
    comparison = tracker.compare_runs(run_ids)
    print(f"[INFO] Comparison table shape: {comparison.shape}")

    return experiment, results, best_run


def demo_model_registry(experiment_results):
    """Demo Model Registry."""
    print("\n[REGISTRY] === MODEL REGISTRY DEMO ===")

    # Inizializza registry
    registry = create_model_registry("demo_registry")

    # Registra modelli dai risultati esperimento
    for result in experiment_results:
        config = result["config"]
        model = result["model"]
        metrics = result["metrics"]

        print(f"[REG] Registrando {config['name']}...")

        metadata = registry.register_model(
            model=model,
            name=config['name'],
            model_type="ARIMA",
            parameters={"order": config['order']},
            performance_metrics=metrics,
            description=f"ARIMA model con order {config['order']}",
            tags=["demo", "arima"],
            author="demo_user"
        )

        print(f"   [OK] Registrato: v{metadata.version} (Stage: {metadata.stage})")

    # Lista modelli
    models = registry.list_models()
    print(f"[INFO] Modelli nel registry: {len(models)}")

    # Promuovi best model a staging
    if experiment_results:
        best_model_name = experiment_results[0]["config"]["name"]  # Prendi il primo per demo

        print(f"[PROMOTE] Promuovendo {best_model_name} a STAGING...")
        promoted = registry.promote_model(
            best_model_name,
            "1.0.0",
            ModelStage.STAGING,
            "demo_user"
        )
        print(f"   [OK] {best_model_name} promosso a {promoted.stage}")

        # Aggiorna performance
        updated_metrics = {"mae": 1.8, "rmse": 2.5, "validation_score": 0.92}
        registry.update_performance(best_model_name, "1.0.0", updated_metrics)
        print(f"   [UPDATE] Performance aggiornate per {best_model_name}")

        return registry, best_model_name

    return registry, None


def demo_deployment_manager(registry, model_name):
    """Demo Deployment Manager."""
    print("\n[DEPLOY] === DEPLOYMENT MANAGER DEMO ===")

    if not model_name:
        print("[WARN] Nessun modello da deployare")
        return None

    # Inizializza deployment manager
    deployment_manager = create_deployment_manager("demo_deployments", registry)

    # Configurazione deployment
    config = DeploymentConfig(
        environment=EnvironmentType.STAGING,
        model_name=model_name,
        model_version="1.0.0",
        replicas=2,
        resources={"cpu": "500m", "memory": "1Gi"},
        environment_vars={"MODEL_ENV": "staging"},
        health_checks=[
            {
                "name": "basic_health",
                "type": "http",
                "url": "http://localhost:8080/health",
                "timeout": 30
            }
        ],
        deployment_strategy="rolling",
        timeout_seconds=300
    )

    print(f"[CONFIG] Configurazione deployment per {model_name}:")
    print(f"   Environment: {config.environment}")
    print(f"   Replicas: {config.replicas}")
    print(f"   Strategy: {config.deployment_strategy}")

    # Crea deployment
    deployment = deployment_manager.create_deployment(
        config=config,
        deployed_by="demo_user",
        auto_deploy=False  # Deploy manuale per demo
    )

    print(f"[OK] Deployment creato: {deployment.deployment_id}")
    print(f"   Status: {deployment.status}")

    # Esegui deployment
    print("[DEPLOY] Eseguendo deployment...")
    deployed = deployment_manager.deploy(deployment.deployment_id)

    print(f"   Status finale: {deployed.status}")
    print(f"   Replicas ready: {deployed.replicas_ready}/{config.replicas}")

    if deployed.health_status:
        print(f"   Health status: {deployed.health_status}")

    # Mostra deployment history
    history = deployment_manager.get_deployment_history(deployment.deployment_id)
    print(f"[HISTORY] Eventi deployment: {len(history)}")
    for event in history[:3]:  # Mostra primi 3 eventi
        print(f"   - {event['event_type']}: {event['timestamp']}")

    return deployment_manager, deployment


def demo_mlops_workflow():
    """Demo workflow MLOps completo end-to-end."""
    print("\n[WORKFLOW] === WORKFLOW MLOPS END-TO-END ===")

    print("Questo demo mostra un workflow completo MLOps:")
    print("1. [EXP] Experiment tracking con multiple configurazioni")
    print("2. [REG] Model registry e versioning")
    print("3. [DEP] Deployment automatico del best model")
    print("4. [MON] Monitoring e management")

    # 1. Experiment Tracking
    experiment, results, best_run = demo_experiment_tracking()

    # 2. Model Registry
    registry, best_model_name = demo_model_registry(results)

    # 3. Deployment Management
    if best_model_name:
        deployment_manager, deployment = demo_deployment_manager(registry, best_model_name)

    # 4. Workflow Summary
    print("\n[SUMMARY] === WORKFLOW SUMMARY ===")
    print(f"[OK] Esperimento completato: {experiment.name}")
    print(f"[OK] {len(results)} modelli testati e registrati")
    if best_run:
        print(f"[BEST] Best model: {best_run.name} (MAE: {best_run.metrics['mae']:.2f})")
    if best_model_name:
        print(f"[DEPLOY] Deployment in staging: {best_model_name}")

    print("\n[SUCCESS] Demo MLOps completato con successo!")

    return {
        "experiment": experiment,
        "registry": registry,
        "deployment_manager": deployment_manager if best_model_name else None,
        "best_model": best_model_name
    }


def demo_advanced_features():
    """Demo funzionalità avanzate MLOps."""
    print("\n[ADVANCED] === FUNZIONALITA' AVANZATE ===")

    # Model Lineage
    registry = create_model_registry("demo_registry")
    models = registry.list_models()

    if models:
        model = models[0]
        print(f"[LINEAGE] Model Lineage per {model.model_name}:")

        lineage = registry.get_model_lineage(model.model_name, model.version)
        print(f"   - Dependencies: {len(lineage.get('dependencies', {}))}")
        print(f"   - Deployments: {len(lineage.get('deployments', []))}")
        print(f"   - Versions: {len(lineage.get('all_versions', []))}")

    # Export per disaster recovery
    print("\n[EXPORT] Export per Disaster Recovery:")

    # Export registry
    registry_export = Path("demo_registry_export.json")
    registry.export_registry(registry_export)
    print(f"   [OK] Registry esportato: {registry_export}")

    # Export experiment
    tracker = create_experiment_tracker("demo_experiments")
    experiments = tracker.list_experiments()

    if experiments:
        exp = experiments[0]
        exp_export = Path("demo_experiment_export.json")
        tracker.export_experiment(exp.experiment_id, exp_export)
        print(f"   [OK] Experiment esportato: {exp_export}")

    print("\n[CLEANUP] Cleanup files...")
    for file in [registry_export, Path("demo_experiment_export.json")]:
        if file.exists():
            file.unlink()
            print(f"   [DELETE] Rimosso: {file}")


if __name__ == "__main__":
    print("[DEMO] ARIMA Forecaster - MLOps Demo Completo")
    print("=" * 50)

    try:
        # Workflow principale
        results = demo_mlops_workflow()

        # Funzionalità avanzate
        demo_advanced_features()

        print(f"\n[DONE] Demo completato con successo!")
        print(f"[INFO] Artifacts creati in: demo_experiments/, demo_registry/, demo_deployments/")

    except Exception as e:
        print(f"\n[ERROR] Errore durante demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n[END] Fine demo MLOps!")