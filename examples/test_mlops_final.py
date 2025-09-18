# ============================================
# FILE TEST FINALE MLOPS
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Test finale MLOps con modello serializzabile
# ============================================

"""
Test finale integrazione MLOps completa.
"""

from arima_forecaster.mlops import (
    ModelRegistry,
    ExperimentTracker,
    DeploymentManager,
    ModelStage,
    EnvironmentType,
    DeploymentConfig,
)


class SimpleTestModel:
    """Modello semplice per test MLOps."""

    def __init__(self, name="test_model"):
        self.name = name
        self.order = (1, 1, 1)
        self.fitted = False
        self.aic = 123.45
        self.bic = 130.67

    def fit(self, data):
        self.fitted = True
        return self

    def predict(self, steps=5):
        return [1.0, 1.1, 1.2, 1.3, 1.4][:steps]


def test_mlops_complete():
    """Test MLOps completo end-to-end."""
    print("[TEST] Avvio test MLOps completo...")

    # 1. Setup componenti
    registry = ModelRegistry("test_registry")
    tracker = ExperimentTracker("test_experiments")
    deployment_manager = DeploymentManager("test_deployments", registry)

    print("[OK] Componenti MLOps inizializzati")

    # 2. Experiment tracking
    experiment = tracker.create_experiment(
        name="Test MLOps Integration", description="Test completo integrazione MLOps"
    )

    run = tracker.start_run(
        experiment_id=experiment.experiment_id,
        parameters={"order": [1, 1, 1], "test": True},
        model_type="TEST_ARIMA",
    )

    # Simula training
    model = SimpleTestModel("test_arima_model")
    model.fit([1, 2, 3, 4, 5])

    # Metriche
    metrics = {"mae": 1.5, "rmse": 2.0, "mape": 7.5}
    tracker.log_metrics(run.run_id, metrics)

    completed_run = tracker.end_run(run.run_id)

    print(f"[OK] Experiment tracking: {completed_run.status}")

    # 3. Model registry
    metadata = registry.register_model(
        model=model,
        name="test_arima",
        model_type="TEST_ARIMA",
        parameters={"order": model.order},
        performance_metrics=metrics,
        description="Test model per MLOps",
    )

    print(f"[OK] Model registry: {metadata.model_name} v{metadata.version}")

    # 4. Model promotion
    promoted = registry.promote_model("test_arima", "1.0.0", ModelStage.STAGING)

    print(f"[OK] Model promotion: {promoted.stage}")

    # 5. Deployment
    config = DeploymentConfig(
        environment=EnvironmentType.STAGING,
        model_name="test_arima",
        model_version="1.0.0",
        replicas=1,
    )

    deployment = deployment_manager.create_deployment(config=config, auto_deploy=False)

    print(f"[OK] Deployment config: {deployment.status}")

    # 6. Test retrieval
    loaded_model, loaded_metadata = registry.get_model("test_arima")
    assert loaded_model.name == "test_arima_model"
    assert loaded_metadata.model_name == "test_arima"

    print("[OK] Model retrieval test passed")

    # 7. Test experiment comparison
    run2 = tracker.start_run(
        experiment_id=experiment.experiment_id,
        parameters={"order": [2, 1, 1], "test": True},
        model_type="TEST_ARIMA",
    )
    tracker.log_metrics(run2.run_id, {"mae": 1.8, "rmse": 2.3})
    tracker.end_run(run2.run_id)

    comparison = tracker.compare_runs([run.run_id, run2.run_id])
    assert len(comparison) == 2

    print("[OK] Experiment comparison test passed")

    # 8. Test best run
    best_run = tracker.get_best_run(experiment.experiment_id, "mae", maximize=False)
    assert best_run.run_id == run.run_id  # primo run ha MAE più basso

    print("[OK] Best run selection test passed")

    # 9. Test export
    registry.export_registry("test_export.json")
    tracker.export_experiment(experiment.experiment_id, "test_exp_export.json")

    print("[OK] Export functionality test passed")

    # 10. Test model lineage
    lineage = registry.get_model_lineage("test_arima", "1.0.0")
    assert "model" in lineage
    assert "deployments" in lineage

    print("[OK] Model lineage test passed")

    print("\n[SUCCESS] Tutti i test MLOps passati!")

    return {
        "registry": registry,
        "tracker": tracker,
        "deployment_manager": deployment_manager,
        "experiment": experiment,
        "model_metadata": metadata,
        "deployment": deployment,
    }


if __name__ == "__main__":
    try:
        results = test_mlops_complete()

        print("\n[SUMMARY] MLOps Test Results:")
        print(f"- Experiment ID: {results['experiment'].experiment_id}")
        print(f"- Model: {results['model_metadata'].model_name}")
        print(f"- Version: {results['model_metadata'].version}")
        print(f"- Stage: {results['model_metadata'].stage}")
        print(f"- Deployment: {results['deployment'].deployment_id}")

        print("\n[CLEANUP] Test artifacts created in:")
        print("- test_registry/")
        print("- test_experiments/")
        print("- test_deployments/")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()
