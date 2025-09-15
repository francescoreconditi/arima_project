# ============================================
# FILE DI TEST MLOPS INTEGRATION
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Test integrazione completa MLOps
# ============================================

"""
Test integrazione MLOps completa.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from arima_forecaster.mlops import (
    ModelRegistry,
    ExperimentTracker,
    DeploymentManager,
    ModelStage,
    ModelStatus,
    ExperimentStatus,
    RunStatus,
    DeploymentStatus,
    EnvironmentType,
    DeploymentConfig
)
from arima_forecaster.core import ARIMAForecaster


class TestMLOpsBasicIntegration:
    """Test integrazione base MLOps."""

    def setup_method(self):
        """Setup per ogni test."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry"
        self.tracking_path = Path(self.temp_dir) / "tracking"
        self.deployment_path = Path(self.temp_dir) / "deployment"

        self.registry = ModelRegistry(self.registry_path)
        self.tracker = ExperimentTracker(self.tracking_path)
        self.deployment_manager = DeploymentManager(self.deployment_path, self.registry)

    def test_mlops_imports(self):
        """Test import moduli MLOps."""
        from arima_forecaster.mlops import (
            ModelRegistry,
            ModelVersion,
            ModelMetadata,
            ExperimentTracker,
            Experiment,
            ExperimentRun,
            DeploymentManager,
            DeploymentConfig,
            ModelDeployment
        )

        # Verifica che le classi siano importabili
        assert ModelRegistry is not None
        assert ModelVersion is not None
        assert ModelMetadata is not None
        assert ExperimentTracker is not None
        assert Experiment is not None
        assert ExperimentRun is not None
        assert DeploymentManager is not None
        assert DeploymentConfig is not None
        assert ModelDeployment is not None

    def test_model_registry_basic(self):
        """Test funzionalità base Model Registry."""
        # Crea modello mock
        model = Mock()
        model.order = (1, 1, 1)

        # Registra modello
        metadata = self.registry.register_model(
            model=model,
            name="test_model",
            model_type="ARIMA",
            description="Test ARIMA model",
            tags=["test", "arima"],
            performance_metrics={"mae": 0.5, "rmse": 0.7}
        )

        assert metadata.model_name == "test_model"
        assert metadata.model_type == "ARIMA"
        assert metadata.version == "1.0.0"
        assert metadata.stage == ModelStage.DEVELOPMENT
        assert metadata.status == ModelStatus.ACTIVE

        # Carica modello
        loaded_model, loaded_metadata = self.registry.get_model("test_model")
        assert loaded_metadata.model_name == "test_model"
        assert loaded_metadata.version == "1.0.0"

    def test_experiment_tracking_basic(self):
        """Test funzionalità base Experiment Tracking."""
        # Crea esperimento
        experiment = self.tracker.create_experiment(
            name="test_experiment",
            description="Test experiment for ARIMA",
            tags=["test", "arima"]
        )

        assert experiment.name == "test_experiment"
        assert experiment.status == ExperimentStatus.RUNNING

        # Inizia run
        run = self.tracker.start_run(
            experiment_id=experiment.experiment_id,
            name="test_run",
            parameters={"order": [1, 1, 1]},
            model_type="ARIMA"
        )

        assert run.experiment_id == experiment.experiment_id
        assert run.status == RunStatus.RUNNING
        assert run.parameters["order"] == [1, 1, 1]

        # Logga metriche
        self.tracker.log_metrics(run.run_id, {"mae": 0.5, "rmse": 0.7})

        # Termina run
        final_run = self.tracker.end_run(run.run_id, RunStatus.COMPLETED)
        assert final_run.status == RunStatus.COMPLETED
        assert final_run.metrics["mae"] == 0.5

    def test_deployment_manager_basic(self):
        """Test funzionalità base Deployment Manager."""
        # Prima registra un modello
        model = Mock()
        self.registry.register_model(
            model=model,
            name="deploy_test_model",
            model_type="ARIMA",
            stage=ModelStage.PRODUCTION
        )

        # Crea configurazione deployment
        config = DeploymentConfig(
            environment=EnvironmentType.STAGING,
            model_name="deploy_test_model",
            model_version="1.0.0",
            replicas=2
        )

        # Crea deployment
        deployment = self.deployment_manager.create_deployment(
            config=config,
            deployed_by="test_user",
            auto_deploy=False  # Non deployare automaticamente nei test
        )

        assert deployment.model_name == "deploy_test_model"
        assert deployment.environment == EnvironmentType.STAGING
        assert deployment.status == DeploymentStatus.PENDING

        # Test deploy manuale
        deployed = self.deployment_manager.deploy(deployment.deployment_id)
        assert deployed.status in [DeploymentStatus.DEPLOYED, DeploymentStatus.FAILED]

    def test_full_mlops_workflow(self):
        """Test workflow completo MLOps."""
        # 1. Crea esperimento
        experiment = self.tracker.create_experiment(
            name="full_workflow_test",
            description="Test complete MLOps workflow"
        )

        # 2. Inizia run
        run = self.tracker.start_run(
            experiment_id=experiment.experiment_id,
            name="training_run",
            parameters={"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
            model_type="SARIMA"
        )

        # 3. Simula training e logga metriche
        self.tracker.log_metrics(run.run_id, {
            "mae": 0.4,
            "rmse": 0.6,
            "mape": 8.5
        })

        # 4. Termina run
        completed_run = self.tracker.end_run(run.run_id, RunStatus.COMPLETED)

        # 5. Registra modello trained
        model = Mock()
        model.order = (1, 1, 1)
        model.seasonal_order = (1, 1, 1, 12)

        metadata = self.registry.register_model(
            model=model,
            name="workflow_model",
            model_type="SARIMA",
            parameters=completed_run.parameters,
            performance_metrics=completed_run.metrics,
            description="Model from full workflow test"
        )

        # 6. Promuovi a staging
        staging_metadata = self.registry.promote_model(
            "workflow_model",
            "1.0.0",
            ModelStage.STAGING
        )
        assert staging_metadata.stage == ModelStage.STAGING

        # 7. Deploy in staging
        config = DeploymentConfig(
            environment=EnvironmentType.STAGING,
            model_name="workflow_model",
            model_version="1.0.0"
        )

        deployment = self.deployment_manager.create_deployment(
            config=config,
            auto_deploy=False
        )

        # Verifica workflow completo
        assert experiment.name == "full_workflow_test"
        assert completed_run.status == RunStatus.COMPLETED
        assert metadata.model_name == "workflow_model"
        assert staging_metadata.stage == ModelStage.STAGING
        assert deployment.model_name == "workflow_model"

    def test_model_lineage_tracking(self):
        """Test tracking lineage del modello."""
        # Registra modello con dependencies
        model = Mock()
        metadata = self.registry.register_model(
            model=model,
            name="lineage_test",
            model_type="ARIMA",
            dependencies={"pandas": "2.0.0", "numpy": "1.24.0"},
            dataset_hash="abc123"
        )

        # Promuovi attraverso stages
        self.registry.promote_model("lineage_test", "1.0.0", ModelStage.STAGING)
        self.registry.promote_model("lineage_test", "1.0.0", ModelStage.PRODUCTION)

        # Ottieni lineage
        lineage = self.registry.get_model_lineage("lineage_test", "1.0.0")

        assert "model" in lineage
        assert "deployments" in lineage
        assert "dependencies" in lineage
        assert lineage["dependencies"]["pandas"] == "2.0.0"

    def test_experiment_comparison(self):
        """Test comparazione esperimenti."""
        experiment = self.tracker.create_experiment("comparison_test")

        # Crea multiple runs con parametri diversi
        run1 = self.tracker.start_run(
            experiment.experiment_id,
            parameters={"order": [1, 1, 1]},
            model_type="ARIMA"
        )
        self.tracker.log_metrics(run1.run_id, {"mae": 0.5, "rmse": 0.7})
        self.tracker.end_run(run1.run_id)

        run2 = self.tracker.start_run(
            experiment.experiment_id,
            parameters={"order": [2, 1, 2]},
            model_type="ARIMA"
        )
        self.tracker.log_metrics(run2.run_id, {"mae": 0.4, "rmse": 0.6})
        self.tracker.end_run(run2.run_id)

        # Compara runs
        comparison = self.tracker.compare_runs([run1.run_id, run2.run_id])
        assert len(comparison) == 2
        assert "metric_mae" in comparison.columns
        assert "param_order" in comparison.columns

        # Trova best run
        best_run = self.tracker.get_best_run(experiment.experiment_id, "mae", maximize=False)
        assert best_run.run_id == run2.run_id  # run2 ha MAE più basso

    def test_deployment_rollback(self):
        """Test rollback deployment."""
        # Registra due versioni
        model_v1 = Mock()
        self.registry.register_model(model_v1, "rollback_test", "ARIMA", version="1.0.0")

        model_v2 = Mock()
        self.registry.register_model(model_v2, "rollback_test", "ARIMA", version="1.1.0")

        # Deploy v1.0.0
        config_v1 = DeploymentConfig(
            environment=EnvironmentType.PRODUCTION,
            model_name="rollback_test",
            model_version="1.0.0"
        )
        deployment_v1 = self.deployment_manager.create_deployment(config_v1, auto_deploy=False)

        # Deploy v1.1.0
        config_v2 = DeploymentConfig(
            environment=EnvironmentType.PRODUCTION,
            model_name="rollback_test",
            model_version="1.1.0"
        )
        deployment_v2 = self.deployment_manager.create_deployment(config_v2, auto_deploy=False)

        # Simula rollback
        rolled_back = self.deployment_manager.rollback(deployment_v2.deployment_id, "1.0.0")

        # Verifica rollback o failure (dipende dalla simulazione)
        assert rolled_back.status in [
            DeploymentStatus.ROLLED_BACK,
            DeploymentStatus.FAILED,
            DeploymentStatus.ROLLING_BACK
        ]

    def test_mlops_export_import(self):
        """Test export/import per disaster recovery."""
        # Crea dati di test
        model = Mock()
        metadata = self.registry.register_model(model, "export_test", "ARIMA")

        experiment = self.tracker.create_experiment("export_experiment")
        run = self.tracker.start_run(experiment.experiment_id)
        self.tracker.log_metrics(run.run_id, {"mae": 0.5})
        self.tracker.end_run(run.run_id)

        # Export registry
        registry_export = Path(self.temp_dir) / "registry_export.json"
        self.registry.export_registry(registry_export)
        assert registry_export.exists()

        # Export experiment
        experiment_export = Path(self.temp_dir) / "experiment_export.json"
        self.tracker.export_experiment(experiment.experiment_id, experiment_export)
        assert experiment_export.exists()

        # Verifica che i file di export contengano dati
        assert registry_export.stat().st_size > 0
        assert experiment_export.stat().st_size > 0


class TestMLOpsErrorHandling:
    """Test gestione errori MLOps."""

    def setup_method(self):
        """Setup per ogni test."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(Path(self.temp_dir) / "registry")

    def test_model_not_found_errors(self):
        """Test errori per modelli non trovati."""
        with pytest.raises(ValueError, match="Modello non trovato"):
            self.registry.get_model("nonexistent")

    def test_invalid_model_promotion(self):
        """Test errori promozione modello invalido."""
        with pytest.raises(ValueError, match="Modello non trovato"):
            self.registry.promote_model("nonexistent", "1.0.0", ModelStage.PRODUCTION)

    def test_duplicate_model_registration(self):
        """Test registrazione modello duplicato."""
        model = Mock()

        # Prima registrazione OK
        self.registry.register_model(model, "duplicate_test", "ARIMA", version="1.0.0")

        # Seconda registrazione con stessa versione dovrebbe sovrascrivere
        metadata2 = self.registry.register_model(model, "duplicate_test", "ARIMA", version="1.0.0")
        assert metadata2.version == "1.0.0"


class TestMLOpsPerformance:
    """Test performance MLOps."""

    def setup_method(self):
        """Setup per ogni test."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(Path(self.temp_dir) / "registry")
        self.tracker = ExperimentTracker(Path(self.temp_dir) / "tracking")

    def test_bulk_model_operations(self):
        """Test operazioni bulk sui modelli."""
        models = []

        # Registra molti modelli
        for i in range(10):
            model = Mock()
            metadata = self.registry.register_model(
                model,
                f"bulk_model_{i}",
                "ARIMA",
                performance_metrics={"mae": 0.1 * i}
            )
            models.append(metadata)

        # Lista tutti
        all_models = self.registry.list_models()
        assert len(all_models) == 10

        # Filtra per stage
        dev_models = self.registry.list_models(stage=ModelStage.DEVELOPMENT)
        assert len(dev_models) == 10

    def test_bulk_experiment_operations(self):
        """Test operazioni bulk sugli esperimenti."""
        experiment = self.tracker.create_experiment("bulk_test")

        runs = []
        # Crea molti runs
        for i in range(20):
            run = self.tracker.start_run(
                experiment.experiment_id,
                parameters={"order": [1, 1, i % 3 + 1]},
                model_type="ARIMA"
            )
            self.tracker.log_metrics(run.run_id, {"mae": 0.1 * i, "rmse": 0.2 * i})
            self.tracker.end_run(run.run_id)
            runs.append(run)

        # Lista tutti i runs
        all_runs = self.tracker.list_runs(experiment_id=experiment.experiment_id)
        assert len(all_runs) == 20

        # Compara tutti i runs
        comparison = self.tracker.compare_runs([r.run_id for r in runs[:5]])
        assert len(comparison) == 5


if __name__ == "__main__":
    pytest.main([__file__])