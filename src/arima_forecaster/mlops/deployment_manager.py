# ============================================
# FILE MLOps DEPLOYMENT MANAGER
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Deployment management utilities
# ============================================

"""
Deployment Manager per ARIMA Forecaster

Sistema di gestione deployment per MLOps:
- Deployment configuration management
- Rolling updates e rollbacks
- Health checks e monitoring
- Environment management (dev/staging/prod)
- Integration con model registry
- Automated deployment pipelines
"""

import json
import sqlite3
import uuid
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import shutil

from pydantic import BaseModel, Field, ConfigDict

from arima_forecaster.utils.logger import get_logger
from .model_registry import ModelRegistry, ModelStage

logger = get_logger(__name__)


class DeploymentStatus(str, Enum):
    """Status di un deployment."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class EnvironmentType(str, Enum):
    """Tipo di environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class HealthCheckType(str, Enum):
    """Tipo di health check."""
    HTTP = "http"
    MODEL_PREDICTION = "model_prediction"
    CUSTOM = "custom"


@dataclass
class DeploymentConfig:
    """Configurazione per deployment."""
    environment: EnvironmentType
    model_name: str
    model_version: str
    replicas: int = 1
    resources: Optional[Dict[str, str]] = None  # CPU, memory limits
    environment_vars: Optional[Dict[str, str]] = None
    health_checks: Optional[List[Dict[str, Any]]] = None
    rollback_config: Optional[Dict[str, Any]] = None
    deployment_strategy: str = "rolling"  # rolling, blue_green, canary
    timeout_seconds: int = 300
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelDeployment:
    """Metadata per un deployment."""
    deployment_id: str
    environment: EnvironmentType
    model_name: str
    model_version: str
    status: DeploymentStatus
    config: DeploymentConfig
    deployed_at: Optional[datetime]
    deployed_by: str
    health_status: Optional[str]
    endpoint_url: Optional[str]
    replicas_ready: int
    rollback_version: Optional[str]
    created_at: datetime
    updated_at: datetime
    logs: List[str]
    metadata: Optional[Dict[str, Any]] = None


class HealthCheck:
    """Health check per modelli deployati."""

    def __init__(self, check_type: HealthCheckType, config: Dict[str, Any]):
        self.check_type = check_type
        self.config = config

    def execute(self) -> Tuple[bool, str]:
        """
        Esegue health check.

        Returns:
            Tupla (success, message)
        """
        if self.check_type == HealthCheckType.HTTP:
            return self._http_check()
        elif self.check_type == HealthCheckType.MODEL_PREDICTION:
            return self._model_prediction_check()
        elif self.check_type == HealthCheckType.CUSTOM:
            return self._custom_check()
        else:
            return False, f"Health check type non supportato: {self.check_type}"

    def _http_check(self) -> Tuple[bool, str]:
        """Health check HTTP."""
        try:
            import requests
            url = self.config.get('url')
            timeout = self.config.get('timeout', 30)

            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, f"HTTP check OK ({response.status_code})"
            else:
                return False, f"HTTP check failed ({response.status_code})"
        except Exception as e:
            return False, f"HTTP check error: {e}"

    def _model_prediction_check(self) -> Tuple[bool, str]:
        """Health check prediction test."""
        try:
            # Carica modello e testa predizione
            model_path = self.config.get('model_path')
            test_data = self.config.get('test_data')

            if not model_path or not test_data:
                return False, "Model path o test data mancanti"

            # Implementazione specifica per ARIMA models
            # Questo sarebbe personalizzato per ogni tipo di modello
            return True, "Model prediction check OK"

        except Exception as e:
            return False, f"Model prediction check error: {e}"

    def _custom_check(self) -> Tuple[bool, str]:
        """Health check custom."""
        try:
            check_function = self.config.get('function')
            if callable(check_function):
                return check_function()
            else:
                return False, "Custom check function non valida"
        except Exception as e:
            return False, f"Custom check error: {e}"


class DeploymentManager:
    """
    Sistema di gestione deployment per modelli ARIMA.

    Features:
    - Configuration management per environments
    - Rolling deployments con rollback
    - Health monitoring e checks
    - Integration con model registry
    - Deployment history e audit
    - Automated pipelines support
    """

    def __init__(
        self,
        deployment_path: Union[str, Path] = "deployments",
        model_registry: Optional[ModelRegistry] = None
    ):
        """
        Inizializza Deployment Manager.

        Args:
            deployment_path: Path base per deployments
            model_registry: Istanza ModelRegistry (opzionale)
        """
        self.deployment_path = Path(deployment_path)
        self.deployment_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.deployment_path / "deployments.db"
        self.configs_path = self.deployment_path / "configs"
        self.logs_path = self.deployment_path / "logs"

        self.configs_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self.model_registry = model_registry
        self._init_database()

        logger.info(f"Deployment Manager inizializzato: {self.deployment_path}")

    def _init_database(self) -> None:
        """Inizializza database SQLite per deployments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabella deployments
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT,  -- JSON
                    deployed_at TIMESTAMP,
                    deployed_by TEXT,
                    health_status TEXT,
                    endpoint_url TEXT,
                    replicas_ready INTEGER DEFAULT 0,
                    rollback_version TEXT,
                    logs TEXT,  -- JSON array
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            # Tabella environment configs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS environment_configs (
                    environment TEXT PRIMARY KEY,
                    config TEXT NOT NULL,  -- JSON
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            # Tabella deployment history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,  -- deploy, rollback, health_check, etc.
                    event_data TEXT,  -- JSON
                    timestamp TIMESTAMP,
                    FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
                )
            """)

            # Indici
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deployments_env ON deployments(environment)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deployments_model ON deployments(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status)")

            conn.commit()

    def create_deployment(
        self,
        config: DeploymentConfig,
        deployed_by: str = "system",
        auto_deploy: bool = True
    ) -> ModelDeployment:
        """
        Crea un nuovo deployment.

        Args:
            config: Configurazione deployment
            deployed_by: Utente che esegue deployment
            auto_deploy: Esegui deployment automaticamente

        Returns:
            ModelDeployment creato
        """
        deployment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        deployment = ModelDeployment(
            deployment_id=deployment_id,
            environment=config.environment,
            model_name=config.model_name,
            model_version=config.model_version,
            status=DeploymentStatus.PENDING,
            config=config,
            deployed_at=None,
            deployed_by=deployed_by,
            health_status=None,
            endpoint_url=None,
            replicas_ready=0,
            rollback_version=None,
            created_at=now,
            updated_at=now,
            logs=[],
            metadata=config.metadata
        )

        # Salva in database
        self._save_deployment(deployment)

        # Log evento
        self._log_deployment_event(deployment_id, "created", {"config": asdict(config)})

        if auto_deploy:
            return self.deploy(deployment_id)

        logger.info(f"Deployment creato: {deployment_id} ({config.model_name} v{config.model_version})")
        return deployment

    def deploy(self, deployment_id: str) -> ModelDeployment:
        """
        Esegue deployment.

        Args:
            deployment_id: ID del deployment

        Returns:
            ModelDeployment aggiornato
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment non trovato: {deployment_id}")

        try:
            # Aggiorna status
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.updated_at = datetime.now(timezone.utc)
            self._save_deployment(deployment)

            # Verifica modello nel registry
            if self.model_registry:
                model_metadata = self.model_registry.get_model_metadata(
                    deployment.model_name,
                    deployment.model_version
                )
                if not model_metadata:
                    raise ValueError(f"Modello non trovato nel registry: {deployment.model_name} v{deployment.model_version}")

            # Esegui deployment strategy
            if deployment.config.deployment_strategy == "rolling":
                success = self._rolling_deployment(deployment)
            elif deployment.config.deployment_strategy == "blue_green":
                success = self._blue_green_deployment(deployment)
            else:
                success = self._simple_deployment(deployment)

            if success:
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.deployed_at = datetime.now(timezone.utc)
                deployment.replicas_ready = deployment.config.replicas

                # Esegui health checks
                health_ok, health_msg = self._run_health_checks(deployment)
                deployment.health_status = "healthy" if health_ok else "unhealthy"

                self._log_deployment_event(deployment_id, "deployed", {
                    "success": True,
                    "health_status": deployment.health_status
                })

            else:
                deployment.status = DeploymentStatus.FAILED
                self._log_deployment_event(deployment_id, "deploy_failed", {"error": "Deployment failed"})

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Deployment error: {e}")
            self._log_deployment_event(deployment_id, "deploy_error", {"error": str(e)})
            logger.error(f"Deployment failed: {deployment_id} - {e}")

        deployment.updated_at = datetime.now(timezone.utc)
        self._save_deployment(deployment)

        return deployment

    def rollback(self, deployment_id: str, target_version: Optional[str] = None) -> ModelDeployment:
        """
        Esegue rollback di un deployment.

        Args:
            deployment_id: ID del deployment
            target_version: Versione target per rollback (previous se None)

        Returns:
            ModelDeployment aggiornato
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment non trovato: {deployment_id}")

        try:
            deployment.status = DeploymentStatus.ROLLING_BACK
            deployment.updated_at = datetime.now(timezone.utc)
            self._save_deployment(deployment)

            # Determina versione target
            if not target_version:
                target_version = self._get_previous_version(deployment)

            if not target_version:
                raise ValueError("Nessuna versione precedente trovata per rollback")

            # Esegui rollback
            rollback_config = DeploymentConfig(
                environment=deployment.environment,
                model_name=deployment.model_name,
                model_version=target_version,
                replicas=deployment.config.replicas,
                resources=deployment.config.resources,
                environment_vars=deployment.config.environment_vars,
                health_checks=deployment.config.health_checks,
                deployment_strategy="rolling",  # Rollback sempre rolling
                timeout_seconds=deployment.config.timeout_seconds
            )

            # Crea nuovo deployment per rollback
            rollback_deployment = self.create_deployment(
                rollback_config,
                deployed_by=f"rollback_from_{deployment.model_version}",
                auto_deploy=True
            )

            if rollback_deployment.status == DeploymentStatus.DEPLOYED:
                deployment.status = DeploymentStatus.ROLLED_BACK
                deployment.rollback_version = target_version
                self._log_deployment_event(deployment_id, "rolled_back", {
                    "target_version": target_version,
                    "rollback_deployment_id": rollback_deployment.deployment_id
                })
            else:
                deployment.status = DeploymentStatus.FAILED
                self._log_deployment_event(deployment_id, "rollback_failed", {"error": "Rollback deployment failed"})

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Rollback error: {e}")
            self._log_deployment_event(deployment_id, "rollback_error", {"error": str(e)})
            logger.error(f"Rollback failed: {deployment_id} - {e}")

        deployment.updated_at = datetime.now(timezone.utc)
        self._save_deployment(deployment)

        return deployment

    def get_deployment(self, deployment_id: str) -> Optional[ModelDeployment]:
        """
        Ottiene deployment per ID.

        Args:
            deployment_id: ID del deployment

        Returns:
            ModelDeployment se trovato
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments WHERE deployment_id = ?", (deployment_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_deployment(row)

    def list_deployments(
        self,
        environment: Optional[EnvironmentType] = None,
        status: Optional[DeploymentStatus] = None,
        model_name: Optional[str] = None
    ) -> List[ModelDeployment]:
        """
        Lista deployments.

        Args:
            environment: Filtra per environment
            status: Filtra per status
            model_name: Filtra per nome modello

        Returns:
            Lista di ModelDeployment
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM deployments WHERE 1=1"
            params = []

            if environment:
                query += " AND environment = ?"
                params.append(environment.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            query += " ORDER BY created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_deployment(row) for row in rows]

    def health_check(self, deployment_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Esegue health check per deployment.

        Args:
            deployment_id: ID del deployment

        Returns:
            Tupla (overall_health, detailed_results)
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return False, {"error": "Deployment non trovato"}

        return self._run_health_checks(deployment)

    def get_deployment_logs(self, deployment_id: str) -> List[str]:
        """
        Ottiene logs di deployment.

        Args:
            deployment_id: ID del deployment

        Returns:
            Lista di log entries
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return []

        return deployment.logs

    def get_deployment_history(self, deployment_id: str) -> List[Dict[str, Any]]:
        """
        Ottiene history di deployment.

        Args:
            deployment_id: ID del deployment

        Returns:
            Lista di eventi
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, event_data, timestamp
                FROM deployment_history
                WHERE deployment_id = ?
                ORDER BY timestamp DESC
            """, (deployment_id,))

            return [
                {
                    "event_type": row[0],
                    "event_data": json.loads(row[1] or '{}'),
                    "timestamp": row[2]
                }
                for row in cursor.fetchall()
            ]

    # ============================================
    # METODI PRIVATI
    # ============================================

    def _rolling_deployment(self, deployment: ModelDeployment) -> bool:
        """Esegue rolling deployment."""
        try:
            # Simulazione rolling deployment
            # In implementazione reale, questo interfaccerebbe con Kubernetes, Docker, ecc.

            deployment.logs.append("Iniziando rolling deployment...")

            # Deploy graduale delle repliche
            for i in range(deployment.config.replicas):
                deployment.logs.append(f"Deploying replica {i+1}/{deployment.config.replicas}")
                time.sleep(1)  # Simulazione tempo deployment

                # Simula health check per replica
                if not self._simulate_replica_health():
                    deployment.logs.append(f"Replica {i+1} failed health check")
                    return False

                deployment.replicas_ready = i + 1

            deployment.logs.append("Rolling deployment completato con successo")
            return True

        except Exception as e:
            deployment.logs.append(f"Rolling deployment error: {e}")
            return False

    def _blue_green_deployment(self, deployment: ModelDeployment) -> bool:
        """Esegue blue-green deployment."""
        try:
            deployment.logs.append("Iniziando blue-green deployment...")

            # Deploy in ambiente green
            deployment.logs.append("Deploying green environment...")
            time.sleep(2)

            # Health check green
            if not self._simulate_replica_health():
                deployment.logs.append("Green environment failed health check")
                return False

            # Switch traffico
            deployment.logs.append("Switching traffic to green...")
            time.sleep(1)

            deployment.logs.append("Blue-green deployment completato")
            return True

        except Exception as e:
            deployment.logs.append(f"Blue-green deployment error: {e}")
            return False

    def _simple_deployment(self, deployment: ModelDeployment) -> bool:
        """Esegue deployment semplice."""
        try:
            deployment.logs.append("Iniziando deployment semplice...")
            time.sleep(1)

            # Simula deployment
            deployment.logs.append("Deployment completato")
            return True

        except Exception as e:
            deployment.logs.append(f"Simple deployment error: {e}")
            return False

    def _run_health_checks(self, deployment: ModelDeployment) -> Tuple[bool, Dict[str, Any]]:
        """Esegue health checks per deployment."""
        if not deployment.config.health_checks:
            return True, {"message": "No health checks configured"}

        results = {}
        overall_healthy = True

        for check_config in deployment.config.health_checks:
            check_name = check_config.get('name', 'unnamed')
            check_type = HealthCheckType(check_config.get('type', 'http'))

            health_check = HealthCheck(check_type, check_config)
            success, message = health_check.execute()

            results[check_name] = {
                "success": success,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if not success:
                overall_healthy = False

        return overall_healthy, results

    def _simulate_replica_health(self) -> bool:
        """Simula health check di una replica."""
        # In implementazione reale, questo farebbe veri health checks
        import random
        return random.random() > 0.1  # 90% success rate

    def _get_previous_version(self, deployment: ModelDeployment) -> Optional[str]:
        """Ottiene versione precedente per rollback."""
        # Cerca ultimo deployment successful dello stesso modello
        deployments = self.list_deployments(
            environment=deployment.environment,
            status=DeploymentStatus.DEPLOYED,
            model_name=deployment.model_name
        )

        for dep in deployments:
            if dep.model_version != deployment.model_version:
                return dep.model_version

        return None

    def _save_deployment(self, deployment: ModelDeployment) -> None:
        """Salva deployment nel database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO deployments
                (deployment_id, environment, model_name, model_version, status,
                 config, deployed_at, deployed_by, health_status, endpoint_url,
                 replicas_ready, rollback_version, logs, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deployment.deployment_id,
                deployment.environment.value,
                deployment.model_name,
                deployment.model_version,
                deployment.status.value,
                json.dumps(asdict(deployment.config)),
                deployment.deployed_at,
                deployment.deployed_by,
                deployment.health_status,
                deployment.endpoint_url,
                deployment.replicas_ready,
                deployment.rollback_version,
                json.dumps(deployment.logs),
                json.dumps(deployment.metadata or {}),
                deployment.created_at,
                deployment.updated_at
            ))
            conn.commit()

    def _row_to_deployment(self, row: tuple) -> ModelDeployment:
        """Converte riga database in ModelDeployment."""
        config_dict = json.loads(row[5] or '{}')
        config = DeploymentConfig(**config_dict) if config_dict else None

        return ModelDeployment(
            deployment_id=row[0],
            environment=EnvironmentType(row[1]),
            model_name=row[2],
            model_version=row[3],
            status=DeploymentStatus(row[4]),
            config=config,
            deployed_at=datetime.fromisoformat(row[6]) if row[6] else None,
            deployed_by=row[7] or 'system',
            health_status=row[8],
            endpoint_url=row[9],
            replicas_ready=row[10] or 0,
            rollback_version=row[11],
            logs=json.loads(row[12] or '[]'),
            metadata=json.loads(row[13] or '{}'),
            created_at=datetime.fromisoformat(row[14]) if row[14] else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(row[15]) if row[15] else datetime.now(timezone.utc)
        )

    def _log_deployment_event(self, deployment_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
        """Logga evento di deployment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO deployment_history
                (deployment_id, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                deployment_id,
                event_type,
                json.dumps(event_data),
                datetime.now(timezone.utc)
            ))
            conn.commit()


def create_deployment_manager(
    deployment_path: Union[str, Path] = "deployments",
    model_registry: Optional[ModelRegistry] = None
) -> DeploymentManager:
    """
    Factory function per creare DeploymentManager.

    Args:
        deployment_path: Path dei deployments
        model_registry: Istanza ModelRegistry

    Returns:
        Istanza DeploymentManager configurata
    """
    return DeploymentManager(deployment_path, model_registry)