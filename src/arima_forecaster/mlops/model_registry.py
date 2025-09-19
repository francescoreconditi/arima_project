# ============================================
# FILE MLOps MODEL REGISTRY
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Model versioning e registry system
# ============================================

"""
Model Registry per ARIMA Forecaster

Sistema di versioning e gestione modelli enterprise-grade:
- Registrazione modelli con metadata
- Versioning automatico con semantic versioning
- Tracking performance e metriche
- Staging environments (dev/staging/production)
- Model lineage e dependencies
"""

import json
import pickle
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class ModelStage(str, Enum):
    """Stage del modello nel lifecycle."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(str, Enum):
    """Status del modello."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata per un modello registrato."""

    model_name: str
    version: str
    model_type: str  # "ARIMA", "SARIMA", "VAR", ecc.
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    tags: List[str]
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    stage: ModelStage
    status: ModelStatus
    dataset_hash: Optional[str] = None
    model_hash: Optional[str] = None
    file_path: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None


class ModelVersion(BaseModel):
    """Pydantic model per validazione model version."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Nome del modello")
    version: str = Field(..., description="Versione (semantic versioning)")
    type: str = Field(..., description="Tipo modello (ARIMA, SARIMA, VAR)")
    stage: ModelStage = Field(default=ModelStage.DEVELOPMENT)
    status: ModelStatus = Field(default=ModelStatus.ACTIVE)
    performance: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRegistry:
    """
    Sistema di registry per modelli ARIMA con versioning enterprise.

    Features:
    - SQLite backend per metadata storage
    - Semantic versioning automatico
    - Performance tracking
    - Stage management (dev/staging/prod)
    - Model lineage e dependencies
    - Export/import per disaster recovery
    """

    def __init__(self, registry_path: Union[str, Path] = "models_registry"):
        """
        Inizializza Model Registry.

        Args:
            registry_path: Path base per registry (default: models_registry)
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.registry_path / "registry.db"
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Model Registry inizializzato: {self.registry_path}")

    def _init_database(self) -> None:
        """Inizializza database SQLite per metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabella modelli
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parameters TEXT,  -- JSON
                    performance TEXT,  -- JSON
                    metadata TEXT,  -- JSON
                    tags TEXT,  -- JSON array
                    description TEXT,
                    author TEXT,
                    dataset_hash TEXT,
                    model_hash TEXT,
                    file_path TEXT,
                    dependencies TEXT,  -- JSON
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    UNIQUE(name, version)
                )
            """)

            # Tabella deployment history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    deployed_at TIMESTAMP,
                    deployed_by TEXT,
                    rollback_version TEXT,
                    deployment_config TEXT,  -- JSON
                    FOREIGN KEY (model_name, model_version) REFERENCES models (name, version)
                )
            """)

            # Indici per performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)")

            conn.commit()

    def register_model(
        self,
        model: Any,
        name: str,
        model_type: str,
        version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: str = "system",
        dataset_hash: Optional[str] = None,
        dependencies: Optional[Dict[str, str]] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
    ) -> ModelMetadata:
        """
        Registra un nuovo modello nel registry.

        Args:
            model: Istanza del modello da registrare
            name: Nome del modello
            model_type: Tipo modello (ARIMA, SARIMA, VAR)
            version: Versione (auto-increment se None)
            parameters: Parametri del modello
            performance_metrics: Metriche di performance
            description: Descrizione del modello
            tags: Tag per categorizzazione
            author: Autore del modello
            dataset_hash: Hash del dataset di training
            dependencies: Dipendenze software
            stage: Stage del modello

        Returns:
            ModelMetadata del modello registrato
        """
        if version is None:
            version = self._get_next_version(name)

        if tags is None:
            tags = []

        if parameters is None:
            parameters = {}

        if performance_metrics is None:
            performance_metrics = {}

        # Genera hash del modello
        model_hash = self._compute_model_hash(model)

        # Salva modello su filesystem
        file_path = self._save_model_file(model, name, version)

        # Crea metadata
        now = datetime.now(timezone.utc)
        metadata = ModelMetadata(
            model_name=name,
            version=version,
            model_type=model_type,
            parameters=parameters,
            performance_metrics=performance_metrics,
            tags=tags,
            description=description,
            author=author,
            created_at=now,
            updated_at=now,
            stage=stage,
            status=ModelStatus.ACTIVE,
            dataset_hash=dataset_hash,
            model_hash=model_hash,
            file_path=str(file_path),
            dependencies=dependencies,
        )

        # Salva in database
        self._save_metadata(metadata)

        logger.info(f"Modello registrato: {name} v{version} ({model_type})")
        return metadata

    def get_model(
        self, name: str, version: Optional[str] = None, stage: Optional[ModelStage] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Carica un modello dal registry.

        Args:
            name: Nome del modello
            version: Versione specifica (latest se None)
            stage: Stage specifico (qualsiasi se None)

        Returns:
            Tupla (modello, metadata)
        """
        metadata = self.get_model_metadata(name, version, stage)
        if not metadata:
            raise ValueError(f"Modello non trovato: {name} v{version or 'latest'}")

        # Carica modello da file
        model_path = Path(metadata.file_path)
        if not model_path.exists():
            raise FileNotFoundError(f"File modello non trovato: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Modello caricato: {name} v{metadata.version}")
        return model, metadata

    def get_model_metadata(
        self, name: str, version: Optional[str] = None, stage: Optional[ModelStage] = None
    ) -> Optional[ModelMetadata]:
        """
        Ottiene metadata di un modello.

        Args:
            name: Nome del modello
            version: Versione specifica (latest se None)
            stage: Stage specifico

        Returns:
            ModelMetadata se trovato, None altrimenti
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if version:
                # Versione specifica
                query = "SELECT * FROM models WHERE name = ? AND version = ?"
                params = [name, version]

                if stage:
                    query += " AND stage = ?"
                    params.append(stage.value)

            else:
                # Ultima versione
                query = """
                    SELECT * FROM models
                    WHERE name = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                params = [name]

                if stage:
                    query = """
                        SELECT * FROM models
                        WHERE name = ? AND stage = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    params.append(stage.value)

            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_metadata(row)

    def list_models(
        self, stage: Optional[ModelStage] = None, status: Optional[ModelStatus] = None
    ) -> List[ModelMetadata]:
        """
        Lista tutti i modelli nel registry.

        Args:
            stage: Filtra per stage specifico
            status: Filtra per status specifico

        Returns:
            Lista di ModelMetadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM models WHERE 1=1"
            params = []

            if stage:
                query += " AND stage = ?"
                params.append(stage.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY name, created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_metadata(row) for row in rows]

    def promote_model(
        self, name: str, version: str, target_stage: ModelStage, author: str = "system"
    ) -> ModelMetadata:
        """
        Promuove un modello a uno stage superiore.

        Args:
            name: Nome del modello
            version: Versione del modello
            target_stage: Stage target
            author: Autore della promozione

        Returns:
            ModelMetadata aggiornato
        """
        metadata = self.get_model_metadata(name, version)
        if not metadata:
            raise ValueError(f"Modello non trovato: {name} v{version}")

        # Registra deployment
        self._log_deployment(name, version, target_stage, author)

        # Aggiorna stage
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE models SET stage = ?, updated_at = ? WHERE name = ? AND version = ?",
                (target_stage.value, datetime.now(timezone.utc), name, version),
            )
            conn.commit()

        logger.info(f"Modello promosso: {name} v{version} -> {target_stage.value}")
        return self.get_model_metadata(name, version)

    def update_performance(
        self, name: str, version: str, metrics: Dict[str, float]
    ) -> ModelMetadata:
        """
        Aggiorna metriche di performance di un modello.

        Args:
            name: Nome del modello
            version: Versione del modello
            metrics: Nuove metriche

        Returns:
            ModelMetadata aggiornato
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE models SET performance = ?, updated_at = ? WHERE name = ? AND version = ?",
                (json.dumps(metrics), datetime.now(timezone.utc), name, version),
            )
            conn.commit()

        logger.info(f"Performance aggiornate: {name} v{version}")
        return self.get_model_metadata(name, version)

    def delete_model(self, name: str, version: str, hard_delete: bool = False) -> bool:
        """
        Elimina un modello dal registry.

        Args:
            name: Nome del modello
            version: Versione del modello
            hard_delete: Se True, elimina fisicamente il file

        Returns:
            True se eliminato con successo
        """
        metadata = self.get_model_metadata(name, version)
        if not metadata:
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if hard_delete:
                # Elimina record dal database
                cursor.execute("DELETE FROM models WHERE name = ? AND version = ?", (name, version))

                # Elimina file se esiste
                if metadata.file_path and Path(metadata.file_path).exists():
                    Path(metadata.file_path).unlink()

            else:
                # Soft delete - marca come archived
                cursor.execute(
                    "UPDATE models SET status = ?, updated_at = ? WHERE name = ? AND version = ?",
                    (ModelStatus.DEPRECATED.value, datetime.now(timezone.utc), name, version),
                )

            conn.commit()

        logger.info(f"Modello {'eliminato' if hard_delete else 'deprecato'}: {name} v{version}")
        return True

    def export_registry(self, export_path: Union[str, Path]) -> None:
        """
        Esporta tutto il registry per backup.

        Args:
            export_path: Path del file di export
        """
        export_path = Path(export_path)

        # Esporta metadata
        models = self.list_models()
        metadata_export = [asdict(model) for model in models]

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "registry_version": "1.0",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "models": metadata_export,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Registry esportato: {export_path}")

    def get_model_lineage(self, name: str, version: str) -> Dict[str, Any]:
        """
        Ottiene lineage di un modello (dependencies, deployments, performance history).

        Args:
            name: Nome del modello
            version: Versione del modello

        Returns:
            Dizionario con informazioni di lineage
        """
        metadata = self.get_model_metadata(name, version)
        if not metadata:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Deployment history
            cursor.execute(
                "SELECT * FROM deployments WHERE model_name = ? AND model_version = ? ORDER BY deployed_at DESC",
                (name, version),
            )
            deployments = [
                dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()
            ]

            # Tutte le versioni del modello
            cursor.execute(
                "SELECT version, stage, status, created_at, performance FROM models WHERE name = ? ORDER BY created_at",
                (name,),
            )
            versions = [
                dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()
            ]

        return {
            "model": asdict(metadata),
            "deployments": deployments,
            "all_versions": versions,
            "dependencies": metadata.dependencies or {},
        }

    # ============================================
    # METODI PRIVATI
    # ============================================

    def _get_next_version(self, name: str) -> str:
        """Genera la prossima versione semantica per un modello."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,),
            )
            row = cursor.fetchone()

            if not row:
                return "1.0.0"

            # Parse semantic version
            last_version = row[0]
            try:
                major, minor, patch = map(int, last_version.split("."))
                return f"{major}.{minor}.{patch + 1}"
            except:
                # Fallback per versioni non semantic
                return f"1.0.{len(self.list_models()) + 1}"

    def _compute_model_hash(self, model: Any) -> str:
        """Calcola hash MD5 del modello serializzato."""
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.md5(model_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"Impossibile calcolare hash modello: {e}")
            return "unknown"

    def _save_model_file(self, model: Any, name: str, version: str) -> Path:
        """Salva modello su filesystem."""
        filename = f"{name}_v{version.replace('.', '_')}.pkl"
        file_path = self.models_path / filename

        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        return file_path

    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Salva metadata nel database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO models
                (name, version, type, stage, status, parameters, performance, metadata,
                 tags, description, author, dataset_hash, model_hash, file_path,
                 dependencies, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.model_name,
                    metadata.version,
                    metadata.model_type,
                    metadata.stage.value,
                    metadata.status.value,
                    json.dumps(metadata.parameters),
                    json.dumps(metadata.performance_metrics),
                    json.dumps({}),  # metadata extra
                    json.dumps(metadata.tags),
                    metadata.description,
                    metadata.author,
                    metadata.dataset_hash,
                    metadata.model_hash,
                    metadata.file_path,
                    json.dumps(metadata.dependencies or {}),
                    metadata.created_at,
                    metadata.updated_at,
                ),
            )
            conn.commit()

    def _row_to_metadata(self, row: tuple) -> ModelMetadata:
        """Converte riga database in ModelMetadata."""
        return ModelMetadata(
            model_name=row[1],
            version=row[2],
            model_type=row[3],
            stage=ModelStage(row[4]),
            status=ModelStatus(row[5]),
            parameters=json.loads(row[6] or "{}"),
            performance_metrics=json.loads(row[7] or "{}"),
            tags=json.loads(row[9] or "[]"),
            description=row[10] or "",
            author=row[11] or "system",
            dataset_hash=row[12],
            model_hash=row[13],
            file_path=row[14],
            dependencies=json.loads(row[15] or "{}"),
            created_at=datetime.fromisoformat(row[16]) if row[16] else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(row[17]) if row[17] else datetime.now(timezone.utc),
        )

    def _log_deployment(self, name: str, version: str, stage: ModelStage, author: str) -> None:
        """Registra deployment nel log."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO deployments
                (model_name, model_version, stage, deployed_at, deployed_by, deployment_config)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (name, version, stage.value, datetime.now(timezone.utc), author, json.dumps({})),
            )
            conn.commit()


def create_model_registry(registry_path: Union[str, Path] = "models_registry") -> ModelRegistry:
    """
    Factory function per creare ModelRegistry.

    Args:
        registry_path: Path del registry

    Returns:
        Istanza ModelRegistry configurata
    """
    return ModelRegistry(registry_path)
