# ============================================
# FILE MLOps EXPERIMENT TRACKING
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Experiment tracking e logging system
# ============================================

"""
Experiment Tracking per ARIMA Forecaster

Sistema di tracking esperimenti per MLOps:
- Logging parametri, metriche, e artifacts
- Comparazione esperimenti
- Tracking dataset versions
- Integration con model registry
- Export risultati per reporting
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import shutil

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    """Status di un esperimento."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(str, Enum):
    """Status di un run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Experiment:
    """Metadata per un esperimento."""

    experiment_id: str
    name: str
    description: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    status: ExperimentStatus
    created_by: str
    runs_count: int = 0
    best_run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentRun:
    """Metadata per un singolo run di esperimento."""

    run_id: str
    experiment_id: str
    name: Optional[str]
    status: RunStatus
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    artifacts: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    model_type: Optional[str]
    dataset_hash: Optional[str]
    git_commit: Optional[str]
    created_by: str
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExperimentRunRequest(BaseModel):
    """Request per creare un nuovo run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment_id: str
    name: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    model_type: Optional[str] = None
    notes: Optional[str] = None


class ExperimentTracker:
    """
    Sistema di tracking esperimenti per ARIMA forecasting.

    Features:
    - Tracking parametri, metriche, artifacts
    - Comparazione esperimenti e runs
    - Integration con model registry
    - Export risultati e reporting
    - Dataset versioning
    - Git integration per code versioning
    """

    def __init__(self, tracking_path: Union[str, Path] = "experiments_tracking"):
        """
        Inizializza Experiment Tracker.

        Args:
            tracking_path: Path base per tracking (default: experiments_tracking)
        """
        self.tracking_path = Path(tracking_path)
        self.tracking_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.tracking_path / "experiments.db"
        self.artifacts_path = self.tracking_path / "artifacts"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Experiment Tracker inizializzato: {self.tracking_path}")

    def _init_database(self) -> None:
        """Inizializza database SQLite per experiments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabella experiments
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,  -- JSON array
                    status TEXT NOT NULL,
                    created_by TEXT,
                    runs_count INTEGER DEFAULT 0,
                    best_run_id TEXT,
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            # Tabella runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    name TEXT,
                    status TEXT NOT NULL,
                    parameters TEXT,  -- JSON
                    metrics TEXT,  -- JSON
                    tags TEXT,  -- JSON array
                    artifacts TEXT,  -- JSON array
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds REAL,
                    model_type TEXT,
                    dataset_hash TEXT,
                    git_commit TEXT,
                    created_by TEXT,
                    notes TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)

            # Tabella metrics history (per tracking nel tempo)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            """)

            # Indici per performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics_history(run_id)")

            conn.commit()

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """
        Crea un nuovo esperimento.

        Args:
            name: Nome dell'esperimento
            description: Descrizione
            tags: Tag per categorizzazione
            created_by: Autore dell'esperimento
            metadata: Metadata aggiuntivi

        Returns:
            Experiment creato
        """
        if tags is None:
            tags = []

        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tags,
            created_at=now,
            updated_at=now,
            status=ExperimentStatus.RUNNING,
            created_by=created_by,
            runs_count=0,
            metadata=metadata,
        )

        # Salva in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments
                (experiment_id, name, description, tags, status, created_by,
                 runs_count, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.experiment_id,
                    experiment.name,
                    experiment.description,
                    json.dumps(experiment.tags),
                    experiment.status.value,
                    experiment.created_by,
                    experiment.runs_count,
                    json.dumps(experiment.metadata or {}),
                    experiment.created_at,
                    experiment.updated_at,
                ),
            )
            conn.commit()

        logger.info(f"Esperimento creato: {name} ({experiment_id})")
        return experiment

    def start_run(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        created_by: str = "system",
        notes: Optional[str] = None,
    ) -> ExperimentRun:
        """
        Inizia un nuovo run per un esperimento.

        Args:
            experiment_id: ID dell'esperimento
            name: Nome del run
            parameters: Parametri del modello/esperimento
            tags: Tag per il run
            model_type: Tipo di modello (ARIMA, SARIMA, ecc.)
            created_by: Autore del run
            notes: Note aggiuntive

        Returns:
            ExperimentRun creato
        """
        if parameters is None:
            parameters = {}
        if tags is None:
            tags = []

        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            name=name or f"run_{now.strftime('%Y%m%d_%H%M%S')}",
            status=RunStatus.RUNNING,
            parameters=parameters,
            metrics={},
            tags=tags,
            artifacts=[],
            start_time=now,
            end_time=None,
            duration_seconds=None,
            model_type=model_type,
            dataset_hash=None,
            git_commit=self._get_git_commit(),
            created_by=created_by,
            notes=notes,
        )

        # Salva in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs
                (run_id, experiment_id, name, status, parameters, metrics, tags,
                 artifacts, start_time, model_type, git_commit, created_by, notes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run.run_id,
                    run.experiment_id,
                    run.name,
                    run.status.value,
                    json.dumps(run.parameters),
                    json.dumps(run.metrics),
                    json.dumps(run.tags),
                    json.dumps(run.artifacts),
                    run.start_time,
                    run.model_type,
                    run.git_commit,
                    run.created_by,
                    run.notes,
                    json.dumps({}),
                ),
            )

            # Aggiorna contatore runs nell'esperimento
            cursor.execute(
                """
                UPDATE experiments
                SET runs_count = runs_count + 1, updated_at = ?
                WHERE experiment_id = ?
            """,
                (now, experiment_id),
            )

            conn.commit()

        logger.info(f"Run iniziato: {run.name} ({run_id})")
        return run

    def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Logga metriche per un run.

        Args:
            run_id: ID del run
            metrics: Dizionario metriche
            step: Step number (per training progressivo)
        """
        now = datetime.now(timezone.utc)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Aggiorna metriche nel run
            cursor.execute("SELECT metrics FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                current_metrics = json.loads(row[0] or "{}")
                current_metrics.update(metrics)

                cursor.execute(
                    "UPDATE runs SET metrics = ? WHERE run_id = ?",
                    (json.dumps(current_metrics), run_id),
                )

                # Logga in metrics_history per tracking nel tempo
                for metric_name, metric_value in metrics.items():
                    cursor.execute(
                        """
                        INSERT INTO metrics_history
                        (run_id, metric_name, metric_value, step, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (run_id, metric_name, metric_value, step, now),
                    )

                conn.commit()

        logger.debug(f"Metriche loggate per run {run_id}: {metrics}")

    def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """
        Logga parametri per un run.

        Args:
            run_id: ID del run
            parameters: Dizionario parametri
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Merge con parametri esistenti
            cursor.execute("SELECT parameters FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                current_params = json.loads(row[0] or "{}")
                current_params.update(parameters)

                cursor.execute(
                    "UPDATE runs SET parameters = ? WHERE run_id = ?",
                    (json.dumps(current_params), run_id),
                )
                conn.commit()

        logger.debug(f"Parametri loggati per run {run_id}: {parameters}")

    def log_artifact(
        self, run_id: str, artifact_path: Union[str, Path], artifact_name: Optional[str] = None
    ) -> str:
        """
        Logga un artifact per un run.

        Args:
            run_id: ID del run
            artifact_path: Path del file artifact
            artifact_name: Nome custom per l'artifact

        Returns:
            Path finale dell'artifact
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact non trovato: {artifact_path}")

        # Crea directory per il run
        run_artifacts_path = self.artifacts_path / run_id
        run_artifacts_path.mkdir(parents=True, exist_ok=True)

        # Copia artifact
        artifact_name = artifact_name or artifact_path.name
        target_path = run_artifacts_path / artifact_name
        shutil.copy2(artifact_path, target_path)

        # Aggiorna database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT artifacts FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                artifacts = json.loads(row[0] or "[]")
                artifacts.append(str(target_path))

                cursor.execute(
                    "UPDATE runs SET artifacts = ? WHERE run_id = ?",
                    (json.dumps(artifacts), run_id),
                )
                conn.commit()

        logger.info(f"Artifact salvato: {artifact_name} per run {run_id}")
        return str(target_path)

    def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> ExperimentRun:
        """
        Termina un run.

        Args:
            run_id: ID del run
            status: Status finale del run

        Returns:
            ExperimentRun aggiornato
        """
        now = datetime.now(timezone.utc)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Calcola durata
            cursor.execute("SELECT start_time FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            duration_seconds = None
            if row and row[0]:
                start_time = datetime.fromisoformat(row[0])
                duration_seconds = (now - start_time).total_seconds()

            cursor.execute(
                """
                UPDATE runs
                SET status = ?, end_time = ?, duration_seconds = ?
                WHERE run_id = ?
            """,
                (status.value, now, duration_seconds, run_id),
            )

            conn.commit()

        run = self.get_run(run_id)
        logger.info(f"Run terminato: {run_id} ({status.value})")
        return run

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Ottiene un esperimento per ID.

        Args:
            experiment_id: ID dell'esperimento

        Returns:
            Experiment se trovato, None altrimenti
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_experiment(row)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Ottiene un run per ID.

        Args:
            run_id: ID del run

        Returns:
            ExperimentRun se trovato, None altrimenti
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_run(row)

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """
        Lista tutti gli esperimenti.

        Args:
            status: Filtra per status specifico

        Returns:
            Lista di Experiment
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM experiments"
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_experiment(row) for row in rows]

    def list_runs(
        self, experiment_id: Optional[str] = None, status: Optional[RunStatus] = None
    ) -> List[ExperimentRun]:
        """
        Lista runs.

        Args:
            experiment_id: Filtra per esperimento specifico
            status: Filtra per status specifico

        Returns:
            Lista di ExperimentRun
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM runs WHERE 1=1"
            params = []

            if experiment_id:
                query += " AND experiment_id = ?"
                params.append(experiment_id)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY start_time DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_run(row) for row in rows]

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compara runs multiple.

        Args:
            run_ids: Lista di run IDs da comparare

        Returns:
            DataFrame con comparazione
        """
        runs_data = []

        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                row_data = {
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "name": run.name,
                    "status": run.status.value,
                    "model_type": run.model_type,
                    "duration_seconds": run.duration_seconds,
                    "start_time": run.start_time,
                }

                # Aggiungi parametri
                for param, value in run.parameters.items():
                    row_data[f"param_{param}"] = value

                # Aggiungi metriche
                for metric, value in run.metrics.items():
                    row_data[f"metric_{metric}"] = value

                runs_data.append(row_data)

        return pd.DataFrame(runs_data)

    def get_best_run(
        self, experiment_id: str, metric_name: str, maximize: bool = True
    ) -> Optional[ExperimentRun]:
        """
        Ottiene il miglior run per una metrica.

        Args:
            experiment_id: ID dell'esperimento
            metric_name: Nome della metrica
            maximize: True per massimizzare, False per minimizzare

        Returns:
            Miglior ExperimentRun
        """
        runs = self.list_runs(experiment_id=experiment_id, status=RunStatus.COMPLETED)

        if not runs:
            return None

        best_run = None
        best_value = None

        for run in runs:
            if metric_name in run.metrics:
                value = run.metrics[metric_name]
                if (
                    best_value is None
                    or (maximize and value > best_value)
                    or (not maximize and value < best_value)
                ):
                    best_value = value
                    best_run = run

        return best_run

    def export_experiment(self, experiment_id: str, export_path: Union[str, Path]) -> None:
        """
        Esporta esperimento completo.

        Args:
            experiment_id: ID dell'esperimento
            export_path: Path del file di export
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Esperimento non trovato: {experiment_id}")

        runs = self.list_runs(experiment_id=experiment_id)

        export_data = {
            "experiment": asdict(experiment),
            "runs": [asdict(run) for run in runs],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        export_path = Path(export_path)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Esperimento esportato: {experiment_id} -> {export_path}")

    # ============================================
    # METODI PRIVATI
    # ============================================

    def _get_git_commit(self) -> Optional[str]:
        """Ottiene commit Git corrente."""
        try:
            import subprocess

            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _row_to_experiment(self, row: tuple) -> Experiment:
        """Converte riga database in Experiment."""
        return Experiment(
            experiment_id=row[0],
            name=row[1],
            description=row[2] or "",
            tags=json.loads(row[3] or "[]"),
            status=ExperimentStatus(row[4]),
            created_by=row[5] or "system",
            runs_count=row[6] or 0,
            best_run_id=row[7],
            metadata=json.loads(row[8] or "{}"),
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(row[10]) if row[10] else datetime.now(timezone.utc),
        )

    def _row_to_run(self, row: tuple) -> ExperimentRun:
        """Converte riga database in ExperimentRun."""
        return ExperimentRun(
            run_id=row[0],
            experiment_id=row[1],
            name=row[2],
            status=RunStatus(row[3]),
            parameters=json.loads(row[4] or "{}"),
            metrics=json.loads(row[5] or "{}"),
            tags=json.loads(row[6] or "[]"),
            artifacts=json.loads(row[7] or "[]"),
            start_time=datetime.fromisoformat(row[8]) if row[8] else datetime.now(timezone.utc),
            end_time=datetime.fromisoformat(row[9]) if row[9] else None,
            duration_seconds=row[10],
            model_type=row[11],
            dataset_hash=row[12],
            git_commit=row[13],
            created_by=row[14] or "system",
            notes=row[15],
            metadata=json.loads(row[16] or "{}"),
        )


def create_experiment_tracker(
    tracking_path: Union[str, Path] = "experiments_tracking",
) -> ExperimentTracker:
    """
    Factory function per creare ExperimentTracker.

    Args:
        tracking_path: Path del tracking

    Returns:
        Istanza ExperimentTracker configurata
    """
    return ExperimentTracker(tracking_path)
