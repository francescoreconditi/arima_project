"""
Router per gestione dati e preprocessing avanzato.

Questo modulo fornisce endpoint per caricamento, validazione, preprocessing
e gestione del ciclo di vita dei dataset per analisi di serie temporali.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import json
import uuid
import asyncio
from io import StringIO

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator

# Simulazione import delle utilità (da implementare nel progetto reale)
from arima_forecaster.utils.logger import get_logger

# Configurazione router
router = APIRouter(
    prefix="/data",
    tags=["Data Management"],
)

logger = get_logger(__name__)

# Storage globale simulato per dataset e jobs
datasets_storage = {}
preprocessing_jobs = {}


class DatasetMetadata(BaseModel):
    """Metadati dataset caricato."""
    
    dataset_id: str = Field(..., description="ID univoco dataset")
    name: str = Field(..., description="Nome descrittivo dataset")
    rows: int = Field(..., ge=1, description="Numero righe")
    columns: int = Field(..., ge=1, description="Numero colonne") 
    size_bytes: int = Field(..., ge=0, description="Dimensione in bytes")
    upload_timestamp: datetime = Field(..., description="Timestamp caricamento")
    file_format: str = Field(..., description="Formato file originale")
    column_info: Dict[str, Any] = Field(..., description="Informazioni colonne")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Score qualità dati")
    missing_values_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio valori mancanti")


class DataUploadRequest(BaseModel):
    """Configurazione caricamento dati."""
    
    dataset_name: str = Field(..., description="Nome descrittivo dataset")
    date_column: Optional[str] = Field(None, description="Nome colonna date/timestamp")
    value_columns: List[str] = Field(..., description="Nome colonne valori numerici")
    separator: str = Field(default=",", description="Separatore CSV")
    date_format: Optional[str] = Field(None, description="Formato date custom")
    encoding: str = Field(default="utf-8", description="Encoding file")
    skip_rows: int = Field(default=0, ge=0, description="Righe da saltare all'inizio")
    validate_data: bool = Field(default=True, description="Eseguire validazione automatica")


class DataValidationRequest(BaseModel):
    """Configurazione validazione dati."""
    
    dataset_id: str = Field(..., description="ID dataset da validare")
    validation_rules: Dict[str, Any] = Field(
        default={
            "check_missing_values": True,
            "check_outliers": True,
            "check_duplicates": True,
            "check_data_types": True,
            "check_time_series_properties": True
        },
        description="Regole validazione da applicare"
    )
    outlier_method: str = Field(
        default="iqr",
        description="Metodo detection outlier: iqr, zscore, modified_zscore"
    )
    outlier_threshold: float = Field(default=3.0, ge=1.0, description="Soglia detection outlier")


class PreprocessingRequest(BaseModel):
    """Configurazione preprocessing avanzato."""
    
    dataset_id: str = Field(..., description="ID dataset da preprocessare")
    preprocessing_steps: List[Dict[str, Any]] = Field(..., description="Step preprocessing sequenziali")
    output_dataset_name: str = Field(..., description="Nome dataset output")
    preserve_original: bool = Field(default=True, description="Mantenere dataset originale")
    
    @validator('preprocessing_steps')
    def validate_steps(cls, v):
        valid_steps = {
            "handle_missing", "remove_outliers", "smooth_data", 
            "difference", "log_transform", "normalize", "resample"
        }
        for step in v:
            if step.get("type") not in valid_steps:
                raise ValueError(f"Step type non valido: {step.get('type')}")
        return v


class DataExploreRequest(BaseModel):
    """Configurazione esplorazione dati."""
    
    dataset_id: str = Field(..., description="ID dataset da esplorare")
    analysis_type: str = Field(
        default="comprehensive",
        description="Tipo analisi: basic, comprehensive, statistical, time_series"
    )
    include_visualizations: bool = Field(default=True, description="Includere visualizzazioni")
    statistical_tests: List[str] = Field(
        default=["stationarity", "seasonality", "normality"],
        description="Test statistici da eseguire"
    )


class DataSplitRequest(BaseModel):
    """Configurazione split train/test."""
    
    dataset_id: str = Field(..., description="ID dataset da dividere")
    split_method: str = Field(
        default="temporal",
        description="Metodo split: temporal, random, stratified"
    )
    train_ratio: float = Field(default=0.8, ge=0.1, le=0.9, description="Ratio training set")
    validation_ratio: Optional[float] = Field(
        None, ge=0.0, le=0.5, 
        description="Ratio validation set (opzionale)"
    )
    split_date: Optional[date] = Field(None, description="Data split per metodo temporal")


class DataQualityRequest(BaseModel):
    """Configurazione analisi qualità dati."""
    
    dataset_id: str = Field(..., description="ID dataset da analizzare")
    quality_dimensions: List[str] = Field(
        default=["completeness", "consistency", "accuracy", "validity", "timeliness"],
        description="Dimensioni qualità da valutare"
    )
    generate_report: bool = Field(default=True, description="Generare report qualità")
    suggest_improvements: bool = Field(
        default=True, 
        description="Suggerire miglioramenti automatici"
    )


class DataJobResponse(BaseModel):
    """Risposta operazione dati."""
    
    job_id: str = Field(..., description="ID univoco job")
    status: str = Field(..., description="Stato: queued, running, completed, failed")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progresso job")
    dataset_id: Optional[str] = Field(None, description="ID dataset risultante")
    results: Dict[str, Any] = Field(default={}, description="Risultati operazione")
    error_message: Optional[str] = Field(None, description="Messaggio errore se failed")


@router.post(
    "/upload",
    response_model=DataJobResponse,
    summary="Carica Dataset da File",
    description="""
    Carica dataset CSV/Excel con validazione automatica e parsing intelligente per time series.
    
    <h4>Formati Supportati:</h4>
    <table>
        <tr><th>Formato</th><th>Estensioni</th><th>Features</th></tr>
        <tr><td>CSV</td><td>.csv, .tsv</td><td>Auto-detect separatore</td></tr>
        <tr><td>Excel</td><td>.xlsx, .xls</td><td>Multiple sheets</td></tr>
        <tr><td>JSON</td><td>.json</td><td>Time-series structure</td></tr>
    </table>
    
    <h4>Esempio Upload:</h4>
    <pre><code>
    {
        "dataset_name": "Sales Data Q4",
        "value_columns": ["sales", "units"],
        "date_column": "timestamp",
        "validate_data": true
    }
    </code></pre>
    """,
)
async def upload_dataset(
    file: UploadFile = File(...),
    config: str = Query(..., description="Configurazione upload JSON-encoded"),
    background_tasks: BackgroundTasks = None
):
    """Carica dataset da file con validazione automatica."""
    
    try:
        # Parse configurazione
        upload_config = DataUploadRequest.parse_raw(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configurazione non valida: {str(e)}")
    
    job_id = f"upload_job_{uuid.uuid4().hex[:8]}"
    dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
    
    # Inizializza job tracking
    preprocessing_jobs[job_id] = {
        "status": "running",
        "progress": 0.1,
        "operation": "upload",
        "dataset_id": dataset_id,
        "created_at": datetime.now(),
        "results": {}
    }
    
    try:
        # Simula lettura file
        content = await file.read()
        file_size = len(content)
        
        # Simula parsing e validazione
        await asyncio.sleep(1)  # Simula processing
        preprocessing_jobs[job_id]["progress"] = 0.5
        
        # Simula inferenza schema e validazione
        await asyncio.sleep(1)
        preprocessing_jobs[job_id]["progress"] = 0.8
        
        # Simula salvataggio dataset
        simulated_rows = 1000
        simulated_cols = len(upload_config.value_columns) + (1 if upload_config.date_column else 0)
        
        # Metadati dataset simulati
        dataset_metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=upload_config.dataset_name,
            rows=simulated_rows,
            columns=simulated_cols,
            size_bytes=file_size,
            upload_timestamp=datetime.now(),
            file_format=file.filename.split('.')[-1].lower(),
            column_info={
                "date_column": upload_config.date_column,
                "value_columns": upload_config.value_columns,
                "inferred_types": {col: "float64" for col in upload_config.value_columns}
            },
            data_quality_score=0.85,
            missing_values_ratio=0.05
        )
        
        # Salva in storage simulato
        datasets_storage[dataset_id] = dataset_metadata
        
        preprocessing_jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "results": {
                "rows_loaded": simulated_rows,
                "columns_detected": simulated_cols,
                "data_quality_score": 0.85,
                "issues_found": ["missing_values_detected"] if dataset_metadata.missing_values_ratio > 0 else [],
                "file_size_mb": round(file_size / (1024*1024), 2)
            }
        })
        
        logger.info(f"Dataset {dataset_id} caricato con successo: {simulated_rows} righe, {simulated_cols} colonne")
        
    except Exception as e:
        preprocessing_jobs[job_id].update({
            "status": "failed",
            "error_message": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Errore caricamento: {str(e)}")
    
    return DataJobResponse(
        job_id=job_id,
        status="completed",
        progress=1.0,
        dataset_id=dataset_id,
        results=preprocessing_jobs[job_id]["results"]
    )


@router.post(
    "/validate",
    response_model=DataJobResponse,
    summary="Valida Qualità Dati",
    description="""Esegue validazione approfondita qualità dati con detection automatica anomalie.
    
    <h4>Validation Checks:</h4>
    <table>
        <tr><th>Check</th><th>Descrizione</th><th>Metodo</th></tr>
        <tr><td>Missing Values</td><td>Pattern valori mancanti</td><td>Count + percentage</td></tr>
        <tr><td>Outliers</td><td>Detection anomalie</td><td>IQR, Z-score, Isolation Forest</td></tr>
        <tr><td>Duplicates</td><td>Record duplicati</td><td>Hash-based detection</td></tr>
        <tr><td>Time Series</td><td>Proprietà temporali</td><td>Frequenza, gap, ordinamento</td></tr>
    </table>
    
    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "dataset_id": "sales_data_001",
        "outlier_method": "iqr",
        "outlier_threshold": 3.0,
        "validation_rules": {"check_missing": true, "check_outliers": true}
    }
    </code></pre>
    """,
)
async def validate_data(
    config: DataValidationRequest,
    background_tasks: BackgroundTasks
):
    """Valida qualità e consistenza dataset."""
    
    if config.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {config.dataset_id} non trovato")
    
    job_id = f"validation_job_{uuid.uuid4().hex[:8]}"
    
    # Inizializza job
    preprocessing_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "operation": "validation",
        "dataset_id": config.dataset_id,
        "created_at": datetime.now(),
        "results": {}
    }
    
    # Simula validazione asincrona
    async def run_validation():
        try:
            dataset = datasets_storage[config.dataset_id]
            
            # Simula diversi check di validazione
            validation_results = {
                "overall_quality_score": 0.82,
                "dataset_info": {
                    "rows": dataset.rows,
                    "columns": dataset.columns,
                    "size_mb": round(dataset.size_bytes / (1024*1024), 2)
                }
            }
            
            if config.validation_rules.get("check_missing_values", True):
                await asyncio.sleep(0.5)  # Simula check
                validation_results["missing_values"] = {
                    "count": int(dataset.rows * dataset.missing_values_ratio),
                    "percentage": round(dataset.missing_values_ratio * 100, 2),
                    "columns_affected": ["value_col_1", "value_col_2"]
                }
                preprocessing_jobs[job_id]["progress"] = 0.2
            
            if config.validation_rules.get("check_outliers", True):
                await asyncio.sleep(0.5)  # Simula outlier detection
                validation_results["outliers_detected"] = {
                    "method": config.outlier_method,
                    "count": 8,
                    "percentage": 0.8,
                    "columns_affected": ["value_col_1"]
                }
                preprocessing_jobs[job_id]["progress"] = 0.4
            
            if config.validation_rules.get("check_duplicates", True):
                await asyncio.sleep(0.3)
                validation_results["duplicates_found"] = {
                    "exact_duplicates": 3,
                    "near_duplicates": 5
                }
                preprocessing_jobs[job_id]["progress"] = 0.6
            
            if config.validation_rules.get("check_data_types", True):
                await asyncio.sleep(0.3)
                validation_results["data_type_issues"] = []  # Nessun problema simulato
                preprocessing_jobs[job_id]["progress"] = 0.8
            
            if config.validation_rules.get("check_time_series_properties", True):
                await asyncio.sleep(0.4)
                validation_results["time_series_issues"] = ["irregular_frequency", "minor_gaps"]
                preprocessing_jobs[job_id]["progress"] = 0.9
            
            # Generazione raccomandazioni automatiche
            recommendations = []
            if validation_results["missing_values"]["count"] > 0:
                recommendations.append("handle_missing_interpolation")
            if validation_results["outliers_detected"]["count"] > 0:
                recommendations.append("remove_outliers_iqr")
            if validation_results["duplicates_found"]["exact_duplicates"] > 0:
                recommendations.append("remove_duplicate_records")
            
            validation_results["recommendations"] = recommendations
            validation_results["validation_timestamp"] = datetime.now().isoformat()
            
            preprocessing_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "results": validation_results
            })
            
            logger.info(f"Validazione completata per dataset {config.dataset_id}")
            
        except Exception as e:
            preprocessing_jobs[job_id].update({
                "status": "failed",
                "error_message": str(e)
            })
            logger.error(f"Errore validazione {job_id}: {str(e)}")
    
    background_tasks.add_task(run_validation)
    
    return DataJobResponse(
        job_id=job_id,
        status="queued",
        progress=0.0,
        dataset_id=config.dataset_id
    )


@router.post(
    "/preprocess",
    response_model=DataJobResponse,
    summary="Preprocessing Avanzato",
    description="""Esegue pipeline preprocessing personalizzabile con step sequenziali per pulizia dati, trasformazioni e preparazione modeling.

    <h4>Step Preprocessing Disponibili:</h4>
    <table>
        <tr><th>Tipo Step</th><th>Parametri</th><th>Descrizione</th></tr>
        <tr><td>handle_missing</td><td>method: interpolate, forward_fill, drop</td><td>Gestione valori mancanti</td></tr>
        <tr><td>remove_outliers</td><td>method: iqr, zscore</td><td>Rimozione outlier</td></tr>
        <tr><td>smooth_data</td><td>window: int</td><td>Smoothing serie temporale</td></tr>
        <tr><td>normalize</td><td>method: minmax, zscore, robust</td><td>Normalizzazione valori</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "dataset_id": "raw_sales_data",
        "preprocessing_steps": [
            {"type": "handle_missing", "method": "interpolate"},
            {"type": "remove_outliers", "method": "iqr", "threshold": 2.5}
        ],
        "output_dataset_name": "preprocessed_sales"
    }
    </code></pre>
    """,
)
async def preprocess_data(
    config: PreprocessingRequest,
    background_tasks: BackgroundTasks
):
    """Esegue pipeline preprocessing configurabile."""
    
    if config.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {config.dataset_id} non trovato")
    
    job_id = f"preprocess_job_{uuid.uuid4().hex[:8]}"
    output_dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
    
    preprocessing_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "operation": "preprocessing",
        "input_dataset_id": config.dataset_id,
        "output_dataset_id": output_dataset_id,
        "created_at": datetime.now(),
        "results": {}
    }
    
    # Simula preprocessing asincrono
    async def run_preprocessing():
        try:
            original_dataset = datasets_storage[config.dataset_id]
            total_steps = len(config.preprocessing_steps)
            
            processing_log = []
            
            for i, step in enumerate(config.preprocessing_steps):
                step_type = step["type"]
                
                # Simula esecuzione step
                await asyncio.sleep(0.5)
                
                # Log step processing
                step_log = {
                    "step": i + 1,
                    "type": step_type,
                    "parameters": {k: v for k, v in step.items() if k != "type"},
                    "status": "completed"
                }
                
                if step_type == "handle_missing":
                    step_log["result"] = {"missing_values_handled": 25, "method": step.get("method", "interpolate")}
                elif step_type == "remove_outliers":
                    step_log["result"] = {"outliers_removed": 8, "method": step.get("method", "iqr")}
                elif step_type == "smooth_data":
                    step_log["result"] = {"smoothing_applied": True, "window": step.get("window", 5)}
                elif step_type == "normalize":
                    step_log["result"] = {"normalization": step.get("method", "minmax"), "range": step.get("feature_range", [0, 1])}
                
                processing_log.append(step_log)
                
                # Aggiorna progresso
                progress = (i + 1) / total_steps * 0.8  # Lascia 20% per finalizzazione
                preprocessing_jobs[job_id]["progress"] = progress
            
            # Finalizza dataset preprocessato
            await asyncio.sleep(0.5)
            
            # Crea metadati nuovo dataset
            processed_metadata = DatasetMetadata(
                dataset_id=output_dataset_id,
                name=config.output_dataset_name,
                rows=original_dataset.rows,  # Potrebbe cambiare con outlier removal
                columns=original_dataset.columns,
                size_bytes=original_dataset.size_bytes,
                upload_timestamp=datetime.now(),
                file_format="processed",
                column_info=original_dataset.column_info,
                data_quality_score=0.92,  # Migliorata dopo preprocessing
                missing_values_ratio=0.0   # Ridotta dopo preprocessing
            )
            
            # Salva dataset processato
            datasets_storage[output_dataset_id] = processed_metadata
            
            preprocessing_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "results": {
                    "processing_log": processing_log,
                    "input_quality_score": original_dataset.data_quality_score,
                    "output_quality_score": processed_metadata.data_quality_score,
                    "improvement": round(processed_metadata.data_quality_score - original_dataset.data_quality_score, 3),
                    "steps_completed": len(processing_log),
                    "output_dataset_id": output_dataset_id
                }
            })
            
            logger.info(f"Preprocessing completato: {config.dataset_id} -> {output_dataset_id}")
            
        except Exception as e:
            preprocessing_jobs[job_id].update({
                "status": "failed",
                "error_message": str(e)
            })
            logger.error(f"Errore preprocessing {job_id}: {str(e)}")
    
    background_tasks.add_task(run_preprocessing)
    
    return DataJobResponse(
        job_id=job_id,
        status="queued", 
        progress=0.0,
        dataset_id=output_dataset_id
    )


@router.post(
    "/explore",
    response_model=DataJobResponse,
    summary="Esplorazione Dati Automatica",
    description="""Genera analisi esplorativa completa con statistiche descrittive, test statistici e visualizzazioni automatiche.

    <h4>Tipi Analisi Disponibili:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Output</th></tr>
        <tr><td>basic</td><td>Statistiche descrittive essenziali</td><td>Mean, std, quantiles</td></tr>
        <tr><td>comprehensive</td><td>Analisi completa con correlazioni</td><td>Stats + correlazioni + distribuzione</td></tr>
        <tr><td>statistical</td><td>Focus su test statistici</td><td>ADF, KPSS, normalità</td></tr>
        <tr><td>time_series</td><td>Specializzata per serie temporali</td><td>Trend, stagionalità, decomposizione</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "dataset_id": "sales_data_001",
        "analysis_type": "comprehensive",
        "include_visualizations": true,
        "statistical_tests": ["stationarity", "seasonality", "normality"]
    }
    </code></pre>
    """,
)
async def explore_data(
    config: DataExploreRequest,
    background_tasks: BackgroundTasks
):
    """Esegue analisi esplorativa automatica del dataset."""
    
    if config.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {config.dataset_id} non trovato")
    
    job_id = f"explore_job_{uuid.uuid4().hex[:8]}"
    
    preprocessing_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "operation": "exploration",
        "dataset_id": config.dataset_id,
        "created_at": datetime.now(),
        "results": {}
    }
    
    # Simula analisi esplorativa asincrona
    async def run_exploration():
        try:
            dataset = datasets_storage[config.dataset_id]
            
            exploration_results = {
                "dataset_summary": {
                    "name": dataset.name,
                    "rows": dataset.rows,
                    "columns": dataset.columns,
                    "data_quality_score": dataset.data_quality_score
                }
            }
            
            # Statistiche descrittive
            await asyncio.sleep(0.5)
            exploration_results["descriptive_statistics"] = {
                "numerical_columns": {
                    "value_col_1": {
                        "mean": 125.4, "std": 23.8, "min": 45.2, "max": 198.7,
                        "q25": 108.3, "q50": 124.1, "q75": 142.9
                    },
                    "value_col_2": {
                        "mean": 87.2, "std": 15.6, "min": 32.1, "max": 134.5,
                        "q25": 76.4, "q50": 86.8, "q75": 97.3
                    }
                }
            }
            preprocessing_jobs[job_id]["progress"] = 0.2
            
            # Analisi correlazioni
            if config.analysis_type in ["comprehensive", "statistical"]:
                await asyncio.sleep(0.5)
                exploration_results["correlation_analysis"] = {
                    "pearson_correlation": {
                        "value_col_1_vs_value_col_2": 0.67
                    },
                    "spearman_correlation": {
                        "value_col_1_vs_value_col_2": 0.63
                    }
                }
                preprocessing_jobs[job_id]["progress"] = 0.4
            
            # Test statistici
            test_results = {}
            for test in config.statistical_tests:
                await asyncio.sleep(0.3)
                
                if test == "stationarity":
                    test_results["stationarity_tests"] = {
                        "adf_test": {"statistic": -3.42, "p_value": 0.012, "is_stationary": True},
                        "kpss_test": {"statistic": 0.34, "p_value": 0.156, "is_stationary": True}
                    }
                elif test == "seasonality":
                    test_results["seasonality_tests"] = {
                        "seasonal_decomposition": {"trend_strength": 0.78, "seasonal_strength": 0.45},
                        "periodicity_detected": {"period": 12, "confidence": 0.82}
                    }
                elif test == "normality":
                    test_results["normality_tests"] = {
                        "shapiro_wilk": {"statistic": 0.987, "p_value": 0.234, "is_normal": True},
                        "jarque_bera": {"statistic": 2.14, "p_value": 0.342, "is_normal": True}
                    }
                
                current_progress = 0.4 + (len(test_results) / len(config.statistical_tests)) * 0.4
                preprocessing_jobs[job_id]["progress"] = current_progress
            
            exploration_results["statistical_tests"] = test_results
            
            # Generazione visualizzazioni
            if config.include_visualizations:
                await asyncio.sleep(0.5)
                exploration_results["visualizations"] = {
                    "generated_plots": [
                        f"/outputs/plots/{job_id}_timeseries.html",
                        f"/outputs/plots/{job_id}_correlation_heatmap.html", 
                        f"/outputs/plots/{job_id}_distribution.html",
                        f"/outputs/plots/{job_id}_seasonal_decomposition.html"
                    ]
                }
                preprocessing_jobs[job_id]["progress"] = 0.9
            
            # Insights automatici
            insights = []
            if exploration_results["statistical_tests"].get("stationarity_tests", {}).get("adf_test", {}).get("is_stationary"):
                insights.append("Serie temporale stazionaria - suitable per ARIMA modeling")
            if exploration_results["statistical_tests"].get("seasonality_tests", {}).get("seasonal_strength", 0) > 0.4:
                insights.append("Componente stagionale significativa - considerare SARIMA")
            if exploration_results["correlation_analysis"]["pearson_correlation"]["value_col_1_vs_value_col_2"] > 0.6:
                insights.append("Correlazione forte tra variabili - possibile multicollinearità")
            
            exploration_results["automated_insights"] = insights
            exploration_results["analysis_timestamp"] = datetime.now().isoformat()
            
            preprocessing_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "results": exploration_results
            })
            
            logger.info(f"Esplorazione completata per dataset {config.dataset_id}")
            
        except Exception as e:
            preprocessing_jobs[job_id].update({
                "status": "failed",
                "error_message": str(e)
            })
            logger.error(f"Errore esplorazione {job_id}: {str(e)}")
    
    background_tasks.add_task(run_exploration)
    
    return DataJobResponse(
        job_id=job_id,
        status="queued",
        progress=0.0,
        dataset_id=config.dataset_id
    )


@router.post(
    "/split",
    response_model=DataJobResponse,
    summary="Dividi Train/Validation/Test",
    description="""Divide dataset in set training/validation/test con metodi appropriati per serie temporali preservando ordine temporale.

    <h4>Metodi Split Disponibili:</h4>
    <table>
        <tr><th>Metodo</th><th>Descrizione</th><th>Uso Raccomandato</th></tr>
        <tr><td>temporal</td><td>Split cronologico ordinato</td><td>Time series (evita data leakage)</td></tr>
        <tr><td>random</td><td>Split casuale con seed</td><td>Dati indipendenti</td></tr>
        <tr><td>stratified</td><td>Split bilanciato</td><td>Classification tasks</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "dataset_id": "sales_data_001",
        "split_method": "temporal",
        "train_ratio": 0.8,
        "validation_ratio": 0.1
    }
    </code></pre>
    """,
)
async def split_dataset(
    config: DataSplitRequest,
    background_tasks: BackgroundTasks
):
    """Divide dataset in training/validation/test sets."""
    
    if config.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {config.dataset_id} non trovato")
    
    job_id = f"split_job_{uuid.uuid4().hex[:8]}"
    
    preprocessing_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "operation": "split",
        "dataset_id": config.dataset_id,
        "created_at": datetime.now(),
        "results": {}
    }
    
    # Simula split asincrono
    async def run_split():
        try:
            original_dataset = datasets_storage[config.dataset_id]
            total_rows = original_dataset.rows
            
            # Calcola dimensioni split
            train_size = int(total_rows * config.train_ratio)
            
            if config.validation_ratio:
                val_size = int(total_rows * config.validation_ratio)
                test_size = total_rows - train_size - val_size
            else:
                val_size = 0
                test_size = total_rows - train_size
            
            await asyncio.sleep(1)  # Simula split processing
            
            # Crea ID per nuovi dataset
            train_dataset_id = f"dataset_train_{uuid.uuid4().hex[:6]}"
            test_dataset_id = f"dataset_test_{uuid.uuid4().hex[:6]}"
            val_dataset_id = f"dataset_val_{uuid.uuid4().hex[:6]}" if val_size > 0 else None
            
            # Crea metadati per dataset split
            def create_split_metadata(split_name, size, dataset_id):
                return DatasetMetadata(
                    dataset_id=dataset_id,
                    name=f"{original_dataset.name}_{split_name}",
                    rows=size,
                    columns=original_dataset.columns,
                    size_bytes=int(original_dataset.size_bytes * size / total_rows),
                    upload_timestamp=datetime.now(),
                    file_format=f"split_{split_name}",
                    column_info=original_dataset.column_info,
                    data_quality_score=original_dataset.data_quality_score,
                    missing_values_ratio=original_dataset.missing_values_ratio
                )
            
            # Salva dataset split
            datasets_storage[train_dataset_id] = create_split_metadata("train", train_size, train_dataset_id)
            datasets_storage[test_dataset_id] = create_split_metadata("test", test_size, test_dataset_id)
            
            if val_dataset_id:
                datasets_storage[val_dataset_id] = create_split_metadata("validation", val_size, val_dataset_id)
            
            preprocessing_jobs[job_id]["progress"] = 0.8
            await asyncio.sleep(0.5)  # Simula finalizzazione
            
            # Risultati split
            split_results = {
                "train_dataset_id": train_dataset_id,
                "test_dataset_id": test_dataset_id,
                "split_summary": {
                    "original_size": total_rows,
                    "train_size": train_size,
                    "test_size": test_size,
                    "split_method": config.split_method,
                    "train_ratio": config.train_ratio
                }
            }
            
            if val_dataset_id:
                split_results["validation_dataset_id"] = val_dataset_id
                split_results["split_summary"]["validation_size"] = val_size
                split_results["split_summary"]["validation_ratio"] = config.validation_ratio
            
            if config.split_date and config.split_method == "temporal":
                split_results["split_summary"]["split_date"] = config.split_date.isoformat()
            
            preprocessing_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "results": split_results
            })
            
            logger.info(f"Split completato: {config.dataset_id} -> train({train_size}), test({test_size})" + 
                       (f", val({val_size})" if val_size > 0 else ""))
            
        except Exception as e:
            preprocessing_jobs[job_id].update({
                "status": "failed",
                "error_message": str(e)
            })
            logger.error(f"Errore split {job_id}: {str(e)}")
    
    background_tasks.add_task(run_split)
    
    return DataJobResponse(
        job_id=job_id,
        status="queued",
        progress=0.0,
        dataset_id=config.dataset_id
    )


@router.post(
    "/quality-report", 
    response_model=DataJobResponse,
    summary="Genera Report Qualità Dati",
    description="""Genera report professionale multi-dimensionale qualità dati con scoring automatico e raccomandazioni miglioramento.

    <h4>Dimensioni Qualità Valutate:</h4>
    <table>
        <tr><th>Dimensione</th><th>Descrizione</th><th>Score Range</th></tr>
        <tr><td>Completeness</td><td>Percentuale dati mancanti</td><td>0.0-1.0</td></tr>
        <tr><td>Consistency</td><td>Uniformità formati, encoding</td><td>0.0-1.0</td></tr>
        <tr><td>Accuracy</td><td>Precisione valori, range validity</td><td>0.0-1.0</td></tr>
        <tr><td>Validity</td><td>Conformità schema, constraint</td><td>0.0-1.0</td></tr>
        <tr><td>Timeliness</td><td>Freshness dati, gap temporali</td><td>0.0-1.0</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "dataset_id": "sales_data_001",
        "quality_dimensions": ["completeness", "consistency", "accuracy"],
        "generate_report": true,
        "suggest_improvements": true
    }
    </code></pre>
    """,
)
async def generate_quality_report(
    config: DataQualityRequest,
    background_tasks: BackgroundTasks
):
    """Genera report completo qualità dati."""
    
    if config.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {config.dataset_id} non trovato")
    
    job_id = f"quality_job_{uuid.uuid4().hex[:8]}"
    
    preprocessing_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "operation": "quality_report",
        "dataset_id": config.dataset_id,
        "created_at": datetime.now(),
        "results": {}
    }
    
    # Simula generazione report qualità
    async def generate_quality_job():
        try:
            dataset = datasets_storage[config.dataset_id]
            
            # Analisi per dimensione qualità
            quality_scores = {}
            detailed_analysis = {}
            
            for dimension in config.quality_dimensions:
                await asyncio.sleep(0.3)  # Simula analisi dimensione
                
                if dimension == "completeness":
                    score = 0.85
                    quality_scores[dimension] = score
                    detailed_analysis[dimension] = {
                        "score": score,
                        "missing_values_ratio": 0.15,
                        "columns_affected": ["value_col_1", "value_col_2"],
                        "missing_patterns": ["random", "consecutive_gaps"]
                    }
                elif dimension == "consistency":
                    score = 0.92
                    quality_scores[dimension] = score
                    detailed_analysis[dimension] = {
                        "score": score,
                        "format_consistency": 0.95,
                        "encoding_issues": 0,
                        "standard_compliance": 0.90
                    }
                elif dimension == "accuracy":
                    score = 0.78
                    quality_scores[dimension] = score
                    detailed_analysis[dimension] = {
                        "score": score,
                        "outliers_detected": 12,
                        "range_violations": 3,
                        "business_rule_violations": 1
                    }
                elif dimension == "validity":
                    score = 0.88
                    quality_scores[dimension] = score
                    detailed_analysis[dimension] = {
                        "score": score,
                        "schema_compliance": 1.0,
                        "constraint_violations": 5,
                        "data_type_mismatches": 0
                    }
                elif dimension == "timeliness":
                    score = 0.82
                    quality_scores[dimension] = score
                    detailed_analysis[dimension] = {
                        "score": score,
                        "data_freshness_days": 2,
                        "temporal_gaps": 3,
                        "update_frequency_compliance": 0.85
                    }
                
                current_progress = len(quality_scores) / len(config.quality_dimensions) * 0.6
                preprocessing_jobs[job_id]["progress"] = current_progress
            
            # Score complessivo
            overall_score = sum(quality_scores.values()) / len(quality_scores)
            
            # Generazione raccomandazioni
            recommendations = []
            if config.suggest_improvements:
                if quality_scores.get("completeness", 1.0) < 0.9:
                    recommendations.append({
                        "priority": "high",
                        "category": "completeness", 
                        "action": "Implement interpolation strategy for missing values",
                        "expected_improvement": 0.15
                    })
                if quality_scores.get("accuracy", 1.0) < 0.8:
                    recommendations.append({
                        "priority": "medium",
                        "category": "accuracy",
                        "action": "Apply outlier detection and removal using IQR method",
                        "expected_improvement": 0.12
                    })
                if quality_scores.get("timeliness", 1.0) < 0.85:
                    recommendations.append({
                        "priority": "low",
                        "category": "timeliness",
                        "action": "Implement automated data pipeline for fresher data",
                        "expected_improvement": 0.08
                    })
            
            preprocessing_jobs[job_id]["progress"] = 0.8
            
            # Generazione report file
            if config.generate_report:
                await asyncio.sleep(0.5)  # Simula generazione report
                report_files = [
                    f"/reports/{job_id}/data_quality_report.html",
                    f"/reports/{job_id}/data_quality_report.pdf",
                    f"/reports/{job_id}/executive_summary.html"
                ]
            else:
                report_files = []
            
            # Risultati finali
            quality_results = {
                "overall_quality_score": round(overall_score, 3),
                "grade": "Good" if overall_score >= 0.8 else "Fair" if overall_score >= 0.7 else "Poor",
                "dimension_scores": quality_scores,
                "detailed_analysis": detailed_analysis,
                "dataset_info": {
                    "name": dataset.name,
                    "rows": dataset.rows,
                    "columns": dataset.columns,
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "recommendations": recommendations,
                "improvement_potential": sum([r["expected_improvement"] for r in recommendations]),
                "report_files": report_files
            }
            
            preprocessing_jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "results": quality_results
            })
            
            logger.info(f"Report qualità generato per {config.dataset_id}: score {overall_score:.3f}")
            
        except Exception as e:
            preprocessing_jobs[job_id].update({
                "status": "failed",
                "error_message": str(e)
            })
            logger.error(f"Errore report qualità {job_id}: {str(e)}")
    
    background_tasks.add_task(generate_quality_job)
    
    return DataJobResponse(
        job_id=job_id,
        status="queued",
        progress=0.0,
        dataset_id=config.dataset_id
    )


@router.get(
    "/datasets",
    response_model=List[DatasetMetadata],
    summary="Lista Tutti i Dataset",
    description="""Restituisce lista completa di tutti i dataset caricati con metadati, statistiche qualità e informazioni uso.

    <h4>Parametri Query Opzionali:</h4>
    <table>
        <tr><th>Parametro</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>format_filter</td><td>str</td><td>Filtra per formato file</td></tr>
        <tr><td>min_quality_score</td><td>float</td><td>Score qualità minimo (0.0-1.0)</td></tr>
        <tr><td>sort_by</td><td>str</td><td>Ordina per: name, upload_time, quality, size</td></tr>
    </table>

    <h4>Esempio Output:</h4>
    <pre><code>
    [
        {
            "dataset_id": "dataset_abc123",
            "name": "Sales Data Q4 2024",
            "rows": 10000,
            "columns": 5,
            "data_quality_score": 0.87,
            "file_format": "csv"
        }
    ]
    </code></pre>
    """,
)
async def list_datasets(
    format_filter: Optional[str] = Query(None, description="Filtra per formato file"),
    min_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Score qualità minimo"),
    sort_by: Optional[str] = Query("upload_timestamp", description="Campo ordinamento")
):
    """Restituisce lista di tutti i dataset disponibili."""
    
    datasets_list = list(datasets_storage.values())
    
    # Applicazione filtri
    if format_filter:
        datasets_list = [d for d in datasets_list if d.file_format == format_filter]
    
    if min_quality_score is not None:
        datasets_list = [d for d in datasets_list if d.data_quality_score >= min_quality_score]
    
    # Ordinamento
    if sort_by == "name":
        datasets_list.sort(key=lambda x: x.name)
    elif sort_by == "quality":
        datasets_list.sort(key=lambda x: x.data_quality_score, reverse=True)
    elif sort_by == "size":
        datasets_list.sort(key=lambda x: x.size_bytes, reverse=True)
    else:  # default upload_timestamp
        datasets_list.sort(key=lambda x: x.upload_timestamp, reverse=True)
    
    return datasets_list


@router.get(
    "/job-status/{job_id}",
    response_model=DataJobResponse,
    summary="Stato Job Data Processing",
    description="""Verifica stato, progresso e risultati di job di processing dati per monitoring operazioni lunghe.

    <h4>Stati Job Possibili:</h4>
    <table>
        <tr><th>Stato</th><th>Descrizione</th><th>Progress</th></tr>
        <tr><td>queued</td><td>Job in coda di processing</td><td>0.0</td></tr>
        <tr><td>running</td><td>Job in esecuzione</td><td>0.0-1.0</td></tr>
        <tr><td>completed</td><td>Job completato con successo</td><td>1.0</td></tr>
        <tr><td>failed</td><td>Job fallito con error message</td><td>Variable</td></tr>
    </table>

    <h4>Esempio Output:</h4>
    <pre><code>
    {
        "job_id": "preprocess_job_abc123",
        "status": "completed",
        "progress": 1.0,
        "dataset_id": "dataset_xyz789",
        "results": {"processed_rows": 1000}
    }
    </code></pre>
    """,
)
async def get_data_job_status(job_id: str):
    """Recupera stato job di data processing."""
    
    if job_id not in preprocessing_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} non trovato")
    
    job_info = preprocessing_jobs[job_id]
    
    return DataJobResponse(
        job_id=job_id,
        status=job_info["status"],
        progress=job_info.get("progress", 0.0),
        dataset_id=job_info.get("dataset_id"),
        results=job_info.get("results", {}),
        error_message=job_info.get("error_message")
    )