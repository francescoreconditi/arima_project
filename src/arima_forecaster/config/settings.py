"""
Sistema di configurazione centralizzato con supporto .env file.
Gestisce tutte le impostazioni globali della libreria ARIMA Forecaster.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Backend types supportati
BackendType = Literal["cpu", "cuda", "auto"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

@dataclass
class ARIMAConfig:
    """
    Configurazione globale per ARIMA Forecaster Library.
    
    Tutte le impostazioni possono essere sovrascritte tramite:
    1. Environment variables (massima priorità)
    2. File .env nella directory di progetto
    3. Valori di default (minima priorità)
    """
    
    # === GPU/CUDA Configuration ===
    backend: BackendType = "auto"
    cuda_device: int = 0
    gpu_memory_limit: Optional[float] = None  # GB, None = unlimited
    enable_mixed_precision: bool = True
    max_gpu_models_parallel: int = 100
    
    # === Performance Configuration ===
    n_jobs: int = -1  # CPU cores, -1 = all available
    chunk_size: int = 1000  # Batch processing size
    cache_enabled: bool = True
    cache_dir: Optional[str] = None
    
    # === Logging Configuration ===
    log_level: LogLevel = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # === Model Training Defaults ===
    default_max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    enable_early_stopping: bool = True
    validation_split: float = 0.2
    
    # === Feature Engineering ===
    max_features: int = 50
    feature_selection_method: str = "stepwise"
    preprocessing_method: str = "auto"
    
    # === Output Configuration ===
    output_dir: str = "outputs"
    save_models: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    
    # === API Configuration ===
    api_host: str = "localhost"
    api_port: int = 8000
    api_workers: int = 1
    enable_swagger: bool = True
    
    # === Development/Debug ===
    debug_mode: bool = False
    profile_performance: bool = False
    seed: Optional[int] = 42
    
    # === Internal ===
    _env_loaded: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Inizializza configurazione caricando .env e validando."""
        if not self._env_loaded:
            self.load_from_env()
            self.validate()
            self._log_configuration()
    
    def load_from_env(self, env_file: Optional[str] = None) -> None:
        """
        Carica configurazione da file .env e environment variables.
        
        Args:
            env_file: Path to .env file. Se None, cerca .env nella directory corrente
        """
        # Cerca file .env nella directory corrente e parent
        if env_file is None:
            potential_paths = [
                Path.cwd() / ".env",
                Path.cwd().parent / ".env",
                Path(__file__).parent.parent.parent.parent / ".env"  # Project root
            ]
            
            for path in potential_paths:
                if path.exists():
                    env_file = str(path)
                    logger.info(f"Caricato file .env da: {env_file}")
                    break
        
        # Carica .env file se esiste
        if env_file and Path(env_file).exists():
            load_dotenv(env_file, override=True)
            logger.info(f"Configurazione caricata da {env_file}")
        
        # Override con environment variables
        self._load_env_vars()
        self._env_loaded = True
    
    def _load_env_vars(self) -> None:
        """Carica tutte le configurazioni da environment variables."""
        
        # GPU/CUDA Settings
        self.backend = os.getenv("ARIMA_BACKEND", self.backend)
        self.cuda_device = int(os.getenv("ARIMA_CUDA_DEVICE", self.cuda_device))
        
        gpu_memory = os.getenv("ARIMA_GPU_MEMORY_LIMIT")
        if gpu_memory:
            self.gpu_memory_limit = float(gpu_memory)
        
        self.enable_mixed_precision = os.getenv("ARIMA_MIXED_PRECISION", "true").lower() == "true"
        self.max_gpu_models_parallel = int(os.getenv("ARIMA_MAX_GPU_PARALLEL", self.max_gpu_models_parallel))
        
        # Performance Settings
        self.n_jobs = int(os.getenv("ARIMA_N_JOBS", self.n_jobs))
        self.chunk_size = int(os.getenv("ARIMA_CHUNK_SIZE", self.chunk_size))
        self.cache_enabled = os.getenv("ARIMA_CACHE_ENABLED", "true").lower() == "true"
        self.cache_dir = os.getenv("ARIMA_CACHE_DIR", self.cache_dir)
        
        # Logging Settings
        self.log_level = os.getenv("ARIMA_LOG_LEVEL", self.log_level)
        self.log_file = os.getenv("ARIMA_LOG_FILE", self.log_file)
        self.log_format = os.getenv("ARIMA_LOG_FORMAT", self.log_format)
        
        # Model Training Settings
        self.default_max_iterations = int(os.getenv("ARIMA_MAX_ITERATIONS", self.default_max_iterations))
        self.convergence_tolerance = float(os.getenv("ARIMA_CONVERGENCE_TOL", self.convergence_tolerance))
        self.enable_early_stopping = os.getenv("ARIMA_EARLY_STOPPING", "true").lower() == "true"
        self.validation_split = float(os.getenv("ARIMA_VALIDATION_SPLIT", self.validation_split))
        
        # Feature Engineering Settings
        self.max_features = int(os.getenv("ARIMA_MAX_FEATURES", self.max_features))
        self.feature_selection_method = os.getenv("ARIMA_FEATURE_SELECTION", self.feature_selection_method)
        self.preprocessing_method = os.getenv("ARIMA_PREPROCESSING", self.preprocessing_method)
        
        # Output Settings
        self.output_dir = os.getenv("ARIMA_OUTPUT_DIR", self.output_dir)
        self.save_models = os.getenv("ARIMA_SAVE_MODELS", "true").lower() == "true"
        self.save_plots = os.getenv("ARIMA_SAVE_PLOTS", "true").lower() == "true"
        self.plot_format = os.getenv("ARIMA_PLOT_FORMAT", self.plot_format)
        
        # API Settings
        self.api_host = os.getenv("ARIMA_API_HOST", self.api_host)
        self.api_port = int(os.getenv("ARIMA_API_PORT", self.api_port))
        self.api_workers = int(os.getenv("ARIMA_API_WORKERS", self.api_workers))
        self.enable_swagger = os.getenv("ARIMA_ENABLE_SWAGGER", "true").lower() == "true"
        
        # Development Settings
        self.debug_mode = os.getenv("ARIMA_DEBUG", "false").lower() == "true"
        self.profile_performance = os.getenv("ARIMA_PROFILE", "false").lower() == "true"
        
        seed_env = os.getenv("ARIMA_SEED")
        if seed_env:
            self.seed = int(seed_env)
    
    def validate(self) -> None:
        """Valida la configurazione e corregge valori non validi."""
        
        # Valida backend
        if self.backend not in ["cpu", "cuda", "auto"]:
            logger.warning(f"Backend '{self.backend}' non valido. Uso 'auto'.")
            self.backend = "auto"
        
        # Valida CUDA device
        if self.cuda_device < 0:
            logger.warning(f"CUDA device {self.cuda_device} non valido. Uso device 0.")
            self.cuda_device = 0
        
        # Valida GPU memory limit
        if self.gpu_memory_limit is not None and self.gpu_memory_limit <= 0:
            logger.warning(f"GPU memory limit {self.gpu_memory_limit} non valido. Rimuovo limite.")
            self.gpu_memory_limit = None
        
        # Valida n_jobs
        if self.n_jobs == 0:
            self.n_jobs = 1
            logger.warning("n_jobs non può essere 0. Impostato a 1.")
        
        # Valida chunk_size
        if self.chunk_size <= 0:
            self.chunk_size = 1000
            logger.warning("chunk_size deve essere positivo. Impostato a 1000.")
        
        # Valida log_level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            logger.warning(f"Log level '{self.log_level}' non valido. Uso 'INFO'.")
            self.log_level = "INFO"
        
        # Crea directory output se non esiste
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Crea cache directory se specificata
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
    
    def _log_configuration(self) -> None:
        """Log della configurazione attuale (solo in debug mode)."""
        if self.debug_mode:
            logger.debug("=== ARIMA Forecaster Configuration ===")
            logger.debug(f"Backend: {self.backend}")
            logger.debug(f"CUDA Device: {self.cuda_device}")
            logger.debug(f"GPU Memory Limit: {self.gpu_memory_limit}")
            logger.debug(f"Mixed Precision: {self.enable_mixed_precision}")
            logger.debug(f"Max GPU Parallel: {self.max_gpu_models_parallel}")
            logger.debug(f"CPU Jobs: {self.n_jobs}")
            logger.debug(f"Cache Enabled: {self.cache_enabled}")
            logger.debug(f"Log Level: {self.log_level}")
            logger.debug(f"Debug Mode: {self.debug_mode}")
            logger.debug("=====================================")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configurazione in dizionario per serializzazione."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
            if not field.name.startswith('_')
        }
    
    def update(self, **kwargs) -> None:
        """Aggiorna configurazione con nuovi valori."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Configurazione aggiornata: {key} = {value}")
            else:
                logger.warning(f"Parametro configurazione sconosciuto: {key}")
        
        self.validate()


# Singleton pattern per configurazione globale
_global_config: Optional[ARIMAConfig] = None

def get_config() -> ARIMAConfig:
    """
    Restituisce l'istanza globale di configurazione.
    
    Returns:
        ARIMAConfig: Configurazione globale
    """
    global _global_config
    if _global_config is None:
        _global_config = ARIMAConfig()
    return _global_config

def reset_config() -> None:
    """Reset della configurazione globale (per testing)."""
    global _global_config
    _global_config = None

def update_config(**kwargs) -> None:
    """
    Aggiorna configurazione globale.
    
    Args:
        **kwargs: Parametri da aggiornare
    """
    config = get_config()
    config.update(**kwargs)