"""
Model Caching System per Performance Optimization.

Sistema di caching intelligente per modelli ARIMA pre-addestrati.
Riduce drasticamente i tempi di ritraining con parametri simili.
"""

import hashlib
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelCache:
    """
    Cache intelligente per modelli ARIMA con gestione automatica della memoria.

    Features:
    - LRU eviction per gestione memoria
    - Persistent cache su disco per restart
    - Hash-based key generation per data similarity
    - TTL (Time To Live) per cache freshness
    - Statistiche performance dettagliate
    """

    def __init__(
        self,
        max_memory_models: int = 50,
        cache_dir: Optional[Path] = None,
        ttl_hours: int = 24,
        enable_disk_cache: bool = True,
    ):
        self.max_memory_models = max_memory_models
        self.ttl_hours = ttl_hours
        self.enable_disk_cache = enable_disk_cache

        # Memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}

        # Disk cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("outputs/cache/models")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "disk_hits": 0,
            "evictions": 0,
            "total_time_saved": 0.0,
        }

        logger.info(f"ModelCache initialized: max_memory={max_memory_models}, ttl={ttl_hours}h")

    def _generate_cache_key(
        self,
        data: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        exog_hash: Optional[str] = None,
    ) -> str:
        """
        Genera chiave cache basata su caratteristiche dei dati e parametri modello.

        Usa hash MD5 di:
        - Statistical fingerprint dei dati (media, std, trend, stagionalità)
        - Parametri modello (order, seasonal_order)
        - Hash variabili esogene se presenti
        """

        # Statistical fingerprint dei dati
        data_stats = {
            "length": len(data),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "trend": float(np.polyfit(range(len(data)), data.values, 1)[0]),
            "autocorr_1": float(data.autocorr(1) if len(data) > 1 else 0),
            "autocorr_12": float(data.autocorr(12) if len(data) > 12 else 0),
        }

        # Parametri modello
        model_params = {"order": order, "seasonal_order": seasonal_order, "exog_hash": exog_hash}

        # Combina tutto
        cache_data = {**data_stats, **model_params}

        # Hash MD5
        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Controlla se entry cache è ancora valida (TTL non scaduto)."""
        created_time = cache_entry.get("created_time")
        if not created_time:
            return False

        elapsed = datetime.now() - created_time
        return elapsed < timedelta(hours=self.ttl_hours)

    def _evict_lru(self):
        """Rimuove il modello Least Recently Used dalla cache."""
        if not self._access_times:
            return

        # Trova chiave con accesso più vecchio
        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]

        # Rimuovi da memoria
        if oldest_key in self._memory_cache:
            del self._memory_cache[oldest_key]
        if oldest_key in self._access_times:
            del self._access_times[oldest_key]

        self.stats["evictions"] += 1
        logger.debug(f"LRU evicted model: {oldest_key[:8]}...")

    def get(
        self,
        data: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        exog_hash: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Recupera modello dalla cache se disponibile.

        Returns:
            Modello cached o None se non trovato/scaduto
        """
        self.stats["total_requests"] += 1

        cache_key = self._generate_cache_key(data, order, seasonal_order, exog_hash)

        # 1. Check memory cache
        if cache_key in self._memory_cache:
            cache_entry = self._memory_cache[cache_key]

            if self._is_cache_valid(cache_entry):
                # Aggiorna access time
                self._access_times[cache_key] = datetime.now()
                self.stats["cache_hits"] += 1

                logger.debug(f"Memory cache hit: {cache_key[:8]}...")
                return cache_entry["model"]
            else:
                # Cache scaduta, rimuovi
                del self._memory_cache[cache_key]
                if cache_key in self._access_times:
                    del self._access_times[cache_key]

        # 2. Check disk cache
        if self.enable_disk_cache:
            disk_path = self.cache_dir / f"{cache_key}.pkl"

            if disk_path.exists():
                try:
                    with open(disk_path, "rb") as f:
                        cache_entry = pickle.load(f)

                    if self._is_cache_valid(cache_entry):
                        # Carica in memory cache
                        self._store_in_memory(cache_key, cache_entry)
                        self.stats["disk_hits"] += 1

                        logger.debug(f"Disk cache hit: {cache_key[:8]}...")
                        return cache_entry["model"]
                    else:
                        # Cache su disco scaduta, rimuovi
                        disk_path.unlink()

                except Exception as e:
                    logger.warning(f"Error loading disk cache {cache_key[:8]}: {e}")

        # Cache miss
        self.stats["cache_misses"] += 1
        logger.debug(f"Cache miss: {cache_key[:8]}...")
        return None

    def _store_in_memory(self, cache_key: str, cache_entry: Dict[str, Any]):
        """Memorizza entry in memory cache con gestione LRU."""

        # Evict se necessario
        while len(self._memory_cache) >= self.max_memory_models:
            self._evict_lru()

        # Store in memory
        self._memory_cache[cache_key] = cache_entry
        self._access_times[cache_key] = datetime.now()

    def store(
        self,
        data: pd.Series,
        order: Tuple[int, int, int],
        model: Any,
        fit_time: float,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        exog_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Memorizza modello addestrato nella cache.

        Args:
            data: Serie temporale usata per training
            order: Parametri ARIMA (p,d,q)
            model: Modello addestrato
            fit_time: Tempo impiegato per il training (per statistiche)
            seasonal_order: Parametri stagionali se SARIMA
            exog_hash: Hash variabili esogene
            metadata: Metadata aggiuntivi
        """
        cache_key = self._generate_cache_key(data, order, seasonal_order, exog_hash)

        cache_entry = {
            "model": model,
            "order": order,
            "seasonal_order": seasonal_order,
            "data_length": len(data),
            "fit_time": fit_time,
            "created_time": datetime.now(),
            "metadata": metadata or {},
        }

        # Store in memory
        self._store_in_memory(cache_key, cache_entry)

        # Store su disco se abilitato
        if self.enable_disk_cache:
            try:
                disk_path = self.cache_dir / f"{cache_key}.pkl"
                with open(disk_path, "wb") as f:
                    pickle.dump(cache_entry, f)

                logger.debug(f"Model cached: {cache_key[:8]}... (fit_time: {fit_time:.3f}s)")

            except Exception as e:
                logger.warning(f"Error saving disk cache {cache_key[:8]}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche performance cache."""
        total_requests = max(self.stats["total_requests"], 1)

        return {
            **self.stats,
            "hit_rate": self.stats["cache_hits"] / total_requests,
            "miss_rate": self.stats["cache_misses"] / total_requests,
            "disk_hit_rate": self.stats["disk_hits"] / total_requests,
            "memory_usage": len(self._memory_cache),
            "memory_capacity": self.max_memory_models,
            "average_time_saved_per_hit": (
                self.stats["total_time_saved"] / max(self.stats["cache_hits"], 1)
            ),
        }

    def clear(self, memory_only: bool = False):
        """
        Pulisce la cache.

        Args:
            memory_only: Se True, pulisce solo memory cache, lascia disk cache
        """
        # Clear memory
        self._memory_cache.clear()
        self._access_times.clear()

        # Clear disk se richiesto
        if not memory_only and self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_file}: {e}")

        # Reset stats
        self.stats = {
            key: 0 if isinstance(value, (int, float)) else value
            for key, value in self.stats.items()
        }

        logger.info(f"Cache cleared (memory_only={memory_only})")


# Istanza globale cache
_global_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Ottiene istanza globale del model cache."""
    global _global_cache

    if _global_cache is None:
        _global_cache = ModelCache()

    return _global_cache


def configure_cache(
    max_memory_models: int = 50,
    cache_dir: Optional[Path] = None,
    ttl_hours: int = 24,
    enable_disk_cache: bool = True,
):
    """Configura istanza globale del cache."""
    global _global_cache

    _global_cache = ModelCache(
        max_memory_models=max_memory_models,
        cache_dir=cache_dir,
        ttl_hours=ttl_hours,
        enable_disk_cache=enable_disk_cache,
    )

    logger.info("Global model cache configured")


@lru_cache(maxsize=20)
def get_smart_starting_params(
    data_length: int,
    has_trend: bool,
    has_seasonality: bool,
    volatility_level: str,  # "low", "medium", "high"
) -> Dict[str, Any]:
    """
    Calcola parametri di starting intelligenti per ottimizzatore ARIMA.

    Usa euristiche basate sulle caratteristiche dei dati per ridurre
    il numero di iterazioni dell'ottimizzatore.

    Returns:
        Dizionario con starting parameters ottimali
    """

    params = {}

    # Starting values per AR parameters
    if volatility_level == "low":
        params["ar_start"] = [0.5, 0.3] if data_length > 100 else [0.7]
    elif volatility_level == "medium":
        params["ar_start"] = [0.3, 0.1] if data_length > 100 else [0.5]
    else:  # high volatility
        params["ar_start"] = [0.1, -0.1] if data_length > 100 else [0.2]

    # Starting values per MA parameters
    if has_seasonality:
        params["ma_start"] = [0.3, 0.1]
    else:
        params["ma_start"] = [0.2] if volatility_level != "high" else [0.1]

    # Variance starting value
    if volatility_level == "low":
        params["sigma2_start"] = 0.1
    elif volatility_level == "medium":
        params["sigma2_start"] = 1.0
    else:
        params["sigma2_start"] = 5.0

    # Ottimizzatore settings
    if data_length < 100:
        params["maxiter"] = 50  # Meno iterazioni per dataset piccoli
    elif data_length < 500:
        params["maxiter"] = 100
    else:
        params["maxiter"] = 200

    params["method"] = "lbfgs"  # Generalmente il più veloce
    params["disp"] = False  # No verbose output

    return params
