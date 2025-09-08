"""
Memory Pool System per ottimizzazione performance ARIMA Forecaster.

Sistema di memory pooling per ridurre garbage collection overhead
e pre-allocazione buffer per operazioni vettorizzate.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List
from threading import Lock
import gc
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryPool:
    """
    Pool di memoria pre-allocata per ridurre overhead GC.
    
    Features:
    - Pre-allocazione array NumPy di dimensioni comuni
    - Pool di Series e DataFrame per operazioni temporanee
    - Buffer riutilizzabili per calcoli intermedi
    - Thread-safe per uso concorrente
    - Monitoring utilizzo memoria
    """
    
    def __init__(self, 
                 max_array_size: int = 10000,
                 max_pool_size: int = 20,
                 enable_monitoring: bool = True):
        
        self.max_array_size = max_array_size
        self.max_pool_size = max_pool_size
        self.enable_monitoring = enable_monitoring
        
        # Thread safety
        self._lock = Lock()
        
        # Memory pools
        self._float_arrays: Dict[int, List[np.ndarray]] = {}
        self._int_arrays: Dict[int, List[np.ndarray]] = {}
        self._series_pool: List[pd.Series] = []
        self._dataframe_pool: List[pd.DataFrame] = []
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'memory_saved_bytes': 0,
            'last_gc_time': datetime.now(),
            'gc_collections': 0
        }
        
        logger.info(f"MemoryPool initialized: max_size={max_array_size}, pool_size={max_pool_size}")
    
    def get_float_array(self, size: int, fill_value: Optional[float] = None) -> np.ndarray:
        """
        Ottiene array float dal pool o crea nuovo.
        
        Args:
            size: Dimensione array richiesta
            fill_value: Valore di riempimento opzionale
            
        Returns:
            Array NumPy float64 della dimensione richiesta
        """
        with self._lock:
            self.stats['total_requests'] += 1
            
            # Controlla se abbiamo array disponibili di questa dimensione
            if size in self._float_arrays and self._float_arrays[size]:
                array = self._float_arrays[size].pop()
                self.stats['pool_hits'] += 1
                self.stats['memory_saved_bytes'] += size * 8  # float64 = 8 bytes
                
                # Reset array se necessario
                if fill_value is not None:
                    array.fill(fill_value)
                else:
                    array.fill(0.0)
                    
                logger.debug(f"Array float[{size}] from pool")
                return array
            
            # Pool miss - crea nuovo array
            self.stats['pool_misses'] += 1
            if fill_value is not None:
                array = np.full(size, fill_value, dtype=np.float64)
            else:
                array = np.zeros(size, dtype=np.float64)
                
            logger.debug(f"Array float[{size}] created new")
            return array
    
    def get_int_array(self, size: int, fill_value: Optional[int] = None) -> np.ndarray:
        """
        Ottiene array int dal pool o crea nuovo.
        
        Args:
            size: Dimensione array richiesta  
            fill_value: Valore di riempimento opzionale
            
        Returns:
            Array NumPy int64 della dimensione richiesta
        """
        with self._lock:
            self.stats['total_requests'] += 1
            
            if size in self._int_arrays and self._int_arrays[size]:
                array = self._int_arrays[size].pop()
                self.stats['pool_hits'] += 1
                self.stats['memory_saved_bytes'] += size * 8  # int64 = 8 bytes
                
                if fill_value is not None:
                    array.fill(fill_value)
                else:
                    array.fill(0)
                    
                logger.debug(f"Array int[{size}] from pool")
                return array
            
            self.stats['pool_misses'] += 1
            if fill_value is not None:
                array = np.full(size, fill_value, dtype=np.int64)
            else:
                array = np.zeros(size, dtype=np.int64)
                
            logger.debug(f"Array int[{size}] created new")  
            return array
    
    def return_array(self, array: np.ndarray) -> None:
        """
        Restituisce array al pool per riutilizzo futuro.
        
        Args:
            array: Array da restituire al pool
        """
        if array is None or array.size > self.max_array_size:
            return  # Array troppo grande per il pool
            
        with self._lock:
            size = array.size
            
            if array.dtype == np.float64:
                if size not in self._float_arrays:
                    self._float_arrays[size] = []
                    
                # Limita dimensione pool
                if len(self._float_arrays[size]) < self.max_pool_size:
                    self._float_arrays[size].append(array)
                    logger.debug(f"Float array[{size}] returned to pool")
                    
            elif array.dtype == np.int64:
                if size not in self._int_arrays:
                    self._int_arrays[size] = []
                    
                if len(self._int_arrays[size]) < self.max_pool_size:
                    self._int_arrays[size].append(array)
                    logger.debug(f"Int array[{size}] returned to pool")
    
    def get_series(self, data: Optional[np.ndarray] = None, 
                   index: Optional[Any] = None, 
                   name: Optional[str] = None) -> pd.Series:
        """
        Ottiene Series dal pool o crea nuova.
        
        Args:
            data: Dati per la Serie
            index: Indice per la Serie
            name: Nome della Serie
            
        Returns:
            pandas Series
        """
        with self._lock:
            self.stats['total_requests'] += 1
            
            if self._series_pool:
                series = self._series_pool.pop()
                self.stats['pool_hits'] += 1
                
                # Reinizializza con nuovi dati
                if data is not None:
                    series = pd.Series(data, index=index, name=name)
                else:
                    # Serie vuota
                    series = pd.Series(dtype=np.float64, index=index, name=name)
                    
                logger.debug("Series from pool")
                return series
            
            # Pool miss
            self.stats['pool_misses'] += 1
            if data is not None:
                series = pd.Series(data, index=index, name=name)
            else:
                series = pd.Series(dtype=np.float64, index=index, name=name)
                
            logger.debug("Series created new")
            return series
    
    def return_series(self, series: pd.Series) -> None:
        """Restituisce Series al pool dopo uso."""
        if series is None or len(series) > self.max_array_size:
            return
            
        with self._lock:
            if len(self._series_pool) < self.max_pool_size:
                # Clear dei dati per riutilizzo
                series.iloc[:] = np.nan
                self._series_pool.append(series)
                logger.debug("Series returned to pool")
    
    def clear_pool(self) -> None:
        """Pulisce tutti i pool per liberare memoria."""
        with self._lock:
            self._float_arrays.clear()
            self._int_arrays.clear() 
            self._series_pool.clear()
            self._dataframe_pool.clear()
            
            # Force garbage collection
            gc.collect()
            self.stats['gc_collections'] += 1
            self.stats['last_gc_time'] = datetime.now()
            
            logger.info("Memory pools cleared and GC forced")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche utilizzo memoria."""
        with self._lock:
            float_arrays_count = sum(len(arrays) for arrays in self._float_arrays.values())
            int_arrays_count = sum(len(arrays) for arrays in self._int_arrays.values())
            
            total_requests = max(self.stats['total_requests'], 1)
            
            return {
                'total_requests': self.stats['total_requests'],
                'pool_hits': self.stats['pool_hits'],
                'pool_misses': self.stats['pool_misses'],
                'hit_rate': self.stats['pool_hits'] / total_requests,
                'memory_saved_mb': self.stats['memory_saved_bytes'] / (1024 * 1024),
                'pooled_float_arrays': float_arrays_count,
                'pooled_int_arrays': int_arrays_count,
                'pooled_series': len(self._series_pool),
                'unique_array_sizes': len(self._float_arrays) + len(self._int_arrays),
                'last_gc_time': self.stats['last_gc_time'],
                'gc_collections': self.stats['gc_collections']
            }


class VectorizedOps:
    """
    Operazioni vettorizzate ottimizzate per forecasting ARIMA.
    
    Collection di funzioni NumPy ottimizzate per operazioni comuni
    nel processo di fitting e forecasting ARIMA.
    """
    
    @staticmethod
    def fast_autocorr(series: np.ndarray, max_lags: int = 20) -> np.ndarray:
        """
        Calcolo autocorrelazione ottimizzato usando FFT.
        
        Args:
            series: Serie temporale
            max_lags: Numero massimo di lag da calcolare
            
        Returns:
            Array di autocorrelazioni per ogni lag
        """
        n = len(series)
        if n < 2:
            return np.array([1.0])
            
        # Mean-center la serie
        mean_centered = series - np.mean(series)
        
        # Pad con zeri per FFT
        padded_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        padded = np.zeros(padded_size)
        padded[:n] = mean_centered
        
        # FFT autocorrelation
        fft_result = np.fft.fft(padded)
        autocorr_full = np.fft.ifft(fft_result * np.conj(fft_result)).real
        
        # Normalizza e prendi solo i primi max_lags
        autocorr_full = autocorr_full[:n] / autocorr_full[0]
        
        return autocorr_full[:min(max_lags + 1, n)]
    
    @staticmethod 
    def fast_diff(series: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Differenziazione ottimizzata per stazionarietà.
        
        Args:
            series: Serie da differenziare
            order: Ordine di differenziazione
            
        Returns:
            Serie differenziata
        """
        result = series.copy()
        
        for _ in range(order):
            if len(result) <= 1:
                break
            result = np.diff(result)
            
        return result
    
    @staticmethod
    def fast_rolling_window(series: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling window ottimizzato usando stride tricks.
        
        Args:
            series: Serie temporale
            window: Dimensione finestra
            
        Returns:
            Array 2D con finestre rolling
        """
        if window > len(series):
            return np.array([])
            
        # Usa numpy stride tricks per evitare copie
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(series, window_shape=window)
    
    @staticmethod
    def fast_moving_average(series: np.ndarray, window: int) -> np.ndarray:
        """
        Media mobile ottimizzata usando convoluzione.
        
        Args:
            series: Serie temporale
            window: Dimensione finestra
            
        Returns:
            Media mobile
        """
        if window > len(series):
            return np.full(len(series), np.mean(series))
            
        # Usa convoluzione per calcolo efficiente
        kernel = np.ones(window) / window
        
        # Mode 'same' mantiene dimensione originale
        return np.convolve(series, kernel, mode='same')
    
    @staticmethod
    def fast_trend_detection(series: np.ndarray) -> Tuple[float, float, bool]:
        """
        Rilevamento trend ottimizzato usando regressione lineare vettorizzata.
        
        Args:
            series: Serie temporale
            
        Returns:
            Tupla (slope, r_squared, has_significant_trend)
        """
        n = len(series)
        if n < 3:
            return 0.0, 0.0, False
            
        # Regressione lineare vettorizzata
        x = np.arange(n)
        x_mean = np.mean(x)
        y_mean = np.mean(series)
        
        # Calcoli vettorizzati
        x_diff = x - x_mean
        y_diff = series - y_mean
        
        slope = np.sum(x_diff * y_diff) / np.sum(x_diff ** 2)
        
        # R-squared
        y_pred = slope * x_diff + y_mean
        ss_res = np.sum((series - y_pred) ** 2)
        ss_tot = np.sum(y_diff ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Significatività trend (R² > 0.1 e slope non trascurabile)
        has_trend = r_squared > 0.1 and abs(slope) > (np.std(series) / n)
        
        return slope, r_squared, has_trend


# Istanza globale memory pool
_global_memory_pool: Optional[MemoryPool] = None


def get_memory_pool() -> MemoryPool:
    """Ottiene istanza globale del memory pool."""
    global _global_memory_pool
    
    if _global_memory_pool is None:
        _global_memory_pool = MemoryPool()
    
    return _global_memory_pool


def configure_memory_pool(max_array_size: int = 10000,
                         max_pool_size: int = 20,
                         enable_monitoring: bool = True):
    """Configura istanza globale memory pool."""
    global _global_memory_pool
    
    _global_memory_pool = MemoryPool(
        max_array_size=max_array_size,
        max_pool_size=max_pool_size,
        enable_monitoring=enable_monitoring
    )
    
    logger.info("Global memory pool configured")


# Context manager per gestione automatica memoria
class ManagedArray:
    """Context manager per gestione automatica array dal pool."""
    
    def __init__(self, size: int, dtype: str = 'float', fill_value: Optional[float] = None):
        self.size = size
        self.dtype = dtype
        self.fill_value = fill_value
        self.array = None
        self.pool = get_memory_pool()
    
    def __enter__(self) -> np.ndarray:
        if self.dtype == 'float':
            self.array = self.pool.get_float_array(self.size, self.fill_value)
        elif self.dtype == 'int':
            self.array = self.pool.get_int_array(self.size, self.fill_value)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return self.array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.array is not None:
            self.pool.return_array(self.array)