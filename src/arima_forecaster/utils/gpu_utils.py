"""
GPU/CUDA Utilities per accelerazione calcoli.
Fornisce interfacce unificate per operazioni GPU-accelerate.
"""

import numpy as np
import warnings
from typing import Optional, Union, Any, List, Tuple
from contextlib import contextmanager

from ..config import get_config, GPUBackend, detect_gpu_capability
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GPUArrayManager:
    """
    Manager per operazioni array GPU/CPU unificate.
    Gestisce automaticamente il backend ottimale e le conversioni.
    """
    
    def __init__(self, backend: Optional[str] = None):
        """
        Inizializza GPUArrayManager.
        
        Args:
            backend: Backend da utilizzare ("cpu", "cuda", "auto")
        """
        self.config = get_config()
        self.backend = backend or self.config.backend
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup del backend array appropriato."""
        if self.backend == "cuda":
            try:
                import cupy as cp
                self.xp = cp
                self.backend_type = GPUBackend.CUDA
                
                # Set GPU device
                cp.cuda.Device(self.config.cuda_device).use()
                
                # Set memory pool if limit specified
                if self.config.gpu_memory_limit:
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=int(self.config.gpu_memory_limit * 1024**3))
                
                logger.info(f"GPUArrayManager: CuPy backend inizializzato (device: {self.config.cuda_device})")
                
            except ImportError:
                logger.warning("CuPy non disponibile - fallback su NumPy")
                self.xp = np
                self.backend_type = GPUBackend.CPU
            except Exception as e:
                logger.error(f"Errore setup CuPy: {e} - fallback su NumPy")
                self.xp = np
                self.backend_type = GPUBackend.CPU
        else:
            self.xp = np
            self.backend_type = GPUBackend.CPU
            logger.info("GPUArrayManager: NumPy backend inizializzato")
    
    def array(self, data: Any, dtype=None) -> Any:
        """Crea array usando il backend appropriato."""
        if dtype is None:
            return self.xp.array(data)
        return self.xp.array(data, dtype=dtype)
    
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype=None) -> Any:
        """Crea array di zeri."""
        if dtype is None:
            return self.xp.zeros(shape)
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Union[int, Tuple[int, ...]], dtype=None) -> Any:
        """Crea array di uno."""
        if dtype is None:
            return self.xp.ones(shape)
        return self.xp.ones(shape, dtype=dtype)
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Sposta array su GPU se backend CUDA."""
        if self.backend_type == GPUBackend.CUDA and isinstance(array, np.ndarray):
            return self.xp.asarray(array)
        return array
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """Sposta array su CPU."""
        if self.backend_type == GPUBackend.CUDA:
            try:
                return self.xp.asnumpy(array)
            except:
                pass
        return np.asarray(array)
    
    def synchronize(self):
        """Sincronizza operazioni GPU."""
        if self.backend_type == GPUBackend.CUDA:
            try:
                self.xp.cuda.Stream.null.synchronize()
            except:
                pass
    
    @contextmanager
    def device_context(self, device_id: Optional[int] = None):
        """Context manager per device GPU."""
        if self.backend_type == GPUBackend.CUDA:
            device_id = device_id or self.config.cuda_device
            try:
                import cupy as cp
                with cp.cuda.Device(device_id):
                    yield
            except:
                yield
        else:
            yield
    
    def get_memory_info(self) -> Optional[Tuple[int, int]]:
        """
        Ottiene informazioni memoria GPU.
        
        Returns:
            Tuple[int, int]: (memoria_libera, memoria_totale) in bytes, None se CPU
        """
        if self.backend_type == GPUBackend.CUDA:
            try:
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                return mempool.free_bytes(), mempool.total_bytes()
            except:
                pass
        return None

class GPUModelTrainer:
    """
    Base class per training modelli GPU-accelerati.
    Gestisce batch processing e parallelizzazione GPU.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Inizializza GPU trainer.
        
        Args:
            use_gpu: Se utilizzare GPU se disponibile
        """
        self.config = get_config()
        self.gpu_manager = GPUArrayManager(
            backend="cuda" if use_gpu else "cpu"
        )
        self.use_gpu = self.gpu_manager.backend_type == GPUBackend.CUDA
        
    def batch_process(self, data_list: List[Any], batch_size: Optional[int] = None) -> List[Any]:
        """
        Elabora dati in batch ottimizzati per GPU.
        
        Args:
            data_list: Lista di dati da elaborare
            batch_size: Dimensione batch (auto se None)
        
        Returns:
            List[Any]: Risultati elaborazione
        """
        if batch_size is None:
            batch_size = self.config.chunk_size if self.use_gpu else len(data_list)
        
        results = []
        
        with self.gpu_manager.device_context():
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                batch_result = self._process_batch(batch)
                results.extend(batch_result)
                
                # Sincronizza per gestire memoria
                if self.use_gpu and i % (batch_size * 10) == 0:
                    self.gpu_manager.synchronize()
        
        return results
    
    def _process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Elabora un singolo batch. Da implementare nelle sottoclassi.
        
        Args:
            batch: Batch di dati
        
        Returns:
            List[Any]: Risultati batch
        """
        raise NotImplementedError("Implementare _process_batch nelle sottoclassi")
    
    def parallel_grid_search(self, series_list: List[Any], param_grids: List[dict]) -> List[dict]:
        """
        Grid search parallelo GPU-accelerato.
        
        Args:
            series_list: Lista serie temporali
            param_grids: Lista griglie parametri
        
        Returns:
            List[dict]: Risultati migliori per ogni serie
        """
        if not self.use_gpu or len(series_list) < 10:
            # Fallback CPU per dataset piccoli
            return self._cpu_grid_search(series_list, param_grids)
        
        return self._gpu_grid_search(series_list, param_grids)
    
    def _cpu_grid_search(self, series_list: List[Any], param_grids: List[dict]) -> List[dict]:
        """Grid search CPU con joblib."""
        try:
            from joblib import Parallel, delayed
            
            def search_single_series(series, param_grid):
                return self._single_series_search(series, param_grid)
            
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(search_single_series)(series, param_grid)
                for series, param_grid in zip(series_list, param_grids)
            )
            
            return results
            
        except ImportError:
            # Sequential fallback
            return [
                self._single_series_search(series, param_grid)
                for series, param_grid in zip(series_list, param_grids)
            ]
    
    def _gpu_grid_search(self, series_list: List[Any], param_grids: List[dict]) -> List[dict]:
        """Grid search GPU-accelerato con batch processing."""
        results = []
        batch_size = min(self.config.max_gpu_models_parallel, len(series_list))
        
        logger.info(f"GPU Grid Search: {len(series_list)} serie, batch_size={batch_size}")
        
        with self.gpu_manager.device_context():
            for i in range(0, len(series_list), batch_size):
                batch_series = series_list[i:i+batch_size]
                batch_grids = param_grids[i:i+batch_size]
                
                # Converti batch su GPU
                gpu_batch = [self.gpu_manager.to_gpu(series) for series in batch_series]
                
                # Elabora batch
                batch_results = [
                    self._single_series_search(series, param_grid)
                    for series, param_grid in zip(gpu_batch, batch_grids)
                ]
                
                results.extend(batch_results)
                
                # Pulizia memoria GPU
                if i % (batch_size * 5) == 0:
                    self.gpu_manager.synchronize()
                    self._cleanup_gpu_memory()
        
        return results
    
    def _single_series_search(self, series: Any, param_grid: dict) -> dict:
        """
        Ricerca parametri ottimali per singola serie. 
        Da implementare nelle sottoclassi.
        """
        raise NotImplementedError("Implementare _single_series_search nelle sottoclassi")
    
    def _cleanup_gpu_memory(self):
        """Pulizia memoria GPU."""
        if self.use_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass

def get_gpu_manager() -> GPUArrayManager:
    """Factory function per GPUArrayManager globale."""
    if not hasattr(get_gpu_manager, '_instance'):
        get_gpu_manager._instance = GPUArrayManager()
    return get_gpu_manager._instance

def reset_gpu_manager():
    """Reset GPU manager globale (per testing)."""
    if hasattr(get_gpu_manager, '_instance'):
        delattr(get_gpu_manager, '_instance')

@contextmanager
def gpu_context(device_id: Optional[int] = None):
    """Context manager globale per operazioni GPU."""
    manager = get_gpu_manager()
    with manager.device_context(device_id):
        yield manager

def benchmark_gpu_vs_cpu(operation_func, data, iterations: int = 10) -> dict:
    """
    Benchmark operazione su GPU vs CPU.
    
    Args:
        operation_func: Funzione da benchmarkare
        data: Dati test
        iterations: Numero iterazioni
    
    Returns:
        dict: Risultati benchmark
    """
    import time
    
    # CPU benchmark
    gpu_manager_cpu = GPUArrayManager(backend="cpu")
    cpu_times = []
    
    for _ in range(iterations):
        start = time.time()
        operation_func(gpu_manager_cpu, data)
        cpu_times.append(time.time() - start)
    
    cpu_avg = np.mean(cpu_times)
    
    # GPU benchmark se disponibile
    gpu_avg = None
    capability = detect_gpu_capability()
    
    if capability.has_cuda:
        gpu_manager_gpu = GPUArrayManager(backend="cuda")
        gpu_times = []
        
        for _ in range(iterations):
            start = time.time()
            operation_func(gpu_manager_gpu, data)
            gpu_manager_gpu.synchronize()  # Ensure GPU completion
            gpu_times.append(time.time() - start)
        
        gpu_avg = np.mean(gpu_times)
    
    return {
        'cpu_time': cpu_avg,
        'gpu_time': gpu_avg,
        'speedup': cpu_avg / gpu_avg if gpu_avg else None,
        'gpu_available': capability.has_cuda
    }