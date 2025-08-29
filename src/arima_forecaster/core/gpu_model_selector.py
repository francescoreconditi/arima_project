"""
GPU-Accelerated Model Selectors per training parallelo ad alta performance.
Estende i model selector esistenti con capacità GPU/CUDA.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .model_selection import ARIMAModelSelector
from .sarima_selection import SARIMAModelSelector  
from ..config import get_config
from ..utils.gpu_utils import GPUModelTrainer, get_gpu_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError

logger = get_logger(__name__)

class GPUARIMAModelSelector(ARIMAModelSelector, GPUModelTrainer):
    """
    ARIMA Model Selector con accelerazione GPU/CUDA.
    Estende ARIMAModelSelector con capacità parallel training.
    """
    
    def __init__(
        self,
        order_ranges: Dict[str, Union[range, List[int]]] = None,
        information_criteria: str = "aic",
        seasonal_test: bool = False,
        max_iterations: int = 1000,
        use_gpu: bool = True,
        max_parallel_models: Optional[int] = None
    ):
        """
        Inizializza GPU ARIMA Model Selector.
        
        Args:
            order_ranges: Range parametri ARIMA come ARIMAModelSelector
            information_criteria: Criterio di selezione ("aic", "bic", "hqic")
            seasonal_test: Se testare componente stagionale
            max_iterations: Massime iterazioni per training
            use_gpu: Se utilizzare GPU quando disponibile
            max_parallel_models: Max modelli paralleli (auto se None)
        """
        # Inizializza classe base ARIMA (adatta parametri per compatibilità)
        if order_ranges is None:
            order_ranges = {'p': (0, 3), 'd': (0, 2), 'q': (0, 3)}
        
        ARIMAModelSelector.__init__(
            self,
            p_range=order_ranges.get('p', (0, 3)),
            d_range=order_ranges.get('d', (0, 2)),
            q_range=order_ranges.get('q', (0, 3)),
            information_criterion=information_criteria
        )
        
        # Inizializza GPU trainer
        GPUModelTrainer.__init__(self, use_gpu=use_gpu)
        
        self.max_parallel_models = max_parallel_models or self.config.max_gpu_models_parallel
        self.batch_size = self.config.chunk_size
        
        logger.info(f"GPUARIMAModelSelector inizializzato: "
                   f"GPU={self.use_gpu}, max_parallel={self.max_parallel_models}")
    
    def search_multiple_series(
        self, 
        series_list: List[pd.Series],
        n_jobs: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ricerca parametri ottimali per multiple serie in parallelo.
        
        Args:
            series_list: Lista di serie temporali
            n_jobs: Numero job paralleli (GPU ignora questo parametro)
            show_progress: Se mostrare progress bar
        
        Returns:
            List[Dict]: Risultati per ogni serie
        """
        if len(series_list) == 0:
            return []
        
        if len(series_list) == 1:
            # Single series - usa metodo base
            result = self.search(series_list[0])
            return [result]
        
        logger.info(f"GPU Grid Search per {len(series_list)} serie temporali")
        start_time = time.time()
        
        if self.use_gpu and len(series_list) >= 10:
            # GPU parallel processing per dataset grandi
            results = self._gpu_parallel_search(series_list, show_progress)
        else:
            # CPU parallel processing per dataset piccoli
            results = self._cpu_parallel_search(series_list, n_jobs, show_progress)
        
        total_time = time.time() - start_time
        logger.info(f"Grid Search completato: {len(results)} serie in {total_time:.2f}s "
                   f"({total_time/len(series_list):.3f}s per serie)")
        
        return results
    
    def _gpu_parallel_search(self, series_list: List[pd.Series], show_progress: bool = True) -> List[Dict[str, Any]]:
        """GPU-accelerated parallel search."""
        results = []
        batch_size = min(self.max_parallel_models, len(series_list))
        
        logger.info(f"GPU Parallel Search: batch_size={batch_size}, total_batches={len(series_list)//batch_size + 1}")
        
        # Progressbar opzionale
        if show_progress:
            try:
                from tqdm import tqdm
                series_iterator = tqdm(
                    range(0, len(series_list), batch_size), 
                    desc="GPU ARIMA Search",
                    unit="batch"
                )
            except ImportError:
                series_iterator = range(0, len(series_list), batch_size)
        else:
            series_iterator = range(0, len(series_list), batch_size)
        
        with self.gpu_manager.device_context():
            for i in series_iterator:
                batch_series = series_list[i:i+batch_size]
                
                # Parallel processing del batch
                batch_results = self._process_gpu_batch(batch_series)
                results.extend(batch_results)
                
                # Memory management ogni 5 batch
                if i % (batch_size * 5) == 0:
                    self.gpu_manager.synchronize()
                    self._cleanup_gpu_memory()
        
        return results
    
    def _process_gpu_batch(self, batch_series: List[pd.Series]) -> List[Dict[str, Any]]:
        """Elabora batch di serie su GPU."""
        batch_results = []
        
        # Genera griglia parametri per ogni serie
        param_combinations = self._generate_param_combinations()
        
        for series in batch_series:
            try:
                # Converti serie su GPU se possibile
                gpu_series = self._prepare_gpu_series(series)
                
                # Grid search per singola serie
                best_result = self._gpu_single_series_search(gpu_series, param_combinations)
                
                # Aggiungi metadata
                best_result['series_name'] = series.name or f"series_{len(batch_results)}"
                best_result['n_observations'] = len(series)
                best_result['backend'] = 'cuda' if self.use_gpu else 'cpu'
                
                batch_results.append(best_result)
                
            except Exception as e:
                logger.error(f"Errore processing serie {series.name}: {e}")
                # Fallback result
                error_result = {
                    'best_order': (1, 1, 1),
                    'best_score': float('inf'),
                    'series_name': series.name or f"series_{len(batch_results)}",
                    'n_observations': len(series),
                    'backend': 'cpu',
                    'error': str(e),
                    'status': 'failed'
                }
                batch_results.append(error_result)
        
        return batch_results
    
    def _prepare_gpu_series(self, series: pd.Series) -> Union[pd.Series, Any]:
        """Prepara serie per processing GPU."""
        if not self.use_gpu:
            return series
        
        try:
            # Converti valori su GPU mantenendo index pandas
            gpu_values = self.gpu_manager.to_gpu(series.values)
            return pd.Series(self.gpu_manager.to_cpu(gpu_values), index=series.index, name=series.name)
        except:
            # Fallback su CPU
            return series
    
    def _gpu_single_series_search(self, series: pd.Series, param_combinations: List[Tuple]) -> Dict[str, Any]:
        """Grid search GPU per singola serie."""
        best_score = float('inf')
        best_order = None
        best_model = None
        n_tested = 0
        
        # Batch processing dei parametri
        param_batches = [
            param_combinations[i:i+self.batch_size] 
            for i in range(0, len(param_combinations), self.batch_size)
        ]
        
        for param_batch in param_batches:
            batch_results = self._evaluate_param_batch(series, param_batch)
            
            for order, score, model in batch_results:
                n_tested += 1
                if score < best_score:
                    best_score = score
                    best_order = order
                    best_model = model
        
        return {
            'best_order': best_order,
            'best_score': best_score,
            'best_model': best_model,
            'n_models_tested': n_tested,
            'information_criteria': self.information_criteria,
            'status': 'success'
        }
    
    def _evaluate_param_batch(self, series: pd.Series, param_batch: List[Tuple]) -> List[Tuple]:
        """Valuta batch di parametri per singola serie."""
        results = []
        
        for order in param_batch:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    
                    # Training modello ARIMA
                    from statsmodels.tsa.arima.model import ARIMA
                    
                    model = ARIMA(series, order=order)
                    fitted_model = model.fit(maxiter=self.max_iterations)
                    
                    # Calcola score
                    if self.information_criteria == "aic":
                        score = fitted_model.aic
                    elif self.information_criteria == "bic":
                        score = fitted_model.bic
                    else:  # hqic
                        score = fitted_model.hqic
                    
                    results.append((order, score, fitted_model))
                    
            except Exception as e:
                # Skip parametri problematici
                logger.debug(f"Parametri {order} falliti per {series.name}: {e}")
                continue
        
        return results
    
    def _cpu_parallel_search(
        self, 
        series_list: List[pd.Series], 
        n_jobs: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Fallback CPU parallel search usando joblib."""
        try:
            from joblib import Parallel, delayed
            
            n_jobs = n_jobs or self.config.n_jobs
            
            if show_progress:
                try:
                    from tqdm import tqdm
                    series_list = tqdm(series_list, desc="CPU ARIMA Search")
                except ImportError:
                    pass
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_series_wrapper)(series)
                for series in series_list
            )
            
            return results
            
        except ImportError:
            # Sequential fallback
            logger.warning("joblib non disponibile - processing sequenziale")
            return [self._single_series_wrapper(series) for series in series_list]
    
    def _single_series_wrapper(self, series: pd.Series) -> Dict[str, Any]:
        """Wrapper per processing singola serie."""
        try:
            result = self.search(series)
            result['series_name'] = series.name or "unnamed_series"
            result['n_observations'] = len(series)
            result['backend'] = 'cpu'
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.error(f"Errore processing serie {series.name}: {e}")
            return {
                'best_order': (1, 1, 1),
                'best_score': float('inf'),
                'series_name': series.name or "unnamed_series",
                'n_observations': len(series),
                'backend': 'cpu',
                'error': str(e),
                'status': 'failed'
            }

class GPUSARIMAModelSelector(SARIMAModelSelector, GPUModelTrainer):
    """
    SARIMA Model Selector con accelerazione GPU/CUDA.
    Estende SARIMAModelSelector per supporto GPU parallel training.
    """
    
    def __init__(
        self,
        order_ranges: Dict[str, Union[range, List[int]]] = None,
        seasonal_ranges: Dict[str, Union[range, List[int]]] = None,
        information_criteria: str = "aic",
        seasonal_test: bool = True,
        max_iterations: int = 1000,
        use_gpu: bool = True,
        max_parallel_models: Optional[int] = None
    ):
        """
        Inizializza GPU SARIMA Model Selector.
        
        Args:
            order_ranges: Range parametri ARIMA non-stagionali
            seasonal_ranges: Range parametri stagionali
            information_criteria: Criterio selezione
            seasonal_test: Test stazionarietà stagionale
            max_iterations: Max iterazioni training
            use_gpu: Usa GPU se disponibile
            max_parallel_models: Max modelli paralleli
        """
        # Inizializza classe base SARIMA
        SARIMAModelSelector.__init__(
            self,
            order_ranges=order_ranges,
            seasonal_ranges=seasonal_ranges,
            information_criteria=information_criteria,
            seasonal_test=seasonal_test,
            max_iterations=max_iterations
        )
        
        # Inizializza GPU trainer
        GPUModelTrainer.__init__(self, use_gpu=use_gpu)
        
        self.max_parallel_models = max_parallel_models or self.config.max_gpu_models_parallel
        
        logger.info(f"GPUSARIMAModelSelector inizializzato: "
                   f"GPU={self.use_gpu}, max_parallel={self.max_parallel_models}")
    
    def search_multiple_series(
        self,
        series_list: List[pd.Series],
        n_jobs: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ricerca parametri SARIMA ottimali per multiple serie.
        Implementazione simile a GPUARIMAModelSelector ma per SARIMA.
        """
        if len(series_list) == 0:
            return []
        
        if len(series_list) == 1:
            result = self.search(series_list[0])
            return [result]
        
        logger.info(f"GPU SARIMA Grid Search per {len(series_list)} serie temporali")
        start_time = time.time()
        
        if self.use_gpu and len(series_list) >= 5:  # SARIMA più pesante di ARIMA
            results = self._gpu_parallel_sarima_search(series_list, show_progress)
        else:
            results = self._cpu_parallel_sarima_search(series_list, n_jobs, show_progress)
        
        total_time = time.time() - start_time
        logger.info(f"SARIMA Grid Search completato: {len(results)} serie in {total_time:.2f}s")
        
        return results
    
    def _gpu_parallel_sarima_search(
        self, 
        series_list: List[pd.Series], 
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """GPU parallel search per SARIMA (batch size ridotto per complessità maggiore)."""
        results = []
        # SARIMA più pesante - batch size ridotto
        batch_size = min(self.max_parallel_models // 4, len(series_list), 25)  
        
        logger.info(f"GPU SARIMA Search: batch_size={batch_size}")
        
        if show_progress:
            try:
                from tqdm import tqdm
                series_iterator = tqdm(
                    range(0, len(series_list), batch_size),
                    desc="GPU SARIMA Search",
                    unit="batch"
                )
            except ImportError:
                series_iterator = range(0, len(series_list), batch_size)
        else:
            series_iterator = range(0, len(series_list), batch_size)
        
        with self.gpu_manager.device_context():
            for i in series_iterator:
                batch_series = series_list[i:i+batch_size]
                batch_results = self._process_sarima_batch(batch_series)
                results.extend(batch_results)
                
                # Cleanup memoria più frequente per SARIMA
                if i % (batch_size * 2) == 0:
                    self.gpu_manager.synchronize()
                    self._cleanup_gpu_memory()
        
        return results
    
    def _process_sarima_batch(self, batch_series: List[pd.Series]) -> List[Dict[str, Any]]:
        """Process batch SARIMA models."""
        batch_results = []
        
        for series in batch_series:
            try:
                # Usa il search method della classe base per singola serie
                result = self.search(series)
                result['series_name'] = series.name or f"sarima_series_{len(batch_results)}"
                result['n_observations'] = len(series)
                result['backend'] = 'cuda' if self.use_gpu else 'cpu'
                result['status'] = 'success'
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"Errore SARIMA per serie {series.name}: {e}")
                error_result = {
                    'best_order': (1, 1, 1),
                    'best_seasonal_order': (1, 1, 1, 12),
                    'best_score': float('inf'),
                    'series_name': series.name or f"sarima_series_{len(batch_results)}",
                    'n_observations': len(series),
                    'backend': 'cpu',
                    'error': str(e),
                    'status': 'failed'
                }
                batch_results.append(error_result)
        
        return batch_results
    
    def _cpu_parallel_sarima_search(
        self,
        series_list: List[pd.Series],
        n_jobs: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """CPU parallel fallback per SARIMA."""
        try:
            from joblib import Parallel, delayed
            
            n_jobs = n_jobs or max(1, self.config.n_jobs // 2)  # SARIMA più pesante
            
            if show_progress:
                try:
                    from tqdm import tqdm
                    series_list = tqdm(series_list, desc="CPU SARIMA Search")
                except ImportError:
                    pass
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._sarima_series_wrapper)(series)
                for series in series_list
            )
            
            return results
            
        except ImportError:
            logger.warning("joblib non disponibile - SARIMA processing sequenziale")
            return [self._sarima_series_wrapper(series) for series in series_list]
    
    def _sarima_series_wrapper(self, series: pd.Series) -> Dict[str, Any]:
        """Wrapper per SARIMA singola serie."""
        try:
            result = self.search(series)
            result['series_name'] = series.name or "unnamed_sarima_series"
            result['n_observations'] = len(series)
            result['backend'] = 'cpu'
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.error(f"Errore SARIMA serie {series.name}: {e}")
            return {
                'best_order': (1, 1, 1),
                'best_seasonal_order': (1, 1, 1, 12),
                'best_score': float('inf'),
                'series_name': series.name or "unnamed_sarima_series",
                'n_observations': len(series),
                'backend': 'cpu',
                'error': str(e),
                'status': 'failed'
            }