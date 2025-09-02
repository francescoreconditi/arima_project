"""
Batch Forecast Processor
Sistema per analisi portfolio massivi con parallelizzazione automatica

Autore: Claude Code
Data: 2025-09-02
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import pickle

from arima_forecaster.automl.auto_selector import AutoForecastSelector
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class BatchStatus(str, Enum):
    """Status di processing batch"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchTaskResult:
    """Risultato singolo task nel batch"""
    task_id: str
    status: BatchStatus
    model_type: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    forecast: Optional[np.ndarray] = None
    explanation: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class BatchProgress:
    """Progress tracking per batch job"""
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_completion: Optional[float] = None
    
    @property
    def completion_percentage(self) -> float:
        return (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
        
    @property
    def eta_seconds(self) -> Optional[float]:
        if self.completed_tasks == 0:
            return None
        rate = self.completed_tasks / self.elapsed_time
        remaining = self.total_tasks - self.completed_tasks
        return remaining / rate if rate > 0 else None


class BatchForecastProcessor:
    """
    Processore batch per portfolio forecasting enterprise
    
    Features:
    - Parallelizzazione automatica (process/thread pool)
    - Progress tracking real-time
    - Error handling robusto  
    - Resume capability per job interrotti
    - Export risultati multiple formats
    """
    
    def __init__(self,
                 max_workers: Optional[int] = None,
                 parallel_mode: str = "process",  # "process", "thread", "serial"
                 progress_callback: Optional[Callable] = None,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Inizializza batch processor
        
        Args:
            max_workers: Numero worker paralleli (default: CPU count)
            parallel_mode: Modalità parallelizzazione
            progress_callback: Callback per progress updates
            cache_dir: Directory per cache risultati
            verbose: Output dettagliato
        """
        self.max_workers = max_workers
        self.parallel_mode = parallel_mode
        self.progress_callback = progress_callback
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = verbose
        
        # State
        self.current_batch_id = None
        self.progress = None
        self.results: Dict[str, BatchTaskResult] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def fit_batch(self,
                  datasets: Dict[str, Union[pd.Series, np.ndarray, List]],
                  automl_config: Optional[Dict[str, Any]] = None,
                  forecast_steps: int = 30,
                  batch_id: Optional[str] = None) -> Dict[str, BatchTaskResult]:
        """
        Processa batch di dataset con AutoML
        
        Args:
            datasets: Dict {task_id: time_series_data}
            automl_config: Configurazione AutoML
            forecast_steps: Steps forecast per ogni serie
            batch_id: ID batch (generato se None)
            
        Returns:
            Dict con risultati per ogni task
        """
        # Setup batch
        batch_id = batch_id or f"batch_{int(time.time())}"
        self.current_batch_id = batch_id
        
        if self.verbose:
            print(f"[BATCH] Starting batch job: {batch_id}")
            print(f"[BATCH] Tasks: {len(datasets)}")
            print(f"[BATCH] Parallel mode: {self.parallel_mode}")
            print(f"[BATCH] Workers: {self._get_worker_count()}")
            
        # Initialize progress
        self.progress = BatchProgress(total_tasks=len(datasets))
        self.results = {}
        
        # Load cached results if available
        if self.cache_dir:
            self._load_cached_results(batch_id)
            
        # Filter uncompleted tasks
        pending_tasks = {
            task_id: data for task_id, data in datasets.items()
            if task_id not in self.results or self.results[task_id].status != BatchStatus.COMPLETED
        }
        
        if self.verbose and len(pending_tasks) < len(datasets):
            cached_count = len(datasets) - len(pending_tasks)
            print(f"[CACHE] Loaded {cached_count} cached results")
            
        # Process tasks
        if pending_tasks:
            if self.parallel_mode == "serial":
                self._process_serial(pending_tasks, automl_config, forecast_steps)
            else:
                self._process_parallel(pending_tasks, automl_config, forecast_steps)
        
        # Save results
        if self.cache_dir:
            self._save_results_cache(batch_id)
            
        if self.verbose:
            self._print_final_summary()
            
        return dict(self.results)
        
    def _get_worker_count(self) -> int:
        """Calcola numero ottimale worker"""
        if self.max_workers:
            return self.max_workers
            
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if self.parallel_mode == "process":
            return min(cpu_count, 8)  # Max 8 processi per evitare overhead
        else:
            return min(cpu_count * 2, 16)  # Thread pool più grande
            
    def _process_serial(self,
                       tasks: Dict[str, Any],
                       automl_config: Dict,
                       forecast_steps: int):
        """Processamento seriale per debug"""
        for task_id, data in tasks.items():
            if self.verbose:
                print(f"[TASK] Processing {task_id}...")
                
            result = self._process_single_task(
                task_id, data, automl_config, forecast_steps
            )
            
            self.results[task_id] = result
            self.progress.completed_tasks += 1
            
            if result.status == BatchStatus.FAILED:
                self.progress.failed_tasks += 1
                
            self._update_progress()
            
    def _process_parallel(self,
                         tasks: Dict[str, Any], 
                         automl_config: Dict,
                         forecast_steps: int):
        """Processamento parallelo con thread/process pool"""
        
        executor_class = ProcessPoolExecutor if self.parallel_mode == "process" else ThreadPoolExecutor
        
        with executor_class(max_workers=self._get_worker_count()) as executor:
            # Submit tutti i task
            future_to_task = {
                executor.submit(
                    self._process_single_task,
                    task_id, data, automl_config, forecast_steps
                ): task_id
                for task_id, data in tasks.items()
            }
            
            # Collect results as completed
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                
                try:
                    result = future.result(timeout=300)  # 5 min timeout per task
                    self.results[task_id] = result
                    
                    if result.status == BatchStatus.FAILED:
                        self.progress.failed_tasks += 1
                        
                except Exception as e:
                    # Handle executor exceptions
                    self.results[task_id] = BatchTaskResult(
                        task_id=task_id,
                        status=BatchStatus.FAILED,
                        error_message=f"Executor error: {str(e)}"
                    )
                    self.progress.failed_tasks += 1
                    
                    if self.verbose:
                        print(f"[ERROR] Task {task_id} failed: {str(e)}")
                
                self.progress.completed_tasks += 1
                self._update_progress()
                
    def _process_single_task(self,
                            task_id: str,
                            data: Any, 
                            automl_config: Optional[Dict],
                            forecast_steps: int) -> BatchTaskResult:
        """
        Processa singolo task con error handling
        """
        start_time = time.time()
        
        try:
            # Convert data to pandas Series if needed
            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            elif not isinstance(data, pd.Series):
                raise ValueError(f"Unsupported data type: {type(data)}")
                
            # Validate data
            if len(data) < 10:
                raise ValueError(f"Insufficient data: {len(data)} observations")
                
            if data.isna().all():
                raise ValueError("All values are NaN")
                
            # Setup AutoML
            config = automl_config or {}
            config.setdefault('verbose', False)  # Quiet per batch
            
            automl = AutoForecastSelector(**config)
            
            # Fit model
            model, explanation = automl.fit(data)
            
            # Generate forecast
            forecast = model.forecast(steps=forecast_steps)
            
            # Extract metadata
            pattern_type = explanation.pattern_detected if explanation else "unknown"
            business_rec = explanation.business_recommendation if explanation else ""
            
            processing_time = time.time() - start_time
            
            return BatchTaskResult(
                task_id=task_id,
                status=BatchStatus.COMPLETED,
                model_type=str(explanation.recommended_model) if explanation else "unknown",
                confidence=explanation.confidence_score if explanation else 0.0,
                processing_time=processing_time,
                forecast=forecast,
                explanation=explanation.why_chosen if explanation else "",
                metadata={
                    'pattern_type': pattern_type,
                    'business_recommendation': business_rec,
                    'data_points': len(data),
                    'forecast_mean': float(np.mean(forecast)),
                    'forecast_std': float(np.std(forecast))
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Add traceback for debugging if verbose
            if self.verbose:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
                
            return BatchTaskResult(
                task_id=task_id,
                status=BatchStatus.FAILED,
                processing_time=processing_time,
                error_message=error_msg
            )
            
    def _update_progress(self):
        """Aggiorna progress e chiama callback"""
        if self.progress_callback:
            self.progress_callback(self.progress)
            
        if self.verbose and self.progress.completed_tasks % 10 == 0:
            pct = self.progress.completion_percentage
            elapsed = self.progress.elapsed_time
            eta = self.progress.eta_seconds
            
            eta_str = f"{eta:.0f}s" if eta else "unknown"
            print(f"[PROGRESS] {pct:.1f}% ({self.progress.completed_tasks}/{self.progress.total_tasks}) | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta_str}")
                  
    def _load_cached_results(self, batch_id: str):
        """Carica risultati dalla cache"""
        cache_file = self.cache_dir / f"{batch_id}_results.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.results = pickle.load(f)
                    
                # Update progress
                completed = sum(1 for r in self.results.values() 
                              if r.status == BatchStatus.COMPLETED)
                failed = sum(1 for r in self.results.values() 
                           if r.status == BatchStatus.FAILED)
                
                self.progress.completed_tasks = completed
                self.progress.failed_tasks = failed
                
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.results = {}
                
    def _save_results_cache(self, batch_id: str):
        """Salva risultati in cache"""
        try:
            cache_file = self.cache_dir / f"{batch_id}_results.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.results, f)
                
            # Save summary as JSON too
            summary_file = self.cache_dir / f"{batch_id}_summary.json"
            summary = self.get_batch_summary()
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            
    def _print_final_summary(self):
        """Stampa summary finale"""
        summary = self.get_batch_summary()
        
        print(f"\n[BATCH] Completed: {self.current_batch_id}")
        print(f"  Total tasks: {summary['total_tasks']}")
        print(f"  Successful: {summary['successful_tasks']} ({summary['success_rate']:.1%})")
        print(f"  Failed: {summary['failed_tasks']}")
        print(f"  Total time: {summary['total_time']:.1f}s")
        print(f"  Avg time/task: {summary['avg_time_per_task']:.1f}s")
        
        if summary['model_distribution']:
            print(f"  Model distribution:")
            for model, count in summary['model_distribution'].items():
                print(f"    {model}: {count}")
                
    def get_batch_summary(self) -> Dict[str, Any]:
        """Genera summary del batch job"""
        completed_results = [r for r in self.results.values() 
                           if r.status == BatchStatus.COMPLETED]
        failed_results = [r for r in self.results.values()
                         if r.status == BatchStatus.FAILED]
        
        # Model distribution
        model_dist = {}
        for result in completed_results:
            model = result.model_type or "unknown"
            model_dist[model] = model_dist.get(model, 0) + 1
            
        # Timing stats
        processing_times = [r.processing_time for r in completed_results 
                          if r.processing_time is not None]
        
        return {
            'batch_id': self.current_batch_id,
            'total_tasks': len(self.results),
            'successful_tasks': len(completed_results),
            'failed_tasks': len(failed_results),
            'success_rate': len(completed_results) / len(self.results) if self.results else 0,
            'total_time': self.progress.elapsed_time if self.progress else 0,
            'avg_time_per_task': np.mean(processing_times) if processing_times else 0,
            'model_distribution': model_dist,
            'avg_confidence': np.mean([r.confidence for r in completed_results 
                                     if r.confidence is not None]) if completed_results else 0
        }
        
    def export_results(self, 
                      format: str = "csv",
                      output_path: Optional[str] = None) -> str:
        """
        Esporta risultati in various formats
        
        Args:
            format: "csv", "excel", "json"
            output_path: Path output file
            
        Returns:
            Path del file generato
        """
        if not self.results:
            raise ValueError("No results to export")
            
        # Prepare data
        export_data = []
        for task_id, result in self.results.items():
            row = {
                'task_id': task_id,
                'status': result.status,
                'model_type': result.model_type,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'explanation': result.explanation,
                'error_message': result.error_message
            }
            
            # Add metadata
            if result.metadata:
                row.update({f"meta_{k}": v for k, v in result.metadata.items()})
                
            # Add forecast statistics
            if result.forecast is not None:
                row.update({
                    'forecast_mean': np.mean(result.forecast),
                    'forecast_std': np.std(result.forecast),
                    'forecast_min': np.min(result.forecast),
                    'forecast_max': np.max(result.forecast)
                })
                
            export_data.append(row)
            
        df = pd.DataFrame(export_data)
        
        # Generate output path
        if not output_path:
            batch_id = self.current_batch_id or "batch_results"
            output_path = f"{batch_id}_results.{format}"
            
        # Export based on format
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        if self.verbose:
            print(f"[EXPORT] Results saved to: {output_path}")
            
        return output_path
        
    def get_failed_tasks(self) -> Dict[str, str]:
        """Ritorna task falliti con error messages"""
        return {
            task_id: result.error_message 
            for task_id, result in self.results.items()
            if result.status == BatchStatus.FAILED
        }
        
    def retry_failed_tasks(self,
                          datasets: Dict[str, Any],
                          automl_config: Optional[Dict] = None,
                          forecast_steps: int = 30) -> Dict[str, BatchTaskResult]:
        """Riprova solo i task falliti"""
        failed_task_ids = [task_id for task_id, result in self.results.items()
                          if result.status == BatchStatus.FAILED]
        
        if not failed_task_ids:
            if self.verbose:
                print("[RETRY] No failed tasks to retry")
            return {}
            
        failed_datasets = {task_id: datasets[task_id] 
                          for task_id in failed_task_ids 
                          if task_id in datasets}
        
        if self.verbose:
            print(f"[RETRY] Retrying {len(failed_datasets)} failed tasks")
            
        return self.fit_batch(failed_datasets, automl_config, forecast_steps)


# Utility functions for common batch scenarios
def process_csv_portfolio(csv_file_path: str,
                         id_column: str,
                         value_column: str,
                         date_column: Optional[str] = None,
                         **batch_kwargs) -> Dict[str, BatchTaskResult]:
    """
    Processa portfolio da file CSV
    
    Args:
        csv_file_path: Path al file CSV
        id_column: Nome colonna ID prodotto/serie
        value_column: Nome colonna valori
        date_column: Nome colonna date (optional)
        **batch_kwargs: Args per BatchForecastProcessor
        
    Returns:
        Risultati batch processing
    """
    # Load data
    df = pd.read_csv(csv_file_path)
    
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values([id_column, date_column])
        
    # Group by ID and create time series
    datasets = {}
    for product_id, group in df.groupby(id_column):
        series = pd.Series(
            group[value_column].values,
            index=group[date_column] if date_column else None,
            name=str(product_id)
        )
        datasets[str(product_id)] = series
        
    # Process batch
    processor = BatchForecastProcessor(**batch_kwargs)
    return processor.fit_batch(datasets)


def process_excel_portfolio(excel_file_path: str,
                           sheet_name: str = 0,
                           **csv_kwargs) -> Dict[str, BatchTaskResult]:
    """Wrapper per file Excel"""
    # Convert Excel to CSV temporarily
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    temp_csv = "temp_portfolio.csv"
    df.to_csv(temp_csv, index=False)
    
    try:
        return process_csv_portfolio(temp_csv, **csv_kwargs)
    finally:
        Path(temp_csv).unlink(missing_ok=True)


if __name__ == "__main__":
    # Quick test
    print("Testing BatchForecastProcessor...")
    
    # Generate test portfolio
    test_data = {}
    for i in range(10):
        # Different patterns for testing
        if i % 3 == 0:
            data = 100 + np.cumsum(np.random.normal(0, 5, 200))  # trend
        elif i % 3 == 1:
            x = np.arange(200)
            data = 100 + 20 * np.sin(2 * np.pi * x / 30) + np.random.normal(0, 5, 200)  # seasonal
        else:
            data = np.random.choice([0, 0, 0, 1, 2, 3], size=200, p=[0.7, 0.1, 0.05, 0.1, 0.03, 0.02])  # intermittent
            
        test_data[f"product_{i:02d}"] = data
        
    # Test batch processing
    processor = BatchForecastProcessor(
        max_workers=4,
        parallel_mode="thread",
        verbose=True
    )
    
    results = processor.fit_batch(test_data, forecast_steps=30)
    
    print(f"\nProcessed {len(results)} products")
    print("Sample results:")
    for i, (task_id, result) in enumerate(list(results.items())[:3]):
        print(f"  {task_id}: {result.model_type} ({result.confidence:.1%})")
        
    # Test export
    output_file = processor.export_results("csv")
    print(f"Results exported to: {output_file}")