"""
Performance Benchmarking Framework per ARIMA Forecaster.

Sistema completo di benchmarking per monitoraggio performance,
comparazione configurazioni e regression testing.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import psutil
import gc
from datetime import datetime
import concurrent.futures
from threading import Lock

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configurazione per benchmark singolo."""
    name: str
    dataset_sizes: List[int] = None
    arima_orders: List[Tuple[int, int, int]] = None
    use_cache: bool = True
    use_smart_params: bool = True
    use_memory_pool: bool = True
    use_vectorized_ops: bool = True
    repetitions: int = 3
    timeout_seconds: int = 300

    def __post_init__(self):
        if self.dataset_sizes is None:
            self.dataset_sizes = [100, 500, 1000]
        if self.arima_orders is None:
            self.arima_orders = [(1,1,1), (2,1,2), (1,1,2)]


@dataclass
class BenchmarkResult:
    """Risultato di un benchmark singolo."""
    config_name: str
    dataset_size: int
    arima_order: Tuple[int, int, int]
    
    # Performance metrics
    fit_time: float
    forecast_time: float
    total_time: float
    memory_peak_mb: float
    
    # Model quality metrics  
    aic: float
    bic: float
    
    # Optimization metrics
    cache_hit: bool
    preprocessing_applied: bool
    vectorized_used: bool
    
    # System metrics
    cpu_percent: float
    timestamp: datetime
    
    # Metadata
    error: Optional[str] = None
    success: bool = True


class DatasetGenerator:
    """Generatore dataset per benchmark con caratteristiche controllate."""
    
    @staticmethod
    def generate_simple_trend(size: int, seed: int = 42) -> pd.Series:
        """Serie semplice con trend lineare."""
        np.random.seed(seed)
        t = np.arange(size)
        trend = 0.1 * t
        noise = np.random.normal(0, 1, size)
        
        values = 100 + trend + noise
        dates = pd.date_range('2020-01-01', periods=size, freq='D')
        return pd.Series(values, index=dates, name='simple_trend')
    
    @staticmethod  
    def generate_seasonal(size: int, seed: int = 42) -> pd.Series:
        """Serie con stagionalità forte."""
        np.random.seed(seed)
        t = np.arange(size)
        trend = 0.05 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 2, size)
        
        values = 100 + trend + seasonal + noise
        dates = pd.date_range('2020-01-01', periods=size, freq='D')
        return pd.Series(values, index=dates, name='seasonal')
    
    @staticmethod
    def generate_volatile(size: int, seed: int = 42) -> pd.Series:
        """Serie ad alta volatilità."""
        np.random.seed(seed)
        values = np.random.normal(100, 20, size).cumsum()
        dates = pd.date_range('2020-01-01', periods=size, freq='D')
        return pd.Series(values, index=dates, name='volatile')
    
    @staticmethod
    def generate_outliers(size: int, outlier_rate: float = 0.05, seed: int = 42) -> pd.Series:
        """Serie con outlier per test preprocessing."""
        np.random.seed(seed)
        t = np.arange(size)
        trend = 0.1 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise = np.random.normal(0, 3, size)
        
        values = 100 + trend + seasonal + noise
        
        # Aggiungi outlier
        n_outliers = int(size * outlier_rate)
        outlier_indices = np.random.choice(size, n_outliers, replace=False)
        
        for idx in outlier_indices:
            multiplier = 8 * (1 if np.random.rand() > 0.5 else -1)
            values[idx] += multiplier * np.std(values)
        
        dates = pd.date_range('2020-01-01', periods=size, freq='D')
        return pd.Series(values, index=dates, name='with_outliers')


class PerformanceBenchmark:
    """
    Framework completo per benchmarking performance ARIMA Forecaster.
    
    Features:
    - Benchmark configurabili per diversi scenari
    - Comparazione ottimizzazioni on/off  
    - Monitoring risorse sistema
    - Export risultati in JSON/CSV
    - Analisi regression performance
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self._lock = Lock()
        
        logger.info(f"PerformanceBenchmark initialized: output_dir={self.output_dir}")
    
    def run_single_benchmark(
        self, 
        config: BenchmarkConfig,
        dataset: pd.Series,
        arima_order: Tuple[int, int, int]
    ) -> List[BenchmarkResult]:
        """
        Esegue benchmark singolo con ripetizioni per statistica robusta.
        
        Args:
            config: Configurazione benchmark
            dataset: Dataset da utilizzare
            arima_order: Ordine ARIMA da testare
            
        Returns:
            Lista risultati (una entry per ripetizione)
        """
        results = []
        
        for rep in range(config.repetitions):
            logger.debug(f"Running {config.name} rep {rep+1}/{config.repetitions} "
                        f"size={len(dataset)} order={arima_order}")
            
            # Import qui per evitare circular import
            from ..core.arima_model import ARIMAForecaster
            
            # Creazione modello con configurazione specifica
            model = ARIMAForecaster(
                order=arima_order,
                use_cache=config.use_cache,
                use_smart_params=config.use_smart_params,
                use_memory_pool=config.use_memory_pool,
                use_vectorized_ops=config.use_vectorized_ops
            )
            
            # Monitora risorse sistema
            process = psutil.Process()
            cpu_start = process.cpu_percent()
            mem_start = process.memory_info().rss / 1024 / 1024  # MB
            
            result = BenchmarkResult(
                config_name=config.name,
                dataset_size=len(dataset),
                arima_order=arima_order,
                fit_time=0.0,
                forecast_time=0.0,
                total_time=0.0,
                memory_peak_mb=0.0,
                aic=0.0,
                bic=0.0,
                cache_hit=False,
                preprocessing_applied=False,
                vectorized_used=config.use_vectorized_ops,
                cpu_percent=0.0,
                timestamp=datetime.now()
            )
            
            try:
                # 1. FIT TIMING
                gc.collect()  # Clean start
                fit_start = time.time()
                
                model.fit(dataset)
                
                fit_end = time.time()
                result.fit_time = fit_end - fit_start
                
                # 2. FORECAST TIMING
                forecast_start = time.time()
                
                forecast = model.forecast(steps=10)
                
                forecast_end = time.time()
                result.forecast_time = forecast_end - forecast_start
                
                result.total_time = result.fit_time + result.forecast_time
                
                # 3. EXTRACT MODEL METRICS
                result.aic = model.fitted_model.aic
                result.bic = model.fitted_model.bic
                result.cache_hit = getattr(model, 'cache_used', False)
                
                # Preprocessing info da metadata
                opt_used = model.training_metadata.get('optimization_used', {})
                result.preprocessing_applied = opt_used.get('preprocessing_applied', False)
                
                # 4. SYSTEM METRICS
                cpu_end = process.cpu_percent()
                mem_end = process.memory_info().rss / 1024 / 1024
                
                result.cpu_percent = max(cpu_end - cpu_start, 0.0)
                result.memory_peak_mb = max(mem_end - mem_start, 0.0)
                
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
                result.error = str(e)
                result.success = False
            
            results.append(result)
            
            # Small delay tra ripetizioni per stabilizzare sistema
            time.sleep(0.1)
        
        return results
    
    def run_configuration_comparison(
        self, 
        dataset_name: str,
        dataset: pd.Series,
        configurations: List[BenchmarkConfig],
        arima_orders: Optional[List[Tuple[int, int, int]]] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Confronta multiple configurazioni sullo stesso dataset.
        
        Args:
            dataset_name: Nome identificativo dataset
            dataset: Serie temporale da testare
            configurations: Lista configurazioni da confrontare
            arima_orders: Ordini ARIMA da testare (default: [(1,1,1), (2,1,1)])
            
        Returns:
            Dizionario {config_name: [results]}
        """
        if arima_orders is None:
            arima_orders = [(1,1,1), (2,1,1)]
        
        logger.info(f"Running configuration comparison on {dataset_name} "
                   f"({len(dataset)} points, {len(configurations)} configs)")
        
        all_results = {}
        
        for config in configurations:
            config_results = []
            
            for order in arima_orders:
                benchmark_results = self.run_single_benchmark(config, dataset, order)
                config_results.extend(benchmark_results)
            
            all_results[config.name] = config_results
            
            # Store risultati incrementalmente
            with self._lock:
                self.results.extend(config_results)
        
        return all_results
    
    def run_scaling_benchmark(
        self,
        config: BenchmarkConfig,
        dataset_generator: Callable[[int], pd.Series],
        sizes: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """
        Testa scalabilità performance al variare della dimensione dataset.
        
        Args:
            config: Configurazione base
            dataset_generator: Funzione che genera dataset data size
            sizes: Liste dimensioni da testare
            
        Returns:
            Lista risultati scaling test
        """
        if sizes is None:
            sizes = [100, 200, 500, 1000, 2000]
        
        logger.info(f"Running scaling benchmark: {config.name} on sizes {sizes}")
        
        scaling_results = []
        
        for size in sizes:
            # Genera dataset della dimensione specifica
            dataset = dataset_generator(size)
            
            # Testa su tutti gli ordini configurati
            for order in config.arima_orders:
                results = self.run_single_benchmark(config, dataset, order)
                scaling_results.extend(results)
        
        with self._lock:
            self.results.extend(scaling_results)
        
        return scaling_results
    
    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """
        Suite completa di benchmark con tutte le configurazioni principali.
        
        Returns:
            Dizionario con risultati aggregati
        """
        logger.info("Starting comprehensive benchmark suite")
        
        # Configurazioni da confrontare
        configs = [
            BenchmarkConfig(
                name="baseline",
                use_cache=False,
                use_smart_params=False,
                use_memory_pool=False,
                use_vectorized_ops=False
            ),
            BenchmarkConfig(
                name="cache_only",
                use_cache=True,
                use_smart_params=False,
                use_memory_pool=False,
                use_vectorized_ops=False
            ),
            BenchmarkConfig(
                name="smart_params_only",
                use_cache=False,
                use_smart_params=True,
                use_memory_pool=False,
                use_vectorized_ops=False
            ),
            BenchmarkConfig(
                name="vectorized_only",
                use_cache=False,
                use_smart_params=False,
                use_memory_pool=True,
                use_vectorized_ops=True
            ),
            BenchmarkConfig(
                name="all_optimizations",
                use_cache=True,
                use_smart_params=True,
                use_memory_pool=True,
                use_vectorized_ops=True
            )
        ]
        
        # Dataset test diversificati
        test_datasets = {
            'simple_trend_500': DatasetGenerator.generate_simple_trend(500),
            'seasonal_500': DatasetGenerator.generate_seasonal(500),
            'volatile_300': DatasetGenerator.generate_volatile(300),
            'outliers_400': DatasetGenerator.generate_outliers(400)
        }
        
        comprehensive_results = {}
        
        # Test su ogni dataset
        for dataset_name, dataset in test_datasets.items():
            logger.info(f"Testing dataset: {dataset_name}")
            
            dataset_results = self.run_configuration_comparison(
                dataset_name=dataset_name,
                dataset=dataset,
                configurations=configs,
                arima_orders=[(1,1,1), (2,1,1)]
            )
            
            comprehensive_results[dataset_name] = dataset_results
        
        # Scaling test con configurazione ottimizzata
        logger.info("Running scaling tests")
        optimized_config = configs[-1]  # all_optimizations
        
        scaling_results = self.run_scaling_benchmark(
            config=optimized_config,
            dataset_generator=DatasetGenerator.generate_simple_trend,
            sizes=[100, 300, 500, 1000]
        )
        
        comprehensive_results['scaling_test'] = scaling_results
        
        # Genera summary
        summary = self.generate_summary()
        comprehensive_results['summary'] = summary
        
        return comprehensive_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Genera summary statistico dei risultati."""
        if not self.results:
            return {"message": "No results available"}
        
        # Converti a DataFrame per analisi
        df_data = []
        for result in self.results:
            if result.success:
                df_data.append({
                    'config': result.config_name,
                    'dataset_size': result.dataset_size,
                    'arima_order': str(result.arima_order),
                    'fit_time': result.fit_time,
                    'forecast_time': result.forecast_time, 
                    'total_time': result.total_time,
                    'memory_mb': result.memory_peak_mb,
                    'aic': result.aic,
                    'cache_hit': result.cache_hit,
                    'preprocessing': result.preprocessing_applied,
                    'vectorized': result.vectorized_used
                })
        
        if not df_data:
            return {"message": "No successful results"}
        
        df = pd.DataFrame(df_data)
        
        # Statistiche aggregate per configurazione
        config_stats = df.groupby('config').agg({
            'fit_time': ['mean', 'std', 'min', 'max'],
            'total_time': ['mean', 'std'],
            'memory_mb': ['mean', 'max'],
            'aic': 'mean',
            'cache_hit': 'mean'
        }).round(4)
        
        # Speedup rispetto baseline se disponibile
        speedups = {}
        if 'baseline' in df['config'].values:
            baseline_time = df[df['config'] == 'baseline']['total_time'].mean()
            
            for config in df['config'].unique():
                if config != 'baseline':
                    config_time = df[df['config'] == config]['total_time'].mean()
                    if config_time > 0:
                        speedups[config] = baseline_time / config_time
        
        return {
            'total_benchmarks': len(self.results),
            'successful_benchmarks': len(df_data),
            'configurations_tested': list(df['config'].unique()),
            'config_stats': config_stats.to_dict(),
            'speedups_vs_baseline': speedups,
            'best_config_by_speed': df.loc[df['total_time'].idxmin(), 'config'],
            'best_config_by_accuracy': df.loc[df['aic'].idxmin(), 'config']
        }
    
    def export_results(self, filename: Optional[str] = None) -> Path:
        """Esporta risultati in formato JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        export_path = self.output_dir / filename
        
        # Converte results a dict per serializzazione JSON
        results_dict = []
        for result in self.results:
            result_dict = {
                'config_name': result.config_name,
                'dataset_size': result.dataset_size,
                'arima_order': str(result.arima_order),  # Convert tuple to string for JSON
                'fit_time': result.fit_time,
                'forecast_time': result.forecast_time,
                'total_time': result.total_time,
                'memory_peak_mb': result.memory_peak_mb,
                'aic': result.aic,
                'bic': result.bic,
                'cache_hit': result.cache_hit,
                'preprocessing_applied': result.preprocessing_applied,
                'vectorized_used': result.vectorized_used,
                'cpu_percent': result.cpu_percent,
                'timestamp': result.timestamp.isoformat(),
                'error': result.error,
                'success': result.success
            }
            results_dict.append(result_dict)
        
        # Include summary
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_results': len(results_dict)
            },
            'summary': self.generate_summary(),
            'results': results_dict
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Benchmark results exported to {export_path}")
        return export_path