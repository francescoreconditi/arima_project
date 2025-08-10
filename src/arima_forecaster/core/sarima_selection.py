"""
Automatic SARIMA model selection with seasonal parameter optimization.
"""

import pandas as pd
import numpy as np
import itertools
from typing import List, Tuple, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from tqdm import tqdm

from .sarima_model import SARIMAForecaster
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class SARIMAModelSelector:
    """
    Automatic SARIMA model selection using grid search with seasonal parameters.
    """
    
    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2),  
        q_range: Tuple[int, int] = (0, 3),
        P_range: Tuple[int, int] = (0, 2),
        D_range: Tuple[int, int] = (0, 1),
        Q_range: Tuple[int, int] = (0, 2),
        seasonal_periods: Optional[List[int]] = None,
        information_criterion: str = 'aic',
        max_models: Optional[int] = None,
        n_jobs: int = 1
    ):
        """
        Initialize SARIMA model selector.
        
        Args:
            p_range: Range of p values (min, max)
            d_range: Range of d values (min, max)
            q_range: Range of q values (min, max)  
            P_range: Range of seasonal P values (min, max)
            D_range: Range of seasonal D values (min, max)
            Q_range: Range of seasonal Q values (min, max)
            seasonal_periods: List of seasonal periods to try (default: [12])
            information_criterion: Criterion for model selection ('aic', 'bic', 'hqic')
            max_models: Maximum number of models to try
            n_jobs: Number of parallel jobs for model fitting
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.P_range = P_range
        self.D_range = D_range
        self.Q_range = Q_range
        self.seasonal_periods = seasonal_periods or [12]
        self.information_criterion = information_criterion.lower()
        self.max_models = max_models
        self.n_jobs = n_jobs
        
        self.results = []
        self.best_model = None
        self.best_order = None
        self.best_seasonal_order = None
        self.logger = get_logger(__name__)
        
        if self.information_criterion not in ['aic', 'bic', 'hqic']:
            raise ValueError("information_criterion must be 'aic', 'bic', or 'hqic'")
    
    def search(
        self, 
        series: pd.Series,
        verbose: bool = True,
        suppress_warnings: bool = True
    ) -> 'SARIMAModelSelector':
        """
        Perform grid search to find optimal SARIMA parameters.
        
        Args:
            series: Time series to fit
            verbose: Whether to show progress
            suppress_warnings: Whether to suppress statsmodels warnings
            
        Returns:
            Self for method chaining
        """
        if suppress_warnings:
            warnings.filterwarnings("ignore")
        
        try:
            self.logger.info(f"Starting SARIMA model selection using {self.information_criterion.upper()}")
            self.logger.info(f"Parameter ranges: p{self.p_range}, d{self.d_range}, q{self.q_range}")
            self.logger.info(f"Seasonal ranges: P{self.P_range}, D{self.D_range}, Q{self.Q_range}")
            self.logger.info(f"Seasonal periods: {self.seasonal_periods}")
            
            # Generate all parameter combinations
            param_combinations = self._generate_param_combinations()
            
            if self.max_models and len(param_combinations) > self.max_models:
                # Randomly sample if too many combinations
                np.random.shuffle(param_combinations)
                param_combinations = param_combinations[:self.max_models]
                self.logger.info(f"Limited to {self.max_models} random parameter combinations")
            
            self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            # Fit models
            if self.n_jobs == 1:
                # Sequential processing
                self.results = self._fit_models_sequential(series, param_combinations, verbose)
            else:
                # Parallel processing
                self.results = self._fit_models_parallel(series, param_combinations, verbose)
            
            # Find best model
            if self.results:
                best_result = min(self.results, key=lambda x: x[self.information_criterion])
                self.best_order = best_result['order']
                self.best_seasonal_order = best_result['seasonal_order']
                
                # Fit the best model
                self.best_model = SARIMAForecaster(
                    order=self.best_order,
                    seasonal_order=self.best_seasonal_order
                )
                self.best_model.fit(series)
                
                self.logger.info(f"Best SARIMA model: {self.best_order}x{self.best_seasonal_order}")
                self.logger.info(f"Best {self.information_criterion.upper()}: {best_result[self.information_criterion]:.2f}")
            else:
                self.logger.error("No models successfully fitted")
            
            return self
            
        except Exception as e:
            self.logger.error(f"SARIMA model selection failed: {e}")
            raise ModelTrainingError(f"SARIMA model selection failed: {e}")
        
        finally:
            if suppress_warnings:
                warnings.resetwarnings()
    
    def _generate_param_combinations(self) -> List[Tuple]:
        """Generate all parameter combinations to test."""
        combinations = []
        
        for s in self.seasonal_periods:
            for p in range(self.p_range[0], self.p_range[1] + 1):
                for d in range(self.d_range[0], self.d_range[1] + 1):
                    for q in range(self.q_range[0], self.q_range[1] + 1):
                        for P in range(self.P_range[0], self.P_range[1] + 1):
                            for D in range(self.D_range[0], self.D_range[1] + 1):
                                for Q in range(self.Q_range[0], self.Q_range[1] + 1):
                                    combinations.append(((p, d, q), (P, D, Q, s)))
        
        return combinations
    
    def _fit_models_sequential(
        self, 
        series: pd.Series, 
        param_combinations: List[Tuple],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """Fit models sequentially."""
        results = []
        
        iterator = tqdm(param_combinations, desc="Testing SARIMA models") if verbose else param_combinations
        
        for order, seasonal_order in iterator:
            try:
                model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
                model.fit(series, validate_input=False)
                
                model_info = model.get_model_info()
                results.append(model_info)
                
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'Best AIC': min(r['aic'] for r in results),
                        'Current': f"{order}x{seasonal_order}"
                    })
                    
            except Exception as e:
                # Model failed to fit, skip it
                continue
        
        return results
    
    def _fit_models_parallel(
        self, 
        series: pd.Series, 
        param_combinations: List[Tuple],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """Fit models in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(_fit_sarima_model, series, order, seasonal_order): (order, seasonal_order)
                for order, seasonal_order in param_combinations
            }
            
            # Collect results
            iterator = tqdm(
                as_completed(future_to_params), 
                total=len(param_combinations),
                desc="Testing SARIMA models"
            ) if verbose else as_completed(future_to_params)
            
            for future in iterator:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        
                        if verbose and hasattr(iterator, 'set_postfix'):
                            iterator.set_postfix({
                                'Completed': len(results),
                                'Best AIC': min(r['aic'] for r in results) if results else 'N/A'
                            })
                except Exception:
                    # Model failed to fit, skip it
                    continue
        
        return results
    
    def get_best_model(self) -> Optional[SARIMAForecaster]:
        """
        Get the best fitted SARIMA model.
        
        Returns:
            Best SARIMAForecaster instance or None if no models fitted
        """
        return self.best_model
    
    def get_results_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get summary of model selection results.
        
        Args:
            top_n: Number of top models to return
            
        Returns:
            DataFrame with model results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Sort by information criterion
        sorted_results = sorted(self.results, key=lambda x: x[self.information_criterion])
        
        # Create summary DataFrame
        summary_data = []
        for result in sorted_results[:top_n]:
            summary_data.append({
                'order': str(result['order']),
                'seasonal_order': str(result['seasonal_order']),
                'aic': result['aic'],
                'bic': result['bic'],
                'hqic': result['hqic'],
                'n_observations': result['n_observations']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_selection_results(self, top_n: int = 20) -> None:
        """
        Plot model selection results.
        
        Args:
            top_n: Number of top models to plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                self.logger.warning("No results to plot")
                return
            
            # Get top results
            sorted_results = sorted(self.results, key=lambda x: x[self.information_criterion])[:top_n]
            
            # Prepare data
            model_names = [f"{r['order']}x{r['seasonal_order']}" for r in sorted_results]
            aic_values = [r['aic'] for r in sorted_results]
            bic_values = [r['bic'] for r in sorted_results]
            hqic_values = [r['hqic'] for r in sorted_results]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(model_names))
            width = 0.25
            
            ax.bar(x - width, aic_values, width, label='AIC', alpha=0.8)
            ax.bar(x, bic_values, width, label='BIC', alpha=0.8)  
            ax.bar(x + width, hqic_values, width, label='HQIC', alpha=0.8)
            
            ax.set_xlabel('SARIMA Models')
            ax.set_ylabel('Information Criterion Value')
            ax.set_title(f'Top {top_n} SARIMA Models Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            self.logger.error(f"Failed to create plot: {e}")


def _fit_sarima_model(
    series: pd.Series, 
    order: Tuple[int, int, int], 
    seasonal_order: Tuple[int, int, int, int]
) -> Optional[Dict[str, Any]]:
    """
    Helper function to fit a single SARIMA model (for parallel processing).
    
    Args:
        series: Time series data
        order: ARIMA order
        seasonal_order: Seasonal ARIMA order
        
    Returns:
        Model info dictionary or None if fitting failed
    """
    try:
        model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
        model.fit(series, validate_input=False)
        return model.get_model_info()
    except Exception:
        return None