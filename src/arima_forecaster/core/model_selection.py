"""
Automatic ARIMA model selection and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
from statsmodels.tsa.arima.model import ARIMA
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class ARIMAModelSelector:
    """
    Automatic ARIMA model selection using grid search and information criteria.
    """
    
    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2), 
        q_range: Tuple[int, int] = (0, 3),
        information_criterion: str = 'aic'
    ):
        """
        Initialize model selector.
        
        Args:
            p_range: Range for autoregressive order (min, max)
            d_range: Range for differencing order (min, max)
            q_range: Range for moving average order (min, max)  
            information_criterion: Criterion for model selection ('aic', 'bic', 'hqic')
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.information_criterion = information_criterion.lower()
        self.results = []
        self.best_order = None
        self.best_model = None
        self.logger = get_logger(__name__)
        
        if self.information_criterion not in ['aic', 'bic', 'hqic']:
            raise ValueError("information_criterion must be one of: 'aic', 'bic', 'hqic'")
    
    def search(
        self, 
        series: pd.Series,
        verbose: bool = True,
        max_models: Optional[int] = None
    ) -> Tuple[int, int, int]:
        """
        Perform grid search to find optimal ARIMA order.
        
        Args:
            series: Time series to fit
            verbose: Whether to print progress
            max_models: Maximum number of models to evaluate
            
        Returns:
            Optimal ARIMA order (p, d, q)
        """
        self.logger.info(f"Starting ARIMA model selection with {self.information_criterion.upper()} criterion")
        self.results = []
        
        # Generate all combinations
        p_values = list(range(self.p_range[0], self.p_range[1] + 1))
        d_values = list(range(self.d_range[0], self.d_range[1] + 1))
        q_values = list(range(self.q_range[0], self.q_range[1] + 1))
        
        all_orders = list(product(p_values, d_values, q_values))
        
        if max_models and len(all_orders) > max_models:
            self.logger.info(f"Limiting search to first {max_models} model combinations")
            all_orders = all_orders[:max_models]
        
        self.logger.info(f"Evaluating {len(all_orders)} model combinations")
        
        best_criterion = float('inf')
        best_order = None
        
        for i, order in enumerate(all_orders):
            try:
                if verbose and (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i + 1}/{len(all_orders)} models evaluated")
                
                # Fit model
                model = ARIMA(series, order=order)
                fitted_model = model.fit(disp=False)
                
                # Get criterion value
                if self.information_criterion == 'aic':
                    criterion_value = fitted_model.aic
                elif self.information_criterion == 'bic':
                    criterion_value = fitted_model.bic
                elif self.information_criterion == 'hqic':
                    criterion_value = fitted_model.hqic
                
                # Store results
                result = {
                    'order': order,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic, 
                    'hqic': fitted_model.hqic,
                    'llf': fitted_model.llf,
                    'params': len(fitted_model.params),
                    'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                }
                
                self.results.append(result)
                
                # Check if this is the best model so far
                if criterion_value < best_criterion:
                    best_criterion = criterion_value
                    best_order = order
                    self.best_model = fitted_model
                
            except Exception as e:
                if verbose:
                    self.logger.debug(f"Failed to fit ARIMA{order}: {e}")
                continue
        
        if not self.results:
            raise ModelTrainingError("No models could be fitted successfully")
        
        self.best_order = best_order
        self.logger.info(f"Best model: ARIMA{best_order} with {self.information_criterion.upper()}={best_criterion:.2f}")
        
        return best_order
    
    def get_results_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get summary of model selection results.
        
        Args:
            top_n: Number of top models to return
            
        Returns:
            DataFrame with model results sorted by information criterion
        """
        if not self.results:
            raise ValueError("No results available. Run search() first.")
        
        df = pd.DataFrame(self.results)
        df = df.sort_values(self.information_criterion).head(top_n)
        
        return df
    
    def plot_selection_results(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot model selection results.
        
        Args:
            figsize: Figure size for plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                raise ValueError("No results available. Run search() first.")
            
            df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('ARIMA Model Selection Results', fontsize=16)
            
            # AIC distribution
            axes[0, 0].hist(df['aic'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(df['aic'].min(), color='red', linestyle='--', label='Best AIC')
            axes[0, 0].set_xlabel('AIC')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('AIC Distribution')
            axes[0, 0].legend()
            
            # BIC distribution
            axes[0, 1].hist(df['bic'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(df['bic'].min(), color='red', linestyle='--', label='Best BIC')
            axes[0, 1].set_xlabel('BIC')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('BIC Distribution')
            axes[0, 1].legend()
            
            # Parameter count vs AIC
            axes[1, 0].scatter(df['params'], df['aic'], alpha=0.6)
            axes[1, 0].set_xlabel('Number of Parameters')
            axes[1, 0].set_ylabel('AIC')
            axes[1, 0].set_title('Parameters vs AIC')
            
            # Top models comparison
            top_models = df.nsmallest(10, self.information_criterion)
            order_labels = [f"({p},{d},{q})" for p, d, q in top_models['order']]
            
            axes[1, 1].bar(range(len(top_models)), top_models[self.information_criterion])
            axes[1, 1].set_xlabel('Model Order')
            axes[1, 1].set_ylabel(self.information_criterion.upper())
            axes[1, 1].set_title(f'Top 10 Models by {self.information_criterion.upper()}')
            axes[1, 1].set_xticks(range(len(top_models)))
            axes[1, 1].set_xticklabels(order_labels, rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            
    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the best selected model.
        
        Returns:
            Dictionary with best model information
        """
        if self.best_order is None:
            raise ValueError("No best model available. Run search() first.")
        
        best_result = next((r for r in self.results if r['order'] == self.best_order), None)
        
        if best_result is None:
            raise ValueError("Best model result not found")
        
        info = {
            'best_order': self.best_order,
            'criterion_used': self.information_criterion,
            'aic': best_result['aic'],
            'bic': best_result['bic'],
            'hqic': best_result['hqic'], 
            'llf': best_result['llf'],
            'parameters': best_result['params'],
            'total_models_evaluated': len(self.results)
        }
        
        return info