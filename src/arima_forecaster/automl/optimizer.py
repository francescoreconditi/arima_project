"""
Advanced hyperparameter optimization for ARIMA models using multiple optimization algorithms.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Integer
    SCIKIT_OPT_AVAILABLE = True
except ImportError:
    SCIKIT_OPT_AVAILABLE = False

from ..core import ARIMAForecaster, SARIMAForecaster, VARForecaster
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class BaseOptimizer:
    """Base class for model optimizers."""
    
    def __init__(
        self,
        objective_metric: str = 'aic',
        cv_folds: int = 3,
        test_size: float = 0.2,
        n_jobs: int = 1,
        random_state: int = 42
    ):
        """
        Initialize base optimizer.
        
        Args:
            objective_metric: Metric to optimize ('aic', 'bic', 'mse', 'mae', 'mape')
            cv_folds: Number of cross-validation folds
            test_size: Proportion of data for testing
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        self.objective_metric = objective_metric
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = get_logger(__name__)
        
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.optimization_history = []
    
    def _time_series_split(self, series: pd.Series, n_splits: int) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Create time series cross-validation splits.
        
        Args:
            series: Time series data
            n_splits: Number of splits
            
        Returns:
            List of (train, test) splits
        """
        splits = []
        n = len(series)
        
        # Calculate split sizes
        min_train_size = n // (n_splits + 1)
        test_size = n // (n_splits + 2)
        
        for i in range(n_splits):
            train_end = min_train_size + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)
            
            if test_end <= test_start:
                break
                
            train_data = series.iloc[:train_end]
            test_data = series.iloc[test_start:test_end]
            
            splits.append((train_data, test_data))
        
        return splits
    
    def _evaluate_model_cv(
        self, 
        model_class: type, 
        params: Dict[str, Any], 
        series: pd.Series
    ) -> float:
        """
        Evaluate model using cross-validation.
        
        Args:
            model_class: Model class to instantiate
            params: Model parameters
            series: Time series data
            
        Returns:
            Average score across CV folds
        """
        try:
            splits = self._time_series_split(series, self.cv_folds)
            scores = []
            
            for train_data, test_data in splits:
                try:
                    # Create and fit model
                    model = model_class(**params)
                    model.fit(train_data, validate_input=False)
                    
                    if self.objective_metric in ['aic', 'bic', 'hqic']:
                        # Use information criteria
                        model_info = model.get_model_info()
                        score = model_info.get(self.objective_metric, float('inf'))
                        
                    else:
                        # Use forecast accuracy metrics
                        forecast_steps = len(test_data)
                        forecast = model.forecast(steps=forecast_steps, confidence_intervals=False)
                        
                        # Align indices
                        forecast_aligned = forecast[:len(test_data)]
                        test_aligned = test_data[:len(forecast_aligned)]
                        
                        if self.objective_metric == 'mse':
                            score = np.mean((forecast_aligned - test_aligned) ** 2)
                        elif self.objective_metric == 'mae':
                            score = np.mean(np.abs(forecast_aligned - test_aligned))
                        elif self.objective_metric == 'mape':
                            score = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
                        else:
                            score = float('inf')
                    
                    scores.append(score)
                    
                except Exception:
                    # Model failed, assign worst possible score
                    scores.append(float('inf'))
            
            return np.mean(scores) if scores else float('inf')
            
        except Exception:
            return float('inf')


class ARIMAOptimizer(BaseOptimizer):
    """
    ARIMA model hyperparameter optimizer using advanced optimization algorithms.
    """
    
    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 5),
        d_range: Tuple[int, int] = (0, 2),
        q_range: Tuple[int, int] = (0, 5),
        **kwargs
    ):
        """
        Initialize ARIMA optimizer.
        
        Args:
            p_range: Range of AR parameters
            d_range: Range of differencing parameters
            q_range: Range of MA parameters
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
    
    def optimize_optuna(
        self, 
        series: pd.Series, 
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize using Optuna.
        
        Args:
            series: Time series data
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        def objective(trial):
            p = trial.suggest_int('p', self.p_range[0], self.p_range[1])
            d = trial.suggest_int('d', self.d_range[0], self.d_range[1])
            q = trial.suggest_int('q', self.q_range[0], self.q_range[1])
            
            params = {'order': (p, d, q)}
            score = self._evaluate_model_cv(ARIMAForecaster, params, series)
            
            return score
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = {
            'order': (
                study.best_params['p'],
                study.best_params['d'],
                study.best_params['q']
            )
        }
        self.best_score = study.best_value
        
        # Train best model
        self.best_model = ARIMAForecaster(**self.best_params)
        self.best_model.fit(series)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials],
            'optimizer': 'optuna'
        }
    
    def optimize_hyperopt(
        self, 
        series: pd.Series, 
        max_evals: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize using Hyperopt.
        
        Args:
            series: Time series data
            max_evals: Maximum number of evaluations
            
        Returns:
            Optimization results
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is not installed. Install with: pip install hyperopt")
        
        def objective(params):
            p = int(params['p'])
            d = int(params['d'])
            q = int(params['q'])
            
            model_params = {'order': (p, d, q)}
            score = self._evaluate_model_cv(ARIMAForecaster, model_params, series)
            
            return {'loss': score, 'status': STATUS_OK}
        
        space = {
            'p': hp.quniform('p', self.p_range[0], self.p_range[1], 1),
            'd': hp.quniform('d', self.d_range[0], self.d_range[1], 1),
            'q': hp.quniform('q', self.q_range[0], self.q_range[1], 1)
        }
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )
        
        self.best_params = {
            'order': (int(best['p']), int(best['d']), int(best['q']))
        }
        self.best_score = trials.best_trial['result']['loss']
        
        # Train best model
        self.best_model = ARIMAForecaster(**self.best_params)
        self.best_model.fit(series)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_evals': len(trials.trials),
            'optimization_history': [trial['result']['loss'] for trial in trials.trials],
            'optimizer': 'hyperopt'
        }
    
    def optimize_skopt(
        self, 
        series: pd.Series, 
        n_calls: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize using scikit-optimize.
        
        Args:
            series: Time series data
            n_calls: Number of optimization calls
            
        Returns:
            Optimization results
        """
        if not SCIKIT_OPT_AVAILABLE:
            raise ImportError("scikit-optimize is not installed. Install with: pip install scikit-optimize")
        
        def objective(params):
            p, d, q = params
            model_params = {'order': (p, d, q)}
            return self._evaluate_model_cv(ARIMAForecaster, model_params, series)
        
        dimensions = [
            Integer(self.p_range[0], self.p_range[1], name='p'),
            Integer(self.d_range[0], self.d_range[1], name='d'),
            Integer(self.q_range[0], self.q_range[1], name='q')
        ]
        
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=self.random_state
        )
        
        self.best_params = {'order': tuple(result.x)}
        self.best_score = result.fun
        
        # Train best model
        self.best_model = ARIMAForecaster(**self.best_params)
        self.best_model.fit(series)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_calls': n_calls,
            'optimization_history': result.func_vals,
            'optimizer': 'scikit-optimize'
        }


class SARIMAOptimizer(BaseOptimizer):
    """
    SARIMA model hyperparameter optimizer.
    """
    
    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2),
        q_range: Tuple[int, int] = (0, 3),
        P_range: Tuple[int, int] = (0, 2),
        D_range: Tuple[int, int] = (0, 1),
        Q_range: Tuple[int, int] = (0, 2),
        seasonal_periods: List[int] = None,
        **kwargs
    ):
        """
        Initialize SARIMA optimizer.
        
        Args:
            p_range: Range of AR parameters
            d_range: Range of differencing parameters
            q_range: Range of MA parameters
            P_range: Range of seasonal AR parameters
            D_range: Range of seasonal differencing parameters
            Q_range: Range of seasonal MA parameters
            seasonal_periods: List of seasonal periods to try
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.P_range = P_range
        self.D_range = D_range
        self.Q_range = Q_range
        self.seasonal_periods = seasonal_periods or [12]
    
    def optimize_optuna(
        self, 
        series: pd.Series, 
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize SARIMA using Optuna.
        
        Args:
            series: Time series data
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        def objective(trial):
            p = trial.suggest_int('p', self.p_range[0], self.p_range[1])
            d = trial.suggest_int('d', self.d_range[0], self.d_range[1])
            q = trial.suggest_int('q', self.q_range[0], self.q_range[1])
            P = trial.suggest_int('P', self.P_range[0], self.P_range[1])
            D = trial.suggest_int('D', self.D_range[0], self.D_range[1])
            Q = trial.suggest_int('Q', self.Q_range[0], self.Q_range[1])
            s = trial.suggest_categorical('s', self.seasonal_periods)
            
            params = {
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, s)
            }
            score = self._evaluate_model_cv(SARIMAForecaster, params, series)
            
            return score
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = {
            'order': (
                study.best_params['p'],
                study.best_params['d'],
                study.best_params['q']
            ),
            'seasonal_order': (
                study.best_params['P'],
                study.best_params['D'],
                study.best_params['Q'],
                study.best_params['s']
            )
        }
        self.best_score = study.best_value
        
        # Train best model
        self.best_model = SARIMAForecaster(**self.best_params)
        self.best_model.fit(series)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials],
            'optimizer': 'optuna'
        }


class VAROptimizer(BaseOptimizer):
    """
    VAR model hyperparameter optimizer.
    """
    
    def __init__(
        self,
        maxlags_range: Tuple[int, int] = (1, 10),
        ic_options: List[str] = None,
        **kwargs
    ):
        """
        Initialize VAR optimizer.
        
        Args:
            maxlags_range: Range of maximum lags to try
            ic_options: Information criteria to try
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.maxlags_range = maxlags_range
        self.ic_options = ic_options or ['aic', 'bic', 'hqic', 'fpe']
    
    def optimize_optuna(
        self, 
        data: pd.DataFrame, 
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize VAR using Optuna.
        
        Args:
            data: Multivariate time series data
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        def objective(trial):
            maxlags = trial.suggest_int('maxlags', self.maxlags_range[0], self.maxlags_range[1])
            ic = trial.suggest_categorical('ic', self.ic_options)
            
            params = {'maxlags': maxlags, 'ic': ic}
            
            try:
                model = VARForecaster(**params)
                model.fit(data, validate_input=False)
                
                model_info = model.get_model_info()
                score = model_info.get(self.objective_metric, float('inf'))
                
                return score
                
            except Exception:
                return float('inf')
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = {
            'maxlags': study.best_params['maxlags'],
            'ic': study.best_params['ic']
        }
        self.best_score = study.best_value
        
        # Train best model
        self.best_model = VARForecaster(**self.best_params)
        self.best_model.fit(data)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials],
            'optimizer': 'optuna'
        }


def optimize_model(
    model_type: str,
    data: Union[pd.Series, pd.DataFrame],
    optimizer_type: str = 'optuna',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to optimize any model type.
    
    Args:
        model_type: Type of model ('arima', 'sarima', 'var')
        data: Time series data
        optimizer_type: Optimization algorithm ('optuna', 'hyperopt', 'skopt')
        **kwargs: Additional arguments passed to optimizer
        
    Returns:
        Optimization results
    """
    if model_type.lower() == 'arima':
        optimizer = ARIMAOptimizer(**kwargs)
        if optimizer_type == 'optuna':
            return optimizer.optimize_optuna(data)
        elif optimizer_type == 'hyperopt':
            return optimizer.optimize_hyperopt(data)
        elif optimizer_type == 'skopt':
            return optimizer.optimize_skopt(data)
    
    elif model_type.lower() == 'sarima':
        optimizer = SARIMAOptimizer(**kwargs)
        return optimizer.optimize_optuna(data)
    
    elif model_type.lower() == 'var':
        optimizer = VAROptimizer(**kwargs)
        return optimizer.optimize_optuna(data)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")