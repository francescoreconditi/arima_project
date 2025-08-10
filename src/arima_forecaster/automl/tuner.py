"""
Advanced hyperparameter tuning with ensemble methods and meta-learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

from .optimizer import ARIMAOptimizer, SARIMAOptimizer, VAROptimizer
from ..core import ARIMAForecaster, SARIMAForecaster, VARForecaster
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class HyperparameterTuner:
    """
    Advanced hyperparameter tuner with ensemble methods and meta-learning.
    """
    
    def __init__(
        self,
        objective_metrics: List[str] = None,
        ensemble_method: str = 'weighted_average',
        meta_learning: bool = True,
        early_stopping_patience: int = 10,
        n_jobs: int = 1
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            objective_metrics: List of metrics to optimize
            ensemble_method: Method for ensemble ('weighted_average', 'rank_based', 'pareto')
            meta_learning: Whether to use meta-learning for warm-starting
            early_stopping_patience: Patience for early stopping
            n_jobs: Number of parallel jobs
        """
        self.objective_metrics = objective_metrics or ['aic', 'bic', 'mse']
        self.ensemble_method = ensemble_method
        self.meta_learning = meta_learning
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        
        self.logger = get_logger(__name__)
        
        # Results storage
        self.optimization_results = {}
        self.ensemble_models = []
        self.meta_knowledge = {}
    
    def multi_objective_optimization(
        self,
        model_type: str,
        data: Union[pd.Series, pd.DataFrame],
        n_trials: int = 100,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Perform multi-objective optimization.
        
        Args:
            model_type: Type of model to optimize
            data: Time series data
            n_trials: Number of trials per objective
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Multi-objective optimization results
        """
        self.logger.info(f"Starting multi-objective optimization for {model_type}")
        
        results = {}
        pareto_solutions = []
        
        # Optimize for each objective
        for metric in self.objective_metrics:
            self.logger.info(f"Optimizing for {metric}")
            
            try:
                if model_type.lower() == 'arima':
                    optimizer = ARIMAOptimizer(objective_metric=metric, **optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=n_trials)
                elif model_type.lower() == 'sarima':
                    optimizer = SARIMAOptimizer(objective_metric=metric, **optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=n_trials)
                elif model_type.lower() == 'var':
                    optimizer = VAROptimizer(objective_metric=metric, **optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=n_trials)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                results[metric] = result
                
                # Evaluate on all metrics for Pareto analysis
                model = optimizer.best_model
                scores = self._evaluate_all_metrics(model, data)
                
                pareto_solutions.append({
                    'params': result['best_params'],
                    'model': model,
                    'scores': scores,
                    'primary_metric': metric
                })
                
            except Exception as e:
                self.logger.error(f"Optimization failed for {metric}: {e}")
                continue
        
        # Find Pareto-optimal solutions
        pareto_front = self._find_pareto_front(pareto_solutions)
        
        # Select best solution based on ensemble method
        best_solution = self._select_best_solution(pareto_front)
        
        return {
            'individual_results': results,
            'pareto_front': pareto_front,
            'best_solution': best_solution,
            'n_pareto_solutions': len(pareto_front)
        }
    
    def ensemble_optimization(
        self,
        model_type: str,
        data: Union[pd.Series, pd.DataFrame],
        n_models: int = 5,
        diversity_threshold: float = 0.1,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Create an ensemble of optimized models.
        
        Args:
            model_type: Type of model to optimize
            data: Time series data
            n_models: Number of models in ensemble
            diversity_threshold: Minimum diversity between models
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Ensemble optimization results
        """
        self.logger.info(f"Creating ensemble of {n_models} {model_type} models")
        
        ensemble_models = []
        seen_params = []
        
        attempts = 0
        max_attempts = n_models * 3
        
        while len(ensemble_models) < n_models and attempts < max_attempts:
            attempts += 1
            
            try:
                # Add randomness to avoid identical solutions
                seed = np.random.randint(0, 10000)
                
                if model_type.lower() == 'arima':
                    optimizer = ARIMAOptimizer(random_state=seed, **optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=50)
                elif model_type.lower() == 'sarima':
                    optimizer = SARIMAOptimizer(random_state=seed, **optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=50)
                elif model_type.lower() == 'var':
                    optimizer = VAROptimizer(random_state=seed, **optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=30)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Check diversity
                params = result['best_params']
                if self._is_diverse(params, seen_params, diversity_threshold):
                    ensemble_models.append({
                        'model': optimizer.best_model,
                        'params': params,
                        'score': result['best_score'],
                        'weight': 1.0 / result['best_score'] if result['best_score'] > 0 else 1.0
                    })
                    seen_params.append(params)
                
            except Exception as e:
                self.logger.warning(f"Ensemble model creation failed (attempt {attempts}): {e}")
                continue
        
        if len(ensemble_models) == 0:
            raise ModelTrainingError("Failed to create any ensemble models")
        
        # Normalize weights
        total_weight = sum(model['weight'] for model in ensemble_models)
        for model in ensemble_models:
            model['weight'] /= total_weight
        
        self.ensemble_models = ensemble_models
        
        return {
            'ensemble_models': ensemble_models,
            'n_models': len(ensemble_models),
            'diversity_metrics': self._calculate_ensemble_diversity(ensemble_models),
            'ensemble_score': self._calculate_ensemble_score(ensemble_models, data)
        }
    
    def adaptive_optimization(
        self,
        model_type: str,
        data: Union[pd.Series, pd.DataFrame],
        max_iterations: int = 10,
        improvement_threshold: float = 0.01,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Perform adaptive optimization with dynamic parameter adjustment.
        
        Args:
            model_type: Type of model to optimize
            data: Time series data
            max_iterations: Maximum optimization iterations
            improvement_threshold: Minimum improvement to continue
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Adaptive optimization results
        """
        self.logger.info(f"Starting adaptive optimization for {model_type}")
        
        iteration_results = []
        best_score = float('inf')
        no_improvement_count = 0
        
        # Initial parameter ranges
        if model_type.lower() == 'arima':
            p_range = optimizer_kwargs.get('p_range', (0, 5))
            d_range = optimizer_kwargs.get('d_range', (0, 2))
            q_range = optimizer_kwargs.get('q_range', (0, 5))
        elif model_type.lower() == 'sarima':
            p_range = optimizer_kwargs.get('p_range', (0, 3))
            d_range = optimizer_kwargs.get('d_range', (0, 2))
            q_range = optimizer_kwargs.get('q_range', (0, 3))
            P_range = optimizer_kwargs.get('P_range', (0, 2))
            D_range = optimizer_kwargs.get('D_range', (0, 1))
            Q_range = optimizer_kwargs.get('Q_range', (0, 2))
        
        for iteration in range(max_iterations):
            self.logger.info(f"Adaptive optimization iteration {iteration + 1}")
            
            try:
                # Create optimizer with current ranges
                if model_type.lower() == 'arima':
                    optimizer = ARIMAOptimizer(
                        p_range=p_range,
                        d_range=d_range,
                        q_range=q_range,
                        **{k: v for k, v in optimizer_kwargs.items() 
                           if k not in ['p_range', 'd_range', 'q_range']}
                    )
                    result = optimizer.optimize_optuna(data, n_trials=50)
                    
                elif model_type.lower() == 'sarima':
                    optimizer = SARIMAOptimizer(
                        p_range=p_range,
                        d_range=d_range,
                        q_range=q_range,
                        P_range=P_range,
                        D_range=D_range,
                        Q_range=Q_range,
                        **{k: v for k, v in optimizer_kwargs.items() 
                           if k not in ['p_range', 'd_range', 'q_range', 'P_range', 'D_range', 'Q_range']}
                    )
                    result = optimizer.optimize_optuna(data, n_trials=50)
                    
                elif model_type.lower() == 'var':
                    optimizer = VAROptimizer(**optimizer_kwargs)
                    result = optimizer.optimize_optuna(data, n_trials=30)
                
                current_score = result['best_score']
                improvement = (best_score - current_score) / best_score if best_score != float('inf') else 0
                
                iteration_results.append({
                    'iteration': iteration + 1,
                    'score': current_score,
                    'improvement': improvement,
                    'params': result['best_params'],
                    'model': optimizer.best_model
                })
                
                # Check for improvement
                if current_score < best_score - improvement_threshold * abs(best_score):
                    best_score = current_score
                    no_improvement_count = 0
                    
                    # Narrow search ranges around best solution (for ARIMA/SARIMA)
                    if model_type.lower() == 'arima':
                        best_order = result['best_params']['order']
                        p_range = (max(0, best_order[0] - 1), min(5, best_order[0] + 2))
                        d_range = (max(0, best_order[1] - 1), min(2, best_order[1] + 2))
                        q_range = (max(0, best_order[2] - 1), min(5, best_order[2] + 2))
                        
                    elif model_type.lower() == 'sarima':
                        best_order = result['best_params']['order']
                        best_seasonal = result['best_params']['seasonal_order']
                        p_range = (max(0, best_order[0] - 1), min(3, best_order[0] + 2))
                        d_range = (max(0, best_order[1] - 1), min(2, best_order[1] + 2))
                        q_range = (max(0, best_order[2] - 1), min(3, best_order[2] + 2))
                        P_range = (max(0, best_seasonal[0] - 1), min(2, best_seasonal[0] + 2))
                        D_range = (max(0, best_seasonal[1] - 1), min(1, best_seasonal[1] + 2))
                        Q_range = (max(0, best_seasonal[2] - 1), min(2, best_seasonal[2] + 2))
                        
                else:
                    no_improvement_count += 1
                    
                    # Early stopping
                    if no_improvement_count >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at iteration {iteration + 1}")
                        break
                
            except Exception as e:
                self.logger.error(f"Iteration {iteration + 1} failed: {e}")
                continue
        
        # Find best iteration
        best_iteration = min(iteration_results, key=lambda x: x['score'])
        
        return {
            'iteration_results': iteration_results,
            'best_iteration': best_iteration,
            'total_iterations': len(iteration_results),
            'final_score': best_score,
            'converged': no_improvement_count < self.early_stopping_patience
        }
    
    def forecast_ensemble(
        self, 
        steps: int, 
        confidence_level: float = 0.95,
        method: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Generate forecasts using ensemble of models.
        
        Args:
            steps: Number of forecast steps
            confidence_level: Confidence level for intervals
            method: Ensemble method ('weighted', 'equal', 'median')
            
        Returns:
            Ensemble forecast results
        """
        if not self.ensemble_models:
            raise ValueError("No ensemble models available. Run ensemble_optimization first.")
        
        forecasts = []
        weights = []
        
        for model_info in self.ensemble_models:
            model = model_info['model']
            weight = model_info['weight']
            
            try:
                if method == 'weighted' or method == 'equal':
                    forecast = model.forecast(steps=steps, confidence_intervals=False)
                    forecasts.append(forecast.values)
                    weights.append(weight if method == 'weighted' else 1.0)
                    
            except Exception as e:
                self.logger.warning(f"Forecast failed for ensemble model: {e}")
                continue
        
        if not forecasts:
            raise ModelTrainingError("All ensemble forecasts failed")
        
        forecasts_array = np.array(forecasts)
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize
        
        if method == 'median':
            ensemble_forecast = np.median(forecasts_array, axis=0)
            ensemble_std = np.std(forecasts_array, axis=0)
        else:
            # Weighted or equal average
            ensemble_forecast = np.average(forecasts_array, axis=0, weights=weights_array)
            ensemble_std = np.sqrt(
                np.average(
                    (forecasts_array - ensemble_forecast) ** 2, 
                    axis=0, 
                    weights=weights_array
                )
            )
        
        # Create confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96  # Approximately for 95% confidence
        
        lower_bound = ensemble_forecast - z_score * ensemble_std
        upper_bound = ensemble_forecast + z_score * ensemble_std
        
        # Create forecast index (simplified)
        forecast_index = range(steps)
        
        return {
            'forecast': ensemble_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'forecast_std': ensemble_std,
            'confidence_level': confidence_level,
            'n_models': len(forecasts),
            'method': method,
            'individual_forecasts': forecasts_array
        }
    
    def _evaluate_all_metrics(
        self, 
        model, 
        data: Union[pd.Series, pd.DataFrame]
    ) -> Dict[str, float]:
        """Evaluate model on all objective metrics."""
        scores = {}
        
        try:
            model_info = model.get_model_info()
            
            # Information criteria
            for metric in ['aic', 'bic', 'hqic']:
                if metric in self.objective_metrics:
                    scores[metric] = model_info.get(metric, float('inf'))
            
            # Forecast accuracy (simplified for demonstration)
            if any(metric in self.objective_metrics for metric in ['mse', 'mae', 'mape']):
                if isinstance(data, pd.Series):
                    test_size = max(1, len(data) // 5)
                    train_data = data[:-test_size]
                    test_data = data[-test_size:]
                    
                    # Retrain on train data
                    if hasattr(model, 'order'):  # ARIMA/SARIMA
                        temp_model = type(model)(
                            order=getattr(model, 'order', (1, 1, 1)),
                            seasonal_order=getattr(model, 'seasonal_order', None)
                        )
                    else:  # VAR
                        temp_model = type(model)(
                            maxlags=getattr(model, 'maxlags', None),
                            ic=getattr(model, 'ic', 'aic')
                        )
                    
                    temp_model.fit(train_data, validate_input=False)
                    forecast = temp_model.forecast(steps=len(test_data), confidence_intervals=False)
                    
                    # Calculate metrics
                    if 'mse' in self.objective_metrics:
                        scores['mse'] = np.mean((forecast[:len(test_data)] - test_data[:len(forecast)]) ** 2)
                    if 'mae' in self.objective_metrics:
                        scores['mae'] = np.mean(np.abs(forecast[:len(test_data)] - test_data[:len(forecast)]))
                    if 'mape' in self.objective_metrics:
                        scores['mape'] = np.mean(
                            np.abs((test_data[:len(forecast)] - forecast[:len(test_data)]) / test_data[:len(forecast)])
                        ) * 100
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate all metrics: {e}")
            for metric in self.objective_metrics:
                if metric not in scores:
                    scores[metric] = float('inf')
        
        return scores
    
    def _find_pareto_front(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto-optimal solutions."""
        pareto_front = []
        
        for i, solution_i in enumerate(solutions):
            is_dominated = False
            
            for j, solution_j in enumerate(solutions):
                if i != j:
                    # Check if solution_j dominates solution_i
                    dominates = True
                    strictly_better = False
                    
                    for metric in self.objective_metrics:
                        score_i = solution_i['scores'].get(metric, float('inf'))
                        score_j = solution_j['scores'].get(metric, float('inf'))
                        
                        if score_i < score_j:  # solution_i is better in this metric
                            dominates = False
                            break
                        elif score_i > score_j:  # solution_j is better in this metric
                            strictly_better = True
                    
                    if dominates and strictly_better:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(solution_i)
        
        return pareto_front
    
    def _select_best_solution(self, pareto_front: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best solution from Pareto front."""
        if not pareto_front:
            return None
        
        if len(pareto_front) == 1:
            return pareto_front[0]
        
        if self.ensemble_method == 'weighted_average':
            # Select solution with best weighted score
            best_solution = None
            best_weighted_score = float('inf')
            
            for solution in pareto_front:
                weighted_score = 0
                total_weight = 0
                
                for metric in self.objective_metrics:
                    score = solution['scores'].get(metric, float('inf'))
                    if score != float('inf'):
                        weight = 1.0 / len(self.objective_metrics)  # Equal weights
                        weighted_score += weight * score
                        total_weight += weight
                
                if total_weight > 0:
                    weighted_score /= total_weight
                    
                    if weighted_score < best_weighted_score:
                        best_weighted_score = weighted_score
                        best_solution = solution
            
            return best_solution or pareto_front[0]
        
        else:
            # Default: return first solution
            return pareto_front[0]
    
    def _is_diverse(
        self, 
        params: Dict[str, Any], 
        seen_params: List[Dict[str, Any]], 
        threshold: float
    ) -> bool:
        """Check if parameters are diverse enough from existing ones."""
        if not seen_params:
            return True
        
        # Calculate diversity based on parameter differences
        for existing_params in seen_params:
            similarity = self._calculate_param_similarity(params, existing_params)
            if similarity > (1 - threshold):
                return False
        
        return True
    
    def _calculate_param_similarity(
        self, 
        params1: Dict[str, Any], 
        params2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between parameter sets."""
        # Simplified similarity calculation
        total_diff = 0
        total_params = 0
        
        for key in params1:
            if key in params2:
                if isinstance(params1[key], tuple) and isinstance(params2[key], tuple):
                    # Handle tuples (like ARIMA orders)
                    diff = sum(abs(a - b) for a, b in zip(params1[key], params2[key]))
                    max_diff = len(params1[key]) * 5  # Assume max difference of 5 per parameter
                    total_diff += diff / max_diff
                    total_params += 1
                elif isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    diff = abs(params1[key] - params2[key])
                    max_diff = max(10, abs(params1[key]), abs(params2[key]))  # Normalize
                    total_diff += diff / max_diff
                    total_params += 1
        
        return 1 - (total_diff / total_params) if total_params > 0 else 0
    
    def _calculate_ensemble_diversity(self, ensemble_models: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate diversity metrics for ensemble."""
        if len(ensemble_models) < 2:
            return {'avg_diversity': 0.0, 'min_diversity': 0.0, 'max_diversity': 0.0}
        
        diversities = []
        
        for i, model1 in enumerate(ensemble_models):
            for j, model2 in enumerate(ensemble_models[i+1:], i+1):
                diversity = 1 - self._calculate_param_similarity(model1['params'], model2['params'])
                diversities.append(diversity)
        
        return {
            'avg_diversity': np.mean(diversities),
            'min_diversity': np.min(diversities),
            'max_diversity': np.max(diversities)
        }
    
    def _calculate_ensemble_score(
        self, 
        ensemble_models: List[Dict[str, Any]], 
        data: Union[pd.Series, pd.DataFrame]
    ) -> float:
        """Calculate weighted ensemble score."""
        weighted_score = 0
        total_weight = 0
        
        for model_info in ensemble_models:
            score = model_info['score']
            weight = model_info['weight']
            
            if score != float('inf'):
                weighted_score += weight * score
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else float('inf')