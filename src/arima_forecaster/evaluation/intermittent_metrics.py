"""
Metriche di Valutazione Specifiche per Domanda Intermittente
Implementa metriche specializzate per spare parts e slow movers

Autore: Claude Code  
Data: 2025-09-02
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntermittentMetrics:
    """Container per metriche domanda intermittente"""
    # Metriche accuracy base
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mae: float  # Mean Absolute Error
    
    # Metriche specifiche intermittent
    mase: float  # Mean Absolute Scaled Error
    periods_in_stock: float  # % periodi con stock disponibile
    fill_rate: float  # % domanda soddisfatta
    
    # Bias metrics
    bias: float  # Mean Error (positive = overforecast)
    pbias: float  # Percentage Bias
    
    # Metriche per periodi con domanda
    mae_demand: float  # MAE solo su periodi con domanda > 0
    mape_demand: float  # MAPE solo su periodi con domanda > 0
    
    # Service level metrics
    achieved_service_level: float  # Livello servizio raggiunto
    stockout_periods: int  # Numero periodi in stockout
    overstock_periods: int  # Numero periodi con eccesso stock
    
    # Cost metrics (se forniti costi)
    total_holding_cost: Optional[float] = None
    total_stockout_cost: Optional[float] = None
    total_cost: Optional[float] = None


class IntermittentEvaluator:
    """
    Valutatore specializzato per modelli Intermittent Demand
    
    Implementa metriche specifiche per spare parts:
    - MASE (Mean Absolute Scaled Error)
    - Fill Rate e Service Level
    - Bias detection
    - Cost optimization metrics
    """
    
    def __init__(self,
                 holding_cost: float = 1.0,
                 stockout_cost: float = 10.0):
        """
        Inizializza evaluator
        
        Args:
            holding_cost: Costo unitario giacenza per periodo
            stockout_cost: Costo unitario stockout per periodo
        """
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        
    def evaluate(self,
                actual: Union[pd.Series, np.ndarray],
                forecast: Union[pd.Series, np.ndarray],
                initial_stock: float = 0) -> IntermittentMetrics:
        """
        Valuta performance forecast su domanda intermittente
        
        Args:
            actual: Valori reali
            forecast: Valori previsti
            initial_stock: Stock iniziale per simulazione
            
        Returns:
            IntermittentMetrics con tutte le metriche
        """
        # Converti in numpy arrays
        y_true = np.array(actual)
        y_pred = np.array(forecast)
        
        # Assicura stessa lunghezza
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calcola metriche base
        errors = y_true - y_pred
        squared_errors = errors ** 2
        absolute_errors = np.abs(errors)
        
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        mae = np.mean(absolute_errors)
        
        # Bias metrics
        bias = np.mean(errors)
        pbias = 100 * bias / np.mean(y_true) if np.mean(y_true) > 0 else 0
        
        # MASE - Mean Absolute Scaled Error
        mase = self._calculate_mase(y_true, y_pred)
        
        # Metriche solo su periodi con domanda
        demand_mask = y_true > 0
        if np.any(demand_mask):
            mae_demand = np.mean(np.abs(errors[demand_mask]))
            mape_demand = 100 * np.mean(
                np.abs(errors[demand_mask]) / y_true[demand_mask]
            )
        else:
            mae_demand = 0
            mape_demand = 0
            
        # Simulazione inventory per service level
        inventory_metrics = self._simulate_inventory(
            y_true, y_pred, initial_stock
        )
        
        # Calcola costi se specificati
        if self.holding_cost > 0 or self.stockout_cost > 0:
            costs = self._calculate_costs(
                inventory_metrics['stock_levels'],
                inventory_metrics['stockouts']
            )
            total_holding_cost = costs['holding']
            total_stockout_cost = costs['stockout']
            total_cost = costs['total']
        else:
            total_holding_cost = None
            total_stockout_cost = None
            total_cost = None
            
        return IntermittentMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            mase=mase,
            periods_in_stock=inventory_metrics['periods_in_stock'],
            fill_rate=inventory_metrics['fill_rate'],
            bias=bias,
            pbias=pbias,
            mae_demand=mae_demand,
            mape_demand=mape_demand,
            achieved_service_level=inventory_metrics['service_level'],
            stockout_periods=inventory_metrics['stockout_periods'],
            overstock_periods=inventory_metrics['overstock_periods'],
            total_holding_cost=total_holding_cost,
            total_stockout_cost=total_stockout_cost,
            total_cost=total_cost
        )
        
    def _calculate_mase(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray) -> float:
        """
        Calcola Mean Absolute Scaled Error
        
        MASE = MAE / MAE_naive
        dove MAE_naive è l'errore del forecast naive (ultimo valore)
        """
        if len(y_true) < 2:
            return np.nan
            
        # Calcola MAE del modello
        mae_model = np.mean(np.abs(y_true - y_pred))
        
        # Calcola MAE naive (forecast = previous value)
        mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
        
        if mae_naive == 0:
            # Se serie costante, usa piccolo valore per evitare divisione zero
            mae_naive = 1e-10
            
        return mae_model / mae_naive
        
    def _simulate_inventory(self,
                           demand: np.ndarray,
                           forecast: np.ndarray,
                           initial_stock: float) -> Dict:
        """
        Simula gestione inventory con forecast
        
        Args:
            demand: Domanda reale
            forecast: Forecast usato per riordini
            initial_stock: Stock iniziale
            
        Returns:
            Dict con metriche inventory
        """
        stock = initial_stock
        stock_levels = []
        stockouts = []
        demand_satisfied = 0
        total_demand = 0
        
        for d, f in zip(demand, forecast):
            # Riordina basandosi su forecast
            order_quantity = max(0, f - stock)
            stock += order_quantity
            
            # Soddisfa domanda
            satisfied = min(stock, d)
            demand_satisfied += satisfied
            total_demand += d
            
            # Aggiorna stock
            stock = max(0, stock - d)
            stock_levels.append(stock)
            
            # Registra stockout
            if d > satisfied:
                stockouts.append(d - satisfied)
            else:
                stockouts.append(0)
                
        # Calcola metriche
        periods_in_stock = 100 * sum(s > 0 for s in stock_levels) / len(stock_levels)
        fill_rate = 100 * demand_satisfied / total_demand if total_demand > 0 else 100
        service_level = 100 * sum(s == 0 for s in stockouts) / len(stockouts)
        stockout_periods = sum(s > 0 for s in stockouts)
        
        # Overstock: stock > 2x average demand
        avg_demand = np.mean(demand[demand > 0]) if np.any(demand > 0) else 1
        overstock_periods = sum(s > 2 * avg_demand for s in stock_levels)
        
        return {
            'stock_levels': stock_levels,
            'stockouts': stockouts,
            'periods_in_stock': periods_in_stock,
            'fill_rate': fill_rate,
            'service_level': service_level,
            'stockout_periods': stockout_periods,
            'overstock_periods': overstock_periods
        }
        
    def _calculate_costs(self,
                        stock_levels: List[float],
                        stockouts: List[float]) -> Dict[str, float]:
        """
        Calcola costi totali inventory
        
        Args:
            stock_levels: Livelli stock per periodo
            stockouts: Quantità stockout per periodo
            
        Returns:
            Dict con costi disaggregati
        """
        holding_cost = self.holding_cost * sum(stock_levels)
        stockout_cost = self.stockout_cost * sum(stockouts)
        total_cost = holding_cost + stockout_cost
        
        return {
            'holding': holding_cost,
            'stockout': stockout_cost,
            'total': total_cost
        }
        
    def compare_methods(self,
                       actual: np.ndarray,
                       forecasts: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Confronta performance di metodi diversi
        
        Args:
            actual: Serie reale
            forecasts: Dict con nome_metodo -> forecast
            
        Returns:
            DataFrame con confronto metriche
        """
        results = []
        
        for method_name, forecast in forecasts.items():
            metrics = self.evaluate(actual, forecast)
            
            results.append({
                'Method': method_name,
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'MASE': metrics.mase,
                'Bias': metrics.bias,
                'Fill Rate %': metrics.fill_rate,
                'Service Level %': metrics.achieved_service_level,
                'Total Cost': metrics.total_cost or 0
            })
            
        df = pd.DataFrame(results)
        df = df.sort_values('MASE')  # MASE è metrica preferita per intermittent
        
        return df
        
    def plot_evaluation(self,
                       actual: np.ndarray,
                       forecast: np.ndarray,
                       method_name: str = "Forecast") -> None:
        """
        Crea grafici diagnostici per valutazione
        
        Args:
            actual: Valori reali
            forecast: Valori previsti
            method_name: Nome metodo per titolo
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Actual vs Forecast
            ax = axes[0, 0]
            ax.plot(actual, 'o-', label='Actual', alpha=0.7)
            ax.plot(forecast, 's-', label=method_name, alpha=0.7)
            ax.set_title('Actual vs Forecast')
            ax.set_xlabel('Period')
            ax.set_ylabel('Demand')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Errors
            ax = axes[0, 1]
            errors = actual - forecast
            ax.bar(range(len(errors)), errors, alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Forecast Errors')
            ax.set_xlabel('Period')
            ax.set_ylabel('Error')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Error Distribution
            ax = axes[1, 0]
            ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Error Distribution')
            ax.set_xlabel('Error')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Metrics Summary
            ax = axes[1, 1]
            ax.axis('off')
            metrics = self.evaluate(actual, forecast)
            
            text = f"""
            {method_name} Performance:
            
            RMSE: {metrics.rmse:.3f}
            MAE: {metrics.mae:.3f}
            MASE: {metrics.mase:.3f}
            Bias: {metrics.bias:.3f}
            
            Fill Rate: {metrics.fill_rate:.1f}%
            Service Level: {metrics.achieved_service_level:.1f}%
            Stockout Periods: {metrics.stockout_periods}
            """
            
            ax.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                   fontfamily='monospace')
            
            plt.suptitle(f'Intermittent Demand Evaluation - {method_name}')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            
    def __repr__(self) -> str:
        return (f"IntermittentEvaluator(holding_cost={self.holding_cost}, "
               f"stockout_cost={self.stockout_cost})")