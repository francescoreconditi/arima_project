"""
Comprehensive visualization utilities for time series forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, Union, List
from ..utils.logger import get_logger


class ForecastPlotter:
    """
    Advanced plotting utilities for time series analysis and forecasting.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter with style settings.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.logger = get_logger(__name__)
        self.default_figsize = figsize
        
        # Set style
        try:
            plt.style.use(style)
        except:
            self.logger.warning(f"Style '{style}' not available, using default")
            
        # Set color palette
        self.colors = {
            'actual': '#2E86C1',
            'forecast': '#E74C3C', 
            'confidence': '#F8C471',
            'residuals': '#7D3C98',
            'fit': '#27AE60'
        }
    
    def plot_forecast(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        confidence_intervals: Optional[pd.DataFrame] = None,
        title: str = "Time Series Forecast",
        figsize: Optional[Tuple[int, int]] = None,
        show_legend: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series with forecast and confidence intervals.
        
        Args:
            actual: Historical time series data
            forecast: Forecast values
            confidence_intervals: Optional confidence intervals DataFrame
            title: Plot title
            figsize: Figure size override
            show_legend: Whether to show legend
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(actual.index, actual.values, 
               color=self.colors['actual'], linewidth=2, label='Historical Data')
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values,
               color=self.colors['forecast'], linewidth=2, 
               linestyle='--', label='Forecast')
        
        # Plot confidence intervals if provided
        if confidence_intervals is not None:
            ax.fill_between(forecast.index,
                          confidence_intervals.iloc[:, 0],
                          confidence_intervals.iloc[:, 1],
                          color=self.colors['confidence'], alpha=0.3,
                          label='Confidence Interval')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if show_legend:
            ax.legend(loc='best', fontsize=10)
        
        # Add vertical line at forecast start
        if len(actual) > 0 and len(forecast) > 0:
            forecast_start = forecast.index[0]
            ax.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_residuals(
        self,
        residuals: pd.Series,
        fitted_values: Optional[pd.Series] = None,
        title: str = "Residual Analysis",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive residual analysis plots.
        
        Args:
            residuals: Model residuals
            fitted_values: Optional fitted values
            title: Plot title
            figsize: Figure size override  
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        figsize = figsize or (15, 10)
        fig = plt.figure(figsize=figsize)
        
        # Residuals time series plot
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(residuals.index, residuals.values, color=self.colors['residuals'])
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(residuals.dropna(), bins=30, color=self.colors['residuals'], alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Residuals')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax3 = plt.subplot(2, 3, 3)
        from scipy.stats import probplot
        probplot(residuals.dropna(), dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal)')
        ax3.grid(True, alpha=0.3)
        
        # ACF plot
        ax4 = plt.subplot(2, 3, 4)
        self._plot_acf(residuals.dropna(), ax=ax4, title='Residuals ACF')
        
        # Residuals vs Fitted (if fitted values provided)
        ax5 = plt.subplot(2, 3, 5)
        if fitted_values is not None:
            ax5.scatter(fitted_values, residuals, alpha=0.6, color=self.colors['residuals'])
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax5.set_xlabel('Fitted Values')
            ax5.set_ylabel('Residuals')
            ax5.set_title('Residuals vs Fitted')
        else:
            ax5.text(0.5, 0.5, 'Fitted values\nnot provided', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Residuals vs Fitted')
        ax5.grid(True, alpha=0.3)
        
        # Scale-Location plot
        ax6 = plt.subplot(2, 3, 6)
        if fitted_values is not None:
            sqrt_abs_resid = np.sqrt(np.abs(residuals))
            ax6.scatter(fitted_values, sqrt_abs_resid, alpha=0.6, color=self.colors['residuals'])
            ax6.set_xlabel('Fitted Values')
            ax6.set_ylabel('√|Residuals|')
            ax6.set_title('Scale-Location Plot')
        else:
            ax6.text(0.5, 0.5, 'Fitted values\nnot provided',
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Scale-Location Plot')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Residual plot saved to {save_path}")
        
        return fig
    
    def plot_decomposition(
        self,
        series: pd.Series,
        model: str = 'additive',
        period: Optional[int] = None,
        title: str = "Time Series Decomposition",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series decomposition.
        
        Args:
            series: Time series to decompose
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period (if None, auto-detect)
            title: Plot title
            figsize: Figure size override
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            figsize = figsize or (12, 10)
            
            # Perform decomposition
            decomposition = seasonal_decompose(series, model=model, period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=figsize)
            
            # Original series
            axes[0].plot(series.index, series.values, color=self.colors['actual'])
            axes[0].set_title('Original Time Series')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                        color=self.colors['fit'])
            axes[1].set_title('Trend Component')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values,
                        color=self.colors['forecast'])
            axes[2].set_title('Seasonal Component')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid.index, decomposition.resid.values,
                        color=self.colors['residuals'])
            axes[3].set_title('Residual Component')
            axes[3].set_xlabel('Time')
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Decomposition plot saved to {save_path}")
            
            return fig
            
        except ImportError:
            self.logger.error("statsmodels required for decomposition plotting")
            raise
    
    def plot_acf_pacf(
        self,
        series: pd.Series,
        lags: int = 20,
        title: str = "ACF and PACF Plots",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Autocorrelation Function (ACF) and Partial ACF.
        
        Args:
            series: Time series data
            lags: Number of lags to display
            title: Plot title
            figsize: Figure size override
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        figsize = figsize or (12, 6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ACF plot
        self._plot_acf(series, ax=ax1, lags=lags, title='Autocorrelation Function (ACF)')
        
        # PACF plot
        self._plot_pacf(series, ax=ax2, lags=lags, title='Partial Autocorrelation Function (PACF)')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ACF/PACF plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'aic',
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of different models.
        
        Args:
            results: Dictionary with model names as keys and metrics as values
            metric: Metric to compare ('aic', 'bic', 'rmse', etc.)
            title: Plot title
            figsize: Figure size override
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        models = list(results.keys())
        values = [results[model].get(metric, float('nan')) for model in models]
        
        # Create bar plot
        bars = ax.bar(models, values, color=self.colors['forecast'], alpha=0.7, edgecolor='black')
        
        # Highlight best model (lowest value for most metrics)
        best_idx = np.nanargmin(values)
        bars[best_idx].set_color(self.colors['actual'])
        
        # Formatting
        title = title or f'Model Comparison - {metric.upper()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-labels if too many models
        if len(models) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def _plot_acf(
        self, 
        series: pd.Series, 
        ax: plt.Axes, 
        lags: int = 20, 
        title: str = 'ACF'
    ) -> None:
        """Plot autocorrelation function."""
        try:
            from statsmodels.tsa.stattools import acf
            
            # Calculate ACF
            clean_series = series.dropna()
            max_lags = min(lags, len(clean_series) // 4)
            
            if max_lags < 1:
                ax.text(0.5, 0.5, 'Insufficient data\nfor ACF plot', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
            
            acf_values = acf(clean_series, nlags=max_lags, fft=False)
            
            # Plot ACF
            ax.stem(range(len(acf_values)), acf_values, basefmt=" ")
            
            # Add confidence intervals
            n = len(clean_series)
            ci = 1.96 / np.sqrt(n)
            ax.axhline(y=ci, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_title(title)
            ax.set_xlabel('Lag')
            ax.set_ylabel('ACF')
            ax.grid(True, alpha=0.3)
            
        except ImportError:
            ax.text(0.5, 0.5, 'statsmodels required\nfor ACF plot',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def _plot_pacf(
        self, 
        series: pd.Series, 
        ax: plt.Axes, 
        lags: int = 20, 
        title: str = 'PACF'
    ) -> None:
        """Plot partial autocorrelation function."""
        try:
            from statsmodels.tsa.stattools import pacf
            
            # Calculate PACF
            clean_series = series.dropna()
            max_lags = min(lags, len(clean_series) // 4)
            
            if max_lags < 1:
                ax.text(0.5, 0.5, 'Insufficient data\nfor PACF plot',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
            
            pacf_values = pacf(clean_series, nlags=max_lags)
            
            # Plot PACF
            ax.stem(range(len(pacf_values)), pacf_values, basefmt=" ")
            
            # Add confidence intervals
            n = len(clean_series)
            ci = 1.96 / np.sqrt(n)
            ax.axhline(y=ci, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_title(title)
            ax.set_xlabel('Lag')
            ax.set_ylabel('PACF')
            ax.grid(True, alpha=0.3)
            
        except ImportError:
            ax.text(0.5, 0.5, 'statsmodels required\nfor PACF plot',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def create_dashboard(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        residuals: Optional[pd.Series] = None,
        confidence_intervals: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, float]] = None,
        title: str = "Forecasting Dashboard",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive forecasting dashboard.
        
        Args:
            actual: Historical time series data
            forecast: Forecast values  
            residuals: Optional model residuals
            confidence_intervals: Optional confidence intervals
            metrics: Optional performance metrics
            title: Dashboard title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Main forecast plot (top half)
        ax1 = plt.subplot(2, 3, (1, 2))
        ax1.plot(actual.index, actual.values, color=self.colors['actual'], 
                linewidth=2, label='Historical')
        ax1.plot(forecast.index, forecast.values, color=self.colors['forecast'], 
                linewidth=2, linestyle='--', label='Forecast')
        
        if confidence_intervals is not None:
            ax1.fill_between(forecast.index,
                           confidence_intervals.iloc[:, 0],
                           confidence_intervals.iloc[:, 1],
                           color=self.colors['confidence'], alpha=0.3,
                           label='Confidence Interval')
        
        # Add vertical line at forecast start
        if len(forecast) > 0:
            ax1.axvline(x=forecast.index[0], color='gray', linestyle=':', alpha=0.5)
        
        ax1.set_title('Time Series Forecast', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics table
        ax2 = plt.subplot(2, 3, 3)
        ax2.axis('off')
        if metrics:
            metrics_text = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text.append(f"{key.upper()}: {value:.4f}")
                else:
                    metrics_text.append(f"{key.upper()}: {value}")
            
            ax2.text(0.1, 0.9, 'Performance Metrics:', fontsize=12, fontweight='bold',
                    transform=ax2.transAxes, verticalalignment='top')
            ax2.text(0.1, 0.8, '\n'.join(metrics_text), fontsize=10,
                    transform=ax2.transAxes, verticalalignment='top',
                    family='monospace')
        else:
            ax2.text(0.5, 0.5, 'No metrics\nprovided', ha='center', va='center',
                    transform=ax2.transAxes)
        
        # Residuals plots (bottom half)
        if residuals is not None:
            # Residuals time series
            ax3 = plt.subplot(2, 3, 4)
            ax3.plot(residuals.index, residuals.values, color=self.colors['residuals'])
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Residuals Over Time', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Residuals histogram
            ax4 = plt.subplot(2, 3, 5)
            ax4.hist(residuals.dropna(), bins=20, color=self.colors['residuals'], 
                    alpha=0.7, edgecolor='black')
            ax4.set_title('Residuals Distribution', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # ACF of residuals
            ax5 = plt.subplot(2, 3, 6)
            self._plot_acf(residuals, ax=ax5, lags=15, title='Residuals ACF')
        else:
            # Placeholder plots if no residuals
            for i, subplot_num in enumerate([4, 5, 6]):
                ax = plt.subplot(2, 3, subplot_num)
                ax.text(0.5, 0.5, 'Residuals\nnot provided', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'Residuals Plot {i+1}', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        return fig