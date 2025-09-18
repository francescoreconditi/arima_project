"""
Utility di visualizzazione complete per previsioni serie temporali.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, Union, List
from ..utils.logger import get_logger
from ..utils.translations import translate as _


class ForecastPlotter:
    """
    Utility di plotting avanzate per analisi serie temporali e previsioni.
    """

    def __init__(
        self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8), language: str = "it"
    ):
        """
        Inizializza plotter con impostazioni di stile.

        Args:
            style: Stile matplotlib da usare
            figsize: Dimensione figura predefinita
            language: Lingua per le label dei grafici ('it', 'en', 'es', 'fr', 'zh')
        """
        self.logger = get_logger(__name__)
        self.default_figsize = figsize
        self.language = language

        # Imposta stile
        try:
            plt.style.use(style)
        except:
            self.logger.warning(f"Stile '{style}' non disponibile, uso predefinito")

        # Imposta palette colori
        self.colors = {
            "actual": "#2E86C1",
            "forecast": "#E74C3C",
            "confidence": "#F8C471",
            "residuals": "#7D3C98",
            "fit": "#27AE60",
        }

    def plot_forecast(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        confidence_intervals: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        show_legend: bool = True,
        save_path: Optional[str] = None,
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

        # Traccia dati storici
        ax.plot(
            actual.index,
            actual.values,
            color=self.colors["actual"],
            linewidth=2,
            label=_("historical_data", self.language),
        )

        # Traccia previsione
        ax.plot(
            forecast.index,
            forecast.values,
            color=self.colors["forecast"],
            linewidth=2,
            linestyle="--",
            label=_("forecast", self.language),
        )

        # Traccia intervalli di confidenza se forniti
        if confidence_intervals is not None:
            ax.fill_between(
                forecast.index,
                confidence_intervals.iloc[:, 0],
                confidence_intervals.iloc[:, 1],
                color=self.colors["confidence"],
                alpha=0.3,
                label=_("confidence_interval", self.language),
            )

        # Formatting
        if title is None:
            title = _("time_series_forecast", self.language)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(_("time", self.language), fontsize=12)
        ax.set_ylabel(_("value", self.language), fontsize=12)
        ax.grid(True, alpha=0.3)

        if show_legend:
            ax.legend(loc="best", fontsize=10)

        # Aggiunge linea verticale all'inizio previsione
        if len(actual) > 0 and len(forecast) > 0:
            forecast_start = forecast.index[0]
            ax.axvline(x=forecast_start, color="gray", linestyle=":", alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_residuals(
        self,
        residuals: pd.Series,
        fitted_values: Optional[pd.Series] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
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
        ax1.plot(residuals.index, residuals.values, color=self.colors["residuals"])
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax1.set_title(_("residuals_over_time", self.language))
        ax1.set_xlabel(_("time", self.language))
        ax1.set_ylabel(_("residuals", self.language))
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(
            residuals.dropna(),
            bins=30,
            color=self.colors["residuals"],
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_title(_("distribution_of_residuals", self.language))
        ax2.set_xlabel(_("residual_value", self.language))
        ax2.set_ylabel(_("frequency", self.language))
        ax2.grid(True, alpha=0.3)

        # Q-Q plot
        ax3 = plt.subplot(2, 3, 3)
        from scipy.stats import probplot

        probplot(residuals.dropna(), dist="norm", plot=ax3)
        ax3.set_title(_("qq_plot", self.language))
        ax3.grid(True, alpha=0.3)

        # ACF plot
        ax4 = plt.subplot(2, 3, 4)
        self._plot_acf(residuals.dropna(), ax=ax4, title=_("residuals_acf", self.language))

        # Residuals vs Fitted (if fitted values provided)
        ax5 = plt.subplot(2, 3, 5)
        if fitted_values is not None:
            ax5.scatter(fitted_values, residuals, alpha=0.6, color=self.colors["residuals"])
            ax5.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax5.set_xlabel(_("fitted_values", self.language))
            ax5.set_ylabel(_("residuals", self.language))
            ax5.set_title(_("residuals_vs_fitted", self.language))
        else:
            ax5.text(
                0.5,
                0.5,
                "Fitted values\nnot provided",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )
            ax5.set_title(_("residuals_vs_fitted", self.language))
        ax5.grid(True, alpha=0.3)

        # Scale-Location plot
        ax6 = plt.subplot(2, 3, 6)
        if fitted_values is not None:
            sqrt_abs_resid = np.sqrt(np.abs(residuals))
            ax6.scatter(fitted_values, sqrt_abs_resid, alpha=0.6, color=self.colors["residuals"])
            ax6.set_xlabel(_("fitted_values", self.language))
            ax6.set_ylabel("√|" + _("residuals", self.language) + "|")
            ax6.set_title(_("scale_location_plot", self.language))
        else:
            ax6.text(
                0.5,
                0.5,
                "Fitted values\nnot provided",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title(_("scale_location_plot", self.language))
        ax6.grid(True, alpha=0.3)

        if title is None:
            title = _("residual_analysis", self.language)
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Grafico residui salvato in {save_path}")

        return fig

    def plot_decomposition(
        self,
        series: pd.Series,
        model: str = "additive",
        period: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
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
            axes[0].plot(series.index, series.values, color=self.colors["actual"])
            axes[0].set_title(_("original_time_series", self.language))
            axes[0].grid(True, alpha=0.3)

            # Trend
            axes[1].plot(
                decomposition.trend.index, decomposition.trend.values, color=self.colors["fit"]
            )
            axes[1].set_title(_("trend_component", self.language))
            axes[1].grid(True, alpha=0.3)

            # Seasonal
            axes[2].plot(
                decomposition.seasonal.index,
                decomposition.seasonal.values,
                color=self.colors["forecast"],
            )
            axes[2].set_title(_("seasonal_component", self.language))
            axes[2].grid(True, alpha=0.3)

            # Residual
            axes[3].plot(
                decomposition.resid.index,
                decomposition.resid.values,
                color=self.colors["residuals"],
            )
            axes[3].set_title(_("residual_component", self.language))
            axes[3].set_xlabel(_("time", self.language))
            axes[3].grid(True, alpha=0.3)

            if title is None:
                title = _("time_series_decomposition", self.language)
            plt.suptitle(title, fontsize=16, fontweight="bold")
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Decomposition plot saved to {save_path}")

            return fig

        except ImportError:
            self.logger.error("statsmodels required for decomposition plotting")
            raise

    def plot_acf_pacf(
        self,
        series: pd.Series,
        lags: int = 20,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
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
        self._plot_acf(
            series, ax=ax1, lags=lags, title=_("autocorrelation_function", self.language)
        )

        # PACF plot
        self._plot_pacf(
            series, ax=ax2, lags=lags, title=_("partial_autocorrelation_function", self.language)
        )

        if title is None:
            title = _("acf_pacf_plots", self.language)
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"ACF/PACF plot saved to {save_path}")

        return fig

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "aic",
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
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
        values = [results[model].get(metric, float("nan")) for model in models]

        # Create bar plot
        bars = ax.bar(models, values, color=self.colors["forecast"], alpha=0.7, edgecolor="black")

        # Highlight best model (lowest value for most metrics)
        best_idx = np.nanargmin(values)
        bars[best_idx].set_color(self.colors["actual"])

        # Formatting
        if title is None:
            title = f"{_('model_comparison', self.language)} - {metric.upper()}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(_("model", self.language), fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-labels if too many models
        if len(models) > 5:
            plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def _plot_acf(
        self, series: pd.Series, ax: plt.Axes, lags: int = 20, title: str = "ACF"
    ) -> None:
        """Traccia funzione di autocorrelazione."""
        try:
            from statsmodels.tsa.stattools import acf

            # Calculate ACF
            clean_series = series.dropna()
            max_lags = min(lags, len(clean_series) // 4)

            if max_lags < 1:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data\nfor ACF plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                return

            acf_values = acf(clean_series, nlags=max_lags, fft=False)

            # Plot ACF
            ax.stem(range(len(acf_values)), acf_values, basefmt=" ")

            # Add confidence intervals
            n = len(clean_series)
            ci = 1.96 / np.sqrt(n)
            ax.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_title(title)
            ax.set_xlabel(_("lag", self.language))
            ax.set_ylabel(_("acf", self.language))
            ax.grid(True, alpha=0.3)

        except ImportError:
            ax.text(
                0.5,
                0.5,
                "statsmodels required\nfor ACF plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)

    def _plot_pacf(
        self, series: pd.Series, ax: plt.Axes, lags: int = 20, title: str = "PACF"
    ) -> None:
        """Traccia funzione di autocorrelazione parziale."""
        try:
            from statsmodels.tsa.stattools import pacf

            # Calculate PACF
            clean_series = series.dropna()
            max_lags = min(lags, len(clean_series) // 4)

            if max_lags < 1:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data\nfor PACF plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                return

            pacf_values = pacf(clean_series, nlags=max_lags)

            # Plot PACF
            ax.stem(range(len(pacf_values)), pacf_values, basefmt=" ")

            # Add confidence intervals
            n = len(clean_series)
            ci = 1.96 / np.sqrt(n)
            ax.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            ax.set_title(title)
            ax.set_xlabel(_("lag", self.language))
            ax.set_ylabel(_("pacf", self.language))
            ax.grid(True, alpha=0.3)

        except ImportError:
            ax.text(
                0.5,
                0.5,
                "statsmodels required\nfor PACF plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)

    def create_dashboard(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        residuals: Optional[pd.Series] = None,
        confidence_intervals: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Crea dashboard completa di previsione.

        Args:
            actual: Dati storici serie temporali
            forecast: Valori previsti
            residuals: Residui modello opzionali
            confidence_intervals: Intervalli di confidenza opzionali
            metrics: Metriche performance opzionali
            title: Titolo dashboard
            save_path: Percorso per salvare figura

        Returns:
            Oggetto figura matplotlib
        """
        fig = plt.figure(figsize=(16, 12))

        # Main forecast plot (top half)
        ax1 = plt.subplot(2, 3, (1, 2))
        ax1.plot(
            actual.index,
            actual.values,
            color=self.colors["actual"],
            linewidth=2,
            label=_("historical_data", self.language),
        )
        ax1.plot(
            forecast.index,
            forecast.values,
            color=self.colors["forecast"],
            linewidth=2,
            linestyle="--",
            label=_("forecast", self.language),
        )

        if confidence_intervals is not None:
            ax1.fill_between(
                forecast.index,
                confidence_intervals.iloc[:, 0],
                confidence_intervals.iloc[:, 1],
                color=self.colors["confidence"],
                alpha=0.3,
                label=_("confidence_interval", self.language),
            )

        # Aggiunge linea verticale all'inizio previsione
        if len(forecast) > 0:
            ax1.axvline(x=forecast.index[0], color="gray", linestyle=":", alpha=0.5)

        ax1.set_title(_("time_series_forecast", self.language), fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Metrics table
        ax2 = plt.subplot(2, 3, 3)
        ax2.axis("off")
        if metrics:
            metrics_text = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text.append(f"{key.upper()}: {value:.4f}")
                else:
                    metrics_text.append(f"{key.upper()}: {value}")

            ax2.text(
                0.1,
                0.9,
                _("performance_metrics", self.language) + ":",
                fontsize=12,
                fontweight="bold",
                transform=ax2.transAxes,
                verticalalignment="top",
            )
            ax2.text(
                0.1,
                0.8,
                "\n".join(metrics_text),
                fontsize=10,
                transform=ax2.transAxes,
                verticalalignment="top",
                family="monospace",
            )
        else:
            ax2.text(
                0.5, 0.5, "No metrics\nprovided", ha="center", va="center", transform=ax2.transAxes
            )

        # Residuals plots (bottom half)
        if residuals is not None:
            # Residuals time series
            ax3 = plt.subplot(2, 3, 4)
            ax3.plot(residuals.index, residuals.values, color=self.colors["residuals"])
            ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax3.set_title(_("residuals_over_time", self.language), fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Residuals histogram
            ax4 = plt.subplot(2, 3, 5)
            ax4.hist(
                residuals.dropna(),
                bins=20,
                color=self.colors["residuals"],
                alpha=0.7,
                edgecolor="black",
            )
            ax4.set_title(_("distribution_of_residuals", self.language), fontsize=10)
            ax4.grid(True, alpha=0.3)

            # ACF of residuals
            ax5 = plt.subplot(2, 3, 6)
            self._plot_acf(residuals, ax=ax5, lags=15, title=_("residuals_acf", self.language))
        else:
            # Placeholder plots if no residuals
            for i, subplot_num in enumerate([4, 5, 6]):
                ax = plt.subplot(2, 3, subplot_num)
                ax.text(
                    0.5,
                    0.5,
                    "Residuals\nnot provided",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Residuals Plot {i + 1}", fontsize=10)

        if title is None:
            title = _("forecasting_dashboard", self.language)
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Dashboard saved to {save_path}")

        return fig

    def plot_exog_analysis(
        self,
        exog_data: pd.DataFrame,
        exog_importance: Optional[pd.DataFrame] = None,
        target_series: Optional[pd.Series] = None,
        title: str = "Analisi Variabili Esogene",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Crea visualizzazione completa per l'analisi delle variabili esogene SARIMAX.

        Args:
            exog_data: DataFrame con variabili esogene
            exog_importance: DataFrame con importanza variabili (coefficienti, p-values)
            target_series: Serie temporale target opzionale
            title: Titolo del grafico
            figsize: Dimensione figura
            save_path: Percorso per salvare figura

        Returns:
            Oggetto figura matplotlib
        """
        figsize = figsize or (16, 10)
        fig = plt.figure(figsize=figsize)

        n_vars = len(exog_data.columns)

        if n_vars <= 4:
            # Layout per poche variabili
            rows, cols = 2, 3
        else:
            # Layout per molte variabili
            rows, cols = 3, 3

        # 1. Serie temporali delle variabili esogene
        ax1 = plt.subplot(rows, cols, 1)
        colors_cycle = plt.cm.tab10(np.linspace(0, 1, n_vars))

        for i, col in enumerate(exog_data.columns[:5]):  # Max 5 variabili nel plot
            ax1.plot(exog_data.index, exog_data[col], color=colors_cycle[i], alpha=0.8, label=col)

        ax1.set_title(_("exog_variables_over_time", self.language), fontweight="bold")
        ax1.set_xlabel(_("time", self.language))
        ax1.set_ylabel(_("value", self.language))
        if n_vars <= 5:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 2. Importanza delle variabili (se disponibile)
        ax2 = plt.subplot(rows, cols, 2)
        if exog_importance is not None and not exog_importance.empty:
            # Ordina per coefficiente assoluto
            importance_sorted = exog_importance.reindex(
                exog_importance["coefficient"].abs().sort_values(ascending=False).index
            )

            colors = ["green" if sig else "red" for sig in importance_sorted["significant"]]
            bars = ax2.bar(
                range(len(importance_sorted)),
                importance_sorted["coefficient"],
                color=colors,
                alpha=0.7,
            )

            ax2.set_title(_("exog_coefficients", self.language), fontweight="bold")
            ax2.set_xlabel(_("variables", self.language))
            ax2.set_ylabel(_("coefficient", self.language))
            ax2.set_xticks(range(len(importance_sorted)))
            ax2.set_xticklabels(importance_sorted["variable"], rotation=45, ha="right")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            # Legenda
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="green", alpha=0.7, label=_("significant_p005", self.language)),
                Patch(facecolor="red", alpha=0.7, label=_("not_significant", self.language)),
            ]
            ax2.legend(handles=legend_elements)
        else:
            ax2.text(
                0.5,
                0.5,
                "Importanza variabili\nnon disponibile",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title(_("variable_importance", self.language), fontweight="bold")

        # 3. Correlazioni tra variabili esogene
        ax3 = plt.subplot(rows, cols, 3)
        numeric_cols = exog_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = exog_data[numeric_cols].corr()
            im = ax3.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

            # Aggiungi valori di correlazione
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    text = ax3.text(
                        j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
                    )

            ax3.set_xticks(range(len(corr_matrix.columns)))
            ax3.set_yticks(range(len(corr_matrix)))
            ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
            ax3.set_yticklabels(corr_matrix.index)
            ax3.set_title(_("variable_correlations", self.language), fontweight="bold")

            # Colorbar
            cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.set_label(_("correlation", self.language))
        else:
            ax3.text(
                0.5,
                0.5,
                "Correlazioni\nnon calcolabili\n(< 2 variabili numeriche)",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title(_("variable_correlations", self.language), fontweight="bold")

        # 4. Distribuzione delle variabili (solo prime 4)
        if rows >= 3:
            for i, col in enumerate(numeric_cols[:4]):
                ax = plt.subplot(rows, cols, 4 + i)
                ax.hist(
                    exog_data[col].dropna(),
                    bins=20,
                    alpha=0.7,
                    color=colors_cycle[i % len(colors_cycle)],
                    edgecolor="black",
                )
                ax.set_title(f"Distribuzione {col}", fontsize=10)
                ax.set_ylabel(_("frequency", self.language))
                ax.grid(True, alpha=0.3)

        # 5. Correlazione con serie target (se disponibile)
        if target_series is not None and rows >= 3:
            ax_target = plt.subplot(rows, cols, cols * (rows - 1))

            target_corr = []
            var_names = []
            for col in numeric_cols[:6]:  # Max 6 variabili
                if len(target_series) == len(exog_data):
                    corr_val = target_series.corr(exog_data[col])
                    if not np.isnan(corr_val):
                        target_corr.append(corr_val)
                        var_names.append(col)

            if target_corr:
                colors_corr = [
                    "green" if abs(c) > 0.5 else "orange" if abs(c) > 0.3 else "red"
                    for c in target_corr
                ]
                bars = ax_target.bar(
                    range(len(target_corr)), target_corr, color=colors_corr, alpha=0.7
                )

                ax_target.set_title(_("target_correlation", self.language), fontweight="bold")
                ax_target.set_xlabel(_("exogenous_variables_axis", self.language))
                ax_target.set_ylabel(_("correlation", self.language))
                ax_target.set_xticks(range(len(var_names)))
                ax_target.set_xticklabels(var_names, rotation=45, ha="right")
                ax_target.grid(True, alpha=0.3)
                ax_target.axhline(y=0, color="black", linestyle="-", alpha=0.5)
                ax_target.set_ylim(-1, 1)

                # Aggiungi valori sui bar
                for bar, val in zip(bars, target_corr):
                    height = bar.get_height()
                    ax_target.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.02 if height >= 0 else height - 0.02,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontsize=9,
                    )
            else:
                ax_target.text(
                    0.5,
                    0.5,
                    "Correlazioni con target\nnon calcolabili",
                    ha="center",
                    va="center",
                    transform=ax_target.transAxes,
                )
                ax_target.set_title(_("target_correlation", self.language), fontweight="bold")

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Exog analysis plot saved to {save_path}")

        return fig

    def plot_sarimax_decomposition(
        self,
        decomposition: Dict[str, pd.Series],
        exog_contributions: Optional[Dict[str, pd.Series]] = None,
        title: str = "Decomposizione SARIMAX",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualizza decomposizione SARIMAX con contributi delle variabili esogene.

        Args:
            decomposition: Dizionario con componenti di decomposizione
            exog_contributions: Contributi individuali delle variabili esogene
            title: Titolo del grafico
            figsize: Dimensione figura
            save_path: Percorso per salvare figura

        Returns:
            Oggetto figura matplotlib
        """
        figsize = figsize or (14, 10)
        n_plots = len(decomposition) + (1 if exog_contributions else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot componenti standard di decomposizione
        for component, values in decomposition.items():
            ax = axes[plot_idx]
            ax.plot(values.index, values.values, linewidth=1.5)
            ax.set_title(f"{component.capitalize()} Component", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            if component == "residual":
                ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

            plot_idx += 1

        # Plot contributi variabili esogene se disponibili
        if exog_contributions:
            ax = axes[plot_idx]
            colors = plt.cm.Set3(np.linspace(0, 1, len(exog_contributions)))

            for i, (var_name, contribution) in enumerate(exog_contributions.items()):
                ax.plot(
                    contribution.index,
                    contribution.values,
                    color=colors[i],
                    linewidth=1.5,
                    label=var_name,
                    alpha=0.8,
                )

            ax.set_title(_("exog_contributions", self.language), fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Formattazione comune
        axes[-1].set_xlabel(_("time", self.language), fontsize=11)
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"SARIMAX decomposition plot saved to {save_path}")

        return fig

    def create_sarimax_dashboard(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        exog_data: pd.DataFrame,
        exog_importance: Optional[pd.DataFrame] = None,
        residuals: Optional[pd.Series] = None,
        confidence_intervals: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, float]] = None,
        title: str = "SARIMAX Dashboard",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Crea dashboard completa specifica per modelli SARIMAX.

        Args:
            actual: Dati storici serie temporali
            forecast: Valori previsti
            exog_data: DataFrame variabili esogene
            exog_importance: Importanza variabili esogene
            residuals: Residui modello
            confidence_intervals: Intervalli di confidenza
            metrics: Metriche performance
            title: Titolo dashboard
            save_path: Percorso per salvare figura

        Returns:
            Oggetto figura matplotlib
        """
        fig = plt.figure(figsize=(18, 12))

        # 1. Main forecast plot (top left, spans 2 columns)
        ax1 = plt.subplot(3, 4, (1, 2))
        ax1.plot(
            actual.index,
            actual.values,
            color=self.colors["actual"],
            linewidth=2,
            label=_("historical", self.language),
        )
        ax1.plot(
            forecast.index,
            forecast.values,
            color=self.colors["forecast"],
            linewidth=2,
            linestyle="--",
            label=_("forecast", self.language),
        )

        if confidence_intervals is not None:
            ax1.fill_between(
                forecast.index,
                confidence_intervals.iloc[:, 0],
                confidence_intervals.iloc[:, 1],
                color=self.colors["confidence"],
                alpha=0.3,
                label=_("confidence_interval", self.language),
            )

        if len(forecast) > 0:
            ax1.axvline(x=forecast.index[0], color="gray", linestyle=":", alpha=0.5)

        ax1.set_title(_("sarimax_forecast", self.language), fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Exog variables importance (top right)
        ax2 = plt.subplot(3, 4, (3, 4))
        if exog_importance is not None and not exog_importance.empty:
            importance_sorted = exog_importance.reindex(
                exog_importance["coefficient"].abs().sort_values(ascending=False).index
            )

            colors = ["green" if sig else "red" for sig in importance_sorted["significant"]]
            bars = ax2.bar(
                range(len(importance_sorted)),
                importance_sorted["coefficient"],
                color=colors,
                alpha=0.7,
            )

            ax2.set_title(_("exog_coefficients", self.language), fontweight="bold")
            ax2.set_xticks(range(len(importance_sorted)))
            ax2.set_xticklabels(importance_sorted["variable"], rotation=45, ha="right")
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
            ax2.grid(True, alpha=0.3)

            # Aggiungi valori sui bar
            for bar, val in zip(bars, importance_sorted["coefficient"]):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 if height >= 0 else height - 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "Importanza variabili\nnon disponibile",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title(_("exog_coefficients", self.language), fontweight="bold")

        # 3. Exog variables time series (middle left)
        ax3 = plt.subplot(3, 4, 5)
        colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(exog_data.columns)))

        for i, col in enumerate(exog_data.columns[:4]):  # Max 4 per leggibilità
            ax3.plot(
                exog_data.index,
                exog_data[col],
                color=colors_cycle[i],
                alpha=0.8,
                label=col,
                linewidth=1,
            )

        ax3.set_title(_("exogenous_variables", self.language), fontsize=12, fontweight="bold")
        if len(exog_data.columns) <= 4:
            ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Residuals (middle center)
        ax4 = plt.subplot(3, 4, 6)
        if residuals is not None:
            ax4.plot(residuals.index, residuals.values, color=self.colors["residuals"])
            ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax4.set_title(
                _("residuals_over_time_title", self.language), fontsize=12, fontweight="bold"
            )
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "Residui\nnon disponibili",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title(
                _("residuals_over_time_title", self.language), fontsize=12, fontweight="bold"
            )

        # 5. Metrics table (middle right)
        ax5 = plt.subplot(3, 4, 7)
        ax5.axis("off")
        if metrics:
            metrics_text = []
            key_metrics = ["mae", "rmse", "mape", "r_squared"]  # Metriche chiave

            for key in key_metrics:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, float):
                        metrics_text.append(f"{key.upper()}: {value:.4f}")

            ax5.text(
                0.1,
                0.9,
                "Metriche Performance:",
                fontsize=12,
                fontweight="bold",
                transform=ax5.transAxes,
                verticalalignment="top",
            )
            ax5.text(
                0.1,
                0.7,
                "\n".join(metrics_text),
                fontsize=10,
                transform=ax5.transAxes,
                verticalalignment="top",
                family="monospace",
            )

            # Aggiungi informazioni modello
            model_info = []
            if "n_exog" in metrics:
                model_info.append(f"Variabili esogene: {metrics['n_exog']}")
            if "n_observations" in metrics:
                model_info.append(f"Osservazioni: {metrics['n_observations']}")

            if model_info:
                ax5.text(
                    0.1,
                    0.4,
                    "Info Modello:",
                    fontsize=11,
                    fontweight="bold",
                    transform=ax5.transAxes,
                    verticalalignment="top",
                )
                ax5.text(
                    0.1,
                    0.3,
                    "\n".join(model_info),
                    fontsize=9,
                    transform=ax5.transAxes,
                    verticalalignment="top",
                )
        else:
            ax5.text(
                0.5,
                0.5,
                "Metriche\nnon disponibili",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )

        # 6. Correlations matrix (middle-right)
        ax6 = plt.subplot(3, 4, 8)
        numeric_cols = exog_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Mostra solo le prime 4 variabili per spazio
            corr_matrix = exog_data[numeric_cols[:4]].corr()
            im = ax6.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

            # Valori di correlazione
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    ax6.text(
                        j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
                    )

            ax6.set_xticks(range(len(corr_matrix.columns)))
            ax6.set_yticks(range(len(corr_matrix)))
            ax6.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
            ax6.set_yticklabels(corr_matrix.index, fontsize=8)
            ax6.set_title(_("variable_correlations", self.language), fontsize=12, fontweight="bold")
        else:
            ax6.text(
                0.5,
                0.5,
                "Correlazioni\nnon disponibili",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title(_("variable_correlations", self.language), fontsize=12, fontweight="bold")

        # 7. Residuals histogram (bottom left)
        ax7 = plt.subplot(3, 4, 9)
        if residuals is not None:
            ax7.hist(
                residuals.dropna(),
                bins=20,
                color=self.colors["residuals"],
                alpha=0.7,
                edgecolor="black",
            )
            ax7.set_title(
                _("residuals_distribution", self.language), fontsize=12, fontweight="bold"
            )
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(
                0.5,
                0.5,
                "Distribuzione\nresidue\nnon disponibile",
                ha="center",
                va="center",
                transform=ax7.transAxes,
            )
            ax7.set_title(
                _("residuals_distribution", self.language), fontsize=12, fontweight="bold"
            )

        # 8. ACF of residuals (bottom center)
        ax8 = plt.subplot(3, 4, 10)
        if residuals is not None:
            self._plot_acf(
                residuals, ax=ax8, lags=12, title=_("residuals_acf_title", self.language)
            )
        else:
            ax8.text(
                0.5,
                0.5,
                "ACF\nresidue\nnon disponibile",
                ha="center",
                va="center",
                transform=ax8.transAxes,
            )
            ax8.set_title(_("residuals_acf_title", self.language), fontsize=12, fontweight="bold")

        # 9. Significatività variabili (bottom right, spans 2)
        ax9 = plt.subplot(3, 4, (11, 12))
        if exog_importance is not None and not exog_importance.empty:
            # Grafico p-values
            pvalues = exog_importance["pvalue"].values
            var_names = exog_importance["variable"].values

            colors_pval = ["green" if p < 0.05 else "orange" if p < 0.1 else "red" for p in pvalues]
            bars = ax9.bar(range(len(pvalues)), pvalues, color=colors_pval, alpha=0.7)

            ax9.set_title(_("variable_significance_pvalues", self.language), fontweight="bold")
            ax9.set_xlabel(_("variables", self.language))
            ax9.set_ylabel(_("p_value", self.language))
            ax9.set_xticks(range(len(var_names)))
            ax9.set_xticklabels(var_names, rotation=45, ha="right")
            ax9.axhline(
                y=0.05, color="red", linestyle="--", alpha=0.7, label=_("alpha_005", self.language)
            )
            ax9.axhline(
                y=0.1,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label=_("alpha_010", self.language),
            )
            ax9.set_yscale("log")
            ax9.grid(True, alpha=0.3)
            ax9.legend(fontsize=8)

            # Aggiungi valori
            for bar, val in zip(bars, pvalues):
                height = bar.get_height()
                ax9.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height * 1.1,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            ax9.text(
                0.5,
                0.5,
                "Significatività\nnon disponibile",
                ha="center",
                va="center",
                transform=ax9.transAxes,
            )
            ax9.set_title(_("variable_significance", self.language), fontweight="bold")

        plt.suptitle(title, fontsize=18, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"SARIMAX dashboard saved to {save_path}")

        return fig
