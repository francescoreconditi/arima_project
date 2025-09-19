"""
Router per diagnostica modelli.

Gestisce analisi residui, test statistici e metriche di performance.
"""

from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
import numpy as np
import pandas as pd

from arima_forecaster.api.models_extra import ModelDiagnostics
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(
    prefix="/models", tags=["Diagnostics"], responses={404: {"description": "Not found"}}
)

"""
🔍 DIAGNOSTICS ROUTER

Analisi approfondita performance e validità modelli:

• POST /models/{model_id}/diagnostics - Analisi diagnostica completa

Analisi incluse:
- Statistiche residui (media, deviazione, skewness, kurtosis)
- Test Ljung-Box per autocorrelazione residui
- Test Jarque-Bera per normalità distribuzione
- Calcolo ACF/PACF per analisi correlazione
- Metriche performance dettagliate (MAE, RMSE, MAPE, R²)
- Validazione assunzioni modello statistico
"""


# Dependency injection dei servizi
def get_services():
    """Dependency per ottenere i servizi necessari."""
    from pathlib import Path

    storage_path = Path("models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    return model_manager, forecast_service


@router.post("/{model_id}/diagnostics", response_model=ModelDiagnostics)
async def get_model_diagnostics(model_id: str, services: tuple = Depends(get_services)):
    """
    Esegue diagnostica completa su un modello addestrato.

    Analizza i residui del modello, esegue test statistici e calcola metriche
    di performance dettagliate per valutare la qualità del fit.

    <h4>Parametri di Ingresso:</h4>
    <table >
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID univoco del modello da analizzare</td></tr>
    </table>

    <h4>Risposta:</h4>
    <table >
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>residuals_stats</td><td>dict</td><td>Statistiche sui residui (mean, std, skewness, kurtosis)</td></tr>
        <tr><td>ljung_box_test</td><td>dict</td><td>Test di Ljung-Box per autocorrelazione residui</td></tr>
        <tr><td>jarque_bera_test</td><td>dict</td><td>Test di Jarque-Bera per normalità dei residui</td></tr>
        <tr><td>acf_values</td><td>list</td><td>Valori di autocorrelazione (ACF)</td></tr>
        <tr><td>pacf_values</td><td>list</td><td>Valori di autocorrelazione parziale (PACF)</td></tr>
        <tr><td>performance_metrics</td><td>dict</td><td>Metriche di performance (MAE, RMSE, MAPE, etc.)</td></tr>
    </table>

    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -X POST "http://localhost:8000/models/abc123/diagnostics"
    </code></pre>

    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "residuals_stats": {
            "mean": 0.002,
            "std": 1.234,
            "skewness": 0.145,
            "kurtosis": 3.021
        },
        "ljung_box_test": {
            "statistic": 15.234,
            "p_value": 0.432,
            "result": "No autocorrelation detected"
        },
        "jarque_bera_test": {
            "statistic": 2.145,
            "p_value": 0.342,
            "result": "Residuals are normally distributed"
        },
        "acf_values": [1.0, 0.05, -0.02, ...],
        "pacf_values": [1.0, 0.05, -0.01, ...],
        "performance_metrics": {
            "mae": 12.34,
            "rmse": 15.67,
            "mape": 5.43,
            "r2": 0.92
        }
    }
    </code></pre>

    <h4>Errori Possibili:</h4>
    <ul>
        <li><strong>404</strong>: Modello non trovato</li>
        <li><strong>500</strong>: Errore durante l'analisi diagnostica</li>
    </ul>
    """
    model_manager, _ = services

    try:
        # Verifica l'esistenza del modello
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Model not found")

        # Carica il modello
        model = model_manager.load_model(model_id)

        # Ottiene i residui del modello
        residuals = model.residuals if hasattr(model, "residuals") else None

        if residuals is None or len(residuals) == 0:
            raise HTTPException(
                status_code=400, detail="Model does not have residuals available for diagnostics"
            )

        # Calcola statistiche sui residui
        residuals_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skewness": float(pd.Series(residuals).skew()),
            "kurtosis": float(pd.Series(residuals).kurtosis()),
        }

        # Test di Ljung-Box per autocorrelazione
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_test = acorr_ljungbox(residuals, lags=10, return_df=False)
        ljung_box_test = {
            "statistic": float(lb_test[0][-1]),
            "p_value": float(lb_test[1][-1]),
            "result": "No autocorrelation detected"
            if lb_test[1][-1] > 0.05
            else "Autocorrelation detected in residuals",
        }

        # Test di Jarque-Bera per normalità
        from scipy import stats

        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        jarque_bera_test = {
            "statistic": float(jb_stat),
            "p_value": float(jb_pvalue),
            "result": "Residuals are normally distributed"
            if jb_pvalue > 0.05
            else "Residuals are not normally distributed",
        }

        # Calcola ACF e PACF
        from statsmodels.stats.stattools import acf, pacf

        acf_values = acf(residuals, nlags=20).tolist()
        pacf_values = pacf(residuals, nlags=20).tolist()

        # Calcola metriche di performance
        evaluator = ModelEvaluator()

        # Ottiene valori fitted se disponibili
        if hasattr(model, "fitted_values") and model.fitted_values is not None:
            fitted = model.fitted_values
            # Trova l'indice comune tra serie originale e fitted
            if hasattr(model, "_last_series"):
                actual = model._last_series
                common_idx = actual.index.intersection(fitted.index)
                if len(common_idx) > 0:
                    metrics = evaluator.calculate_metrics(actual[common_idx], fitted[common_idx])
                else:
                    metrics = {}
            else:
                metrics = {}
        else:
            metrics = {}

        # Converte metriche in float per JSON
        performance_metrics = {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}

        return ModelDiagnostics(
            residuals_stats=residuals_stats,
            ljung_box_test=ljung_box_test,
            jarque_bera_test=jarque_bera_test,
            acf_values=acf_values,
            pacf_values=pacf_values,
            performance_metrics=performance_metrics,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diagnostics failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
