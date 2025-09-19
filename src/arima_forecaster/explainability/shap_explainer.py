"""
SHAP Explainer per Forecast

Genera spiegazioni SHAP per predizioni dei modelli di forecasting.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import shap

    _has_shap = True
except ImportError:
    _has_shap = False

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SHAPConfig:
    """Configurazione SHAP explainer"""

    explainer_type: str = "auto"  # "auto", "kernel", "tree", "linear"
    max_evals: int = 1000
    feature_names: Optional[List[str]] = None
    background_samples: int = 100
    confidence_level: float = 0.95
    cache_explanations: bool = True

    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = []


class SHAPExplainer:
    """
    SHAP Explainer per modelli di forecasting

    Features:
    - SHAP values per singole predizioni
    - Feature importance globale e locale
    - Spiegazioni visuali e testuali
    - Cache per performance
    - Supporto modelli multipli
    """

    def __init__(self, config: SHAPConfig = None):
        if not _has_shap:
            raise ImportError("SHAP non disponibile. Installare: pip install shap")

        self.config = config or SHAPConfig()
        self.explainer = None
        self.model = None
        self.feature_names = []
        self.background_data = None
        self.explanation_cache = {} if self.config.cache_explanations else None

        # Statistiche
        self.explanations_generated = 0
        self.cache_hits = 0

    def fit(
        self, model: Any, background_data: np.ndarray, feature_names: Optional[List[str]] = None
    ):
        """
        Addestra SHAP explainer su modello e dati background

        Args:
            model: Modello da spiegare
            background_data: Dati per background distribution
            feature_names: Nomi delle features
        """
        try:
            self.model = model
            self.background_data = background_data
            self.feature_names = (
                feature_names
                or self.config.feature_names
                or [f"feature_{i}" for i in range(background_data.shape[1])]
            )

            # Seleziona tipo explainer
            explainer_type = self._determine_explainer_type(model)
            logger.info(f"Inizializzazione SHAP explainer tipo: {explainer_type}")

            if explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(
                    model=self._model_predict_wrapper,
                    data=background_data[: self.config.background_samples],
                    feature_names=self.feature_names,
                )

            elif explainer_type == "linear":
                self.explainer = shap.LinearExplainer(
                    model=model, data=background_data, feature_names=self.feature_names
                )

            elif explainer_type == "tree":
                self.explainer = shap.TreeExplainer(model=model, feature_names=self.feature_names)

            else:
                # Fallback a KernelExplainer
                self.explainer = shap.KernelExplainer(
                    model=self._model_predict_wrapper,
                    data=background_data[: self.config.background_samples],
                    feature_names=self.feature_names,
                )

            logger.info(f"SHAP explainer addestrato con {len(self.feature_names)} features")

        except Exception as e:
            logger.error(f"Errore training SHAP explainer: {e}")
            raise

    def _determine_explainer_type(self, model: Any) -> str:
        """Determina tipo explainer ottimale per modello"""
        if self.config.explainer_type != "auto":
            return self.config.explainer_type

        model_type = str(type(model)).lower()

        if any(
            tree_model in model_type for tree_model in ["random", "forest", "tree", "xgb", "lgb"]
        ):
            return "tree"
        elif any(linear_model in model_type for linear_model in ["linear", "ridge", "lasso"]):
            return "linear"
        else:
            return "kernel"

    def _model_predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper per compatibilità modello con SHAP"""
        try:
            if hasattr(self.model, "predict"):
                predictions = self.model.predict(X)
            elif hasattr(self.model, "forecast"):
                # Per modelli ARIMA custom
                predictions = []
                for row in X:
                    pred = self.model.forecast(
                        steps=1, exog_variables=row if len(row) > 0 else None
                    )
                    predictions.append(pred["values"][0] if "values" in pred else pred)
                predictions = np.array(predictions)
            else:
                raise ValueError("Modello non supporta predict() o forecast()")

            # Assicura che output sia array 1D
            if predictions.ndim > 1:
                predictions = predictions.flatten()

            return predictions

        except Exception as e:
            logger.error(f"Errore wrapper predict: {e}")
            # Fallback: restituisce array di zeri
            return np.zeros(X.shape[0])

    def explain_instance(
        self, instance: np.ndarray, return_dict: bool = True, cache_key: Optional[str] = None
    ) -> Union[Dict[str, Any], shap.Explanation]:
        """
        Spiega singola istanza

        Args:
            instance: Dati input per spiegazione (1D array)
            return_dict: Se restituire dict invece di Explanation object
            cache_key: Chiave per cache (opzionale)

        Returns:
            Dict con spiegazione o oggetto SHAP Explanation
        """
        try:
            # Controlla cache se abilitata
            if self.explanation_cache and cache_key:
                if cache_key in self.explanation_cache:
                    self.cache_hits += 1
                    logger.debug(f"SHAP explanation da cache: {cache_key}")
                    return self.explanation_cache[cache_key]

            if self.explainer is None:
                raise ValueError("SHAP explainer non addestrato. Chiamare fit() prima.")

            # Assicura dimensioni corrette
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)

            # Genera SHAP values
            shap_values = self.explainer.shap_values(X=instance, nsamples=self.config.max_evals)

            # Se output è 3D, prendi primo elemento
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]

            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Prima istanza

            # Calcola baseline
            if hasattr(self.explainer, "expected_value"):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = float(expected_value[0])
            else:
                expected_value = 0.0

            # Predizione per istanza
            prediction = self._model_predict_wrapper(instance)[0]

            self.explanations_generated += 1

            if not return_dict:
                # Restituisce oggetto SHAP nativo
                return shap.Explanation(
                    values=shap_values,
                    base_values=expected_value,
                    data=instance[0],
                    feature_names=self.feature_names,
                )

            # Crea spiegazione strutturata
            explanation = self._create_structured_explanation(
                shap_values=shap_values,
                feature_values=instance[0],
                baseline=expected_value,
                prediction=prediction,
            )

            # Salva in cache se abilitata
            if self.explanation_cache and cache_key:
                self.explanation_cache[cache_key] = explanation

            logger.debug(f"SHAP explanation generata per predizione: {prediction:.3f}")
            return explanation

        except Exception as e:
            logger.error(f"Errore generazione SHAP explanation: {e}")
            return self._create_fallback_explanation(instance)

    def _create_structured_explanation(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        baseline: float,
        prediction: float,
    ) -> Dict[str, Any]:
        """Crea spiegazione strutturata"""

        # Feature contributions
        feature_contributions = {}
        for i, (name, shap_val, feature_val) in enumerate(
            zip(self.feature_names, shap_values, feature_values)
        ):
            feature_contributions[name] = {
                "shap_value": float(shap_val),
                "feature_value": float(feature_val),
                "contribution_percentage": abs(shap_val) / (abs(shap_values).sum() + 1e-10) * 100,
            }

        # Ranking features per importanza
        feature_ranking = sorted(
            [
                {
                    "feature": name,
                    "importance": abs(contrib["shap_value"]),
                    "direction": "positive" if contrib["shap_value"] > 0 else "negative",
                    "shap_value": contrib["shap_value"],
                    "feature_value": contrib["feature_value"],
                }
                for name, contrib in feature_contributions.items()
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

        # Confidence factors
        confidence_factors = self._calculate_confidence_factors(shap_values, prediction, baseline)

        # Summary testuale
        top_features = feature_ranking[:3]
        summary_text = self._generate_text_summary(top_features, prediction, baseline)

        explanation = {
            "prediction": float(prediction),
            "baseline": float(baseline),
            "shap_values": {name: float(val) for name, val in zip(self.feature_names, shap_values)},
            "feature_contributions": feature_contributions,
            "feature_ranking": feature_ranking,
            "confidence_factors": confidence_factors,
            "summary": summary_text,
            "metadata": {
                "explainer_type": self.config.explainer_type,
                "num_features": len(self.feature_names),
                "timestamp": datetime.now().isoformat(),
            },
        }

        return explanation

    def _calculate_confidence_factors(
        self, shap_values: np.ndarray, prediction: float, baseline: float
    ) -> Dict[str, float]:
        """Calcola fattori di confidenza per spiegazione"""

        # Stabilità: quanto sono distribuite le SHAP values
        shap_std = float(np.std(shap_values))
        shap_mean = float(np.mean(np.abs(shap_values)))
        stability = 1.0 / (1.0 + shap_std / (shap_mean + 1e-10))

        # Consistenza: predizione vs somma SHAP + baseline
        expected_prediction = baseline + np.sum(shap_values)
        consistency = 1.0 - min(
            abs(prediction - expected_prediction) / (abs(prediction) + 1e-10), 1.0
        )

        # Coverage: quante features contribuiscono significativamente
        significant_features = np.sum(np.abs(shap_values) > np.mean(np.abs(shap_values)))
        coverage = significant_features / len(shap_values)

        # Score globale
        overall_confidence = stability * 0.4 + consistency * 0.4 + coverage * 0.2

        return {
            "stability": float(np.clip(stability, 0, 1)),
            "consistency": float(np.clip(consistency, 0, 1)),
            "coverage": float(np.clip(coverage, 0, 1)),
            "overall": float(np.clip(overall_confidence, 0, 1)),
        }

    def _generate_text_summary(
        self, top_features: List[Dict], prediction: float, baseline: float
    ) -> str:
        """Genera riassunto testuale della spiegazione"""

        if not top_features:
            return f"Predizione: {prediction:.2f} (baseline: {baseline:.2f}). Nessuna feature significativa."

        deviation = prediction - baseline
        direction = "superiore" if deviation > 0 else "inferiore"

        summary = f"Predizione: {prediction:.2f} ({direction} alla baseline {baseline:.2f}).\n\n"

        summary += "Principali fattori:\n"
        for i, feature in enumerate(top_features[:3], 1):
            impact = "aumenta" if feature["shap_value"] > 0 else "diminuisce"
            summary += f"{i}. {feature['feature']} (valore: {feature['feature_value']:.2f}) "
            summary += f"{impact} la predizione di {abs(feature['shap_value']):.2f}\n"

        return summary

    def _create_fallback_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """Crea spiegazione fallback in caso di errore"""
        return {
            "prediction": 0.0,
            "baseline": 0.0,
            "shap_values": {name: 0.0 for name in self.feature_names},
            "feature_contributions": {},
            "feature_ranking": [],
            "confidence_factors": {
                "overall": 0.0,
                "stability": 0.0,
                "consistency": 0.0,
                "coverage": 0.0,
            },
            "summary": "Spiegazione non disponibile a causa di errore interno.",
            "error": True,
            "metadata": {
                "explainer_type": "fallback",
                "num_features": len(self.feature_names),
                "timestamp": datetime.now().isoformat(),
            },
        }

    def explain_batch(self, instances: np.ndarray, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Spiega batch di istanze

        Args:
            instances: Batch di istanze (2D array)
            batch_size: Dimensione batch per processamento

        Returns:
            Lista di spiegazioni
        """
        explanations = []

        # Processa in mini-batch per memoria
        for i in range(0, len(instances), batch_size):
            batch = instances[i : i + batch_size]

            for j, instance in enumerate(batch):
                try:
                    cache_key = f"batch_{i + j}_{hash(instance.tobytes())}"
                    explanation = self.explain_instance(instance=instance, cache_key=cache_key)
                    explanations.append(explanation)

                except Exception as e:
                    logger.error(f"Errore spiegazione istanza {i + j}: {e}")
                    explanations.append(self._create_fallback_explanation(instance))

        logger.info(f"Batch explanation completato: {len(explanations)} istanze")
        return explanations

    def get_global_feature_importance(self) -> Dict[str, float]:
        """Ottiene importanza globale features dal background data"""
        try:
            if self.explainer is None or self.background_data is None:
                return {}

            # Calcola SHAP values su sample del background
            sample_size = min(50, len(self.background_data))
            sample_indices = np.random.choice(len(self.background_data), sample_size, replace=False)
            sample_data = self.background_data[sample_indices]

            shap_values = self.explainer.shap_values(sample_data)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Media delle importanze assolute
            global_importance = np.mean(np.abs(shap_values), axis=0)

            return {
                name: float(importance)
                for name, importance in zip(self.feature_names, global_importance)
            }

        except Exception as e:
            logger.error(f"Errore calcolo importanza globale: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Statistiche explainer"""
        return {
            "explainer_fitted": self.explainer is not None,
            "explainer_type": self.config.explainer_type,
            "num_features": len(self.feature_names),
            "background_samples": len(self.background_data)
            if self.background_data is not None
            else 0,
            "explanations_generated": self.explanations_generated,
            "cache_enabled": self.explanation_cache is not None,
            "cache_size": len(self.explanation_cache) if self.explanation_cache else 0,
            "cache_hit_rate": (self.cache_hits / max(self.explanations_generated, 1)) * 100,
        }

    def clear_cache(self):
        """Pulisce cache spiegazioni"""
        if self.explanation_cache:
            self.explanation_cache.clear()
            logger.info("Cache SHAP explanations pulita")


# Utility functions
def create_forecast_features(
    series_data: pd.Series,
    lags: List[int] = [1, 2, 3, 7],
    include_time_features: bool = True,
    include_stats: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Crea features per spiegazione forecast ARIMA

    Args:
        series_data: Serie temporale
        lags: Lista di lag da includere
        include_time_features: Include features temporali
        include_stats: Include statistiche rolling

    Returns:
        (features_array, feature_names)
    """
    features = []
    feature_names = []

    # Lag features
    for lag in lags:
        if len(series_data) > lag:
            lag_values = series_data.shift(lag).fillna(method="bfill")
            features.append(lag_values.values)
            feature_names.append(f"lag_{lag}")

    # Time features se abilitato
    if include_time_features and hasattr(series_data.index, "day_of_week"):
        features.append(series_data.index.day_of_week.values)
        feature_names.append("day_of_week")

        features.append(series_data.index.month.values)
        feature_names.append("month")

        features.append(series_data.index.dayofyear.values)
        feature_names.append("day_of_year")

    # Statistiche rolling se abilitate
    if include_stats:
        rolling_mean = series_data.rolling(window=7, min_periods=1).mean()
        features.append(rolling_mean.values)
        feature_names.append("rolling_mean_7d")

        rolling_std = series_data.rolling(window=7, min_periods=1).std().fillna(0)
        features.append(rolling_std.values)
        feature_names.append("rolling_std_7d")

    # Combina features
    if features:
        features_array = np.column_stack(features)
    else:
        features_array = np.zeros((len(series_data), 1))
        feature_names = ["dummy_feature"]

    return features_array, feature_names


def explain_arima_forecast(
    model: Any,
    series_data: pd.Series,
    forecast_instance: Optional[np.ndarray] = None,
    config: SHAPConfig = None,
) -> Dict[str, Any]:
    """
    Utility per spiegare forecast ARIMA con SHAP

    Args:
        model: Modello ARIMA addestrato
        series_data: Serie temporale usata per training
        forecast_instance: Istanza specifica da spiegare (opzionale)
        config: Configurazione SHAP

    Returns:
        Spiegazione SHAP
    """
    # Crea features
    features, feature_names = create_forecast_features(series_data)

    # Istanza da spiegare (ultima osservazione se non specificata)
    if forecast_instance is None:
        forecast_instance = features[-1]

    # Crea e addestra explainer
    explainer = SHAPExplainer(config or SHAPConfig())
    explainer.fit(model, features, feature_names)

    # Genera spiegazione
    explanation = explainer.explain_instance(instance=forecast_instance, cache_key="arima_forecast")

    return explanation


if __name__ == "__main__":
    # Test SHAP explainer
    print("Test SHAP Explainer per ARIMA Forecasting")

    # Dati demo
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.random.normal(1000, 100, 100) + np.sin(np.arange(100) * 2 * np.pi / 7) * 50
    series = pd.Series(values, index=dates)

    print(f"Serie temporale creata: {len(series)} punti")

    # Crea features
    features, feature_names = create_forecast_features(series)
    print(f"Features create: {feature_names}")

    # Test senza modello reale (demo)
    class MockARIMAModel:
        def predict(self, X):
            return np.random.normal(1000, 50, len(X))

    model = MockARIMAModel()

    try:
        explainer = SHAPExplainer()
        explainer.fit(model, features, feature_names)

        explanation = explainer.explain_instance(features[-1])
        print(f"Spiegazione generata: {explanation['summary']}")
        print(f"Top 3 features: {[f['feature'] for f in explanation['feature_ranking'][:3]]}")

    except Exception as e:
        print(f"Errore test: {e}")
        print("Nota: Per test completo installare SHAP: pip install shap")
