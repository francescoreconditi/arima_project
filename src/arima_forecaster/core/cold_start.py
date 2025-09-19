"""
Modulo per gestire il Cold Start Problem nel forecasting di serie temporali.

Fornisce funzioni generiche per il transfer learning e forecasting di nuovi prodotti
senza dati storici, utilizzando pattern e caratteristiche di prodotti simili esistenti.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

from .arima_model import ARIMAForecaster
from .sarima_model import SARIMAForecaster
from .prophet_model import ProphetForecaster
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProductSimilarity:
    """Struttura per memorizzare informazioni di similarità tra prodotti"""

    source_product: str
    target_product: str
    similarity_score: float
    similarity_method: str
    features_used: List[str]
    confidence_level: float


@dataclass
class TransferPattern:
    """Pattern di trasferimento estratto da un prodotto sorgente"""

    source_product: str
    seasonal_pattern: Optional[Dict[str, float]]
    trend_parameters: Dict[str, float]
    volatility_profile: Dict[str, float]
    demand_characteristics: Dict[str, Any]
    model_parameters: Dict[str, Any]


class ColdStartForecaster:
    """
    Classe principale per gestire il Cold Start Problem nel forecasting.

    Implementa diversi metodi di transfer learning:
    1. Pattern Transfer: Trasferimento diretto di pattern stagionali e trend
    2. Analogical Forecasting: Scaling basato su caratteristiche prodotto
    3. Multi-Product VAR: Modellazione delle relazioni tra prodotti
    4. Hybrid Transfer: Combinazione di più approcci
    """

    def __init__(self, similarity_threshold: float = 0.7, min_history_days: int = 30):
        """
        Inizializza il Cold Start Forecaster.

        Args:
            similarity_threshold: Soglia minima per considerare due prodotti simili
            min_history_days: Giorni minimi di storia richiesti per prodotto sorgente
        """
        self.similarity_threshold = similarity_threshold
        self.min_history_days = min_history_days
        self.product_patterns = {}
        self.similarity_matrix = {}
        self.scaler = StandardScaler()

    def extract_product_features(
        self, product_data: pd.Series, product_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estrae features caratteristiche da un prodotto per il matching.

        Args:
            product_data: Serie temporale vendite del prodotto
            product_info: Informazioni aggiuntive (prezzo, categoria, etc.)

        Returns:
            Dizionario con features estratte
        """
        try:
            features = {}

            # Features dalla serie temporale
            if len(product_data) > 7:
                features["mean_demand"] = product_data.mean()
                features["std_demand"] = product_data.std()
                features["cv_demand"] = (
                    features["std_demand"] / features["mean_demand"]
                    if features["mean_demand"] > 0
                    else 0
                )
                features["max_demand"] = product_data.max()
                features["trend_slope"] = self._calculate_trend_slope(product_data)

                # Stagionalità settimanale
                if len(product_data) >= 14:
                    weekly_pattern = self._detect_weekly_seasonality(product_data)
                    features["weekly_seasonality"] = weekly_pattern

                # Stagionalità mensile (se abbastanza dati)
                if len(product_data) >= 60:
                    monthly_pattern = self._detect_monthly_seasonality(product_data)
                    features["monthly_seasonality"] = monthly_pattern

            # Features da informazioni prodotto
            # Supporta sia 'prezzo' che 'prezzo_medio' per compatibilità
            if "prezzo" in product_info:
                features["price"] = float(product_info["prezzo"])
            elif "prezzo_medio" in product_info:
                features["price"] = float(product_info["prezzo_medio"])

            if "categoria" in product_info:
                features["category_encoded"] = hash(product_info["categoria"]) % 1000
            if "peso" in product_info:
                features["weight"] = float(product_info.get("peso", 0))
            if "volume" in product_info:
                features["volume"] = float(product_info.get("volume", 0))

            return features

        except Exception as e:
            logger.error(f"Errore estrazione features: {e}")
            return {}

    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calcola la pendenza del trend usando regressione lineare semplice"""
        try:
            if len(series) < 3:
                return 0.0

            x = np.arange(len(series))
            y = series.values

            # Rimuovi NaN
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                return 0.0

            x_clean = x[mask]
            y_clean = y[mask]

            # Regressione lineare manuale
            n = len(x_clean)
            slope = (n * np.sum(x_clean * y_clean) - np.sum(x_clean) * np.sum(y_clean)) / (
                n * np.sum(x_clean**2) - np.sum(x_clean) ** 2
            )

            return slope

        except Exception as e:
            logger.warning(f"Errore calcolo trend slope: {e}")
            return 0.0

    def _detect_weekly_seasonality(self, series: pd.Series) -> float:
        """Rileva pattern settimanale e restituisce intensità"""
        try:
            if len(series) < 14:
                return 0.0

            # Raggruppa per giorno della settimana
            if hasattr(series.index, "dayofweek"):
                weekly_groups = series.groupby(series.index.dayofweek).mean()
            else:
                # Se non c'è indice datetime, simula pattern settimanale
                days = np.arange(len(series)) % 7
                weekly_groups = series.groupby(days).mean()

            # Calcola coefficiente di variazione settimanale
            weekly_cv = weekly_groups.std() / weekly_groups.mean()
            return min(weekly_cv, 2.0)  # Cap a 2.0

        except Exception as e:
            logger.warning(f"Errore rilevamento stagionalità settimanale: {e}")
            return 0.0

    def _detect_monthly_seasonality(self, series: pd.Series) -> float:
        """Rileva pattern mensile e restituisce intensità"""
        try:
            if len(series) < 60:
                return 0.0

            # Raggruppa per mese
            if hasattr(series.index, "month"):
                monthly_groups = series.groupby(series.index.month).mean()
            else:
                # Se non c'è indice datetime, simula pattern mensile
                months = (np.arange(len(series)) // 30) % 12
                monthly_groups = series.groupby(months).mean()

            # Calcola coefficiente di variazione mensile
            monthly_cv = monthly_groups.std() / monthly_groups.mean()
            return min(monthly_cv, 1.5)  # Cap a 1.5

        except Exception as e:
            logger.warning(f"Errore rilevamento stagionalità mensile: {e}")
            return 0.0

    def calculate_product_similarity(
        self,
        source_features: Dict[str, float],
        target_features: Dict[str, float],
        method: str = "cosine",
    ) -> float:
        """
        Calcola similarità tra due prodotti basata sulle features.

        Args:
            source_features: Features del prodotto sorgente
            target_features: Features del prodotto target
            method: Metodo di calcolo ('cosine', 'correlation', 'euclidean')

        Returns:
            Score di similarità (0-1, dove 1 = identici)
        """
        try:
            # Trova features comuni
            common_features = set(source_features.keys()) & set(target_features.keys())

            if len(common_features) < 2:
                logger.warning("Meno di 2 features comuni per calcolo similarità")
                return 0.0

            # Estrai vettori features
            source_vector = np.array([source_features[f] for f in common_features])
            target_vector = np.array([target_features[f] for f in common_features])

            # Gestisci NaN e infiniti
            if np.any(np.isnan(source_vector)) or np.any(np.isnan(target_vector)):
                return 0.0
            if np.any(np.isinf(source_vector)) or np.any(np.isinf(target_vector)):
                return 0.0

            if method == "cosine":
                # Similarità coseno
                if np.linalg.norm(source_vector) == 0 or np.linalg.norm(target_vector) == 0:
                    return 0.0
                similarity = cosine_similarity([source_vector], [target_vector])[0][0]
                return max(0.0, similarity)  # Assicura non negativo

            elif method == "correlation":
                # Correlazione di Pearson
                if len(source_vector) < 3:
                    return 0.0
                correlation, _ = pearsonr(source_vector, target_vector)
                return max(0.0, correlation)  # Solo correlazioni positive

            elif method == "euclidean":
                # Distanza euclidea normalizzata
                # Normalizza i vettori
                source_norm = (source_vector - source_vector.mean()) / (source_vector.std() + 1e-8)
                target_norm = (target_vector - target_vector.mean()) / (target_vector.std() + 1e-8)

                # Calcola distanza e converti in similarità
                distance = np.linalg.norm(source_norm - target_norm)
                similarity = 1.0 / (1.0 + distance)
                return similarity

            else:
                logger.error(f"Metodo similarità non supportato: {method}")
                return 0.0

        except Exception as e:
            logger.error(f"Errore calcolo similarità: {e}")
            return 0.0

    def find_similar_products(
        self,
        target_product_info: Dict[str, Any],
        products_database: Dict[str, Dict],
        top_n: int = 3,
    ) -> List[ProductSimilarity]:
        """
        Trova i prodotti più simili al prodotto target.

        Args:
            target_product_info: Info del prodotto per cui fare forecasting
            products_database: Database con info e dati di tutti i prodotti
            top_n: Numero di prodotti simili da restituire

        Returns:
            Lista di ProductSimilarity ordinata per score
        """
        try:
            similarities = []

            # Estrai features del prodotto target
            target_features = target_product_info.get("features", {})
            if not target_features:
                logger.warning("Nessuna feature disponibile per prodotto target")
                return []

            # Confronta con tutti i prodotti nel database
            for source_product, source_data in products_database.items():
                # Verifica che ci siano abbastanza dati storici
                if "vendite" not in source_data:
                    continue

                vendite_series = source_data["vendite"]
                if len(vendite_series) < self.min_history_days:
                    continue

                # Estrai features del prodotto sorgente
                source_features = source_data.get("features", {})
                if not source_features:
                    continue

                # Calcola similarità
                similarity_score = self.calculate_product_similarity(
                    source_features, target_features, method="cosine"
                )

                if similarity_score >= self.similarity_threshold:
                    similarity = ProductSimilarity(
                        source_product=source_product,
                        target_product=target_product_info.get("codice", "unknown"),
                        similarity_score=similarity_score,
                        similarity_method="cosine",
                        features_used=list(
                            set(source_features.keys()) & set(target_features.keys())
                        ),
                        confidence_level=min(similarity_score, len(vendite_series) / 365.0),
                    )
                    similarities.append(similarity)

            # Ordina per score e restituisci top_n
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            return similarities[:top_n]

        except Exception as e:
            logger.error(f"Errore ricerca prodotti simili: {e}")
            return []

    def extract_transfer_pattern(
        self, source_product: str, source_data: pd.Series, source_info: Dict[str, Any]
    ) -> TransferPattern:
        """
        Estrae pattern trasferibili da un prodotto sorgente.

        Args:
            source_product: Codice prodotto sorgente
            source_data: Serie temporale vendite
            source_info: Informazioni aggiuntive prodotto

        Returns:
            TransferPattern con tutti i pattern estratti
        """
        try:
            # Pattern stagionali
            seasonal_pattern = {}
            if len(source_data) >= 14:
                # Pattern settimanale
                if hasattr(source_data.index, "dayofweek"):
                    weekly_avg = source_data.groupby(source_data.index.dayofweek).mean()
                    seasonal_pattern["weekly"] = weekly_avg.to_dict()

                # Pattern mensile se abbastanza dati
                if len(source_data) >= 60 and hasattr(source_data.index, "month"):
                    monthly_avg = source_data.groupby(source_data.index.month).mean()
                    seasonal_pattern["monthly"] = monthly_avg.to_dict()

            # Parametri di trend
            trend_params = {
                "slope": self._calculate_trend_slope(source_data),
                "intercept": source_data.iloc[0] if len(source_data) > 0 else 0,
                "r_squared": self._calculate_trend_rsquared(source_data),
            }

            # Profilo di volatilità
            volatility_profile = {
                "overall_std": source_data.std(),
                "rolling_volatility_30": source_data.rolling(30).std().mean()
                if len(source_data) >= 30
                else source_data.std(),
                "max_volatility": source_data.rolling(7).std().max()
                if len(source_data) >= 7
                else source_data.std(),
            }

            # Caratteristiche domanda
            demand_chars = {
                "mean_demand": source_data.mean(),
                "median_demand": source_data.median(),
                "demand_skewness": source_data.skew(),
                "demand_kurtosis": source_data.kurtosis(),
                "zero_demand_ratio": (source_data == 0).mean(),
            }

            # Parametri modello (prova a fittare ARIMA semplice)
            model_params = {}
            try:
                arima_model = ARIMAForecaster(order=(1, 1, 1))
                arima_model.fit(source_data)
                model_params["arima_order"] = (1, 1, 1)
                model_params["aic"] = (
                    arima_model.model_fit_.aic if hasattr(arima_model, "model_fit_") else None
                )
            except:
                model_params["arima_order"] = (1, 0, 0)  # Fallback semplice

            return TransferPattern(
                source_product=source_product,
                seasonal_pattern=seasonal_pattern if seasonal_pattern else None,
                trend_parameters=trend_params,
                volatility_profile=volatility_profile,
                demand_characteristics=demand_chars,
                model_parameters=model_params,
            )

        except Exception as e:
            logger.error(f"Errore estrazione pattern da {source_product}: {e}")
            # Restituisci pattern minimale
            return TransferPattern(
                source_product=source_product,
                seasonal_pattern=None,
                trend_parameters={"slope": 0, "intercept": 1, "r_squared": 0},
                volatility_profile={"overall_std": 1},
                demand_characteristics={"mean_demand": 1},
                model_parameters={"arima_order": (1, 0, 0)},
            )

    def _calculate_trend_rsquared(self, series: pd.Series) -> float:
        """Calcola R-squared del trend lineare"""
        try:
            if len(series) < 3:
                return 0.0

            x = np.arange(len(series))
            y = series.values

            # Rimuovi NaN
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                return 0.0

            x_clean = x[mask]
            y_clean = y[mask]

            # Calcola R-squared
            correlation_matrix = np.corrcoef(x_clean, y_clean)
            if correlation_matrix.shape[0] < 2:
                return 0.0

            correlation = correlation_matrix[0, 1]
            return correlation**2 if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Errore calcolo R-squared: {e}")
            return 0.0

    def pattern_transfer_forecast(
        self,
        transfer_pattern: TransferPattern,
        target_product_info: Dict[str, Any],
        forecast_days: int = 30,
        scaling_factor: float = 1.0,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Genera forecast usando Pattern Transfer.

        Args:
            transfer_pattern: Pattern estratto da prodotto simile
            target_product_info: Informazioni prodotto target
            forecast_days: Giorni da prevedere
            scaling_factor: Fattore di scala per adattare alle caratteristiche target

        Returns:
            Tupla (forecast_series, metadata)
        """
        try:
            # Base forecast dal pattern di domanda
            base_demand = transfer_pattern.demand_characteristics["mean_demand"] * scaling_factor

            # Genera serie base con trend
            trend_slope = transfer_pattern.trend_parameters["slope"] * scaling_factor
            trend_component = np.arange(forecast_days) * trend_slope

            # Serie base
            base_series = np.full(forecast_days, base_demand) + trend_component

            # Applica stagionalità se disponibile
            if transfer_pattern.seasonal_pattern:
                # Stagionalità settimanale
                if "weekly" in transfer_pattern.seasonal_pattern:
                    weekly_pattern = transfer_pattern.seasonal_pattern["weekly"]
                    for i in range(forecast_days):
                        day_of_week = i % 7
                        if day_of_week in weekly_pattern:
                            weekly_factor = (
                                weekly_pattern[day_of_week] / base_demand if base_demand > 0 else 1
                            )
                            base_series[i] *= weekly_factor

                # Stagionalità mensile
                if "monthly" in transfer_pattern.seasonal_pattern:
                    monthly_pattern = transfer_pattern.seasonal_pattern["monthly"]
                    for i in range(forecast_days):
                        month = ((i // 30) % 12) + 1  # Approssimazione mese
                        if month in monthly_pattern:
                            monthly_factor = (
                                monthly_pattern[month] / base_demand if base_demand > 0 else 1
                            )
                            base_series[i] *= monthly_factor

            # Aggiungi noise basato sulla volatilità
            volatility = transfer_pattern.volatility_profile["overall_std"] * scaling_factor
            noise = np.random.normal(0, volatility * 0.1, forecast_days)  # Noise ridotto
            base_series += noise

            # Assicura valori non negativi
            base_series = np.maximum(base_series, 0)

            # Crea serie pandas con indice temporale
            dates = pd.date_range(start=pd.Timestamp.now().date(), periods=forecast_days, freq="D")
            forecast_series = pd.Series(base_series, index=dates)

            # Metadata
            metadata = {
                "method": "pattern_transfer",
                "source_product": transfer_pattern.source_product,
                "scaling_factor": scaling_factor,
                "base_demand": base_demand,
                "trend_slope": trend_slope,
                "volatility": volatility,
                "seasonality_applied": transfer_pattern.seasonal_pattern is not None,
                "confidence": "medium",
            }

            return forecast_series, metadata

        except Exception as e:
            logger.error(f"Errore Pattern Transfer forecast: {e}")
            # Forecast fallback semplice
            fallback_series = pd.Series(
                np.full(forecast_days, 1.0 * scaling_factor),
                index=pd.date_range(
                    start=pd.Timestamp.now().date(), periods=forecast_days, freq="D"
                ),
            )
            return fallback_series, {"method": "fallback", "error": str(e)}

    def analogical_forecast(
        self,
        similar_products: List[ProductSimilarity],
        products_database: Dict[str, Dict],
        target_product_info: Dict[str, Any],
        forecast_days: int = 30,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Genera forecast usando Analogical Forecasting con scaling intelligente.

        Args:
            similar_products: Lista prodotti simili
            products_database: Database completo prodotti
            target_product_info: Info prodotto target
            forecast_days: Giorni da prevedere

        Returns:
            Tupla (forecast_series, metadata)
        """
        try:
            if not similar_products:
                raise ValueError("Nessun prodotto simile disponibile")

            # Calcola scaling factors basati sulle caratteristiche
            scaling_factors = []
            forecasts = []
            weights = []

            for similarity in similar_products:
                source_product = similarity.source_product
                source_data = products_database[source_product]

                # Calcola scaling factor
                scaling_factor = self._calculate_analogical_scaling(
                    source_data["info"], target_product_info
                )

                # Genera forecast dal prodotto simile
                source_series = source_data["vendite"]

                # Usa ultimi N giorni come base per il pattern
                recent_days = min(30, len(source_series))
                recent_pattern = source_series[-recent_days:].values

                # Proietta il pattern
                if len(recent_pattern) > 0:
                    # Estendi pattern ripetendo e applicando trend
                    trend_slope = self._calculate_trend_slope(
                        source_series[-60:] if len(source_series) >= 60 else source_series
                    )

                    forecast_values = []
                    for i in range(forecast_days):
                        pattern_index = i % len(recent_pattern)
                        base_value = recent_pattern[pattern_index]
                        trend_adjustment = trend_slope * i
                        scaled_value = (base_value + trend_adjustment) * scaling_factor
                        forecast_values.append(max(scaled_value, 0))

                    forecasts.append(forecast_values)
                    scaling_factors.append(scaling_factor)
                    weights.append(similarity.similarity_score)

            if not forecasts:
                raise ValueError("Nessun forecast generato dai prodotti simili")

            # Combina forecasts con pesi
            forecasts_array = np.array(forecasts)
            weights_array = np.array(weights)
            weights_normalized = weights_array / weights_array.sum()

            # Media pesata
            final_forecast = np.average(forecasts_array, axis=0, weights=weights_normalized)

            # Crea serie pandas
            dates = pd.date_range(start=pd.Timestamp.now().date(), periods=forecast_days, freq="D")
            forecast_series = pd.Series(final_forecast, index=dates)

            # Metadata
            metadata = {
                "method": "analogical_forecasting",
                "source_products": [s.source_product for s in similar_products],
                "scaling_factors": scaling_factors,
                "similarity_scores": [s.similarity_score for s in similar_products],
                "weights_used": weights_normalized.tolist(),
                "confidence": "high" if len(similar_products) >= 2 else "medium",
            }

            return forecast_series, metadata

        except Exception as e:
            logger.error(f"Errore Analogical forecast: {e}")
            # Forecast fallback
            fallback_series = pd.Series(
                np.full(forecast_days, 1.0),
                index=pd.date_range(
                    start=pd.Timestamp.now().date(), periods=forecast_days, freq="D"
                ),
            )
            return fallback_series, {"method": "fallback", "error": str(e)}

    def _calculate_analogical_scaling(
        self, source_info: Dict[str, Any], target_info: Dict[str, Any]
    ) -> float:
        """
        Calcola fattore di scala per analogical forecasting basato sulle caratteristiche.

        Args:
            source_info: Caratteristiche prodotto sorgente
            target_info: Caratteristiche prodotto target

        Returns:
            Fattore di scala da applicare al forecast
        """
        try:
            scaling_factor = 1.0

            # Scaling basato su prezzo (prodotti più costosi = domanda minore)
            if "prezzo" in source_info and "prezzo" in target_info:
                price_ratio = target_info["prezzo"] / source_info["prezzo"]
                # Elasticità price semplificata (inversa)
                scaling_factor *= (1.0 / price_ratio) ** 0.3  # Elasticità 0.3

            # Scaling basato su caratteristiche fisiche
            if "peso" in source_info and "peso" in target_info:
                weight_ratio = target_info["peso"] / source_info["peso"]
                scaling_factor *= weight_ratio**0.1  # Peso ha impatto minore

            # Scaling basato su categoria (stesso codice categoria = scaling 1.0)
            if "categoria" in source_info and "categoria" in target_info:
                if source_info["categoria"] != target_info["categoria"]:
                    scaling_factor *= 0.8  # Penalità per categoria diversa

            # Limita scaling factor a range ragionevole
            scaling_factor = max(0.1, min(10.0, scaling_factor))

            return scaling_factor

        except Exception as e:
            logger.warning(f"Errore calcolo scaling analogico: {e}")
            return 1.0

    def cold_start_forecast(
        self,
        target_product_info: Dict[str, Any],
        products_database: Dict[str, Dict],
        forecast_days: int = 30,
        method: str = "hybrid",
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Metodo principale per Cold Start Forecasting.

        Args:
            target_product_info: Informazioni del nuovo prodotto
            products_database: Database con tutti i prodotti storici
            forecast_days: Giorni da prevedere
            method: Metodo da usare ('pattern', 'analogical', 'hybrid')

        Returns:
            Tupla (forecast_series, metadata_completo)
        """
        try:
            logger.info(
                f"Avvio Cold Start forecast per prodotto: {target_product_info.get('codice', 'unknown')}"
            )

            # Step 1: Trova prodotti simili
            similar_products = self.find_similar_products(
                target_product_info, products_database, top_n=3
            )

            if not similar_products:
                logger.warning("Nessun prodotto simile trovato, uso forecast base")
                return self._base_forecast(target_product_info, forecast_days)

            logger.info(f"Trovati {len(similar_products)} prodotti simili")

            # Step 2: Genera forecast secondo il metodo scelto
            if method == "pattern":
                # Usa solo il prodotto più simile per pattern transfer
                best_match = similar_products[0]
                source_data = products_database[best_match.source_product]["vendite"]
                source_info = products_database[best_match.source_product]["info"]

                transfer_pattern = self.extract_transfer_pattern(
                    best_match.source_product, source_data, source_info
                )

                scaling_factor = self._calculate_analogical_scaling(
                    source_info, target_product_info
                )
                return self.pattern_transfer_forecast(
                    transfer_pattern, target_product_info, forecast_days, scaling_factor
                )

            elif method == "analogical":
                return self.analogical_forecast(
                    similar_products, products_database, target_product_info, forecast_days
                )

            elif method == "hybrid":
                # Combina pattern transfer e analogical
                return self._hybrid_forecast(
                    similar_products, products_database, target_product_info, forecast_days
                )

            else:
                raise ValueError(f"Metodo non supportato: {method}")

        except Exception as e:
            logger.error(f"Errore Cold Start forecast: {e}")
            return self._base_forecast(target_product_info, forecast_days)

    def _hybrid_forecast(
        self,
        similar_products: List[ProductSimilarity],
        products_database: Dict[str, Dict],
        target_product_info: Dict[str, Any],
        forecast_days: int,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Genera forecast ibrido combinando pattern transfer e analogical"""
        try:
            # Forecast con pattern transfer
            best_match = similar_products[0]
            source_data = products_database[best_match.source_product]["vendite"]
            source_info = products_database[best_match.source_product]["info"]

            transfer_pattern = self.extract_transfer_pattern(
                best_match.source_product, source_data, source_info
            )
            scaling_factor = self._calculate_analogical_scaling(source_info, target_product_info)

            pattern_forecast, pattern_meta = self.pattern_transfer_forecast(
                transfer_pattern, target_product_info, forecast_days, scaling_factor
            )

            # Forecast analogical
            analogical_forecast, analogical_meta = self.analogical_forecast(
                similar_products, products_database, target_product_info, forecast_days
            )

            # Combina con pesi
            pattern_weight = 0.6  # Peso maggiore al pattern transfer
            analogical_weight = 0.4

            hybrid_forecast = (
                pattern_forecast * pattern_weight + analogical_forecast * analogical_weight
            )

            # Metadata combinato
            hybrid_meta = {
                "method": "hybrid",
                "pattern_forecast_meta": pattern_meta,
                "analogical_forecast_meta": analogical_meta,
                "weights": {"pattern": pattern_weight, "analogical": analogical_weight},
                "confidence": "high",
            }

            return hybrid_forecast, hybrid_meta

        except Exception as e:
            logger.error(f"Errore hybrid forecast: {e}")
            # Fallback su analogical se pattern transfer fallisce
            return self.analogical_forecast(
                similar_products, products_database, target_product_info, forecast_days
            )

    def _base_forecast(
        self, target_product_info: Dict[str, Any], forecast_days: int
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Forecast base quando non ci sono prodotti simili"""
        try:
            # Stima domanda base dalle caratteristiche del prodotto
            base_demand = 1.0

            # Aggiustamenti basati su caratteristiche
            if "prezzo" in target_product_info:
                prezzo = target_product_info["prezzo"]
                if prezzo > 100:
                    base_demand *= 0.5  # Prodotti costosi = domanda minore
                elif prezzo < 20:
                    base_demand *= 2.0  # Prodotti economici = domanda maggiore

            if "categoria" in target_product_info:
                categoria = target_product_info["categoria"].lower()
                if "carrozzina" in categoria:
                    base_demand *= 0.8  # Prodotti specializzati
                elif "materasso" in categoria:
                    base_demand *= 1.2  # Prodotti di consumo
                elif "saturimetro" in categoria:
                    base_demand *= 1.5  # Prodotti di uso comune

            # Genera serie con leggero trend crescente (lancio nuovo prodotto)
            trend_values = np.linspace(base_demand * 0.5, base_demand * 1.2, forecast_days)

            # Aggiungi variabilità
            noise = np.random.normal(0, base_demand * 0.1, forecast_days)
            final_values = np.maximum(trend_values + noise, 0.1)  # Min 0.1

            # Crea serie
            dates = pd.date_range(start=pd.Timestamp.now().date(), periods=forecast_days, freq="D")
            forecast_series = pd.Series(final_values, index=dates)

            metadata = {
                "method": "base_forecast",
                "base_demand": base_demand,
                "adjustments_applied": list(target_product_info.keys()),
                "confidence": "low",
            }

            return forecast_series, metadata

        except Exception as e:
            logger.error(f"Errore base forecast: {e}")
            # Forecast di emergenza ultra-semplice
            dates = pd.date_range(start=pd.Timestamp.now().date(), periods=forecast_days, freq="D")
            emergency_series = pd.Series(np.full(forecast_days, 1.0), index=dates)
            return emergency_series, {"method": "emergency", "error": str(e)}
