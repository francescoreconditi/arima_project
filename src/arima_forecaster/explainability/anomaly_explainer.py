"""
Anomaly Explainer

Spiega le cause e i motivi delle anomalie rilevate nei forecast.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel
import json

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalySeverity(Enum):
    """Livelli di severità anomalie"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Tipi di anomalie rilevabili"""
    OUTLIER = "outlier"                    # Valore singolo anomalo
    TREND_SHIFT = "trend_shift"            # Cambio di trend
    SEASONAL_DEVIATION = "seasonal_deviation"  # Deviazione pattern stagionale
    VARIANCE_CHANGE = "variance_change"    # Cambio varianza
    LEVEL_SHIFT = "level_shift"            # Shift di livello permanente
    PATTERN_BREAK = "pattern_break"        # Rottura pattern consolidato


class AnomalyExplanation(BaseModel):
    """Schema spiegazione anomalia"""
    anomaly_id: str
    timestamp: datetime
    predicted_value: float
    actual_value: Optional[float] = None
    expected_range: Tuple[float, float]
    anomaly_score: float
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    deviation_magnitude: float
    confidence_level: float
    
    # Spiegazioni
    primary_cause: str
    contributing_factors: List[Dict[str, Any]]
    similar_historical_cases: List[Dict[str, Any]]
    recommended_actions: List[str]
    
    # Metadati
    model_id: str
    detection_method: str
    metadata: Dict[str, Any] = {}


@dataclass  
class ExplanationConfig:
    """Configurazione per spiegazioni anomalie"""
    historical_window_days: int = 90
    similarity_threshold: float = 0.7
    max_contributing_factors: int = 5
    max_similar_cases: int = 3
    confidence_threshold: float = 0.8
    seasonal_periods: List[int] = None
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [7, 30, 365]  # Weekly, monthly, yearly


class AnomalyExplainer:
    """
    Explainer per anomalie nei forecast
    
    Features:
    - Classificazione tipo anomalia
    - Identificazione cause probabili
    - Analisi fattori contribuenti
    - Ricerca casi storici simili
    - Raccomandazioni azioni correttive
    - Spiegazioni in linguaggio naturale
    """
    
    def __init__(self, config: ExplanationConfig = None):
        self.config = config or ExplanationConfig()
        self.historical_data = {}
        self.anomaly_patterns = {}
        self.explanation_cache = {}
        
        # Statistiche
        self.explanations_generated = 0
        self.pattern_matches = 0
        
        # Inizializza knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Inizializza knowledge base con pattern anomalie comuni"""
        self.anomaly_patterns = {
            AnomalyType.OUTLIER: {
                "description": "Valore singolo significativamente diverso dalla norma",
                "typical_causes": [
                    "Errore di misurazione",
                    "Evento eccezionale",
                    "Anomalia nei dati di input",
                    "Malfunzionamento sistema"
                ],
                "detection_criteria": "z_score > 3 OR iqr_outlier = True"
            },
            AnomalyType.TREND_SHIFT: {
                "description": "Cambiamento significativo nella tendenza generale",
                "typical_causes": [
                    "Cambiamento nelle condizioni di mercato",
                    "Nuova strategia business",
                    "Fattori stagionali inaspettati",
                    "Eventi macroeconomici"
                ],
                "detection_criteria": "trend_change > threshold AND sustained_duration > min_period"
            },
            AnomalyType.SEASONAL_DEVIATION: {
                "description": "Deviazione dal pattern stagionale atteso",
                "typical_causes": [
                    "Cambiamento nelle abitudini dei consumatori",
                    "Eventi meteorologici anomali",
                    "Campagne marketing fuori calendario",
                    "Modifiche nei prodotti stagionali"
                ],
                "detection_criteria": "seasonal_residual > seasonal_threshold"
            },
            AnomalyType.LEVEL_SHIFT: {
                "description": "Cambiamento permanente nel livello base",
                "typical_causes": [
                    "Lancio nuovo prodotto",
                    "Acquisizione clienti importanti",
                    "Cambiamento strutturale nel business",
                    "Modifiche operative permanenti"
                ],
                "detection_criteria": "level_change > threshold AND persistence > min_duration"
            },
            AnomalyType.VARIANCE_CHANGE: {
                "description": "Cambiamento nella variabilità dei dati",
                "typical_causes": [
                    "Instabilità operativa",
                    "Cambiamenti nei processi",
                    "Nuovi canali di vendita",
                    "Aumento incertezza mercato"
                ],
                "detection_criteria": "variance_ratio > threshold OR volatility_change > limit"
            },
            AnomalyType.PATTERN_BREAK: {
                "description": "Rottura di pattern consolidati",
                "typical_causes": [
                    "Disruption tecnologica",
                    "Cambiamento comportamento consumatori",
                    "Eventi black swan",
                    "Crisi settoriale"
                ],
                "detection_criteria": "pattern_correlation < min_correlation"
            }
        }
    
    def add_historical_data(
        self, 
        model_id: str, 
        data: pd.Series,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Aggiunge dati storici per confronto"""
        self.historical_data[model_id] = {
            "data": data.copy(),
            "metadata": metadata or {},
            "statistics": self._calculate_historical_stats(data),
            "added_at": datetime.now()
        }
        logger.info(f"Dati storici aggiunti per {model_id}: {len(data)} punti")
    
    def _calculate_historical_stats(self, data: pd.Series) -> Dict[str, Any]:
        """Calcola statistiche dati storici"""
        try:
            stats = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "count": len(data),
                "last_value": float(data.iloc[-1]) if len(data) > 0 else 0.0
            }
            
            # Statistiche rolling
            if len(data) >= 30:
                rolling_30 = data.rolling(30, min_periods=10)
                stats["rolling_30_mean"] = float(rolling_30.mean().iloc[-1])
                stats["rolling_30_std"] = float(rolling_30.std().iloc[-1])
            
            # Trend
            if len(data) >= 10:
                x = np.arange(len(data))
                trend_coef = np.polyfit(x, data.values, 1)[0]
                stats["trend_slope"] = float(trend_coef)
            
            return stats
            
        except Exception as e:
            logger.error(f"Errore calcolo statistiche storiche: {e}")
            return {"error": str(e)}
    
    def explain_anomaly(
        self,
        model_id: str,
        predicted_value: float,
        actual_value: Optional[float] = None,
        timestamp: datetime = None,
        anomaly_score: float = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> AnomalyExplanation:
        """
        Spiega anomalia rilevata
        
        Args:
            model_id: ID del modello
            predicted_value: Valore predetto dal modello
            actual_value: Valore reale osservato (se disponibile)
            timestamp: Timestamp anomalia
            anomaly_score: Score di anomalia (0-1)
            context_data: Dati di contesto aggiuntivi
            
        Returns:
            Spiegazione strutturata dell'anomalia
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if anomaly_score is None:
                anomaly_score = self._estimate_anomaly_score(model_id, predicted_value, actual_value)
            
            # Determina severità
            severity = self._determine_severity(anomaly_score, predicted_value, actual_value)
            
            # Classifica tipo anomalia
            anomaly_type = self._classify_anomaly_type(
                model_id, predicted_value, actual_value, context_data
            )
            
            # Calcola range atteso
            expected_range = self._calculate_expected_range(model_id, predicted_value, context_data)
            
            # Calcola deviazione
            deviation = self._calculate_deviation_magnitude(predicted_value, actual_value, expected_range)
            
            # Identifica causa primaria
            primary_cause = self._identify_primary_cause(
                anomaly_type, predicted_value, actual_value, context_data
            )
            
            # Trova fattori contribuenti
            contributing_factors = self._find_contributing_factors(
                model_id, predicted_value, actual_value, anomaly_type, context_data
            )
            
            # Cerca casi storici simili
            similar_cases = self._find_similar_historical_cases(
                model_id, predicted_value, anomaly_type, timestamp
            )
            
            # Genera raccomandazioni
            recommendations = self._generate_recommendations(
                anomaly_type, severity, primary_cause, contributing_factors
            )
            
            # Stima confidenza spiegazione
            confidence = self._estimate_explanation_confidence(
                anomaly_type, contributing_factors, similar_cases
            )
            
            # Crea spiegazione
            explanation = AnomalyExplanation(
                anomaly_id=f"anomaly_{model_id}_{timestamp.timestamp()}",
                timestamp=timestamp,
                predicted_value=predicted_value,
                actual_value=actual_value,
                expected_range=expected_range,
                anomaly_score=anomaly_score,
                severity=severity,
                anomaly_type=anomaly_type,
                deviation_magnitude=deviation,
                confidence_level=confidence,
                primary_cause=primary_cause,
                contributing_factors=contributing_factors,
                similar_historical_cases=similar_cases,
                recommended_actions=recommendations,
                model_id=model_id,
                detection_method="comprehensive_analysis",
                metadata=context_data or {}
            )
            
            self.explanations_generated += 1
            logger.info(f"Spiegazione anomalia generata: {explanation.anomaly_id}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Errore generazione spiegazione anomalia: {e}")
            return self._create_fallback_explanation(
                model_id, predicted_value, actual_value, timestamp, anomaly_score
            )
    
    def _estimate_anomaly_score(
        self, 
        model_id: str, 
        predicted_value: float, 
        actual_value: Optional[float]
    ) -> float:
        """Stima score di anomalia se non fornito"""
        try:
            if model_id not in self.historical_data:
                return 0.5  # Score neutro se non ci sono dati storici
            
            hist_stats = self.historical_data[model_id]["statistics"]
            mean = hist_stats.get("mean", predicted_value)
            std = hist_stats.get("std", 1.0)
            
            # Z-score per predicted value
            if std > 0:
                z_score = abs(predicted_value - mean) / std
                score = min(z_score / 3.0, 1.0)  # Normalizza a [0,1]
            else:
                score = 0.0
            
            # Se abbiamo actual value, considera anche quello
            if actual_value is not None:
                actual_z_score = abs(actual_value - mean) / max(std, 1e-10)
                actual_score = min(actual_z_score / 3.0, 1.0)
                score = max(score, actual_score)  # Prende il più alto
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            logger.error(f"Errore stima anomaly score: {e}")
            return 0.5
    
    def _determine_severity(
        self, 
        anomaly_score: float, 
        predicted_value: float, 
        actual_value: Optional[float]
    ) -> AnomalySeverity:
        """Determina severità anomalia"""
        if anomaly_score >= 0.9:
            return AnomalySeverity.CRITICAL
        elif anomaly_score >= 0.7:
            return AnomalySeverity.HIGH
        elif anomaly_score >= 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _classify_anomaly_type(
        self,
        model_id: str,
        predicted_value: float,
        actual_value: Optional[float],
        context_data: Optional[Dict[str, Any]]
    ) -> AnomalyType:
        """Classifica tipo di anomalia"""
        try:
            # Default: outlier semplice
            if model_id not in self.historical_data:
                return AnomalyType.OUTLIER
            
            hist_data = self.historical_data[model_id]["data"]
            hist_stats = self.historical_data[model_id]["statistics"]
            
            # Controlla level shift
            recent_mean = hist_data.tail(min(30, len(hist_data))).mean()
            overall_mean = hist_stats["mean"]
            if abs(predicted_value - recent_mean) > 2 * abs(recent_mean - overall_mean):
                return AnomalyType.LEVEL_SHIFT
            
            # Controlla trend shift (se abbiamo abbastanza dati)
            if len(hist_data) >= 30:
                recent_trend = np.polyfit(range(30), hist_data.tail(30).values, 1)[0]
                overall_trend = hist_stats.get("trend_slope", 0)
                if abs(recent_trend - overall_trend) > abs(overall_trend) * 0.5:
                    return AnomalyType.TREND_SHIFT
            
            # Controlla seasonal deviation
            if context_data and "expected_seasonal" in context_data:
                expected_seasonal = context_data["expected_seasonal"]
                seasonal_dev = abs(predicted_value - expected_seasonal) / max(abs(expected_seasonal), 1)
                if seasonal_dev > 0.3:
                    return AnomalyType.SEASONAL_DEVIATION
            
            # Controlla variance change
            recent_std = hist_data.tail(min(30, len(hist_data))).std()
            overall_std = hist_stats["std"]
            if recent_std > overall_std * 1.5:
                return AnomalyType.VARIANCE_CHANGE
            
            # Default: outlier
            return AnomalyType.OUTLIER
            
        except Exception as e:
            logger.error(f"Errore classificazione anomalia: {e}")
            return AnomalyType.OUTLIER
    
    def _calculate_expected_range(
        self,
        model_id: str,
        predicted_value: float,
        context_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calcola range di valori attesi"""
        try:
            if model_id not in self.historical_data:
                # Fallback: ±20% del valore predetto
                margin = abs(predicted_value) * 0.2
                return (predicted_value - margin, predicted_value + margin)
            
            hist_stats = self.historical_data[model_id]["statistics"]
            mean = hist_stats["mean"]
            std = hist_stats["std"]
            
            # Range basato su 2 deviazioni standard
            lower = mean - 2 * std
            upper = mean + 2 * std
            
            # Se abbiamo confidence interval dal contesto, usa quello
            if context_data and "confidence_interval" in context_data:
                ci = context_data["confidence_interval"]
                if len(ci) == 2:
                    lower, upper = ci
            
            return (float(lower), float(upper))
            
        except Exception as e:
            logger.error(f"Errore calcolo range atteso: {e}")
            margin = abs(predicted_value) * 0.2
            return (predicted_value - margin, predicted_value + margin)
    
    def _calculate_deviation_magnitude(
        self,
        predicted_value: float,
        actual_value: Optional[float],
        expected_range: Tuple[float, float]
    ) -> float:
        """Calcola magnitudine della deviazione"""
        lower, upper = expected_range
        range_size = upper - lower
        
        if range_size <= 0:
            return 0.0
        
        # Se abbiamo actual value, usa quello
        if actual_value is not None:
            if actual_value < lower:
                deviation = (lower - actual_value) / range_size
            elif actual_value > upper:
                deviation = (actual_value - upper) / range_size
            else:
                deviation = 0.0
        else:
            # Usa predicted value
            if predicted_value < lower:
                deviation = (lower - predicted_value) / range_size
            elif predicted_value > upper:
                deviation = (predicted_value - upper) / range_size
            else:
                deviation = 0.0
        
        return float(deviation)
    
    def _identify_primary_cause(
        self,
        anomaly_type: AnomalyType,
        predicted_value: float,
        actual_value: Optional[float],
        context_data: Optional[Dict[str, Any]]
    ) -> str:
        """Identifica causa primaria dell'anomalia"""
        try:
            # Pattern base da knowledge base
            pattern_info = self.anomaly_patterns.get(anomaly_type, {})
            typical_causes = pattern_info.get("typical_causes", [])
            
            # Se abbiamo contesto, cerca indizi specifici
            if context_data:
                # Controlla indicatori nel contesto
                if "external_events" in context_data:
                    events = context_data["external_events"]
                    if events:
                        return f"Evento esterno identificato: {events[0]}"
                
                if "data_quality_issues" in context_data:
                    issues = context_data["data_quality_issues"]
                    if issues:
                        return f"Problema qualità dati: {issues[0]}"
                
                if "system_changes" in context_data:
                    changes = context_data["system_changes"]
                    if changes:
                        return f"Cambiamento sistema: {changes[0]}"
            
            # Usa prima causa tipica come fallback
            if typical_causes:
                return typical_causes[0]
            
            return f"Causa non determinata per anomalia di tipo {anomaly_type.value}"
            
        except Exception as e:
            logger.error(f"Errore identificazione causa primaria: {e}")
            return "Causa non identificabile"
    
    def _find_contributing_factors(
        self,
        model_id: str,
        predicted_value: float,
        actual_value: Optional[float],
        anomaly_type: AnomalyType,
        context_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Trova fattori che contribuiscono all'anomalia"""
        factors = []
        
        try:
            # Fattori dal contesto
            if context_data:
                # Variabili esogene anomale
                if "exog_anomalies" in context_data:
                    for var, score in context_data["exog_anomalies"].items():
                        factors.append({
                            "factor": f"Variabile esogena: {var}",
                            "contribution_score": float(score),
                            "type": "exogenous_variable",
                            "evidence": f"Valore anomalo per {var}"
                        })
                
                # Problemi dati
                if "missing_data_pct" in context_data:
                    missing_pct = context_data["missing_data_pct"]
                    if missing_pct > 0.1:  # >10% missing
                        factors.append({
                            "factor": "Dati mancanti",
                            "contribution_score": min(missing_pct, 1.0),
                            "type": "data_quality",
                            "evidence": f"{missing_pct*100:.1f}% dati mancanti"
                        })
                
                # Drift del modello
                if "model_drift_score" in context_data:
                    drift_score = context_data["model_drift_score"]
                    if drift_score > 0.3:
                        factors.append({
                            "factor": "Drift del modello",
                            "contribution_score": float(drift_score),
                            "type": "model_degradation",
                            "evidence": f"Score drift: {drift_score:.3f}"
                        })
            
            # Fattori basati su dati storici
            if model_id in self.historical_data:
                hist_data = self.historical_data[model_id]["data"]
                hist_stats = self.historical_data[model_id]["statistics"]
                
                # Recency bias (dati recenti diversi da storico)
                if len(hist_data) >= 30:
                    recent_mean = hist_data.tail(10).mean()
                    overall_mean = hist_stats["mean"]
                    if abs(recent_mean - overall_mean) > hist_stats["std"]:
                        factors.append({
                            "factor": "Shift nei dati recenti",
                            "contribution_score": min(abs(recent_mean - overall_mean) / hist_stats["std"], 1.0),
                            "type": "data_pattern_change",
                            "evidence": f"Media recente: {recent_mean:.2f} vs storica: {overall_mean:.2f}"
                        })
            
            # Ordina per contribution score e limita
            factors.sort(key=lambda x: x["contribution_score"], reverse=True)
            factors = factors[:self.config.max_contributing_factors]
            
            return factors
            
        except Exception as e:
            logger.error(f"Errore ricerca fattori contribuenti: {e}")
            return [{"factor": "Analisi fattori non disponibile", "contribution_score": 0.0, "type": "error"}]
    
    def _find_similar_historical_cases(
        self,
        model_id: str,
        predicted_value: float,
        anomaly_type: AnomalyType,
        current_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Trova casi storici simili"""
        similar_cases = []
        
        try:
            if model_id not in self.historical_data:
                return []
            
            hist_data = self.historical_data[model_id]["data"]
            hist_stats = self.historical_data[model_id]["statistics"]
            
            # Trova valori storici simili (entro 20% del valore predetto)
            threshold = abs(predicted_value) * 0.2
            similar_values = hist_data[
                abs(hist_data - predicted_value) <= threshold
            ]
            
            if len(similar_values) == 0:
                return []
            
            # Seleziona casi più rappresentativi
            for i, (timestamp, value) in enumerate(similar_values.items()):
                if len(similar_cases) >= self.config.max_similar_cases:
                    break
                
                # Calcola similarità
                value_similarity = 1 - abs(value - predicted_value) / max(abs(predicted_value), 1)
                
                # Preferisci casi recenti
                days_ago = (current_timestamp - timestamp).days
                recency_score = 1 / (1 + days_ago / 30)  # Decay su 30 giorni
                
                overall_similarity = value_similarity * 0.7 + recency_score * 0.3
                
                if overall_similarity >= self.config.similarity_threshold:
                    similar_cases.append({
                        "timestamp": timestamp.isoformat(),
                        "value": float(value),
                        "days_ago": days_ago,
                        "similarity_score": float(overall_similarity),
                        "context": f"Valore {value:.2f} osservato {days_ago} giorni fa"
                    })
            
            # Ordina per similarità
            similar_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            if similar_cases:
                self.pattern_matches += 1
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Errore ricerca casi simili: {e}")
            return []
    
    def _generate_recommendations(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        primary_cause: str,
        contributing_factors: List[Dict[str, Any]]
    ) -> List[str]:
        """Genera raccomandazioni azioni"""
        recommendations = []
        
        try:
            # Raccomandazioni per severità
            if severity == AnomalySeverity.CRITICAL:
                recommendations.append("URGENTE: Investigare immediatamente la causa dell'anomalia")
                recommendations.append("Considerar sospendere processi automatici fino a risoluzione")
            elif severity == AnomalySeverity.HIGH:
                recommendations.append("Investigare la causa entro 24 ore")
                recommendations.append("Monitorare strettamente prossime predizioni")
            
            # Raccomandazioni per tipo anomalia
            pattern_info = self.anomaly_patterns.get(anomaly_type, {})
            typical_causes = pattern_info.get("typical_causes", [])
            
            if anomaly_type == AnomalyType.OUTLIER:
                recommendations.append("Verificare qualità dei dati di input")
                recommendations.append("Controllare per errori di misurazione")
            
            elif anomaly_type == AnomalyType.TREND_SHIFT:
                recommendations.append("Analizzare cambiamenti nelle condizioni di mercato")
                recommendations.append("Considerare ricalibrazione del modello")
            
            elif anomaly_type == AnomalyType.SEASONAL_DEVIATION:
                recommendations.append("Verificare fattori stagionali inaspettati")
                recommendations.append("Aggiornare pattern stagionali nel modello")
            
            elif anomaly_type == AnomalyType.LEVEL_SHIFT:
                recommendations.append("Identificare cambiamenti strutturali nel business")
                recommendations.append("Considerare retraining del modello")
            
            elif anomaly_type == AnomalyType.VARIANCE_CHANGE:
                recommendations.append("Investigare cause di maggiore instabilità")
                recommendations.append("Valutare aggiustamenti ai parametri di volatilità")
            
            elif anomaly_type == AnomalyType.PATTERN_BREAK:
                recommendations.append("Analizzare disruption o cambiamenti fondamentali")
                recommendations.append("Considerare sviluppo nuovo modello")
            
            # Raccomandazioni per fattori contribuenti
            for factor in contributing_factors:
                factor_type = factor.get("type", "")
                
                if factor_type == "data_quality":
                    recommendations.append("Migliorare qualità dei dati di input")
                elif factor_type == "model_degradation":
                    recommendations.append("Pianificare retraining o aggiornamento modello")
                elif factor_type == "exogenous_variable":
                    recommendations.append(f"Investigare variabile: {factor.get('factor', '')}")
            
            # Rimuovi duplicati e limita
            recommendations = list(dict.fromkeys(recommendations))[:8]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Errore generazione raccomandazioni: {e}")
            return ["Consultare esperto di dominio per analisi approfondita"]
    
    def _estimate_explanation_confidence(
        self,
        anomaly_type: AnomalyType,
        contributing_factors: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]]
    ) -> float:
        """Stima confidenza nella spiegazione"""
        try:
            confidence_factors = []
            
            # Confidenza basata su knowledge base
            if anomaly_type in self.anomaly_patterns:
                confidence_factors.append(0.7)  # Alta confidenza per tipi noti
            else:
                confidence_factors.append(0.3)  # Bassa per tipi sconosciuti
            
            # Confidenza basata su fattori contribuenti
            if contributing_factors:
                avg_contribution = np.mean([f.get("contribution_score", 0) for f in contributing_factors])
                confidence_factors.append(avg_contribution)
            else:
                confidence_factors.append(0.2)
            
            # Confidenza basata su casi simili
            if similar_cases:
                avg_similarity = np.mean([c.get("similarity_score", 0) for c in similar_cases])
                confidence_factors.append(avg_similarity)
            else:
                confidence_factors.append(0.1)
            
            # Confidenza globale (media pesata)
            overall_confidence = np.average(confidence_factors, weights=[0.4, 0.3, 0.3])
            
            return float(np.clip(overall_confidence, 0, 1))
            
        except Exception as e:
            logger.error(f"Errore stima confidenza: {e}")
            return 0.5
    
    def _create_fallback_explanation(
        self,
        model_id: str,
        predicted_value: float,
        actual_value: Optional[float],
        timestamp: datetime,
        anomaly_score: Optional[float]
    ) -> AnomalyExplanation:
        """Crea spiegazione fallback in caso di errore"""
        return AnomalyExplanation(
            anomaly_id=f"anomaly_fallback_{model_id}_{timestamp.timestamp()}",
            timestamp=timestamp or datetime.now(),
            predicted_value=predicted_value,
            actual_value=actual_value,
            expected_range=(predicted_value * 0.8, predicted_value * 1.2),
            anomaly_score=anomaly_score or 0.5,
            severity=AnomalySeverity.MEDIUM,
            anomaly_type=AnomalyType.OUTLIER,
            deviation_magnitude=0.5,
            confidence_level=0.1,
            primary_cause="Causa non determinabile a causa di errore interno",
            contributing_factors=[],
            similar_historical_cases=[],
            recommended_actions=["Consultare esperto per analisi manuale"],
            model_id=model_id,
            detection_method="fallback",
            metadata={"error": "Spiegazione fallback generata"}
        )
    
    def generate_text_summary(self, explanation: AnomalyExplanation) -> str:
        """Genera riassunto testuale della spiegazione"""
        try:
            summary = f"ANOMALIA RILEVATA - {explanation.severity.value.upper()}\n\n"
            
            summary += f"Modello: {explanation.model_id}\n"
            summary += f"Timestamp: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"Valore predetto: {explanation.predicted_value:.2f}\n"
            
            if explanation.actual_value is not None:
                summary += f"Valore reale: {explanation.actual_value:.2f}\n"
            
            summary += f"Range atteso: [{explanation.expected_range[0]:.2f}, {explanation.expected_range[1]:.2f}]\n"
            summary += f"Score anomalia: {explanation.anomaly_score:.3f}\n"
            summary += f"Tipo: {explanation.anomaly_type.value}\n\n"
            
            summary += f"CAUSA PRIMARIA:\n{explanation.primary_cause}\n\n"
            
            if explanation.contributing_factors:
                summary += "FATTORI CONTRIBUENTI:\n"
                for factor in explanation.contributing_factors[:3]:
                    summary += f"• {factor['factor']} (score: {factor['contribution_score']:.2f})\n"
                summary += "\n"
            
            if explanation.similar_historical_cases:
                summary += "CASI STORICI SIMILI:\n"
                for case in explanation.similar_historical_cases[:2]:
                    summary += f"• {case['context']}\n"
                summary += "\n"
            
            summary += "RACCOMANDAZIONI:\n"
            for i, rec in enumerate(explanation.recommended_actions[:3], 1):
                summary += f"{i}. {rec}\n"
            
            summary += f"\nConfidenza spiegazione: {explanation.confidence_level:.1%}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Errore generazione text summary: {e}")
            return "Errore generazione riassunto spiegazione"
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche explainer"""
        return {
            "explanations_generated": self.explanations_generated,
            "pattern_matches": self.pattern_matches,
            "models_with_historical_data": len(self.historical_data),
            "anomaly_types_supported": len(self.anomaly_patterns),
            "cache_size": len(self.explanation_cache),
            "match_rate": (self.pattern_matches / max(self.explanations_generated, 1)) * 100
        }


# Utility functions
def explain_forecast_anomaly(
    model_id: str,
    predicted_value: float,
    historical_series: pd.Series,
    actual_value: Optional[float] = None,
    confidence_interval: Optional[Tuple[float, float]] = None,
    config: ExplanationConfig = None
) -> AnomalyExplanation:
    """
    Utility per spiegare anomalia in forecast
    
    Args:
        model_id: ID modello
        predicted_value: Valore predetto
        historical_series: Serie storica per contesto
        actual_value: Valore reale (opzionale)
        confidence_interval: Intervallo confidenza (opzionale)
        config: Configurazione
        
    Returns:
        Spiegazione anomalia
    """
    explainer = AnomalyExplainer(config or ExplanationConfig())
    explainer.add_historical_data(model_id, historical_series)
    
    context_data = {}
    if confidence_interval:
        context_data["confidence_interval"] = confidence_interval
    
    explanation = explainer.explain_anomaly(
        model_id=model_id,
        predicted_value=predicted_value,
        actual_value=actual_value,
        context_data=context_data
    )
    
    return explanation


if __name__ == "__main__":
    # Test anomaly explainer
    print("Test Anomaly Explainer")
    
    # Dati demo
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    normal_values = np.random.normal(1000, 50, 100)
    series = pd.Series(normal_values, index=dates)
    
    # Test explanation
    explainer = AnomalyExplainer()
    explainer.add_historical_data("test_model", series)
    
    # Valore anomalo
    anomaly_value = 1500.0  # Molto diverso dalla media
    
    explanation = explainer.explain_anomaly(
        model_id="test_model",
        predicted_value=anomaly_value,
        actual_value=None,
        anomaly_score=0.85
    )
    
    print(f"Spiegazione generata: {explanation.anomaly_id}")
    print(f"Tipo anomalia: {explanation.anomaly_type.value}")
    print(f"Severità: {explanation.severity.value}")
    print(f"Causa primaria: {explanation.primary_cause}")
    print(f"Raccomandazioni: {len(explanation.recommended_actions)}")
    print(f"Confidenza: {explanation.confidence_level:.2f}")
    
    # Test text summary
    summary = explainer.generate_text_summary(explanation)
    print(f"\nRiassunto testuale ({len(summary)} caratteri):")
    print(summary[:300] + "...")