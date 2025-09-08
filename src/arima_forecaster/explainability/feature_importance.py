"""
Feature Importance Analyzer

Analizza l'importanza delle features per modelli di forecasting.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ImportanceConfig:
    """Configurazione analisi feature importance"""
    methods: List[str] = None  # ["correlation", "mutual_info", "permutation", "univariate"]
    correlation_threshold: float = 0.1
    significance_level: float = 0.05
    n_permutations: int = 100
    random_state: int = 42
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["correlation", "mutual_info", "univariate"]


class FeatureImportanceAnalyzer:
    """
    Analizzatore importanza features per modelli forecasting
    
    Features:
    - Correlation analysis (Pearson, Spearman)
    - Mutual information
    - Permutation importance
    - Univariate statistical tests
    - Feature ranking e selezione
    - Visualizzazioni interpretabili
    """
    
    def __init__(self, config: ImportanceConfig = None):
        self.config = config or ImportanceConfig()
        self.feature_names = []
        self.target_name = "target"
        self.importance_scores = {}
        self.analysis_results = {}
        
        np.random.seed(self.config.random_state)
    
    def analyze_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target"
    ) -> Dict[str, Any]:
        """
        Analizza importanza features rispetto al target
        
        Args:
            X: Matrix features (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Nomi features (opzionale)
            target_name: Nome target
            
        Returns:
            Dict con risultati analisi
        """
        try:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            self.target_name = target_name
            
            logger.info(f"Analisi importanza per {len(self.feature_names)} features")
            
            # Inizializza risultati
            results = {
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "n_samples": len(y),
                "n_features": X.shape[1],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Esegui metodi di analisi
            for method in self.config.methods:
                try:
                    if method == "correlation":
                        results[method] = self._analyze_correlation(X, y)
                    elif method == "mutual_info":
                        results[method] = self._analyze_mutual_information(X, y)
                    elif method == "permutation":
                        results[method] = self._analyze_permutation_importance(X, y)
                    elif method == "univariate":
                        results[method] = self._analyze_univariate_tests(X, y)
                    else:
                        logger.warning(f"Metodo non riconosciuto: {method}")
                        
                except Exception as e:
                    logger.error(f"Errore analisi {method}: {e}")
                    results[method] = {"error": str(e)}
            
            # Combina ranking da tutti i metodi
            results["combined_ranking"] = self._combine_rankings(results)
            
            # Identifica features più importanti
            results["top_features"] = self._get_top_features(results)
            
            # Genera summary
            results["summary"] = self._generate_importance_summary(results)
            
            self.analysis_results = results
            return results
            
        except Exception as e:
            logger.error(f"Errore analisi features: {e}")
            return {"error": str(e)}
    
    def _analyze_correlation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analizza correlazioni Pearson e Spearman"""
        try:
            correlations = {}
            
            for i, feature_name in enumerate(self.feature_names):
                feature_values = X[:, i]
                
                # Rimuovi NaN per calcolo correlazione
                mask = ~(np.isnan(feature_values) | np.isnan(y))
                if np.sum(mask) < 3:  # Troppi pochi dati
                    correlations[feature_name] = {
                        "pearson_r": 0.0,
                        "pearson_p": 1.0,
                        "spearman_r": 0.0,
                        "spearman_p": 1.0,
                        "significant": False
                    }
                    continue
                
                clean_feature = feature_values[mask]
                clean_target = y[mask]
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(clean_feature, clean_target)
                
                # Spearman correlation
                spearman_r, spearman_p = spearmanr(clean_feature, clean_target)
                
                # Significatività
                significant = (pearson_p < self.config.significance_level and 
                             abs(pearson_r) > self.config.correlation_threshold)
                
                correlations[feature_name] = {
                    "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
                    "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else 1.0,
                    "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else 0.0,
                    "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
                    "significant": significant
                }
            
            # Ranking per correlazione assoluta
            ranking = sorted(
                correlations.items(),
                key=lambda x: abs(x[1]["pearson_r"]),
                reverse=True
            )
            
            return {
                "correlations": correlations,
                "ranking": [{"feature": name, "score": abs(data["pearson_r"])} for name, data in ranking],
                "significant_features": [name for name, data in correlations.items() if data["significant"]],
                "method": "correlation"
            }
            
        except Exception as e:
            logger.error(f"Errore analisi correlazione: {e}")
            return {"error": str(e), "method": "correlation"}
    
    def _analyze_mutual_information(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analizza mutual information"""
        try:
            # Import condizionale per mutual_info
            try:
                from sklearn.feature_selection import mutual_info_regression
                _has_sklearn = True
            except ImportError:
                logger.warning("Sklearn non disponibile per mutual information")
                _has_sklearn = False
            
            if not _has_sklearn:
                return {
                    "error": "Sklearn richiesto per mutual information",
                    "method": "mutual_info"
                }
            
            # Calcola mutual information
            mi_scores = mutual_info_regression(
                X, y, 
                random_state=self.config.random_state
            )
            
            # Crea risultati
            mutual_info = {}
            for i, feature_name in enumerate(self.feature_names):
                mutual_info[feature_name] = {
                    "mi_score": float(mi_scores[i]),
                    "normalized_score": float(mi_scores[i] / (np.max(mi_scores) + 1e-10))
                }
            
            # Ranking
            ranking = sorted(
                mutual_info.items(),
                key=lambda x: x[1]["mi_score"],
                reverse=True
            )
            
            return {
                "mutual_information": mutual_info,
                "ranking": [{"feature": name, "score": data["mi_score"]} for name, data in ranking],
                "method": "mutual_info"
            }
            
        except Exception as e:
            logger.error(f"Errore mutual information: {e}")
            return {"error": str(e), "method": "mutual_info"}
    
    def _analyze_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analizza permutation importance (senza modello specifico)"""
        try:
            # Implementazione semplificata senza modello
            # Usa correlazione come proxy per importanza
            
            baseline_correlation = np.corrcoef(np.mean(X, axis=1), y)[0, 1]
            baseline_score = abs(baseline_correlation) if not np.isnan(baseline_correlation) else 0
            
            permutation_scores = {}
            
            for i, feature_name in enumerate(self.feature_names):
                scores = []
                
                for _ in range(min(self.config.n_permutations, 20)):  # Limit per performance
                    # Permuta feature
                    X_perm = X.copy()
                    X_perm[:, i] = np.random.permutation(X_perm[:, i])
                    
                    # Calcola score con feature permutata
                    perm_correlation = np.corrcoef(np.mean(X_perm, axis=1), y)[0, 1]
                    perm_score = abs(perm_correlation) if not np.isnan(perm_correlation) else 0
                    
                    # Importanza = degradazione performance
                    importance = baseline_score - perm_score
                    scores.append(importance)
                
                permutation_scores[feature_name] = {
                    "importance_mean": float(np.mean(scores)),
                    "importance_std": float(np.std(scores)),
                    "baseline_score": float(baseline_score)
                }
            
            # Ranking
            ranking = sorted(
                permutation_scores.items(),
                key=lambda x: x[1]["importance_mean"],
                reverse=True
            )
            
            return {
                "permutation_importance": permutation_scores,
                "ranking": [{"feature": name, "score": data["importance_mean"]} for name, data in ranking],
                "method": "permutation"
            }
            
        except Exception as e:
            logger.error(f"Errore permutation importance: {e}")
            return {"error": str(e), "method": "permutation"}
    
    def _analyze_univariate_tests(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analizza test statistici univariati"""
        try:
            test_results = {}
            
            for i, feature_name in enumerate(self.feature_names):
                feature_values = X[:, i]
                
                # Rimuovi NaN
                mask = ~(np.isnan(feature_values) | np.isnan(y))
                if np.sum(mask) < 5:
                    test_results[feature_name] = {
                        "f_statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "r_squared": 0.0
                    }
                    continue
                
                clean_feature = feature_values[mask]
                clean_target = y[mask]
                
                try:
                    # Test F per regressione lineare univariata
                    # H0: feature non ha relazione lineare con target
                    
                    # Calcola coefficienti regressione
                    feature_mean = np.mean(clean_feature)
                    target_mean = np.mean(clean_target)
                    
                    numerator = np.sum((clean_feature - feature_mean) * (clean_target - target_mean))
                    denominator = np.sum((clean_feature - feature_mean) ** 2)
                    
                    if denominator == 0:
                        slope = 0
                        r_squared = 0
                    else:
                        slope = numerator / denominator
                        
                        # R-squared
                        y_pred = slope * (clean_feature - feature_mean) + target_mean
                        ss_res = np.sum((clean_target - y_pred) ** 2)
                        ss_tot = np.sum((clean_target - target_mean) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                    
                    # F-statistic
                    n = len(clean_target)
                    f_statistic = (r_squared / (1 - r_squared + 1e-10)) * (n - 2)
                    
                    # P-value usando distribuzione F
                    p_value = 1 - stats.f.cdf(f_statistic, 1, n - 2) if f_statistic > 0 else 1.0
                    
                    test_results[feature_name] = {
                        "f_statistic": float(f_statistic),
                        "p_value": float(p_value),
                        "significant": p_value < self.config.significance_level,
                        "r_squared": float(max(0, r_squared)),
                        "slope": float(slope)
                    }
                    
                except Exception as e:
                    logger.warning(f"Errore test per feature {feature_name}: {e}")
                    test_results[feature_name] = {
                        "f_statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "r_squared": 0.0,
                        "slope": 0.0
                    }
            
            # Ranking per F-statistic
            ranking = sorted(
                test_results.items(),
                key=lambda x: x[1]["f_statistic"],
                reverse=True
            )
            
            return {
                "univariate_tests": test_results,
                "ranking": [{"feature": name, "score": data["f_statistic"]} for name, data in ranking],
                "significant_features": [name for name, data in test_results.items() if data["significant"]],
                "method": "univariate"
            }
            
        except Exception as e:
            logger.error(f"Errore test univariati: {e}")
            return {"error": str(e), "method": "univariate"}
    
    def _combine_rankings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combina ranking da diversi metodi"""
        try:
            # Raccoglie tutti i ranking
            all_rankings = {}
            method_weights = {
                "correlation": 0.3,
                "mutual_info": 0.3,
                "permutation": 0.2,
                "univariate": 0.2
            }
            
            for method in self.config.methods:
                if method in results and "ranking" in results[method]:
                    ranking = results[method]["ranking"]
                    weight = method_weights.get(method, 0.25)
                    
                    for rank, item in enumerate(ranking):
                        feature = item["feature"]
                        # Score normalizzato per rank (1.0 per primo, decrescente)
                        normalized_score = (len(ranking) - rank) / len(ranking)
                        
                        if feature not in all_rankings:
                            all_rankings[feature] = {"total_score": 0.0, "method_scores": {}}
                        
                        all_rankings[feature]["total_score"] += normalized_score * weight
                        all_rankings[feature]["method_scores"][method] = {
                            "rank": rank + 1,
                            "score": item["score"],
                            "normalized_score": normalized_score
                        }
            
            # Ranking finale
            final_ranking = sorted(
                all_rankings.items(),
                key=lambda x: x[1]["total_score"],
                reverse=True
            )
            
            return {
                "combined_scores": all_rankings,
                "final_ranking": [
                    {
                        "feature": name,
                        "combined_score": data["total_score"],
                        "method_scores": data["method_scores"]
                    }
                    for name, data in final_ranking
                ]
            }
            
        except Exception as e:
            logger.error(f"Errore combinazione ranking: {e}")
            return {"error": str(e)}
    
    def _get_top_features(
        self, 
        results: Dict[str, Any], 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Identifica top features più importanti"""
        try:
            if "combined_ranking" not in results or "final_ranking" not in results["combined_ranking"]:
                return {"error": "Ranking non disponibile"}
            
            final_ranking = results["combined_ranking"]["final_ranking"]
            top_features = final_ranking[:top_k]
            
            # Analisi caratteristiche top features
            top_analysis = {}
            for feature_info in top_features:
                feature_name = feature_info["feature"]
                analysis = {
                    "combined_score": feature_info["combined_score"],
                    "rank": top_features.index(feature_info) + 1,
                    "strong_methods": [],
                    "weak_methods": []
                }
                
                # Analizza performance per metodo
                for method, method_data in feature_info.get("method_scores", {}).items():
                    if method_data["normalized_score"] > 0.7:  # Top 30%
                        analysis["strong_methods"].append(method)
                    elif method_data["normalized_score"] < 0.3:  # Bottom 70%
                        analysis["weak_methods"].append(method)
                
                top_analysis[feature_name] = analysis
            
            return {
                "top_features_list": [f["feature"] for f in top_features],
                "top_features_analysis": top_analysis,
                "selection_threshold": top_features[-1]["combined_score"] if top_features else 0.0
            }
            
        except Exception as e:
            logger.error(f"Errore identificazione top features: {e}")
            return {"error": str(e)}
    
    def _generate_importance_summary(self, results: Dict[str, Any]) -> str:
        """Genera summary testuale dell'analisi"""
        try:
            n_features = results.get("n_features", 0)
            n_samples = results.get("n_samples", 0)
            
            summary = f"Analisi Feature Importance ({n_features} features, {n_samples} campioni):\n\n"
            
            # Top features
            if "top_features" in results and "top_features_list" in results["top_features"]:
                top_features = results["top_features"]["top_features_list"]
                summary += f"Top {len(top_features)} Features:\n"
                for i, feature in enumerate(top_features[:5], 1):
                    summary += f"{i}. {feature}\n"
                summary += "\n"
            
            # Metodi usati
            methods_used = [m for m in self.config.methods if m in results and "error" not in results[m]]
            if methods_used:
                summary += f"Metodi di analisi: {', '.join(methods_used)}\n"
            
            # Features significative per ogni metodo
            for method in methods_used:
                method_results = results.get(method, {})
                if "significant_features" in method_results:
                    sig_features = method_results["significant_features"]
                    summary += f"{method.capitalize()}: {len(sig_features)} features significative\n"
            
            summary += f"\nAnalisi completata: {results.get('analysis_timestamp', 'N/A')}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Errore generazione summary: {e}")
            return "Errore generazione summary dell'analisi."
    
    def select_features(
        self,
        threshold: float = 0.1,
        max_features: Optional[int] = None
    ) -> List[str]:
        """
        Seleziona features basandosi sui risultati dell'analisi
        
        Args:
            threshold: Soglia combinata per selezione
            max_features: Numero massimo features da selezionare
            
        Returns:
            Lista features selezionate
        """
        if not self.analysis_results or "combined_ranking" not in self.analysis_results:
            logger.warning("Analisi non disponibile. Eseguire analyze_features() prima.")
            return []
        
        try:
            final_ranking = self.analysis_results["combined_ranking"]["final_ranking"]
            
            # Filtra per threshold
            selected = [
                item["feature"] 
                for item in final_ranking 
                if item["combined_score"] >= threshold
            ]
            
            # Limita numero se specificato
            if max_features and len(selected) > max_features:
                selected = selected[:max_features]
            
            logger.info(f"Features selezionate: {len(selected)}/{len(self.feature_names)}")
            return selected
            
        except Exception as e:
            logger.error(f"Errore selezione features: {e}")
            return []
    
    def get_feature_insights(self, feature_name: str) -> Dict[str, Any]:
        """Ottiene insights dettagliati per feature specifica"""
        if not self.analysis_results:
            return {"error": "Analisi non disponibile"}
        
        insights = {
            "feature_name": feature_name,
            "insights": {},
            "recommendations": []
        }
        
        try:
            # Raccoglie informazioni da tutti i metodi
            for method in self.config.methods:
                if method in self.analysis_results and "error" not in self.analysis_results[method]:
                    method_results = self.analysis_results[method]
                    
                    if method == "correlation":
                        corr_data = method_results["correlations"].get(feature_name, {})
                        insights["insights"]["correlation"] = {
                            "pearson_r": corr_data.get("pearson_r", 0),
                            "significant": corr_data.get("significant", False),
                            "interpretation": self._interpret_correlation(corr_data.get("pearson_r", 0))
                        }
                    
                    elif method == "mutual_info":
                        mi_data = method_results["mutual_information"].get(feature_name, {})
                        insights["insights"]["mutual_info"] = {
                            "mi_score": mi_data.get("mi_score", 0),
                            "normalized": mi_data.get("normalized_score", 0),
                            "interpretation": self._interpret_mutual_info(mi_data.get("normalized_score", 0))
                        }
            
            # Raccomandazioni
            insights["recommendations"] = self._generate_feature_recommendations(feature_name, insights["insights"])
            
            return insights
            
        except Exception as e:
            logger.error(f"Errore insights per {feature_name}: {e}")
            return {"error": str(e)}
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpreta valore correlazione"""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "correlazione molto debole"
        elif abs_corr < 0.3:
            return "correlazione debole"
        elif abs_corr < 0.7:
            return "correlazione moderata"
        else:
            return "correlazione forte"
    
    def _interpret_mutual_info(self, normalized_mi: float) -> str:
        """Interpreta mutual information normalizzata"""
        if normalized_mi < 0.1:
            return "dipendenza molto debole"
        elif normalized_mi < 0.3:
            return "dipendenza debole"
        elif normalized_mi < 0.7:
            return "dipendenza moderata"
        else:
            return "dipendenza forte"
    
    def _generate_feature_recommendations(self, feature_name: str, insights: Dict[str, Any]) -> List[str]:
        """Genera raccomandazioni per feature"""
        recommendations = []
        
        # Basato su correlazione
        if "correlation" in insights:
            corr_info = insights["correlation"]
            if corr_info.get("significant", False):
                if abs(corr_info.get("pearson_r", 0)) > 0.5:
                    recommendations.append("Feature altamente predittiva - mantieni nel modello")
                else:
                    recommendations.append("Feature moderatamente utile - considera per ensemble")
            else:
                recommendations.append("Correlazione non significativa - valuta rimozione")
        
        # Basato su mutual information
        if "mutual_info" in insights:
            mi_info = insights["mutual_info"]
            if mi_info.get("normalized", 0) > 0.3:
                recommendations.append("Cattura dipendenze non-lineari importanti")
        
        if not recommendations:
            recommendations.append("Importanza incerta - richiede analisi aggiuntiva")
        
        return recommendations


# Utility functions
def analyze_forecast_features(
    series_data: pd.Series,
    exog_data: Optional[pd.DataFrame] = None,
    lags: List[int] = [1, 2, 3, 7, 30],
    config: ImportanceConfig = None
) -> Dict[str, Any]:
    """
    Analizza importanza features per forecasting
    
    Args:
        series_data: Serie temporale target
        exog_data: Variabili esogene (opzionale)
        lags: Lista di lag da analizzare
        config: Configurazione analisi
        
    Returns:
        Risultati analisi importanza
    """
    # Crea features lag
    features = []
    feature_names = []
    
    # Lag features
    for lag in lags:
        if len(series_data) > lag:
            lag_series = series_data.shift(lag)
            features.append(lag_series.values)
            feature_names.append(f"lag_{lag}")
    
    # Features esogene se disponibili
    if exog_data is not None:
        for col in exog_data.columns:
            features.append(exog_data[col].values)
            feature_names.append(f"exog_{col}")
    
    # Combina features
    if features:
        X = np.column_stack(features)
        
        # Rimuovi NaN (usa forward fill)
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(series_data.values)
        X_clean = X[mask]
        y_clean = series_data.values[mask]
        
        if len(X_clean) < 10:
            return {"error": "Dati insufficienti per analisi (meno di 10 campioni validi)"}
        
        # Esegui analisi
        analyzer = FeatureImportanceAnalyzer(config)
        results = analyzer.analyze_features(X_clean, y_clean, feature_names, "forecast_target")
        
        return results
    
    else:
        return {"error": "Nessuna feature creata per analisi"}


if __name__ == "__main__":
    # Test feature importance analyzer
    print("Test Feature Importance Analyzer")
    
    # Dati demo
    np.random.seed(42)
    n_samples = 100
    
    # Features con diversi livelli di importanza
    X = np.random.randn(n_samples, 5)
    X[:, 0] = X[:, 0] * 2 + 1  # Feature importante
    X[:, 1] = X[:, 1] * 0.5    # Feature meno importante
    X[:, 2] = np.random.randn(n_samples)  # Feature random
    
    # Target correlato alle prime 2 features
    y = X[:, 0] * 3 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    
    feature_names = ["important_feature", "moderate_feature", "random_feature", "noise_1", "noise_2"]
    
    # Test analyzer
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.analyze_features(X, y, feature_names)
    
    print(f"Analisi completata: {len(results)} componenti")
    print(f"Top 3 features: {results.get('top_features', {}).get('top_features_list', [])[:3]}")
    print(f"Summary: {results.get('summary', 'N/A')[:200]}...")
    
    # Test selezione features
    selected = analyzer.select_features(threshold=0.2, max_features=3)
    print(f"Features selezionate: {selected}")
    
    # Test insights per feature specifica
    if selected:
        insights = analyzer.get_feature_insights(selected[0])
        print(f"Insights per {selected[0]}: {len(insights.get('recommendations', []))} raccomandazioni")