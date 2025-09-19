"""
Explainable AI Module

Fornisce spiegazioni interpretabili per forecast e decisioni del modello.
"""

from .shap_explainer import SHAPExplainer, SHAPConfig
from .feature_importance import FeatureImportanceAnalyzer
from .anomaly_explainer import AnomalyExplainer, AnomalyExplanation
from .business_rules import BusinessRulesEngine, Rule, RuleAction, BusinessContext, RuleType

__all__ = [
    "SHAPExplainer",
    "SHAPConfig",
    "FeatureImportanceAnalyzer",
    "AnomalyExplainer",
    "AnomalyExplanation",
    "BusinessRulesEngine",
    "Rule",
    "RuleAction",
    "BusinessContext",
    "RuleType",
]
