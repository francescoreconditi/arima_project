"""
Test per moduli explainability v0.4.0.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch


def test_explainability_imports():
    """Test import moduli explainability."""
    from arima_forecaster.explainability import (
        SHAPExplainer,
        SHAPConfig,
        FeatureImportanceAnalyzer,
        AnomalyExplainer,
        AnomalyExplanation,
        BusinessRulesEngine,
        BusinessContext,
        Rule,
        RuleAction,
        RuleType
    )
    
    # Verifica che le classi siano importabili
    assert SHAPExplainer is not None
    assert SHAPConfig is not None
    assert FeatureImportanceAnalyzer is not None
    assert AnomalyExplainer is not None
    assert AnomalyExplanation is not None
    assert BusinessRulesEngine is not None
    assert BusinessContext is not None
    assert Rule is not None
    assert RuleAction is not None
    assert RuleType is not None


def test_feature_importance_analyzer():
    """Test analyzer importanza feature."""
    from arima_forecaster.explainability import FeatureImportanceAnalyzer
    
    analyzer = FeatureImportanceAnalyzer()
    assert analyzer is not None
    
    # Test con dati demo
    X = np.random.randn(50, 3)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(50) * 0.1
    feature_names = ["feature_1", "feature_2", "feature_3"]
    
    results = analyzer.analyze_features(X, y, feature_names)
    
    assert isinstance(results, dict)
    assert 'feature_importance' in results
    assert 'top_features' in results
    assert len(results['feature_importance']) == 3


def test_anomaly_explainer():
    """Test spiegazione anomalie."""
    from arima_forecaster.explainability import AnomalyExplainer
    
    explainer = AnomalyExplainer()
    assert explainer is not None
    
    # Aggiungi dati storici demo
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    historical_data = pd.Series(np.random.normal(100, 10, 30), index=dates)
    
    explainer.add_historical_data("test_model", historical_data)
    
    # Test spiegazione anomalia
    explanation = explainer.explain_anomaly(
        model_id="test_model",
        predicted_value=200.0,  # Valore anomalo
        anomaly_score=0.85
    )
    
    assert explanation is not None
    assert hasattr(explanation, 'anomaly_id')
    assert hasattr(explanation, 'anomaly_type')
    assert hasattr(explanation, 'severity')
    assert hasattr(explanation, 'confidence_level')
    assert hasattr(explanation, 'recommended_actions')


def test_business_rules_engine():
    """Test business rules engine."""
    from arima_forecaster.explainability import BusinessRulesEngine, BusinessContext
    
    engine = BusinessRulesEngine()
    assert engine is not None
    
    # Test context
    context = BusinessContext(
        model_id="test_product",
        product_id="PROD-001",
        forecast_date=datetime.now(),
        max_capacity=1000,
        is_weekend=False
    )
    
    # Test applicazione regole - forecast normale
    normal_forecast = 800.0
    final_value, results = engine.apply_rules(normal_forecast, context)
    
    assert isinstance(final_value, float)
    assert isinstance(results, list)
    assert final_value == normal_forecast  # Nessuna modifica prevista
    
    # Test applicazione regole - forecast che supera capacità
    high_forecast = 1200.0
    final_value_high, results_high = engine.apply_rules(high_forecast, context)
    
    assert final_value_high <= context.max_capacity
    assert any(r.applied for r in results_high)  # Almeno una regola applicata


def test_shap_explainer_config():
    """Test configurazione SHAP explainer."""
    from arima_forecaster.explainability import SHAPConfig
    
    config = SHAPConfig()
    assert config is not None
    assert hasattr(config, 'explainer_type')
    assert hasattr(config, 'background_samples')
    
    # Configurazione custom
    custom_config = SHAPConfig(
        explainer_type="tree",
        background_samples=50,
        max_evals=100
    )
    assert custom_config.explainer_type == "tree"
    assert custom_config.background_samples == 50


def test_business_context_creation():
    """Test creazione business context."""
    from arima_forecaster.explainability import BusinessContext
    
    # Context minimo
    context = BusinessContext(
        model_id="test_model",
        forecast_date=datetime.now()
    )
    assert context.model_id == "test_model"
    assert isinstance(context.forecast_date, datetime)
    
    # Context completo
    full_context = BusinessContext(
        model_id="full_test",
        product_id="PROD-123",
        forecast_date=datetime.now(),
        max_capacity=5000,
        min_capacity=100,
        is_weekend=True,
        historical_average=1200.5,
        seasonal_factor=1.15,
        budget_constraint=10000.0
    )
    assert full_context.product_id == "PROD-123"
    assert full_context.max_capacity == 5000
    assert full_context.min_capacity == 100
    assert full_context.is_weekend == True
    assert full_context.historical_average == 1200.5
    assert full_context.seasonal_factor == 1.15
    assert full_context.budget_constraint == 10000.0


def test_rule_types():
    """Test tipi di regole disponibili."""
    from arima_forecaster.explainability import RuleType, RuleAction
    
    # Test enum RuleType
    assert hasattr(RuleType, 'CAPACITY_LIMIT')
    assert hasattr(RuleType, 'VALIDATION')
    assert hasattr(RuleType, 'ADJUSTMENT')
    
    # Test enum RuleAction
    assert hasattr(RuleAction, 'LIMIT')
    assert hasattr(RuleAction, 'ADJUST')
    assert hasattr(RuleAction, 'ALERT')
    assert hasattr(RuleAction, 'REJECT')


@patch('arima_forecaster.explainability.shap_explainer.shap')
def test_shap_explainer_mock(mock_shap):
    """Test SHAP explainer con mock (evita dipendenza SHAP per test)."""
    from arima_forecaster.explainability import SHAPExplainer, SHAPConfig
    
    # Mock SHAP
    mock_shap.Explainer.return_value = Mock()
    mock_shap.Explainer.return_value.return_value = Mock()
    mock_shap.Explainer.return_value.return_value.values = np.array([0.1, 0.2, -0.1])
    
    config = SHAPConfig(explainer_type="linear")
    explainer = SHAPExplainer(config)
    
    # Test con dati demo
    X_train = np.random.randn(50, 3)
    y_train = np.random.randn(50)
    
    explainer.fit(X_train, y_train)
    assert explainer._fitted == True
    
    # Test spiegazione
    instance = np.random.randn(1, 3)
    explanation = explainer.explain_instance(instance)
    
    assert isinstance(explanation, dict)
    assert 'shap_values' in explanation or 'feature_importance' in explanation