"""
Eccezioni personalizzate per il package ARIMA forecaster.
"""


class ARIMAForecasterError(Exception):
    """Classe eccezione base per ARIMA forecaster."""

    pass


class DataProcessingError(ARIMAForecasterError):
    """Eccezione sollevata quando il processing dei dati fallisce."""

    pass


class ModelTrainingError(ARIMAForecasterError):
    """Eccezione sollevata quando l'addestramento del modello fallisce."""

    pass


class ForecastError(ARIMAForecasterError):
    """Eccezione sollevata quando il forecasting fallisce."""

    pass


class ConfigurationError(ARIMAForecasterError):
    """Eccezione sollevata quando la configurazione non è valida."""

    pass


class ValidationError(ARIMAForecasterError):
    """Eccezione sollevata quando la validazione dati fallisce."""

    pass
