"""
Utility di logging per il package ARIMA forecaster.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "arima_forecaster", level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configura un logger con handler console e file.

    Args:
        name: Nome del logger
        level: Livello di logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Percorso opzionale al file di log

    Returns:
        Istanza logger configurata
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Evita di aggiungere handler più volte
    if logger.handlers:
        return logger

    # Crea formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler file se specificato
    if log_file:
        # Crea directory logs se non esiste
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "arima_forecaster") -> logging.Logger:
    """Ottieni un logger esistente o creane uno nuovo con impostazioni predefinite."""
    return logging.getLogger(name) if logging.getLogger(name).handlers else setup_logger(name)
