"""
Utility per il caricamento dati di serie temporali.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
from ..utils.logger import get_logger
from ..utils.exceptions import DataProcessingError


class DataLoader:
    """
    Carica e valida dati di serie temporali da varie fonti.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def load_csv(
        self,
        file_path: Union[str, Path],
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        parse_dates: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Carica dati di serie temporali da file CSV.

        Args:
            file_path: Percorso del file CSV
            date_column: Nome colonna data (se None, assume indice)
            value_column: Nome colonna valore
            parse_dates: Se parsare le date automaticamente
            **kwargs: Parametri aggiuntivi per pd.read_csv

        Returns:
            DataFrame con indice datetime

        Raises:
            DataProcessingError: Se il file non può essere caricato o elaborato
        """
        try:
            self.logger.info(f"Caricamento dati da {file_path}")

            # Parametri predefiniti
            default_params = {
                "parse_dates": parse_dates,
                "index_col": 0 if date_column is None else date_column,
            }
            default_params.update(kwargs)

            df = pd.read_csv(file_path, **default_params)

            # Valida dati
            if df.empty:
                raise DataProcessingError("DataFrame caricato è vuoto")

            # Assicura indice datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise DataProcessingError(f"Impossibile convertire indice a datetime: {e}")

            # Ordina per data
            df = df.sort_index()

            self.logger.info(
                f"Caricati con successo {len(df)} record da {df.index.min()} a {df.index.max()}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Errore caricamento dati da {file_path}: {e}")
            raise DataProcessingError(f"Fallito caricamento dati: {e}")

    def validate_time_series(self, df: pd.DataFrame, value_column: str) -> Dict[str, Any]:
        """
        Valida dati serie temporali e restituisce metriche di qualità.

        Args:
            df: DataFrame con dati serie temporali
            value_column: Nome della colonna target

        Returns:
            Dizionario con risultati validazione e statistiche
        """
        validation_results = {"is_valid": True, "issues": [], "statistics": {}}

        try:
            # Controlla se la colonna esiste
            if value_column not in df.columns:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Colonna '{value_column}' non trovata")
                return validation_results

            series = df[value_column]

            # Controlla valori mancanti
            missing_count = series.isnull().sum()
            if missing_count > 0:
                validation_results["issues"].append(f"{missing_count} valori mancanti trovati")

            # Controlla valori costanti
            if series.nunique() == 1:
                validation_results["issues"].append("La serie ha valori costanti")

            # Controlla dati sufficienti
            if len(series) < 10:
                validation_results["issues"].append("Punti dati insufficienti (< 10)")

            # Calcola statistiche
            validation_results["statistics"] = {
                "count": len(series),
                "missing_values": missing_count,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "unique_values": series.nunique(),
            }

            if validation_results["issues"]:
                validation_results["is_valid"] = False

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Errore validazione: {e}")

        return validation_results
