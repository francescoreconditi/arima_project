"""
Servizi business logic per l'API FastAPI di forecasting ARIMA/SARIMA/VAR.

Questo modulo implementa i servizi core che gestiscono:
- Ciclo di vita completo dei modelli (addestramento, salvataggio, caricamento)
- Generazione di previsioni per tutti i tipi di modello
- Selezione automatica dei parametri ottimali
- Diagnostiche avanzate dei modelli
- Generazione di report completi
- Gestione persistente dello stato dei modelli

Architettura dei servizi:
- **ModelManager**: Gestisce storage, registry e operazioni CRUD sui modelli
- **ForecastService**: Coordina generazione previsioni, diagnostiche e report
- **Registry JSON**: Persiste metadati dei modelli su filesystem
- **Background processing**: Supporta addestramento asincrono
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from .models import *
from ..core import ARIMAForecaster, SARIMAForecaster, VARForecaster, SARIMAXForecaster
from ..core import ARIMAModelSelector, SARIMAModelSelector, SARIMAXModelSelector
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class ModelManager:
    """
    Gestore centralizzato per tutti i modelli addestrati dell'applicazione.

    Questo servizio implementa un pattern Repository per la gestione persistente
    dei modelli, fornendo un'interfaccia unificata per operazioni CRUD,
    registry in-memory per performance e persistenza su disco per durabilità.

    <h4>Responsabilità Principali:</h4>
    - **Storage Management**: Salvataggio e caricamento modelli da filesystem
    - **Model Registry**: Tracciamento in-memory dei metadati per accesso veloce
    - **Lifecycle Management**: Gestione stati (training, completed, failed)
    - **Concurrent Training**: Supporto per addestramento parallelo di modelli
    - **Data Integrity**: Validazione coerenza tra registry e file system

    <h4>Architettura Storage:</h4>
    ```
    storage_path/
    ├── registry.json           # Metadati centrali di tutti i modelli
    ├── model_id_1.pkl         # Modello serializzato con pickle
    ├── model_id_1.metadata.pkl # Metadati estesi del modello
    ├── model_id_2.pkl         # Altri modelli...
    └── ...
    ```

    <h4>Stati del Modello:</h4>
    - **training**: Addestramento in corso (background task)
    - **completed**: Modello completato e disponibile
    - **failed**: Addestramento fallito (dettagli in error field)

    <h4>Registry Schema:</h4>
    ```json
    {
      "model_id": {
        "model_type": "sarima",
        "status": "completed",
        "created_at": "2024-08-23T22:30:00",
        "n_observations": 365,
        "parameters": {"order": [1,1,1], "seasonal_order": [1,1,1,12]},
        "metrics": {"aic": 1875.42, "bic": 1891.33},
        "model_path": "/path/to/model.pkl"
      }
    }
    ```
    """

    def __init__(self, storage_path: Path):
        """
        Inizializza il gestore modelli con il percorso di storage specificato.

        Crea la directory se non esiste e carica il registry esistente.
        Il registry viene mantenuto in memoria per accesso veloce e
        sincronizzato su disco ad ogni modifica.

        Args:
            storage_path: Percorso della directory per salvare modelli e metadati

        Raises:
            PermissionError: Se non ha permessi di scrittura nella directory
            OSError: Se non può creare la directory di storage
        """
        self.storage_path = Path(storage_path)
        # Crea la directory con permessi per directory intermedie
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        # Registry in-memory per accesso veloce ai metadati
        # Evita letture frequenti dal filesystem per operazioni comuni
        self.model_registry: Dict[str, Dict[str, Any]] = {}

        # Carica modelli esistenti dal registry persistente
        self._load_existing_models()

    def _load_existing_models(self):
        """
        Carica il registry dei modelli esistenti dal file JSON su disco.

        Questo metodo viene chiamato all'inizializzazione per ripristinare
        lo stato del registry da sessioni precedenti. Gestisce gracefully
        casi di registry corrotto o mancante.

        Registry file format: JSON con metadati completi per ogni modello
        Conversion automatica delle date da ISO string a datetime objects
        """
        try:
            registry_path = self.storage_path / "registry.json"
            if registry_path.exists():
                with open(registry_path, "r", encoding="utf-8") as f:
                    loaded_registry = json.load(f)

                # Converte le date da stringa ISO a datetime objects
                for model_id, info in loaded_registry.items():
                    if "created_at" in info and isinstance(info["created_at"], str):
                        try:
                            info["created_at"] = datetime.fromisoformat(info["created_at"])
                        except ValueError:
                            # Fallback per formati date non standard
                            info["created_at"] = datetime.now()

                self.model_registry = loaded_registry
                self.logger.info(f"Caricati {len(self.model_registry)} modelli dal registry")
            else:
                self.logger.info("Nessun registry esistente trovato, inizializzato registry vuoto")

        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Impossibile caricare il registry modelli: {e}")
            self.logger.info("Inizializzato nuovo registry vuoto")
            self.model_registry = {}
        except Exception as e:
            self.logger.error(f"Errore inaspettato durante il caricamento del registry: {e}")
            self.model_registry = {}

    def _save_registry(self):
        """
        Salva il registry corrente su disco in formato JSON.

        Questo metodo viene chiamato dopo ogni modifica del registry per
        garantire persistenza. Gestisce la serializzazione di oggetti datetime
        e fornisce error handling robusto per problemi di I/O.

        Thread Safety: Scrive prima su file temporaneo poi sposta atomicamente
        per evitare corruzione durante scritture concorrenti.
        """
        try:
            registry_path = self.storage_path / "registry.json"
            temp_path = registry_path.with_suffix(".tmp")

            # Crea una copia per la serializzazione senza modificare l'originale
            registry_copy = {}
            for model_id, info in self.model_registry.items():
                registry_copy[model_id] = info.copy()
                # Converte datetime objects a stringa ISO per JSON serialization
                if "created_at" in registry_copy[model_id] and isinstance(
                    registry_copy[model_id]["created_at"], datetime
                ):
                    registry_copy[model_id]["created_at"] = registry_copy[model_id][
                        "created_at"
                    ].isoformat()

            # Scrittura atomica: prima su file temporaneo, poi rinomina
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(registry_copy, f, indent=2, ensure_ascii=False)

            # Sposta atomicamente per evitare corruzione
            temp_path.replace(registry_path)

        except (IOError, OSError) as e:
            self.logger.error(f"Impossibile salvare il registry modelli: {e}")
            # Non solleva l'eccezione per non bloccare l'operazione principale
        except Exception as e:
            self.logger.error(f"Errore inaspettato durante il salvataggio del registry: {e}")

    def save_model(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        metadata: Dict[str, Any]
    ):
        """
        Salva un modello addestrato sul filesystem e aggiorna il registry.

        Args:
            model_id: Identificatore univoco del modello
            model: Istanza del modello addestrato (ARIMAForecaster, SARIMAForecaster, etc.)
            model_type: Tipo di modello ("arima", "sarima", "sarimax", "var")
            metadata: Metadati del modello (parametri, metriche, status, etc.)

        Raises:
            IOError: Errore durante il salvataggio del modello
            ValueError: Parametri non validi
        """
        try:
            # Percorso per salvare il modello serializzato
            model_path = self.storage_path / f"{model_id}.pkl"

            # Salva il modello usando il metodo save della classe specifica
            model.save(model_path)

            # Aggiorna il registry con i metadati
            self.model_registry[model_id] = {
                "model_id": model_id,
                "model_type": model_type,
                "model_path": str(model_path),
                "created_at": metadata.get("created_at", datetime.now()),
                "training_observations": metadata.get("training_observations", 0),
                "parameters": metadata.get("parameters", {}),
                "metrics": metadata.get("metrics", {}),
                "status": metadata.get("status", "completed"),
            }

            # Persiste il registry su disco
            self._save_registry()

            self.logger.info(f"Modello {model_id} ({model_type}) salvato con successo")

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio del modello {model_id}: {e}")
            raise

    async def train_model(self, model_id: str, series: pd.Series, request: ModelTrainingRequest):
        """
        Addestra un modello ARIMA, SARIMA o SARIMAX in modalità asincrona.

        Questo metodo implementa il workflow completo di addestramento:
        1. Validazione dei dati di input
        2. Inizializzazione entry nel registry con status "training"
        3. Configurazione del modello in base ai parametri o auto-selezione
        4. Addestramento del modello con gestione errori
        5. Salvataggio del modello serializzato su disco
        6. Aggiornamento del registry con metriche e parametri finali

        <h4>Supporto per Auto-Selection:</h4>
        Quando `request.auto_select=True`, utilizza grid search per trovare
        i parametri ottimali basati sul criterio informativo.

        <h4>Gestione Stati:</h4>
        - Inizializza con status "training" per indicare processing
        - Aggiorna a "completed" al successo con parametri e metriche
        - Aggiorna a "failed" in caso di errore con dettagli

        Args:
            model_id: Identificatore univoco per il modello (tipicamente UUID)
            series: Serie temporale pandas con index datetime e valori numerici
            request: Configurazione completa per l'addestramento del modello

        Raises:
            ModelTrainingError: Errore durante creazione o addestramento del modello
            ValueError: Parametri di configurazione non validi
            IOError: Errore durante il salvataggio del modello

        Example:
            ```python
            # Addestramento SARIMA manuale
            request = ModelTrainingRequest(
                data=TimeSeriesData(...),
                model_type="sarima",
                order=ARIMAOrder(p=1, d=1, q=1),
                seasonal_order=SARIMAOrder(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
            )
            await manager.train_model("model-123", series, request)

            # Addestramento con auto-selezione
            request.auto_select = True
            await manager.train_model("model-456", series, request)
            ```
        """
        try:
            self.logger.info(
                f"Avvio addestramento modello {request.model_type.upper()} con ID {model_id}"
            )

            # Registra subito il modello con stato "training" per tracking
            # Questo permette all'API di restituire immediatamente un response
            # mentre l'addestramento continua in background
            self.model_registry[model_id] = {
                "model_type": request.model_type,
                "status": "training",
                "created_at": datetime.now(),
                "n_observations": len(series),
                "parameters": {},  # Popolato dopo l'addestramento
                "metrics": {},  # Popolato dopo l'addestramento
            }
            self._save_registry()

            model = None

            # Prepara i dati esogeni se forniti (per modelli SARIMAX)
            # Sincronizza l'index con la serie principale per consistency
            exog_df = None
            if request.exogenous_data:
                exog_df = pd.DataFrame(request.exogenous_data.variables, index=series.index)
                self.logger.debug(f"Dati esogeni preparati: {exog_df.shape}")

            if request.auto_select:
                # Modalità selezione automatica: usa grid search per parametri ottimali
                self.logger.info(f"Avvio selezione automatica parametri per {request.model_type}")

                if request.model_type == "arima":
                    selector = ARIMAModelSelector()
                    selector.search(series)
                    model = selector.get_best_model()
                    params = {"order": selector.best_order}

                elif request.model_type == "sarima":
                    selector = SARIMAModelSelector()
                    selector.search(series)
                    model = selector.get_best_model()
                    params = {
                        "order": selector.best_order,
                        "seasonal_order": selector.best_seasonal_order,
                    }

                elif request.model_type == "sarimax":
                    if exog_df is None:
                        raise ModelTrainingError("SARIMAX richiede dati esogeni per auto-selezione")
                    selector = SARIMAXModelSelector()
                    selector.search(series, exog=exog_df)
                    model = selector.get_best_model()
                    params = {
                        "order": selector.best_order,
                        "seasonal_order": selector.best_seasonal_order,
                        "exog_variables": list(exog_df.columns),
                    }

                self.logger.info(f"Auto-selezione completata, parametri ottimali: {params}")

            else:
                # Modalità parametri manuali: usa esattamente i parametri specificati
                self.logger.info(f"Addestramento con parametri manuali per {request.model_type}")

                if request.model_type == "arima":
                    order = (request.order.p, request.order.d, request.order.q)
                    model = ARIMAForecaster(order=order)
                    model.fit(series)
                    params = {"order": order}

                elif request.model_type == "sarima":
                    order = (request.order.p, request.order.d, request.order.q)
                    seasonal_order = (
                        request.seasonal_order.P,
                        request.seasonal_order.D,
                        request.seasonal_order.Q,
                        request.seasonal_order.s,
                    )
                    model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
                    model.fit(series)
                    params = {"order": order, "seasonal_order": seasonal_order}

                elif request.model_type == "sarimax":
                    if exog_df is None:
                        raise ModelTrainingError(
                            "SARIMAX richiede dati esogeni per l'addestramento"
                        )
                    order = (request.order.p, request.order.d, request.order.q)
                    seasonal_order = (
                        request.seasonal_order.P,
                        request.seasonal_order.D,
                        request.seasonal_order.Q,
                        request.seasonal_order.s,
                    )
                    model = SARIMAXForecaster(order=order, seasonal_order=seasonal_order)
                    model.fit(series, exog=exog_df)
                    params = {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "exog_variables": list(exog_df.columns),
                    }

            # Verifica che il modello sia stato creato correttamente
            if model is None:
                raise ModelTrainingError(f"Impossibile creare il modello {request.model_type}")

            # Salva il modello serializzato su disco
            # Usa pickle per serializzazione completa inclusi metadati statsmodels
            model_path = self.storage_path / f"{model_id}.pkl"
            model.save(model_path)
            self.logger.info(f"Modello salvato in: {model_path}")

            # Estrae le metriche di performance dal modello addestrato
            # Queste saranno utilizzate per valutazioni comparative e reporting
            model_info = model.get_model_info()
            metrics = {
                "aic": model_info.get("aic", None),
                "bic": model_info.get("bic", None),
                "hqic": model_info.get("hqic", None),
            }
            # Rimuove metriche None per JSON serialization pulita
            metrics = {k: v for k, v in metrics.items() if v is not None}

            # Aggiorna il registry con lo stato finale del modello
            self.model_registry[model_id].update(
                {
                    "status": "completed",
                    "parameters": params,
                    "metrics": metrics,
                    "model_path": str(model_path),
                }
            )
            self._save_registry()

            self.logger.info(f"Addestramento modello {model_id} completato con successo")
            self.logger.debug(f"Metriche finali: {metrics}")

        except Exception as e:
            # Gestione errori: aggiorna lo stato a "failed" con dettagli dell'errore
            if model_id in self.model_registry:
                self.model_registry[model_id]["status"] = "failed"
                self.model_registry[model_id]["error"] = str(e)
                self._save_registry()

            self.logger.error(f"Addestramento modello {model_id} fallito: {e}")
            # Re-solleva l'eccezione per propagazione al chiamante
            raise

    async def train_var_model(self, model_id: str, data: pd.DataFrame, request: VARTrainingRequest):
        """
        Addestra un modello VAR (Vector Autoregression) per serie multivariate.

        I modelli VAR sono specificamente progettati per catturare le relazioni
        dinamiche bidirezionali tra multiple serie temporali. Questo metodo
        gestisce la complessità aggiuntiva dei modelli multivariati.

        <h4>Caratteristiche VAR:</h4>
        - **Multivariate**: Gestisce N serie temporali simultaneamente
        - **Interdipendenze**: Cattura come ogni variabile influenza le altre
        - **Lag Selection**: Determina automaticamente il numero ottimale di lag
        - **Impulse Response**: Permette analisi di shock e propagazione

        <h4>Processo di Addestramento:</h4>
        1. Validazione che ci siano ≥2 variabili (requisito VAR)
        2. Stima del numero ottimale di lag usando criterio informativo
        3. Addestramento del sistema di equazioni VAR
        4. Estrazione di metriche per ogni equazione
        5. Salvataggio con metadati estesi per variabili multiple

        Args:
            model_id: Identificatore univoco per il modello VAR
            data: DataFrame con colonne per ogni variabile del sistema
            request: Configurazione VAR (maxlags, criterio informativo)

        Raises:
            ModelTrainingError: Errore durante l'addestramento VAR
            ValueError: Dati non adatti per modelli VAR (es. <2 variabili)

        Example:
            ```python
            # Dati multivariati: vendite, marketing, economia
            data = pd.DataFrame({
                'vendite': [...],
                'marketing': [...],
                'economia': [...]
            }, index=pd.date_range('2020-01-01', periods=100))

            request = VARTrainingRequest(
                data=MultivariateTimeSeriesData(...),
                maxlags=5,
                ic='aic'
            )
            await manager.train_var_model("var-789", data, request)
            ```
        """
        try:
            self.logger.info(f"Avvio addestramento modello VAR con ID {model_id}")
            self.logger.info(
                f"Serie multivariate: {data.shape[1]} variabili, {len(data)} osservazioni"
            )

            # Registra il modello VAR con metadati specifici multivariati
            # Include informazioni sulle variabili e loro nomi
            self.model_registry[model_id] = {
                "model_type": "var",
                "status": "training",
                "created_at": datetime.now(),
                "n_observations": len(data),
                "n_variables": data.shape[1],
                "variable_names": list(data.columns),
                "parameters": {"maxlags": request.maxlags, "ic": request.ic},
                "metrics": {},
            }
            self._save_registry()

            # Addestra il modello VAR con parametri specificati
            # Il modello determinerà automaticamente il lag ottimale se maxlags=None
            model = VARForecaster(maxlags=request.maxlags, ic=request.ic)
            model.fit(data)

            # Salva il modello VAR con tutte le equazioni stimate
            model_path = self.storage_path / f"{model_id}.pkl"
            model.save(model_path)

            # Estrae metriche aggregate per il sistema VAR
            # Ogni criterio informativo è calcolato per l'intero sistema
            model_info = model.get_model_info()
            metrics = {
                "aic": model_info.get("aic", None),
                "bic": model_info.get("bic", None),
                "hqic": model_info.get("hqic", None),
                "fpe": model_info.get("fpe", None),  # Final Prediction Error (specifico VAR)
            }
            # Filtra metriche None
            metrics = {k: v for k, v in metrics.items() if v is not None}

            # Aggiorna il registry con informazioni complete del modello VAR
            self.model_registry[model_id].update(
                {
                    "status": "completed",
                    "metrics": metrics,
                    "model_path": str(model_path),
                    "selected_lag": model_info.get(
                        "lag_order", 0
                    ),  # Lag effettivamente selezionato
                }
            )
            self._save_registry()

            self.logger.info(f"Modello VAR {model_id} addestrato con successo")
            self.logger.info(f"Lag selezionato: {model_info.get('lag_order', 0)}")

        except Exception as e:
            # Gestione errori specifica per modelli VAR
            if model_id in self.model_registry:
                self.model_registry[model_id]["status"] = "failed"
                self.model_registry[model_id]["error"] = str(e)
                self._save_registry()

            self.logger.error(f"Addestramento modello VAR {model_id} fallito: {e}")
            raise

    def load_model(self, model_id: str):
        """
        Carica un modello addestrato dal filesystem.

        Questo metodo implementa il pattern Factory per creare l'istanza
        corretta della classe modello basata sui metadati del registry.
        Gestisce la deserializzazione completa inclusi fitted parameters.

        <h4>Processo di Caricamento:</h4>
        1. Verifica esistenza nel registry
        2. Determina il tipo di modello dai metadati
        3. Carica il file pickle corrispondente
        4. Istanzia la classe corretta del modello
        5. Ripristina stato completo (parametri, fitted model, metadati)

        <h4>Supporto Multi-Type:</h4>
        - **ARIMA**: ARIMAForecaster con parametri (p,d,q)
        - **SARIMA**: SARIMAForecaster con parametri stagionali
        - **SARIMAX**: SARIMAXForecaster con variabili esogene
        - **VAR**: VARForecaster per serie multivariate

        Args:
            model_id: Identificatore univoco del modello da caricare

        Returns:
            Istanza del modello caricato con stato completo ripristinato

        Raises:
            ValueError: Modello non trovato o tipo non riconosciuto
            FileNotFoundError: File del modello non esiste sul filesystem
            pickle.UnpicklingError: Errore durante deserializzazione

        Example:
            ```python
            # Carica qualsiasi tipo di modello
            model = manager.load_model("sarima-123")

            # Il modello è pronto per generare previsioni
            forecast = model.forecast(steps=12)
            ```
        """
        try:
            # Verifica esistenza nel registry in-memory per accesso veloce
            if model_id not in self.model_registry:
                raise ValueError(f"Modello {model_id} non trovato nel registry")

            # Estrae metadati dal registry per determinare tipo e percorso
            model_info = self.model_registry[model_id]
            model_path = Path(model_info["model_path"])

            # Verifica esistenza fisica del file del modello
            if not model_path.exists():
                raise FileNotFoundError(f"File del modello non trovato: {model_path}")

            # Determina la classe corretta basata sul tipo di modello
            model_type = model_info["model_type"]

            # Factory pattern per istanziare la classe corretta
            if model_type == "arima":
                model = ARIMAForecaster.load(model_path)
            elif model_type == "sarima":
                model = SARIMAForecaster.load(model_path)
            elif model_type == "sarimax":
                model = SARIMAXForecaster.load(model_path)
            elif model_type == "var":
                model = VARForecaster.load(model_path)
            else:
                raise ValueError(f"Tipo di modello non riconosciuto: {model_type}")

            self.logger.debug(f"Modello {model_id} ({model_type}) caricato con successo")
            return model

        except Exception as e:
            self.logger.error(f"Impossibile caricare il modello {model_id}: {e}")
            raise

    def model_exists(self, model_id: str) -> bool:
        """
        Verifica se un modello esiste nel sistema.

        Controllo veloce basato sul registry in-memory senza accesso al filesystem.
        Utile per validazione rapida prima di operazioni costose.

        Args:
            model_id: ID del modello da verificare

        Returns:
            True se il modello esiste nel registry, False altrimenti
        """
        return model_id in self.model_registry

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Ottiene informazioni complete su un modello specifico.

        Restituisce una copia dei metadati per evitare modifiche accidentali
        al registry interno. Include tutti i dettagli: parametri, metriche,
        stato, date di creazione, percorsi file.

        Args:
            model_id: ID del modello

        Returns:
            Dizionario con metadati completi del modello

        Raises:
            ValueError: Modello non trovato nel registry
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Modello {model_id} non trovato")
        # Restituisce una copia per evitare modifiche accidentali al registry
        return self.model_registry[model_id].copy()

    def list_models(self) -> List[str]:
        """
        Elenca tutti gli ID dei modelli disponibili nel sistema.

        Returns:
            Lista di stringhe con gli identificatori di tutti i modelli
        """
        return list(self.model_registry.keys())

    def delete_model(self, model_id: str):
        """
        Elimina completamente un modello dal sistema.

        Rimuove sia i file dal filesystem che l'entry dal registry.
        Operazione irreversibile che pulisce completamente ogni traccia
        del modello per liberare spazio su disco e nel registry.

        <h4>File Eliminati:</h4>
        - File principale del modello (.pkl)
        - File di metadati (.metadata.pkl) se presente
        - Entry dal registry JSON

        Args:
            model_id: ID del modello da eliminare

        Raises:
            ValueError: Modello non trovato
            OSError: Errore durante eliminazione file
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Modello {model_id} non trovato")

            # Elimina i file fisici dal filesystem
            model_info = self.model_registry[model_id]
            if "model_path" in model_info:
                model_path = Path(model_info["model_path"])

                # Elimina file principale del modello
                if model_path.exists():
                    model_path.unlink()
                    self.logger.debug(f"Eliminato file modello: {model_path}")

                # Elimina anche file di metadati se esiste
                metadata_path = model_path.with_suffix(".metadata.pkl")
                if metadata_path.exists():
                    metadata_path.unlink()
                    self.logger.debug(f"Eliminato file metadati: {metadata_path}")

            # Rimuove dal registry in-memory e salva
            del self.model_registry[model_id]
            self._save_registry()

            self.logger.info(f"Modello {model_id} eliminato completamente")

        except Exception as e:
            self.logger.error(f"Impossibile eliminare il modello {model_id}: {e}")
            raise


class ForecastService:
    """
    Servizio coordinatore per generazione previsioni e analisi avanzate.

    Questo servizio orchestra tutte le operazioni di forecasting e analisi,
    fornendo un'interfaccia unificata che astrae la complessità dei diversi
    tipi di modello e le loro specifiche API.

    <h4>Responsabilità Principali:</h4>
    - **Forecast Generation**: Previsioni per ARIMA, SARIMA, SARIMAX, VAR
    - **Auto Model Selection**: Grid search per parametri ottimali
    - **Advanced Diagnostics**: Test statistici e analisi residui
    - **Report Generation**: Report completi multi-formato
    - **Error Handling**: Gestione unificata errori di forecasting

    <h4>Supporto Multi-Model:</h4>
    - **Univariati**: ARIMA, SARIMA, SARIMAX con intervalli di confidenza
    - **Multivariati**: VAR con previsioni per ogni variabile del sistema
    - **Esogeni**: Gestione automatica variabili esogene per SARIMAX
    - **Confidence Intervals**: Calcolo intervalli personalizzabili (50%-99%)

    <h4>Pattern di Integrazione:</h4>
    ```python
    # Dependency injection del ModelManager
    service = ForecastService(model_manager)

    # Operazioni coordinate attraverso il servizio
    forecast = await service.generate_forecast(model_id, steps=12)
    diagnostics = await service.generate_diagnostics(model_id)
    report = await service.generate_report(model_id, format="pdf")
    ```
    """

    def __init__(self, model_manager: ModelManager):
        """
        Inizializza il servizio di forecasting con dependency injection.

        Utilizza il pattern Dependency Injection per ricevere il ModelManager,
        consentendo testing semplificato e loose coupling tra servizi.

        Args:
            model_manager: Istanza del ModelManager per accesso ai modelli
        """
        self.model_manager = model_manager
        self.logger = get_logger(__name__)

    async def generate_forecast(
        self,
        model_id: str,
        steps: int,
        confidence_level: float = 0.95,
        return_intervals: bool = True,
        exogenous_future: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """
        Genera previsioni da modelli univariati (ARIMA/SARIMA/SARIMAX).

        Questo metodo fornisce un'interfaccia unificata per la generazione
        di previsioni, gestendo automaticamente le differenze tra tipi di modello
        e i requisiti specifici di ciascuno (es. variabili esogene per SARIMAX).

        <h4>Processo di Generazione:</h4>
        1. Caricamento e validazione del modello
        2. Controllo requisiti specifici del tipo (es. exog per SARIMAX)
        3. Generazione previsioni con parametri appropriati
        4. Calcolo intervalli di confidenza se richiesti
        5. Formattazione risultati in struttura standardizzata

        <h4>Gestione Intervalli di Confidenza:</h4>
        - Calcolo basato su distribuzione normale degli errori
        - Supporto per livelli personalizzati (50%-99%)
        - Gestione automatica di alpha = 1 - confidence_level
        - Propagazione incertezza attraverso i passi futuri

        <h4>Supporto SARIMAX:</h4>
        Per modelli SARIMAX, le variabili esogene future sono obbligatorie:
        - Validazione presenza exogenous_future
        - Controllo coerenza con variabili di training
        - Propagazione attraverso tutti i passi di previsione

        Args:
            model_id: ID del modello da utilizzare per le previsioni
            steps: Numero di passi futuri da prevedere (1-100)
            confidence_level: Livello di confidenza per intervalli (0.5-0.99)
            return_intervals: Se includere intervalli di confidenza
            exogenous_future: DataFrame con valori futuri variabili esogene (SARIMAX)

        Returns:
            ForecastResult con previsioni, timestamp e intervalli se richiesti

        Raises:
            ForecastError: Errore durante generazione previsioni
            ValueError: Parametri non validi o variabili esogene mancanti

        Example:
            ```python
            # Previsione ARIMA/SARIMA semplice
            forecast = await service.generate_forecast(
                model_id="sarima-123",
                steps=12,
                confidence_level=0.95
            )

            # Previsione SARIMAX con variabili esogene
            exog_future = pd.DataFrame({
                'temperatura': [23, 22, 21, 20],
                'promozioni': [1, 0, 1, 0]
            })
            forecast = await service.generate_forecast(
                model_id="sarimax-456",
                steps=4,
                exogenous_future=exog_future
            )
            ```
        """
        try:
            # Carica il modello utilizzando il ModelManager
            model = self.model_manager.load_model(model_id)

            # Ottiene metadati per determinare requisiti specifici del tipo
            model_info = self.model_manager.get_model_info(model_id)
            model_type = model_info.get("model_type", "").lower()

            # Calcola il parametro alpha per intervalli di confidenza
            # Alpha rappresenta la probabilità nelle code (es. 0.05 per 95% confidence)
            alpha = 1 - confidence_level

            if model_type == "sarimax":
                # Modelli SARIMAX richiedono obbligatoriamente variabili esogene future
                if exogenous_future is None:
                    raise ForecastError(
                        "I modelli SARIMAX richiedono variabili esogene future (exogenous_future) "
                        "per generare previsioni. Fornire un DataFrame con le stesse variabili "
                        "utilizzate durante l'addestramento."
                    )

                # Valida che le variabili esogene future abbiano la lunghezza corretta
                if len(exogenous_future) != steps:
                    raise ForecastError(
                        f"Le variabili esogene future devono avere {steps} righe "
                        f"(uguale al numero di passi), ma ne sono state fornite {len(exogenous_future)}"
                    )

                # Genera previsioni SARIMAX con variabili esogene
                if return_intervals:
                    forecast, conf_int = model.forecast(
                        steps=steps, exog_future=exogenous_future, alpha=alpha, return_conf_int=True
                    )
                else:
                    forecast = model.forecast(
                        steps=steps, exog_future=exogenous_future, confidence_intervals=False
                    )
                    conf_int = None

            else:
                # Modelli ARIMA/SARIMA standard senza variabili esogene
                if return_intervals:
                    forecast, conf_int = model.forecast(
                        steps=steps, alpha=alpha, return_conf_int=True
                    )
                else:
                    forecast = model.forecast(steps=steps, confidence_intervals=False)
                    conf_int = None

            # Converte risultati pandas in liste per serializzazione JSON
            forecast_timestamps = [str(ts) for ts in forecast.index]
            forecast_values = forecast.tolist()

            # Estrae limiti degli intervalli di confidenza se disponibili
            lower_bounds = None
            upper_bounds = None
            if conf_int is not None:
                # Confidence intervals tipicamente hanno 2 colonne: lower, upper
                lower_bounds = conf_int.iloc[:, 0].tolist()
                upper_bounds = conf_int.iloc[:, 1].tolist()

            # Crea il risultato strutturato per l'API
            result = ForecastResult(
                model_id=model_id,
                forecast_timestamps=forecast_timestamps,
                forecast_values=forecast_values,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level if return_intervals else None,
                generated_at=datetime.now(),
            )

            self.logger.info(f"Previsione generata per modello {model_id}: {steps} passi")
            return result

        except Exception as e:
            self.logger.error(f"Generazione previsione fallita per modello {model_id}: {e}")
            raise ForecastError(f"Impossibile generare previsione: {e}")

    async def generate_var_forecast(
        self,
        model_id: str,
        steps: int,
        confidence_level: float = 0.95,
        return_intervals: bool = True,
    ) -> VARForecastResult:
        """
        Genera previsioni da modelli VAR multivariati.

        I modelli VAR generano previsioni simultanee per tutte le variabili
        del sistema, catturando le interdipendenze dinamiche. Questo metodo
        gestisce la complessità delle previsioni multivariate.

        <h4>Caratteristiche VAR Forecasting:</h4>
        - **Multi-Variable**: Una previsione per ogni variabile del sistema
        - **Cross-Correlations**: Considera influenze reciproche tra variabili
        - **System Consistency**: Previsioni coerenti con dinamiche storiche
        - **Confidence Intervals**: Intervalli per ogni variabile individualmente

        <h4>Struttura Output:</h4>
        ```python
        {
            "forecasts": {
                "vendite": [1050, 1100, 1080, ...],
                "marketing": [520, 550, 530, ...],
                "economia": [102, 105, 103, ...]
            },
            "lower_bounds": { ... },  # Se return_intervals=True
            "upper_bounds": { ... }   # Se return_intervals=True
        }
        ```

        Args:
            model_id: ID del modello VAR
            steps: Numero di passi futuri per ogni variabile
            confidence_level: Livello di confidenza per intervalli
            return_intervals: Se calcolare intervalli di confidenza

        Returns:
            VARForecastResult con previsioni per tutte le variabili

        Raises:
            ForecastError: Errore durante generazione previsioni VAR

        Example:
            ```python
            # Previsioni VAR per sistema a 3 variabili
            forecast = await service.generate_var_forecast(
                model_id="var-789",
                steps=6,
                confidence_level=0.90
            )

            # Accesso previsioni per variabile specifica
            vendite_forecast = forecast.forecasts["vendite"]
            vendite_lower = forecast.lower_bounds["vendite"]
            ```
        """
        try:
            # Carica il modello VAR
            model = self.model_manager.load_model(model_id)
            model_info = self.model_manager.get_model_info(model_id)

            # Verifica che sia effettivamente un modello VAR
            if model_info.get("model_type") != "var":
                raise ForecastError(f"Il modello {model_id} non è un modello VAR")

            # Genera previsioni VAR con calcolo incertezza
            alpha = 1 - confidence_level
            forecast_result = model.forecast(steps=steps, alpha=alpha)

            # Estrae DataFrame delle previsioni
            forecast_df = forecast_result["forecast"]
            forecast_timestamps = [str(ts) for ts in forecast_df.index]

            # Converte previsioni in dizionario variabile → lista valori
            forecasts = {}
            for col in forecast_df.columns:
                forecasts[col] = forecast_df[col].tolist()

            # Gestisce intervalli di confidenza se richiesti
            lower_bounds = None
            upper_bounds = None

            if return_intervals:
                lower_df = forecast_result["lower_bounds"]
                upper_df = forecast_result["upper_bounds"]

                lower_bounds = {}
                upper_bounds = {}

                # Converte intervalli per ogni variabile
                for col in forecast_df.columns:
                    lower_bounds[col] = lower_df[col].tolist()
                    upper_bounds[col] = upper_df[col].tolist()

            # Crea risultato strutturato per API
            result = VARForecastResult(
                model_id=model_id,
                forecast_timestamps=forecast_timestamps,
                forecasts=forecasts,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level if return_intervals else None,
                generated_at=datetime.now(),
            )

            variables_list = list(forecasts.keys())
            self.logger.info(
                f"Previsione VAR generata per {len(variables_list)} variabili: {variables_list}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Generazione previsione VAR fallita per modello {model_id}: {e}")
            raise ForecastError(f"Impossibile generare previsione VAR: {e}")

    async def auto_select_model(
        self,
        series: pd.Series,
        model_type: str,
        max_models: int = 50,
        information_criterion: str = "aic",
        exogenous_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Esegue selezione automatica dei parametri ottimali tramite grid search.

        Questo metodo implementa un approccio sistematico per trovare la migliore
        combinazione di parametri, testando multiple configurazioni e selezionando
        quella con il miglior score del criterio informativo.

        <h4>Algoritmo di Selezione:</h4>
        1. **Grid Generation**: Genera griglia di combinazioni parametri
        2. **Model Training**: Addestra ogni combinazione sui dati
        3. **Scoring**: Calcola criterio informativo per ogni modello
        4. **Ranking**: Ordina modelli dal miglior score al peggiore
        5. **Selection**: Sceglie il modello con miglior performance
        6. **Persistence**: Salva il modello migliore per uso futuro

        <h4>Criteri Informativi Supportati:</h4>
        - **AIC**: Akaike Information Criterion (bilanciato fit/complessità)
        - **BIC**: Bayesian Information Criterion (penalizza modelli complessi)
        - **HQIC**: Hannan-Quinn Information Criterion (compromesso AIC/BIC)

        <h4>Spazio di Ricerca:</h4>
        - **ARIMA**: p∈[0,3], d∈[0,2], q∈[0,3]
        - **SARIMA**: + P∈[0,2], D∈[0,1], Q∈[0,2], s∈[4,12,52]
        - **SARIMAX**: SARIMA + gestione variabili esogene

        Args:
            series: Serie temporale per l'addestramento
            model_type: Tipo di modello da ottimizzare
            max_models: Limite massimo di modelli da testare
            information_criterion: Criterio per la selezione
            exogenous_data: Variabili esogene per SARIMAX

        Returns:
            Dict con ID modello migliore, parametri, score e tutti i risultati

        Raises:
            ForecastError: Errore durante la selezione automatica
            ValueError: Tipo di modello non supportato o dati mancanti

        Example:
            ```python
            # Auto-selezione SARIMA
            result = await service.auto_select_model(
                series=my_timeseries,
                model_type="sarima",
                max_models=100,
                information_criterion="aic"
            )

            print(f"Miglior modello: {result['best_model_id']}")
            print(f"Parametri ottimali: {result['best_parameters']}")
            print(f"AIC Score: {result['best_score']}")

            # Il modello è già salvato e pronto per l'uso
            forecast = await service.generate_forecast(
                result['best_model_id'],
                steps=12
            )
            ```
        """
        try:
            self.logger.info(f"Avvio selezione automatica per modello {model_type.upper()}")
            self.logger.info(
                f"Parametri: max_models={max_models}, criterio={information_criterion}"
            )

            # Crea il selettore appropriato basato sul tipo di modello
            if model_type == "arima":
                selector = ARIMAModelSelector(
                    information_criterion=information_criterion, max_models=max_models
                )
                # Esegue grid search su spazio parametri ARIMA
                selector.search(series)

            elif model_type == "sarima":
                selector = SARIMAModelSelector(
                    information_criterion=information_criterion, max_models=max_models
                )
                # Esegue grid search su spazio parametri SARIMA
                selector.search(series)

            elif model_type == "sarimax":
                # SARIMAX richiede variabili esogene per la selezione
                if exogenous_data is None:
                    raise ValueError(
                        "La selezione automatica di modelli SARIMAX richiede exogenous_data"
                    )
                selector = SARIMAXModelSelector(
                    information_criterion=information_criterion, max_models=max_models
                )
                # Esegue grid search con variabili esogene
                selector.search(series, exog=exogenous_data)

            else:
                raise ValueError(f"Tipo di modello non supportato per auto-selezione: {model_type}")

            # Ottiene il modello con i migliori parametri
            best_model = selector.get_best_model()
            if best_model is None:
                raise ForecastError(
                    "Nessun modello adatto trovato durante la selezione automatica. "
                    "Verificare i dati di input o aumentare max_models."
                )

            # Genera ID univoco e salva il modello migliore
            import uuid

            model_id = str(uuid.uuid4())

            model_path = self.model_manager.storage_path / f"{model_id}.pkl"
            best_model.save(model_path)

            # Ottiene risultati completi del processo di selezione
            results_df = selector.get_results_summary()
            all_results = results_df.to_dict("records")

            # Registra il modello migliore nel registry
            best_info = best_model.get_model_info()
            parameters = {
                "order": selector.best_order,
            }
            # Aggiunge parametri stagionali se disponibili
            if hasattr(selector, "best_seasonal_order") and selector.best_seasonal_order:
                parameters["seasonal_order"] = selector.best_seasonal_order

            # Aggiunge variabili esogene per SARIMAX
            if model_type == "sarimax" and exogenous_data is not None:
                parameters["exog_variables"] = list(exogenous_data.columns)

            self.model_manager.model_registry[model_id] = {
                "model_type": model_type,
                "status": "completed",
                "created_at": datetime.now(),
                "n_observations": len(series),
                "parameters": parameters,
                "metrics": {
                    "aic": best_info.get("aic", None),
                    "bic": best_info.get("bic", None),
                    "hqic": best_info.get("hqic", None),
                },
                "model_path": str(model_path),
                "auto_selected": True,  # Flag per indicare selezione automatica
            }
            self.model_manager._save_registry()

            # Prepara risultato della selezione
            result = {
                "best_model_id": model_id,
                "best_parameters": parameters,
                "best_score": best_info.get(information_criterion, float("inf")),
                "all_results": all_results,
                "tested_models": len(all_results),
                "selection_criterion": information_criterion,
            }

            self.logger.info(f"Selezione automatica completata: {len(all_results)} modelli testati")
            self.logger.info(
                f"Miglior modello {model_id} con {information_criterion}={result['best_score']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Selezione automatica modelli fallita: {e}")
            raise ForecastError(f"Selezione automatica modelli fallita: {e}")

    async def generate_diagnostics(
        self, model_id: str, include_residuals: bool = True, include_acf_pacf: bool = True
    ) -> ModelDiagnostics:
        """
        Genera diagnostiche statistiche avanzate per un modello addestrato.

        Le diagnostiche sono fondamentali per valutare la qualità del modello
        e identificare potenziali problemi come autocorrelazione residua,
        non-normalità degli errori, o eteroschedasticità.

        <h4>Analisi Diagnostiche Implementate:</h4>

        **Analisi Residui:**
        - Statistiche descrittive (media, deviazione standard, skewness, kurtosis)
        - Distribuzione degli errori di previsione
        - Identificazione outliers e pattern sistematici

        **Test di Normalità:**
        - **Jarque-Bera Test**: Verifica normalità basata su skewness e kurtosis
        - Interpretazione: p-value > 0.05 → residui approssimativamente normali

        **Test di Autocorrelazione:**
        - **Ljung-Box Test**: Detecta autocorrelazione seriale nei residui
        - Interpretazione: p-value > 0.05 → no autocorrelazione significativa

        **Funzioni di Correlazione:**
        - **ACF**: Autocorrelation Function per identificare pattern temporali
        - **PACF**: Partial ACF per determinare ordine ottimale AR

        <h4>Interpretazione Risultati:</h4>
        - **Residui Normali**: Media ≈ 0, simmetrici, code leggere
        - **No Autocorrelazione**: Test Ljung-Box non significativo
        - **Stabilità**: Varianza costante, no trend nei residui
        - **Adeguatezza**: ACF/PACF entro bande di confidenza

        Args:
            model_id: ID del modello da diagnosticare
            include_residuals: Include analisi dettagliata dei residui
            include_acf_pacf: Include calcolo ACF e PACF

        Returns:
            ModelDiagnostics con risultati completi delle analisi

        Raises:
            ForecastError: Errore durante generazione diagnostiche

        Example:
            ```python
            # Diagnostiche complete
            diagnostics = await service.generate_diagnostics(
                model_id="sarima-123",
                include_residuals=True,
                include_acf_pacf=True
            )

            # Verifica qualità modello
            if diagnostics.normality_test["p_value"] > 0.05:
                print("✓ Residui normalmente distribuiti")

            if diagnostics.ljung_box_test["p_value"] > 0.05:
                print("✓ No autocorrelazione significativa nei residui")

            # Analisi ACF/PACF per miglioramenti
            acf_values = diagnostics.acf_values
            ```
        """
        try:
            # Carica il modello e i suoi metadati
            model = self.model_manager.load_model(model_id)
            model_info = self.model_manager.get_model_info(model_id)

            # I modelli VAR hanno diagnostiche diverse (non implementate in questa versione)
            if model_info["model_type"] == "var":
                self.logger.info(f"Diagnostiche VAR non implementate per modello {model_id}")
                return ModelDiagnostics(model_id=model_id)

            # Inizializza oggetto diagnostiche
            diagnostics = ModelDiagnostics(model_id=model_id)

            # Analisi dei residui se il modello ha fitted_model (statsmodels)
            if (
                include_residuals
                and hasattr(model, "fitted_model")
                and hasattr(model.fitted_model, "resid")
            ):
                residuals = model.fitted_model.resid

                # Statistiche descrittive dei residui
                diagnostics.residual_stats = {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "min": float(residuals.min()),
                    "max": float(residuals.max()),
                    "skewness": float(residuals.skew()) if hasattr(residuals, "skew") else 0.0,
                    "kurtosis": float(residuals.kurtosis())
                    if hasattr(residuals, "kurtosis")
                    else 0.0,
                    "n_observations": len(residuals),
                }

                # Test statistici sui residui (richiede scipy)
                try:
                    from scipy import stats

                    # Test di normalità Jarque-Bera
                    # H0: i residui seguono distribuzione normale
                    # H1: i residui non seguono distribuzione normale
                    clean_residuals = residuals.dropna()
                    if len(clean_residuals) > 0:
                        jb_stat, jb_p_value = stats.jarque_bera(clean_residuals)
                        diagnostics.normality_test = {
                            "test_name": "Jarque-Bera",
                            "test_statistic": float(jb_stat),
                            "p_value": float(jb_p_value),
                            "is_normal": jb_p_value > 0.05,
                            "interpretation": "p>0.05: residui normali, p≤0.05: residui non normali",
                        }

                    # Test di Ljung-Box per autocorrelazione
                    # H0: no autocorrelazione nei residui
                    # H1: presenza di autocorrelazione
                    try:
                        from statsmodels.stats.diagnostic import acorr_ljungbox

                        lb_result = acorr_ljungbox(
                            clean_residuals, lags=min(10, len(clean_residuals) // 4)
                        )
                        # Prende il p-value per lag 10 (o il massimo disponibile)
                        lb_p_value = lb_result["lb_pvalue"].iloc[-1]
                        lb_stat = lb_result["lb_stat"].iloc[-1]

                        diagnostics.ljung_box_test = {
                            "test_name": "Ljung-Box",
                            "test_statistic": float(lb_stat),
                            "p_value": float(lb_p_value),
                            "no_autocorrelation": lb_p_value > 0.05,
                            "interpretation": "p>0.05: no autocorrelazione, p≤0.05: autocorrelazione presente",
                        }
                    except ImportError:
                        self.logger.warning(
                            "statsmodels.stats.diagnostic non disponibile per test Ljung-Box"
                        )

                except ImportError:
                    self.logger.warning("scipy non disponibile per test statistici avanzati")
                except Exception as e:
                    self.logger.warning(f"Errore durante test statistici: {e}")

            # Analisi ACF/PACF per identificazione pattern temporali
            if include_acf_pacf:
                try:
                    from statsmodels.tsa.stattools import acf, pacf

                    # Utilizza i dati di training per ACF/PACF
                    if hasattr(model, "training_data") and model.training_data is not None:
                        series = model.training_data

                        # Calcola numero di lag appropriato (regola empirica)
                        max_lags = min(20, len(series) // 4, 50)

                        # Autocorrelation Function
                        acf_vals = acf(series, nlags=max_lags, fft=True)

                        # Partial Autocorrelation Function
                        pacf_vals = pacf(series, nlags=max_lags)

                        # Converte in liste per serializzazione JSON
                        diagnostics.acf_values = acf_vals.tolist()
                        diagnostics.pacf_values = pacf_vals.tolist()

                        self.logger.debug(f"Calcolate ACF/PACF per {max_lags} lag")
                    else:
                        self.logger.warning("Dati di training non disponibili per calcolo ACF/PACF")

                except ImportError:
                    self.logger.warning("statsmodels.tsa.stattools non disponibile per ACF/PACF")
                except Exception as e:
                    self.logger.warning(f"Errore durante calcolo ACF/PACF: {e}")

            self.logger.info(f"Diagnostiche generate per modello {model_id}")
            return diagnostics

        except Exception as e:
            self.logger.error(f"Generazione diagnostiche fallita per modello {model_id}: {e}")
            raise ForecastError(f"Impossibile generare diagnostiche: {e}")

    async def generate_report(
        self,
        model_id: str,
        report_title: Optional[str] = None,
        output_filename: Optional[str] = None,
        format_type: str = "html",
        include_diagnostics: bool = True,
        include_forecast: bool = True,
        forecast_steps: int = 12,
    ) -> Dict[str, Any]:
        """
        Genera un report completo e professionale per un modello addestrato.

        Questo metodo coordina la generazione di report multi-formato che includono
        analisi dettagliate, visualizzazioni interattive, diagnostiche statistiche
        e previsioni con intervalli di confidenza.

        <h4>Contenuti del Report:</h4>

        **Executive Summary:**
        - Sintesi dei risultati principali
        - Metriche di performance chiave
        - Raccomandazioni immediate

        **Model Overview:**
        - Tipo e parametri del modello
        - Dati di addestramento utilizzati
        - Processo di selezione parametri

        **Performance Analysis:**
        - Metriche complete (AIC, BIC, MAE, RMSE, MAPE)
        - Comparazione con benchmark
        - Analisi di accuratezza

        **Diagnostics (opzionale):**
        - Analisi residui dettagliata
        - Test statistici e loro interpretazione
        - Grafici ACF/PACF
        - Raccomandazioni per miglioramenti

        **Forecasts (opzionale):**
        - Previsioni future con intervalli
        - Visualizzazioni interattive
        - Analisi di incertezza
        - Scenari alternativi

        **Technical Appendix:**
        - Dettagli implementativi
        - Parametri di configurazione
        - Log di addestramento
        - Metadati completi

        <h4>Formati Supportati:</h4>
        - **HTML**: Report interattivo con grafici dinamici Plotly
        - **PDF**: Document formattato per stampa e condivisione
        - **DOCX**: Report editabile in Microsoft Word

        <h4>Personalizzazione:</h4>
        - Titoli e nomi file personalizzabili
        - Sezioni includibili/escludibili
        - Numero di passi di previsione configurabile
        - Template personalizzabili

        Args:
            model_id: ID del modello per cui generare il report
            report_title: Titolo personalizzato del report
            output_filename: Nome file senza estensione
            format_type: Formato output (html, pdf, docx)
            include_diagnostics: Include sezione diagnostiche
            include_forecast: Include sezione previsioni
            forecast_steps: Numero di passi di previsione

        Returns:
            Dict con percorso report, formato e metadati

        Raises:
            ForecastError: Errore durante generazione report

        Example:
            ```python
            # Report HTML completo
            result = await service.generate_report(
                model_id="sarima-123",
                report_title="Analisi Vendite Q4 2024",
                output_filename="vendite_q4_sarima",
                format_type="html",
                include_diagnostics=True,
                include_forecast=True,
                forecast_steps=24
            )

            # Report PDF essenziale
            result = await service.generate_report(
                model_id="arima-456",
                format_type="pdf",
                include_diagnostics=False,
                forecast_steps=6
            )

            print(f"Report generato: {result['report_path']}")
            ```
        """
        try:
            # Carica il modello per accesso alle funzionalità di reporting
            model = self.model_manager.load_model(model_id)
            model_info = self.model_manager.get_model_info(model_id)

            # Genera titolo e nome file di default se non specificati
            if report_title is None:
                model_type = model_info.get("model_type", "ARIMA").upper()
                created_date = model_info.get("created_at", datetime.now())
                if isinstance(created_date, str):
                    created_date = datetime.fromisoformat(created_date)
                date_str = created_date.strftime("%d/%m/%Y")
                report_title = f"Report di Analisi {model_type} - {date_str}"

            if output_filename is None:
                # Genera nome file con timestamp e ID modello abbreviato
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                model_short_id = model_id[:8] if len(model_id) >= 8 else model_id
                output_filename = (
                    f"report_{model_info.get('model_type', 'model')}_{model_short_id}_{timestamp}"
                )

            # Prepara dati per le visualizzazioni (opzionale)
            plots_data = None

            # Pre-genera forecast per visualizzazioni se richiesto
            if include_forecast and hasattr(model, "forecast"):
                try:
                    self.logger.debug(f"Pre-generazione forecast per visualizzazioni report")

                    # Utilizza parametri di default per il forecast del report
                    forecast_result = model.forecast(
                        steps=forecast_steps,
                        confidence_intervals=True,
                        alpha=0.05,  # 95% confidence
                    )

                    # Prepara dati per le visualizzazioni
                    # Il generatore di report utilizzerà questi dati per grafici
                    plots_data = {"forecast_preview": True, "forecast_steps": forecast_steps}

                    self.logger.debug("Forecast pre-generato per visualizzazioni")

                except Exception as e:
                    self.logger.warning(f"Impossibile pre-generare forecast per report: {e}")
                    # Continua senza forecast pre-generato

            # Delega la generazione del report al modello specifico
            # Ogni tipo di modello ha la propria implementazione di report
            # che gestisce le specifiche esigenze di visualizzazione e analisi
            self.logger.info(
                f"Avvio generazione report {format_type.upper()} per modello {model_id}"
            )

            report_path = model.generate_report(
                plots_data=plots_data,
                report_title=report_title,
                output_filename=output_filename,
                format_type=format_type,
                include_diagnostics=include_diagnostics,
                include_forecast=include_forecast,
                forecast_steps=forecast_steps,
            )

            # Prepara risultato con metadati completi
            result = {
                "model_id": model_id,
                "report_path": str(report_path),
                "format_type": format_type,
                "report_title": report_title,
                "generated_at": datetime.now().isoformat(),
                "model_type": model_info.get("model_type"),
                "sections": {
                    "diagnostics_included": include_diagnostics,
                    "forecast_included": include_forecast,
                    "forecast_steps": forecast_steps if include_forecast else 0,
                },
            }

            self.logger.info(f"Report {format_type.upper()} generato con successo: {report_path}")
            return result

        except Exception as e:
            self.logger.error(f"Generazione report fallita per modello {model_id}: {e}")
            raise ForecastError(f"Impossibile generare report: {e}")
