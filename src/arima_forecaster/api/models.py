"""
Modelli Pydantic per la validazione delle richieste e risposte dell'API REST.

Questo modulo definisce tutti i data model utilizzati dall'API FastAPI per:
- Validazione automatica dei dati in ingresso
- Serializzazione delle risposte in formato JSON
- Generazione automatica della documentazione OpenAPI/Swagger
- Type hints per il supporto IDE e controlli statici

Caratteristiche principali:
- Validazione rigida dei tipi e dei valori
- Messaggi di errore personalizzati e informativi
- Supporto per nested models complessi
- Gestione di dati temporali e serie multivariate
- Compatibilità con pandas DataFrame e Series
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime
import pandas as pd
from pathlib import Path
from arima_forecaster.api.examples import TIMESERIES_EXAMPLES, MULTIVARIATE_EXAMPLES


class TimeSeriesData(BaseModel):
    """
    Modello per dati di serie temporali univariate.

    Utilizzato per l'addestramento di modelli ARIMA, SARIMA e SARIMAX.
    Garantisce la consistenza tra timestamps e valori e la presenza di dati validi.

    <h4>Attributi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Vincoli</th></tr>
        <tr><td>timestamps</td><td>List[str]</td><td>Lista di timestamp in formato stringa</td><td>Non vuota, parsabile come date</td></tr>
        <tr><td>values</td><td>List[float]</td><td>Lista di valori numerici della serie temporale</td><td>Stessa lunghezza di timestamps</td></tr>
    </table>

    <h4>Esempio di Utilizzo:</h4>
    <pre><code>
    {
        "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "values": [100.5, 102.3, 98.7]
    }
    </code></pre>

    <h4>Validazioni:</h4>
    - Timestamps non può essere vuoto
    - Values e timestamps devono avere la stessa lunghezza
    - Values non può essere vuoto
    - Tutti i valori devono essere numerici validi
    """

    timestamps: List[str] = Field(
        ...,
        description="Lista di timestamp in formato stringa (ISO 8601 raccomandato)",
        example=TIMESERIES_EXAMPLES["esempio_base"]["value"]["timestamps"][:5],
    )
    values: List[float] = Field(
        ...,
        description="Lista di valori numerici della serie temporale",
        example=TIMESERIES_EXAMPLES["esempio_base"]["value"]["values"][:5],
    )

    @validator("timestamps")
    def validate_timestamps(cls, v):
        """
        Valida la lista dei timestamps.

        Controlla che la lista non sia vuota e che tutti i timestamp
        siano stringhe valide che possono essere parsate come date.
        """
        if len(v) == 0:
            raise ValueError("La lista dei timestamps non può essere vuota")

        # Verifica che tutti i timestamp possano essere parsati
        try:
            pd.to_datetime(v[:5])  # Testa i primi 5 per performance
        except Exception:
            raise ValueError("I timestamps devono essere in formato data valido (es. ISO 8601)")

        return v

    @validator("values")
    def validate_values(cls, v, values):
        """
        Valida la lista dei valori della serie temporale.

        Controlla che:
        - La lista non sia vuota
        - Abbia la stessa lunghezza dei timestamps
        - Tutti i valori siano numerici finiti (no NaN/Inf)
        """
        if len(v) == 0:
            raise ValueError("La lista dei valori non può essere vuota")

        if "timestamps" in values and len(v) != len(values["timestamps"]):
            raise ValueError("I valori e i timestamps devono avere la stessa lunghezza")

        # Controlla che tutti i valori siano numerici finiti
        import math

        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Il valore alla posizione {i} non è un numero finito valido")

        return v


class MultivariateTimeSeriesData(BaseModel):
    """
    Modello per dati di serie temporali multivariate.

    Utilizzato specificamente per i modelli VAR (Vector Autoregression)
    che analizzano le relazioni dinamiche tra multiple serie temporali.

    <h4>Attributi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Vincoli</th></tr>
        <tr><td>timestamps</td><td>List[str]</td><td>Lista di timestamp condivisi</td><td>Non vuota, formato date valido</td></tr>
        <tr><td>data</td><td>Dict[str, List[float]]</td><td>Dizionario variabile → valori</td><td>Almeno 2 variabili, stessa lunghezza</td></tr>
    </table>

    <h4>Esempio di Utilizzo:</h4>
    <pre><code>
    {
        "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "data": {
            "vendite": [1000, 1100, 950],
            "marketing": [500, 600, 450],
            "temperatura": [22.5, 24.1, 21.8]
        }
    }
    </code></pre>

    <h4>Validazioni:</h4>
    - Minimo 2 variabili (per modelli VAR significativi)
    - Tutte le variabili devono avere la stessa lunghezza dei timestamps
    - Nomi delle variabili devono essere stringhe non vuote
    - Tutti i valori devono essere numerici finiti
    """

    timestamps: List[str] = Field(
        ...,
        description="Lista di timestamp condivisi tra tutte le variabili",
        example=MULTIVARIATE_EXAMPLES["esempio_var"]["value"]["timestamps"][:5],
    )
    data: Dict[str, List[float]] = Field(
        ...,
        description="Dizionario che mappa nomi delle variabili ai loro valori temporali",
        example={
            "vendite": [1000, 1100, 950],
            "marketing": [500, 600, 450],
            "temperatura": [22.5, 24.1, 21.8],
        },
    )

    @validator("data")
    def validate_data(cls, v, values):
        """
        Valida i dati delle serie multivariate.

        Controlla che:
        - Ci siano almeno 2 variabili (requisito per VAR)
        - Tutte le variabili abbiano la stessa lunghezza
        - I nomi delle variabili siano validi
        - Tutti i valori siano numerici finiti
        """
        if len(v) < 2:
            raise ValueError("I dati multivariati devono avere almeno 2 variabili per modelli VAR")

        # Controlla che i nomi delle variabili siano validi
        for var_name in v.keys():
            if not var_name or not isinstance(var_name, str):
                raise ValueError("I nomi delle variabili devono essere stringhe non vuote")

        # Controlla che tutte le variabili abbiano la stessa lunghezza
        if "timestamps" in values:
            expected_len = len(values["timestamps"])
            for var_name, var_values in v.items():
                if len(var_values) != expected_len:
                    raise ValueError(
                        f"La variabile '{var_name}' ha {len(var_values)} valori, "
                        f"ma sono attesi {expected_len} (lunghezza dei timestamps)"
                    )

        # Controlla che tutti i valori siano numerici finiti
        import math

        for var_name, var_values in v.items():
            for i, val in enumerate(var_values):
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(
                        f"Valore non valido alla posizione {i} nella variabile '{var_name}'"
                    )

        return v


class ARIMAOrder(BaseModel):
    """
    Specifica i parametri di ordine per un modello ARIMA.

    I modelli ARIMA sono caratterizzati da tre parametri (p,d,q):
    - p: ordine autoregressivo (AR)
    - d: grado di differenziazione (I)
    - q: ordine della media mobile (MA)

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Parametro</th><th>Tipo</th><th>Descrizione</th><th>Range</th></tr>
        <tr><td>p</td><td>int</td><td>Ordine autoregressivo (AR)</td><td>0-5</td></tr>
        <tr><td>d</td><td>int</td><td>Grado di differenziazione (I)</td><td>0-2</td></tr>
        <tr><td>q</td><td>int</td><td>Ordine media mobile (MA)</td><td>0-5</td></tr>
    </table>

    <h4>Esempio:</h4>
    <pre><code>
    {"p": 1, "d": 1, "q": 1}
    </code></pre>

    <h4>Linee Guida:</h4>
    - p alto: Serie con forte autocorrelazione
    - d=1: Serie con trend lineare
    - d=2: Serie con trend quadratico
    - q alto: Serie con errori correlati
    """

    p: int = Field(
        ...,
        ge=0,
        le=5,
        description="Ordine del componente autoregressivo (AR) - numero di osservazioni passate",
        example=1,
    )
    d: int = Field(
        ...,
        ge=0,
        le=2,
        description="Grado di differenziazione integrata (I) - per rendere la serie stazionaria",
        example=1,
    )
    q: int = Field(
        ...,
        ge=0,
        le=5,
        description="Ordine del componente media mobile (MA) - numero di errori di previsione passati",
        example=1,
    )


class SARIMAOrder(BaseModel):
    """
    Specifica i parametri di ordine per un modello SARIMA.

    I modelli SARIMA estendono ARIMA con componenti stagionali (P,D,Q,s):
    - P: ordine AR stagionale
    - D: differenziazione stagionale
    - Q: ordine MA stagionale
    - s: periodo stagionale

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Parametro</th><th>Tipo</th><th>Descrizione</th><th>Range</th></tr>
        <tr><td>p</td><td>int</td><td>Ordine AR non stagionale</td><td>0-5</td></tr>
        <tr><td>d</td><td>int</td><td>Differenziazione non stagionale</td><td>0-2</td></tr>
        <tr><td>q</td><td>int</td><td>Ordine MA non stagionale</td><td>0-5</td></tr>
        <tr><td>P</td><td>int</td><td>Ordine AR stagionale</td><td>0-2</td></tr>
        <tr><td>D</td><td>int</td><td>Differenziazione stagionale</td><td>0-1</td></tr>
        <tr><td>Q</td><td>int</td><td>Ordine MA stagionale</td><td>0-2</td></tr>
        <tr><td>s</td><td>int</td><td>Periodo stagionale</td><td>2-365</td></tr>
    </table>

    <h4>Esempi Comuni:</h4>
    <pre><code>
    {
        "p": 1, "d": 1, "q": 1,
        "P": 1, "D": 1, "Q": 1, "s": 12
    }
    </code></pre>

    <h4>Periodi Stagionali Tipici:</h4>
    - s=4: Dati trimestrali
    - s=7: Dati giornalieri con stagionalità settimanale
    - s=12: Dati mensili con stagionalità annuale
    - s=52: Dati settimanali con stagionalità annuale
    """

    p: int = Field(..., ge=0, le=5, description="Ordine autoregressivo non stagionale")
    d: int = Field(..., ge=0, le=2, description="Grado di differenziazione non stagionale")
    q: int = Field(..., ge=0, le=5, description="Ordine media mobile non stagionale")
    P: int = Field(..., ge=0, le=2, description="Ordine autoregressivo stagionale")
    D: int = Field(..., ge=0, le=1, description="Grado di differenziazione stagionale")
    Q: int = Field(..., ge=0, le=2, description="Ordine media mobile stagionale")
    s: int = Field(..., ge=2, le=365, description="Periodo stagionale (es. 12 per dati mensili)")


class ExogenousData(BaseModel):
    """
    Modello per le variabili esogene utilizzate nei modelli SARIMAX.

    Le variabili esogene sono fattori esterni che influenzano la serie temporale
    ma non sono predetti dal modello (temperatura, promozioni, festività, etc.).

    <h4>Attributi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Vincoli</th></tr>
        <tr><td>variables</td><td>Dict[str, List[float]]</td><td>Mappa variabile → valori</td><td>Non vuoto, stessa lunghezza</td></tr>
    </table>

    <h4>Esempio di Utilizzo:</h4>
    <pre><code>
    {
        "variables": {
            "temperatura": [22.5, 24.1, 21.8, 19.3],
            "promozioni": [0, 1, 0, 1],
            "festivo": [0, 0, 1, 0]
        }
    }
    </code></pre>

    <h4>Tipi di Variabili Esogene:</h4>
    - <strong>Continue</strong>: Temperatura, prezzi, indici economici
    - <strong>Binarie</strong>: Festività, promozioni, eventi speciali
    - <strong>Categoriche</strong>: Giorni della settimana, stagioni
    - <strong>Lag</strong>: Valori ritardati di altre serie temporali

    <h4>Validazioni:</h4>
    - Tutte le variabili devono avere la stessa lunghezza
    - I valori devono essere numerici finiti
    - I nomi delle variabili devono essere identificatori validi
    """

    variables: Dict[str, List[float]] = Field(
        ...,
        description="Dizionario che mappa nomi delle variabili esogene ai loro valori",
        example={"temperatura": [22.5, 24.1, 21.8], "promozioni": [0, 1, 0], "festivo": [0, 0, 1]},
    )

    @validator("variables")
    def validate_variables(cls, v):
        """
        Valida le variabili esogene.

        Controlla che:
        - Non sia vuoto
        - Tutte le variabili abbiano la stessa lunghezza
        - I nomi delle variabili siano identificatori validi
        - Tutti i valori siano numerici finiti
        """
        if len(v) == 0:
            raise ValueError("Le variabili esogene non possono essere vuote")

        # Controlla i nomi delle variabili
        for var_name in v.keys():
            if not var_name or not isinstance(var_name, str):
                raise ValueError("I nomi delle variabili esogene devono essere stringhe non vuote")
            if not var_name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(
                    f"Il nome della variabile '{var_name}' contiene caratteri non validi. "
                    "Usare solo lettere, numeri, underscore e trattini."
                )

        # Controlla che tutte le variabili abbiano la stessa lunghezza
        lengths = [len(values) for values in v.values()]
        if len(set(lengths)) > 1:
            vars_lengths = {name: len(vals) for name, vals in v.items()}
            raise ValueError(
                f"Tutte le variabili esogene devono avere la stessa lunghezza. "
                f"Lunghezze trovate: {vars_lengths}"
            )

        # Controlla che tutti i valori siano numerici finiti
        import math

        for var_name, var_values in v.items():
            for i, val in enumerate(var_values):
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(
                        f"Valore non valido alla posizione {i} nella variabile esogena '{var_name}'"
                    )

        return v


class ExogenousFutureData(BaseModel):
    """
    Modello per i valori futuri delle variabili esogene.

    Necessario per generare previsioni con modelli SARIMAX, in quanto
    i valori futuri delle variabili esogene devono essere forniti
    per il numero di passi di previsione richiesti.

    <h4>Esempio:</h4>
    <pre><code>
    {
        "variables": {
            "temperatura": [23.0, 22.5, 21.0],
            "promozioni": [1, 0, 1]
        }
    }
    </code></pre>
    """

    variables: Dict[str, List[float]] = Field(
        ..., description="Valori futuri delle variabili esogene per le previsioni SARIMAX"
    )


class ModelTrainingRequest(BaseModel):
    """
    Richiesta per l'addestramento di un modello di forecasting.

    Questo modello gestisce la validazione di tutti i parametri necessari
    per addestrare modelli ARIMA, SARIMA o SARIMAX con controlli di coerenza
    automatici tra tipo di modello e parametri richiesti.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
        <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td><td>Sì</td></tr>
        <tr><td>model_type</td><td>str</td><td>Tipo di modello: arima, sarima, sarimax</td><td>Sì</td></tr>
        <tr><td>order</td><td>ARIMAOrder</td><td>Parametri (p,d,q)</td><td>Solo se non auto_select</td></tr>
        <tr><td>seasonal_order</td><td>SARIMAOrder</td><td>Parametri stagionali (P,D,Q,s)</td><td>Per SARIMA/SARIMAX</td></tr>
        <tr><td>exogenous_data</td><td>ExogenousData</td><td>Variabili esogene</td><td>Per SARIMAX</td></tr>
        <tr><td>auto_select</td><td>bool</td><td>Selezione automatica parametri</td><td>No (default: false)</td></tr>
    </table>

    <h4>Esempi per Tipo di Modello:</h4>

    <strong>ARIMA:</strong>
    <pre><code>
    {
        "data": {...},
        "model_type": "arima",
        "order": {"p": 1, "d": 1, "q": 1}
    }
    </code></pre>

    <strong>SARIMA:</strong>
    <pre><code>
    {
        "data": {...},
        "model_type": "sarima",
        "order": {"p": 1, "d": 1, "q": 1},
        "seasonal_order": {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12}
    }
    </code></pre>

    <strong>SARIMAX:</strong>
    <pre><code>
    {
        "data": {...},
        "model_type": "sarimax",
        "order": {"p": 1, "d": 1, "q": 1},
        "seasonal_order": {...},
        "exogenous_data": {...}
    }
    </code></pre>

    <h4>Validazioni Automatiche:</h4>
    - Coerenza tra model_type e parametri richiesti
    - Presenza di dati esogeni per modelli SARIMAX
    - Parametri di ordine quando auto_select=false
    - Lunghezza coerente tra serie principale e variabili esogene
    """

    data: TimeSeriesData = Field(..., description="Dati della serie temporale per l'addestramento")
    model_type: str = Field(..., description="Tipo di modello da addestrare")
    order: Optional[ARIMAOrder] = Field(None, description="Parametri di ordine ARIMA (p,d,q)")
    seasonal_order: Optional[SARIMAOrder] = Field(
        None, description="Parametri di ordine stagionale SARIMA"
    )
    exogenous_data: Optional[ExogenousData] = Field(
        None, description="Variabili esogene per modelli SARIMAX"
    )
    auto_select: bool = Field(
        default=False,
        description="Se true, seleziona automaticamente i migliori parametri tramite grid search",
    )

    @validator("model_type")
    def validate_model_type(cls, v):
        """Valida che il tipo di modello sia supportato."""
        valid_types = ["arima", "sarima", "sarimax"]
        if v.lower() not in valid_types:
            raise ValueError(f"model_type deve essere uno tra: {', '.join(valid_types)}")
        return v.lower()

    @validator("exogenous_data")
    def validate_exogenous(cls, v, values):
        """Valida la coerenza delle variabili esogene con il tipo di modello."""
        model_type = values.get("model_type", "").lower()

        if model_type == "sarimax" and v is None:
            raise ValueError("exogenous_data è obbligatorio per i modelli SARIMAX")
        if model_type in ["arima", "sarima"] and v is not None:
            raise ValueError(f"exogenous_data non è consentito per modelli {model_type.upper()}")

        # Controlla che la lunghezza delle variabili esogene corrisponda ai dati principali
        if v is not None and "data" in values:
            main_data_len = len(values["data"].values)
            for var_name, var_values in v.variables.items():
                if len(var_values) != main_data_len:
                    raise ValueError(
                        f"La variabile esogena '{var_name}' ha {len(var_values)} valori, "
                        f"ma la serie principale ne ha {main_data_len}"
                    )

        return v

    @validator("order")
    def validate_order(cls, v, values):
        """Valida che i parametri ARIMA siano forniti quando necessario."""
        model_type = values.get("model_type", "").lower()
        auto_select = values.get("auto_select", False)

        if model_type == "arima" and not auto_select and v is None:
            raise ValueError(
                "I parametri 'order' sono obbligatori per modelli ARIMA quando auto_select=false"
            )

        return v

    @validator("seasonal_order")
    def validate_seasonal_order(cls, v, values):
        """Valida che i parametri stagionali siano forniti per modelli SARIMA/SARIMAX."""
        model_type = values.get("model_type", "").lower()
        auto_select = values.get("auto_select", False)

        if model_type in ["sarima", "sarimax"] and not auto_select and v is None:
            raise ValueError(
                f"I parametri 'seasonal_order' sono obbligatori per modelli {model_type.upper()} "
                "quando auto_select=false"
            )

        return v


class VARTrainingRequest(BaseModel):
    """
    Richiesta per l'addestramento di un modello VAR (Vector Autoregression).

    I modelli VAR analizzano le relazioni dinamiche tra multiple serie temporali,
    catturando come ciascuna variabile influenzi le altre nel tempo.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Range</th></tr>
        <tr><td>data</td><td>MultivariateTimeSeriesData</td><td>Serie temporali multivariate</td><td>≥2 variabili</td></tr>
        <tr><td>maxlags</td><td>int</td><td>Numero massimo di lag da considerare</td><td>1-20</td></tr>
        <tr><td>ic</td><td>str</td><td>Criterio informativo per selezione</td><td>aic, bic, hqic, fpe</td></tr>
    </table>

    <h4>Esempio:</h4>
    <pre><code>
    {
        "data": {
            "timestamps": ["2023-01-01", "2023-01-02"],
            "data": {
                "vendite": [1000, 1100],
                "marketing": [500, 600],
                "economia": [100, 105]
            }
        },
        "maxlags": 5,
        "ic": "aic"
    }
    </code></pre>

    <h4>Criteri Informativi:</h4>
    - AIC: Akaike Information Criterion (bilanciato)
    - BIC: Bayesian Information Criterion (più parsimonioso)
    - HQIC: Hannan-Quinn Information Criterion
    - FPE: Final Prediction Error
    """

    data: MultivariateTimeSeriesData = Field(
        ..., description="Dati delle serie temporali multivariate"
    )
    maxlags: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Numero massimo di lag da considerare (None per selezione automatica)",
    )
    ic: str = Field(
        default="aic",
        description="Criterio informativo per la selezione del numero ottimale di lag",
    )

    @validator("ic")
    def validate_ic(cls, v):
        """Valida il criterio informativo."""
        valid_criteria = ["aic", "bic", "hqic", "fpe"]
        if v.lower() not in valid_criteria:
            raise ValueError(f"ic deve essere uno tra: {', '.join(valid_criteria)}")
        return v.lower()


class ProphetTrainingRequest(BaseModel):
    """
    Richiesta per l'addestramento di un modello Prophet (Facebook Prophet).

    Facebook Prophet è un modello di forecasting robusto che gestisce automaticamente:
    - Trend non lineari con punti di cambio
    - Stagionalità multiple (giornaliera, settimanale, annuale)
    - Effetti delle festività
    - Valori mancanti e outliers

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Opzioni</th></tr>
        <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td><td>Richiesto</td></tr>
        <tr><td>growth</td><td>str</td><td>Tipo di crescita del trend</td><td>linear, logistic, flat</td></tr>
        <tr><td>yearly_seasonality</td><td>str|bool</td><td>Stagionalità annuale</td><td>auto, true, false</td></tr>
        <tr><td>weekly_seasonality</td><td>str|bool</td><td>Stagionalità settimanale</td><td>auto, true, false</td></tr>
        <tr><td>daily_seasonality</td><td>str|bool</td><td>Stagionalità giornaliera</td><td>auto, true, false</td></tr>
        <tr><td>seasonality_mode</td><td>str</td><td>Modalità stagionalità</td><td>additive, multiplicative</td></tr>
        <tr><td>country_holidays</td><td>str</td><td>Codice paese festività</td><td>IT, US, UK, DE, FR, ES</td></tr>
    </table>

    <h4>Esempio Base:</h4>
    <pre><code>
    {
        "data": {
            "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "values": [100.5, 102.3, 98.7]
        },
        "growth": "linear",
        "yearly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": false,
        "seasonality_mode": "additive",
        "country_holidays": "IT"
    }
    </code></pre>

    <h4>Parametri Avanzati (Opzionali):</h4>
    - changepoint_prior_scale: Flessibilità trend (default: 0.05)
    - seasonality_prior_scale: Flessibilità stagionalità (default: 10.0)
    - holidays_prior_scale: Flessibilità festività (default: 10.0)
    """

    data: TimeSeriesData = Field(..., description="Dati della serie temporale per l'addestramento")

    # Core Prophet parameters
    growth: str = Field(
        default="linear",
        description="Tipo di crescita del trend: linear (lineare), logistic (logistico), flat (piatto)",
    )

    yearly_seasonality: Union[str, bool] = Field(
        default="auto", description="Stagionalità annuale: auto (automatica), true, false"
    )

    weekly_seasonality: Union[str, bool] = Field(
        default="auto", description="Stagionalità settimanale: auto (automatica), true, false"
    )

    daily_seasonality: Union[str, bool] = Field(
        default="auto", description="Stagionalità giornaliera: auto (automatica), true, false"
    )

    seasonality_mode: str = Field(
        default="additive",
        description="Modalità stagionalità: additive (additiva), multiplicative (moltiplicativa)",
    )

    country_holidays: Optional[str] = Field(
        default=None, description="Codice paese per festività (IT, US, UK, DE, FR, ES)"
    )

    # Advanced parameters
    changepoint_prior_scale: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Flessibilità del trend (maggiore = più flessibile)",
    )

    seasonality_prior_scale: float = Field(
        default=10.0,
        gt=0.0,
        le=50.0,
        description="Flessibilità della stagionalità (maggiore = più flessibile)",
    )

    holidays_prior_scale: float = Field(
        default=10.0,
        gt=0.0,
        le=50.0,
        description="Flessibilità degli effetti festività (maggiore = più flessibile)",
    )

    @validator("growth")
    def validate_growth(cls, v):
        """Valida il tipo di crescita."""
        valid_growth = ["linear", "logistic", "flat"]
        if v not in valid_growth:
            raise ValueError(f"growth deve essere uno tra: {', '.join(valid_growth)}")
        return v

    @validator("seasonality_mode")
    def validate_seasonality_mode(cls, v):
        """Valida la modalità di stagionalità."""
        valid_modes = ["additive", "multiplicative"]
        if v not in valid_modes:
            raise ValueError(f"seasonality_mode deve essere uno tra: {', '.join(valid_modes)}")
        return v

    @validator("country_holidays")
    def validate_country_holidays(cls, v):
        """Valida il codice paese per le festività."""
        if v is None:
            return v
        valid_countries = ["IT", "US", "UK", "DE", "FR", "ES"]
        if v not in valid_countries:
            raise ValueError(f"country_holidays deve essere uno tra: {', '.join(valid_countries)}")
        return v


class ProphetAutoSelectionRequest(BaseModel):
    """
    Richiesta per selezione automatica di parametri ottimali per modelli Prophet.

    Esegue una ricerca su griglia o casuale per trovare la migliore combinazione
    di parametri Prophet che minimizza l'errore di cross-validazione.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>data</td><td>TimeSeriesData</td><td>Dati serie temporale</td><td>Richiesto</td></tr>
        <tr><td>growth_types</td><td>List[str]</td><td>Tipi crescita da testare</td><td>["linear", "logistic"]</td></tr>
        <tr><td>seasonality_modes</td><td>List[str]</td><td>Modalità stagionalità</td><td>["additive"]</td></tr>
        <tr><td>country_holidays</td><td>List[str]</td><td>Paesi festività da testare</td><td>["IT", None]</td></tr>
        <tr><td>max_models</td><td>int</td><td>Numero max modelli</td><td>50</td></tr>
        <tr><td>cv_horizon</td><td>str</td><td>Orizzonte cross-validation</td><td>"30 days"</td></tr>
    </table>

    <h4>Esempio:</h4>
    <pre><code>
    {
        "data": {...},
        "growth_types": ["linear", "logistic"],
        "seasonality_modes": ["additive", "multiplicative"],
        "country_holidays": ["IT", "US", null],
        "max_models": 30,
        "cv_horizon": "30 days"
    }
    </code></pre>

    <h4>Cross-Validation:</h4>
    - Valuta performance con dati storici
    - Utilizza rolling forecast origin
    - Restituisce metriche MAE, RMSE, MAPE
    """

    data: TimeSeriesData = Field(..., description="Dati della serie temporale per l'addestramento")

    growth_types: List[str] = Field(
        default=["linear", "logistic"], description="Lista di tipi di crescita da testare"
    )

    seasonality_modes: List[str] = Field(
        default=["additive"], description="Lista di modalità di stagionalità da testare"
    )

    country_holidays: List[Optional[str]] = Field(
        default=["IT", None],
        description="Lista di codici paese per festività da testare (None = no festività)",
    )

    max_models: int = Field(
        default=50, ge=5, le=200, description="Numero massimo di combinazioni di modelli da testare"
    )

    cv_horizon: str = Field(
        default="30 days",
        description="Orizzonte temporale per cross-validation (es: '30 days', '7 days')",
    )

    @validator("growth_types")
    def validate_growth_types(cls, v):
        """Valida i tipi di crescita."""
        valid_growth = ["linear", "logistic", "flat"]
        for growth in v:
            if growth not in valid_growth:
                raise ValueError(f"Ogni growth_type deve essere uno tra: {', '.join(valid_growth)}")
        return v

    @validator("seasonality_modes")
    def validate_seasonality_modes(cls, v):
        """Valida le modalità di stagionalità."""
        valid_modes = ["additive", "multiplicative"]
        for mode in v:
            if mode not in valid_modes:
                raise ValueError(
                    f"Ogni seasonality_mode deve essere uno tra: {', '.join(valid_modes)}"
                )
        return v

    @validator("country_holidays")
    def validate_country_holidays(cls, v):
        """Valida i codici paese."""
        valid_countries = ["IT", "US", "UK", "DE", "FR", "ES"]
        for country in v:
            if country is not None and country not in valid_countries:
                raise ValueError(
                    f"Ogni country_holiday deve essere uno tra: {', '.join(valid_countries + ['null'])}"
                )
        return v


class ForecastRequest(BaseModel):
    """
    Richiesta per la generazione di previsioni da un modello addestrato.

    Supporta tutti i tipi di modello con opzioni per intervalli di confidenza
    e gestione delle variabili esogene future per modelli SARIMAX.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Range/Vincoli</th></tr>
        <tr><td>steps</td><td>int</td><td>Numero di passi futuri</td><td>1-100</td></tr>
        <tr><td>confidence_level</td><td>float</td><td>Livello di confidenza</td><td>0.5-0.99</td></tr>
        <tr><td>return_intervals</td><td>bool</td><td>Include intervalli confidenza</td><td>true/false</td></tr>
        <tr><td>exogenous_future</td><td>ExogenousFutureData</td><td>Valori futuri variabili esogene</td><td>Solo SARIMAX</td></tr>
    </table>

    <h4>Esempio Base:</h4>
    <pre><code>
    {
        "steps": 12,
        "confidence_level": 0.95,
        "return_intervals": true
    }
    </code></pre>

    <h4>Esempio SARIMAX:</h4>
    <pre><code>
    {
        "steps": 6,
        "confidence_level": 0.90,
        "return_intervals": true,
        "exogenous_future": {
            "variables": {
                "temperatura": [23.0, 22.5, 21.0, 20.5, 19.0, 18.5],
                "promozioni": [1, 0, 1, 0, 1, 0]
            }
        }
    }
    </code></pre>

    <h4>Note Importanti:</h4>
    - Per modelli SARIMAX, exogenous_future è obbligatorio
    - Le variabili esogene future devono avere lunghezza = steps
    - I modelli VAR non richiedono parametri aggiuntivi
    """

    steps: int = Field(
        ..., ge=1, le=100, description="Numero di passi temporali futuri da prevedere"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Livello di confidenza per gli intervalli (es. 0.95 = 95%)",
    )
    return_intervals: bool = Field(
        default=True, description="Se true, include gli intervalli di confidenza nella risposta"
    )
    exogenous_future: Optional[ExogenousFutureData] = Field(
        None, description="Valori futuri delle variabili esogene (richiesto per modelli SARIMAX)"
    )

    @validator("exogenous_future")
    def validate_exogenous_future(cls, v, values):
        """Valida che i valori futuri delle variabili esogene abbiano la lunghezza corretta."""
        if v is not None and "steps" in values:
            steps = values["steps"]
            for var_name, var_values in v.variables.items():
                if len(var_values) != steps:
                    raise ValueError(
                        f"La variabile esogena futura '{var_name}' ha {len(var_values)} valori, "
                        f"ma sono richiesti {steps} valori (uguale a steps)"
                    )
        return v


class ModelInfo(BaseModel):
    """
    Informazioni complete su un modello addestrato.

    Contiene tutti i metadati, parametri di configurazione e metriche
    di performance di un modello salvato nel sistema.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>Identificatore univoco del modello (UUID)</td></tr>
        <tr><td>model_type</td><td>str</td><td>Tipo di modello (arima, sarima, sarimax, var)</td></tr>
        <tr><td>status</td><td>str</td><td>Stato corrente (training, completed, failed)</td></tr>
        <tr><td>created_at</td><td>datetime</td><td>Timestamp di creazione</td></tr>
        <tr><td>training_observations</td><td>int</td><td>Numero di osservazioni utilizzate per il training</td></tr>
        <tr><td>parameters</td><td>Dict</td><td>Parametri di configurazione del modello</td></tr>
        <tr><td>metrics</td><td>Dict</td><td>Metriche di valutazione e performance</td></tr>
    </table>

    <h4>Esempio:</h4>
    <pre><code>
    {
        "model_id": "abc123-def456-ghi789",
        "model_type": "sarima",
        "status": "completed",
        "created_at": "2024-08-23T22:30:00Z",
        "training_observations": 365,
        "parameters": {
            "order": [1, 1, 1],
            "seasonal_order": [1, 1, 1, 12]
        },
        "metrics": {
            "aic": 1875.42,
            "bic": 1891.33,
            "mae": 2.34,
            "rmse": 3.12
        }
    }
    </code></pre>
    """

    model_id: str = Field(..., description="ID univoco del modello")
    model_type: str = Field(..., description="Tipo di modello")
    status: str = Field(..., description="Stato del modello")
    created_at: datetime = Field(..., description="Timestamp di creazione")
    training_observations: int = Field(..., description="Numero di osservazioni per il training")
    parameters: Dict[str, Any] = Field(..., description="Parametri di configurazione")
    metrics: Dict[str, float] = Field(..., description="Metriche di performance")


class ForecastResult(BaseModel):
    """
    Risultato delle previsioni per modelli univariati (ARIMA/SARIMA/SARIMAX).

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello utilizzato</td></tr>
        <tr><td>forecast_timestamps</td><td>List[str]</td><td>Timestamp delle previsioni</td></tr>
        <tr><td>forecast_values</td><td>List[float]</td><td>Valori previsti</td></tr>
        <tr><td>lower_bounds</td><td>List[float]</td><td>Limiti inferiori intervalli confidenza</td></tr>
        <tr><td>upper_bounds</td><td>List[float]</td><td>Limiti superiori intervalli confidenza</td></tr>
        <tr><td>confidence_level</td><td>float</td><td>Livello di confidenza utilizzato</td></tr>
        <tr><td>generated_at</td><td>datetime</td><td>Timestamp generazione previsione</td></tr>
    </table>
    """

    model_id: str = Field(..., description="ID del modello utilizzato")
    forecast_timestamps: List[str] = Field(..., description="Timestamp delle previsioni")
    forecast_values: List[float] = Field(..., description="Valori delle previsioni")
    lower_bounds: Optional[List[float]] = Field(
        None, description="Limiti inferiori intervalli di confidenza"
    )
    upper_bounds: Optional[List[float]] = Field(
        None, description="Limiti superiori intervalli di confidenza"
    )
    confidence_level: Optional[float] = Field(None, description="Livello di confidenza")
    generated_at: datetime = Field(..., description="Timestamp di generazione")


class VARForecastResult(BaseModel):
    """
    Risultato delle previsioni per modelli VAR multivariati.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello VAR utilizzato</td></tr>
        <tr><td>forecast_timestamps</td><td>List[str]</td><td>Timestamp delle previsioni</td></tr>
        <tr><td>forecasts</td><td>Dict[str, List[float]]</td><td>Previsioni per ogni variabile</td></tr>
        <tr><td>lower_bounds</td><td>Dict[str, List[float]]</td><td>Limiti inferiori per variabile</td></tr>
        <tr><td>upper_bounds</td><td>Dict[str, List[float]]</td><td>Limiti superiori per variabile</td></tr>
        <tr><td>confidence_level</td><td>float</td><td>Livello di confidenza</td></tr>
        <tr><td>generated_at</td><td>datetime</td><td>Timestamp generazione</td></tr>
    </table>
    """

    model_id: str = Field(..., description="ID del modello VAR utilizzato")
    forecast_timestamps: List[str] = Field(..., description="Timestamp delle previsioni")
    forecasts: Dict[str, List[float]] = Field(
        ..., description="Previsioni per ogni variabile del sistema"
    )
    lower_bounds: Optional[Dict[str, List[float]]] = Field(
        None, description="Limiti inferiori per variabile"
    )
    upper_bounds: Optional[Dict[str, List[float]]] = Field(
        None, description="Limiti superiori per variabile"
    )
    confidence_level: Optional[float] = Field(None, description="Livello di confidenza")
    generated_at: datetime = Field(..., description="Timestamp di generazione")


class ErrorResponse(BaseModel):
    """
    Modello standardizzato per le risposte di errore dell'API.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>error</td><td>str</td><td>Tipo/codice dell'errore</td></tr>
        <tr><td>message</td><td>str</td><td>Messaggio descrittivo dell'errore</td></tr>
        <tr><td>details</td><td>Dict</td><td>Dettagli aggiuntivi (opzionale)</td></tr>
    </table>

    <h4>Esempio:</h4>
    <pre><code>
    {
        "error": "ValidationError",
        "message": "I parametri di input non sono validi",
        "details": {
            "field": "order.p",
            "constraint": "deve essere tra 0 e 5"
        }
    }
    </code></pre>
    """

    error: str = Field(..., description="Tipo o codice dell'errore")
    message: str = Field(..., description="Messaggio descrittivo dell'errore")
    details: Optional[Dict[str, Any]] = Field(None, description="Dettagli aggiuntivi sull'errore")


class ModelListResponse(BaseModel):
    """
    Risposta per l'endpoint di elenco modelli.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>models</td><td>List[ModelInfo]</td><td>Lista di informazioni sui modelli</td></tr>
        <tr><td>total</td><td>int</td><td>Numero totale di modelli</td></tr>
    </table>
    """

    models: List[ModelInfo] = Field(
        ..., description="Lista di informazioni sui modelli disponibili"
    )
    total: int = Field(..., description="Numero totale di modelli nel sistema")


class AutoSelectionRequest(BaseModel):
    """
    Richiesta per la selezione automatica dei parametri ottimali del modello.

    Esegue una grid search per trovare la migliore combinazione di parametri
    basata sul criterio informativo specificato.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Vincoli</th></tr>
        <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td><td>Richiesto</td></tr>
        <tr><td>model_type</td><td>str</td><td>Tipo di modello</td><td>arima, sarima, sarimax</td></tr>
        <tr><td>max_models</td><td>int</td><td>Numero massimo di modelli da testare</td><td>1-200</td></tr>
        <tr><td>information_criterion</td><td>str</td><td>Criterio di selezione</td><td>aic, bic, hqic</td></tr>
        <tr><td>exogenous_data</td><td>ExogenousData</td><td>Variabili esogene</td><td>Solo per SARIMAX</td></tr>
    </table>

    <h4>Esempio:</h4>
    <pre><code>
    {
        "data": {...},
        "model_type": "sarima",
        "max_models": 100,
        "information_criterion": "aic"
    }
    </code></pre>
    """

    data: TimeSeriesData = Field(..., description="Dati della serie temporale")
    model_type: str = Field(..., description="Tipo di modello per la selezione automatica")
    exogenous_data: Optional[ExogenousData] = Field(
        None, description="Variabili esogene per SARIMAX"
    )
    max_models: Optional[int] = Field(
        default=50,
        ge=1,
        le=200,
        description="Numero massimo di combinazioni di parametri da testare",
    )
    information_criterion: str = Field(
        default="aic", description="Criterio informativo per la selezione del modello migliore"
    )
    # Parametri per grid search (usati da /training/auto-select endpoint)
    max_p: Optional[int] = Field(default=3, ge=0, le=10, description="Valore massimo per p (AR)")
    max_d: Optional[int] = Field(default=2, ge=0, le=3, description="Valore massimo per d (differencing)")
    max_q: Optional[int] = Field(default=3, ge=0, le=10, description="Valore massimo per q (MA)")
    seasonal: Optional[bool] = Field(default=False, description="Se considerare componente stagionale")
    seasonal_period: Optional[int] = Field(default=12, ge=2, description="Periodo stagionale (es. 12 per mensile)")
    criterion: Optional[str] = Field(default="aic", description="Criterio di selezione: aic o bic")

    @validator("model_type")
    def validate_model_type(cls, v):
        """Valida il tipo di modello per la selezione automatica."""
        valid_types = ["arima", "sarima", "sarimax"]
        if v.lower() not in valid_types:
            raise ValueError(f"model_type deve essere uno tra: {', '.join(valid_types)}")
        return v.lower()

    @validator("exogenous_data")
    def validate_exogenous(cls, v, values):
        """Valida la coerenza delle variabili esogene per la selezione automatica."""
        model_type = values.get("model_type", "").lower()

        if model_type == "sarimax" and v is None:
            raise ValueError(
                "exogenous_data è obbligatorio per la selezione automatica di modelli SARIMAX"
            )
        if model_type in ["arima", "sarima"] and v is not None:
            raise ValueError(
                f"exogenous_data non è consentito per la selezione automatica di modelli {model_type.upper()}"
            )

        return v

    @validator("information_criterion")
    def validate_ic(cls, v):
        """Valida il criterio informativo."""
        valid_criteria = ["aic", "bic", "hqic"]
        if v.lower() not in valid_criteria:
            raise ValueError(
                f"information_criterion deve essere uno tra: {', '.join(valid_criteria)}"
            )
        return v.lower()


class AutoSelectionResult(BaseModel):
    """
    Risultato della selezione automatica dei parametri del modello.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>best_model_id</td><td>str</td><td>ID del modello con i migliori parametri</td></tr>
        <tr><td>best_parameters</td><td>Dict</td><td>Parametri ottimali trovati</td></tr>
        <tr><td>best_score</td><td>float</td><td>Miglior valore del criterio informativo</td></tr>
        <tr><td>all_results</td><td>List[Dict]</td><td>Risultati di tutti i modelli testati</td></tr>
        <tr><td>selection_time</td><td>float</td><td>Tempo impiegato per la selezione (secondi)</td></tr>
    </table>
    """

    best_model_id: str = Field(..., description="ID del modello con i parametri ottimali")
    best_parameters: Dict[str, Any] = Field(..., description="Migliori parametri trovati")
    best_score: float = Field(
        ..., description="Valore del criterio informativo per il modello migliore"
    )
    all_results: List[Dict[str, Any]] = Field(
        ..., description="Risultati completi di tutti i modelli testati"
    )
    selection_time: float = Field(..., description="Tempo impiegato per la selezione in secondi")


class ModelDiagnosticsRequest(BaseModel):
    """
    Richiesta per la generazione di diagnostiche avanzate del modello.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>include_residuals</td><td>bool</td><td>Include analisi dettagliata residui</td><td>true</td></tr>
        <tr><td>include_acf_pacf</td><td>bool</td><td>Include grafici ACF/PACF</td><td>true</td></tr>
    </table>
    """

    include_residuals: bool = Field(
        default=True, description="Include l'analisi statistica dettagliata dei residui"
    )
    include_acf_pacf: bool = Field(
        default=True,
        description="Include i grafici ACF (autocorrelazione) e PACF (autocorrelazione parziale)",
    )


class ModelDiagnostics(BaseModel):
    """
    Risultati completi delle diagnostiche del modello.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello analizzato</td></tr>
        <tr><td>residual_stats</td><td>Dict[str, float]</td><td>Statistiche descrittive dei residui</td></tr>
        <tr><td>normality_test</td><td>Dict[str, float]</td><td>Test di normalità (Jarque-Bera)</td></tr>
        <tr><td>ljung_box_test</td><td>Dict[str, float]</td><td>Test di autocorrelazione dei residui</td></tr>
        <tr><td>heteroscedasticity_test</td><td>Dict[str, float]</td><td>Test di eteroschedasticità</td></tr>
        <tr><td>acf_values</td><td>List[float]</td><td>Valori della funzione di autocorrelazione</td></tr>
        <tr><td>pacf_values</td><td>List[float]</td><td>Valori della funzione di autocorrelazione parziale</td></tr>
    </table>
    """

    model_id: str = Field(..., description="ID del modello diagnosticato")
    residual_stats: Optional[Dict[str, float]] = Field(None, description="Statistiche dei residui")
    normality_test: Optional[Dict[str, float]] = Field(None, description="Test di normalità")
    ljung_box_test: Optional[Dict[str, float]] = Field(None, description="Test di Ljung-Box")
    heteroscedasticity_test: Optional[Dict[str, float]] = Field(
        None, description="Test di eteroschedasticità"
    )
    acf_values: Optional[List[float]] = Field(None, description="Valori ACF")
    pacf_values: Optional[List[float]] = Field(None, description="Valori PACF")


class ReportGenerationRequest(BaseModel):
    """
    Richiesta per la generazione di un report completo del modello.

    <h4>Parametri:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>report_title</td><td>str</td><td>Titolo personalizzato del report</td><td>Auto-generato</td></tr>
        <tr><td>output_filename</td><td>str</td><td>Nome del file (senza estensione)</td><td>Auto-generato</td></tr>
        <tr><td>format_type</td><td>str</td><td>Formato di output</td><td>html</td></tr>
        <tr><td>include_diagnostics</td><td>bool</td><td>Include sezione diagnostiche</td><td>true</td></tr>
        <tr><td>include_forecast</td><td>bool</td><td>Include sezione previsioni</td><td>true</td></tr>
        <tr><td>forecast_steps</td><td>int</td><td>Numero di passi di previsione</td><td>12</td></tr>
    </table>

    <h4>Formati Supportati:</h4>
    - html: Report interattivo con grafici dinamici
    - pdf: Report per stampa e condivisione
    - docx: Report editabile in Microsoft Word

    <h4>Esempio:</h4>
    <pre><code>
    {
        "report_title": "Analisi Vendite Q4 2024",
        "output_filename": "vendite_q4_sarima",
        "format_type": "html",
        "include_diagnostics": true,
        "include_forecast": true,
        "forecast_steps": 24
    }
    </code></pre>
    """

    report_title: Optional[str] = Field(
        default=None, description="Titolo personalizzato per il report"
    )
    output_filename: Optional[str] = Field(
        default=None, description="Nome personalizzato per il file di output (senza estensione)"
    )
    format_type: str = Field(default="html", description="Formato di output del report")
    include_diagnostics: bool = Field(
        default=True, description="Include la sezione di analisi diagnostiche"
    )
    include_forecast: bool = Field(default=True, description="Include la sezione di previsioni")
    forecast_steps: int = Field(
        default=12, ge=1, le=100, description="Numero di passi futuri per le previsioni"
    )

    @validator("format_type")
    def validate_format_type(cls, v):
        """Valida il formato di output del report."""
        valid_formats = ["html", "pdf", "docx"]
        if v.lower() not in valid_formats:
            raise ValueError(f"format_type deve essere uno tra: {', '.join(valid_formats)}")
        return v.lower()

    @validator("output_filename")
    def validate_output_filename(cls, v):
        """Valida il nome del file di output."""
        if v is not None:
            # Rimuove caratteri non validi per i nomi file
            import re

            if not re.match(r"^[a-zA-Z0-9_\-\s]+$", v):
                raise ValueError(
                    "output_filename può contenere solo lettere, numeri, underscore, "
                    "trattini e spazi"
                )
        return v


class ReportGenerationResponse(BaseModel):
    """
    Risposta per la generazione di report.

    <h4>Campi:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello utilizzato</td></tr>
        <tr><td>report_path</td><td>str</td><td>Percorso completo del file generato</td></tr>
        <tr><td>format_type</td><td>str</td><td>Formato del report generato</td></tr>
        <tr><td>generation_time</td><td>float</td><td>Tempo di generazione in secondi</td></tr>
        <tr><td>file_size_mb</td><td>float</td><td>Dimensione del file in MB</td></tr>
        <tr><td>download_url</td><td>str</td><td>URL per il download del report</td></tr>
    </table>
    """

    model_id: str = Field(..., description="ID del modello utilizzato per il report")
    report_path: str = Field(..., description="Percorso completo del file di report generato")
    format_type: str = Field(..., description="Formato del report generato")
    generation_time: float = Field(..., description="Tempo impiegato per la generazione in secondi")
    file_size_mb: Optional[float] = Field(None, description="Dimensione del file in megabytes")
    download_url: Optional[str] = Field(None, description="URL relativo per il download del report")
