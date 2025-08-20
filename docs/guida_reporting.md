# Guida Completa al Sistema di Reporting Quarto

La libreria ARIMA Forecaster include un sistema avanzato di reporting basato su Quarto che genera automaticamente report professionali per l'analisi di modelli ARIMA e SARIMA.

## Indice

1. [Introduzione](#introduzione)
2. [Installazione e Setup](#installazione-e-setup)
3. [Report Base](#report-base)
4. [Report Avanzati](#report-avanzati)
5. [Personalizzazione](#personalizzazione)
6. [Template e Struttura](#template-e-struttura)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduzione

Il sistema di reporting Quarto trasforma automaticamente i risultati dei tuoi modelli ARIMA/SARIMA in report professionali pronti per la presentazione. I report includono:

- **Analisi automatiche** con interpretazione intelligente dei risultati
- **Visualizzazioni integrate** con grafici professionali
- **Export multi-formato** (HTML, PDF, DOCX)
- **Template personalizzabili** per diversi casi d'uso
- **Report comparativi** per confrontare modelli multipli

### Vantaggi Chiave

✅ **Automatico**: Generazione report senza intervento manuale  
✅ **Professionale**: Output di qualità publication-ready  
✅ **Intelligente**: Interpretazione automatica dei risultati  
✅ **Flessibile**: Template e formati personalizzabili  
✅ **Riproducibile**: Metadata completi per audit trail  

## Installazione e Setup

### 1. Dipendenze Python

```bash
# Installa dipendenze reporting con UV (raccomandato)
uv sync --extra reports

# Oppure con pip tradizionale
pip install -e ".[reports]"
```

Le dipendenze principali installate:
- **quarto**: Engine di rendering documenti
- **jupyter**: Supporto esecuzione codice
- **nbformat**: Gestione formato notebook

### 2. Quarto CLI

Il rendering dei report richiede Quarto CLI installato nel sistema:

#### Windows
```powershell
# Con Chocolatey
choco install quarto

# Con Scoop
scoop install quarto

# Download diretto da https://quarto.org/docs/get-started/
```

#### macOS
```bash
# Con Homebrew
brew install --cask quarto

# Download diretto da https://quarto.org/docs/get-started/
```

#### Linux
```bash
# Ubuntu/Debian
curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb
sudo gdebi quarto-linux-amd64.deb

# Download altri OS: https://quarto.org/docs/get-started/
```

### 3. Dipendenze Opzionali per Export Avanzato

#### PDF Export (LaTeX)
```bash
# Windows (MiKTeX)
winget install MiKTeX.MiKTeX

# macOS (MacTeX)
brew install --cask mactex

# Linux (TeX Live)
sudo apt-get install texlive-full  # Ubuntu/Debian
```

#### DOCX Export (Pandoc)
```bash
# Installato automaticamente con Quarto
# Per installazione separata:
# Windows: winget install --id JohnMacFarlane.Pandoc
# macOS: brew install pandoc
# Linux: sudo apt install pandoc
```

### 4. Verifica Installazione

```python
# Test rapido funzionalità reporting
from arima_forecaster import ARIMAForecaster
import pandas as pd
import numpy as np

# Genera dati test
dates = pd.date_range('2020-01-01', periods=50, freq='M')
values = 100 + np.cumsum(np.random.normal(0, 5, len(dates)))
data = pd.Series(values, index=dates)

# Test modello e report
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(data)

try:
    report_path = model.generate_report(
        report_title="Test Installation",
        output_filename="test_report",
        format_type="html"
    )
    print(f"✅ Reporting funzionale: {report_path}")
except ImportError:
    print("❌ Dipendenze mancanti - run: uv sync --extra reports")
except Exception as e:
    print(f"⚠️ Possibile problema setup: {e}")
```

## Report Base

### Report Singolo Modello

Il modo più semplice per generare un report:

```python
from arima_forecaster import ARIMAForecaster
import pandas as pd

# Carica dati e addestra modello
data = pd.read_csv('vendite.csv', index_col='data', parse_dates=True)
model = ARIMAForecaster(order=(2, 1, 2))
model.fit(data['vendite'])

# Genera report base HTML
report_path = model.generate_report(
    report_title="Analisi Vendite Mensili",
    output_filename="vendite_analysis",
    format_type="html"
)

print(f"Report disponibile: {report_path}")
```

### Parametri Base

```python
report_path = model.generate_report(
    # Contenuto
    report_title="Titolo Personalizzato",           # Titolo del report
    include_diagnostics=True,                       # Include test diagnostici
    include_forecast=True,                          # Include sezione forecasting
    forecast_steps=24,                              # Passi di previsione
    
    # Output
    output_filename="custom_report",                # Nome file output
    format_type="html",                            # Formato: html, pdf, docx
    
    # Visualizzazioni (opzionale)
    plots_data=None                                # Dictionary con path grafici
)
```

### Report con Visualizzazioni Custom

```python
from arima_forecaster.visualization import ForecastPlotter

# Genera visualizzazioni
plotter = ForecastPlotter()
forecast_result = model.forecast(steps=12, confidence_intervals=True)

# Salva grafici per integrazione
plots_data = {
    'forecast': plotter.plot_forecast(
        actual=data['vendite'],
        forecast=forecast_result['forecast'],
        confidence_intervals=forecast_result['confidence_intervals'],
        title="Forecast Vendite 12 Mesi",
        save_path="outputs/plots/custom_forecast.png"
    ),
    
    'residuals': plotter.plot_residuals(
        residuals=model.fitted_model.resid,
        title="Analisi Residui Dettagliata",
        save_path="outputs/plots/custom_residuals.png"
    ),
    
    'acf_pacf': plotter.plot_acf_pacf(
        series=data['vendite'],
        lags=20,
        save_path="outputs/plots/acf_pacf_analysis.png"
    )
}

# Report con grafici custom
enhanced_report = model.generate_report(
    plots_data=plots_data,
    report_title="Analisi Completa con Visualizzazioni Custom",
    output_filename="enhanced_analysis",
    format_type="html",
    include_diagnostics=True,
    include_forecast=True,
    forecast_steps=36
)
```

## Report Avanzati

### Report SARIMA con Decomposizione

I modelli SARIMA hanno funzionalità aggiuntive per l'analisi stagionale:

```python
from arima_forecaster import SARIMAForecaster

# Modello stagionale
sarima_model = SARIMAForecaster(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)  # Stagionalità annuale
)
sarima_model.fit(data['vendite'])

# Report con decomposizione stagionale
sarima_report = sarima_model.generate_report(
    report_title="Analisi SARIMA - Componenti Stagionali",
    output_filename="sarima_seasonal_analysis", 
    format_type="html",
    include_diagnostics=True,
    include_forecast=True,
    include_seasonal_decomposition=True,  # 🔥 Specifico SARIMA
    forecast_steps=48
)
```

### Report Comparativo Multi-Modello

Confronta automaticamente diversi modelli:

```python
from arima_forecaster.reporting import QuartoReportGenerator
from arima_forecaster.evaluation import ModelEvaluator

# Addestra modelli diversi
models = {
    'ARIMA(1,1,1)': ARIMAForecaster(order=(1, 1, 1)),
    'ARIMA(2,1,2)': ARIMAForecaster(order=(2, 1, 2)), 
    'SARIMA(1,1,1)x(1,1,1,12)': SARIMAForecaster(
        order=(1, 1, 1), 
        seasonal_order=(1, 1, 1, 12)
    )
}

# Addestra e valuta tutti i modelli
evaluator = ModelEvaluator()
models_results = {}

for name, model in models.items():
    model.fit(data['vendite'])
    predictions = model.predict()
    metrics = evaluator.calculate_forecast_metrics(data['vendite'], predictions)
    
    models_results[name] = {
        'model_type': 'SARIMA' if 'SARIMA' in name else 'ARIMA',
        'order': model.order,
        'seasonal_order': getattr(model, 'seasonal_order', None),
        'model_info': model.get_model_info(),
        'metrics': metrics,
        'training_data': {
            'observations': len(data['vendite']),
            'start_date': str(data.index.min()),
            'end_date': str(data.index.max())
        }
    }

# Genera report comparativo
generator = QuartoReportGenerator()
comparison_report = generator.create_comparison_report(
    models_results=models_results,
    report_title="Studio Comparativo: Selezione Modello Ottimale",
    output_filename="models_comparison_study",
    format_type="html"
)

print(f"Report comparativo: {comparison_report}")
```

### Export Multi-Formato

Genera lo stesso report in formati diversi:

```python
base_config = {
    'report_title': "Executive Summary - Q4 Forecast",
    'include_diagnostics': True,
    'include_forecast': True,
    'forecast_steps': 12,
    'plots_data': plots_data
}

# HTML per condivisione web
html_report = model.generate_report(
    **base_config,
    output_filename="q4_forecast_web",
    format_type="html"
)

# PDF per presentazioni executive
try:
    pdf_report = model.generate_report(
        **base_config,
        output_filename="q4_forecast_executive",
        format_type="pdf"
    )
    print(f"📄 PDF Executive: {pdf_report}")
except Exception as e:
    print(f"⚠️ PDF requires LaTeX: {e}")

# DOCX per editing collaborativo
try:
    docx_report = model.generate_report(
        **base_config,
        output_filename="q4_forecast_editable",
        format_type="docx"  
    )
    print(f"📝 DOCX Editable: {docx_report}")
except Exception as e:
    print(f"⚠️ DOCX requires pandoc: {e}")
```

## Personalizzazione

### Personalizzazione Parametri Report

```python
# Report altamente personalizzato
custom_report = model.generate_report(
    # Metadata
    report_title="Previsioni Vendite E-commerce - Analisi Dettagliata",
    
    # Contenuto specifico
    include_diagnostics=True,          # Test residui, normalità, autocorrelazione
    include_forecast=True,             # Sezione forecasting con intervalli
    forecast_steps=52,                 # 52 settimane (1 anno)
    
    # File output
    output_filename="ecommerce_weekly_forecast_2024",
    format_type="html",
    
    # Visualizzazioni integrate
    plots_data={
        'trend_analysis': 'outputs/plots/trend_decomposition.png',
        'seasonal_patterns': 'outputs/plots/seasonal_analysis.png', 
        'forecast_confidence': 'outputs/plots/forecast_intervals.png',
        'residuals_diagnostics': 'outputs/plots/residuals_complete.png',
        'model_comparison': 'outputs/plots/models_performance.png'
    }
)
```

### Template Personalizzati (Avanzato)

Per personalizzazioni estreme, puoi modificare i template Quarto:

```python
from arima_forecaster.reporting import QuartoReportGenerator
from pathlib import Path

# Crea template personalizzato
custom_template = Path("templates/custom_arima_template.qmd")
custom_template.parent.mkdir(exist_ok=True)

# Template base personalizzato
custom_qmd_content = """---
title: "{{title}}"
subtitle: "Analisi Personalizzata Serie Temporali"
author: "La Tua Azienda"
date: "{{date}}"
format:
  html:
    theme: flatly
    toc: true
    toc-depth: 2
    code-fold: show
    fig-width: 12
    fig-height: 8
  pdf:
    geometry: margin=0.8in
    toc: true
---

# Executive Summary Personalizzato

{{executive_summary}}

# Analisi Tecnica Approfondita

{{technical_analysis}}

# Raccomandazioni Business

{{business_recommendations}}
"""

with open(custom_template, 'w') as f:
    f.write(custom_qmd_content)

# Usa template personalizzato (implementazione futura)
# generator = QuartoReportGenerator(custom_template=custom_template)
```

## Template e Struttura

### Struttura Report Standard

Ogni report generato segue questa struttura:

```
📄 Report ARIMA/SARIMA
├── 📋 Executive Summary
│   ├── Panoramica modello e parametri
│   ├── Metriche chiave (MAE, RMSE, MAPE, R², AIC, BIC)
│   └── Data analisi e informazioni tecniche
│
├── 🔧 Metodologia e Approccio  
│   ├── Processo selezione modello
│   ├── Parametri con interpretazione
│   └── Preprocessing applicato
│
├── 📊 Analisi Risultati
│   ├── Performance con grafici confronto
│   ├── Diagnostica residui con test
│   └── Interpretazione automatica
│
├── 📈 Visualizzazioni
│   ├── Grafici forecast con intervalli
│   ├── Analisi residui multi-pannello
│   └── Decomposizione stagionale (SARIMA)
│
├── 💡 Raccomandazioni
│   ├── Interpretazione performance
│   ├── Suggerimenti operativi
│   └── Avvisi problemi potenziali
│
└── 🔍 Dettagli Tecnici
    ├── Configurazione completa
    ├── Ambiente esecuzione
    └── Metadata riproducibilità
```

### Sezioni Report Comparativo

I report comparativi includono sezioni aggiuntive:

```
📊 Report Comparativo Multi-Modello
├── 🏆 Panoramica Modelli
│   └── Tabella riassuntiva tutti i modelli
│
├── 📈 Confronto Performance  
│   ├── Grafici metriche comparative
│   └── Analisi statistica differenze
│
├── 🥇 Ranking Modelli
│   ├── Classifica per performance
│   ├── Score composito multi-criterio
│   └── Analisi trade-off
│
├── 🎯 Raccomandazioni Finali
│   ├── Modello raccomandato
│   ├── Modelli alternativi
│   └── Considerazioni implementative
│
└── 📋 Dettagli Tecnici Comparativi
    └── Configurazioni complete tutti i modelli
```

## Best Practices

### 1. Organizzazione Output

Struttura consigliata per output:

```
project/
├── outputs/
│   ├── models/           # Modelli salvati (.pkl)
│   ├── plots/            # Grafici per report (.png, .svg)
│   └── reports/          # Report generati
│       ├── html/         # Report HTML
│       ├── pdf/          # Report PDF  
│       └── docx/         # Report DOCX
├── data/
│   ├── raw/              # Dati originali
│   └── processed/        # Dati preprocessati
└── scripts/
    └── generate_reports.py # Script automazione report
```

### 2. Naming Convention

Usa naming consistenti per tracciabilità:

```python
# Pattern: {dataset}_{model}_{date}_{version}
report_names = [
    "vendite_arima211_20240315_v1",
    "vendite_sarima111x1112_20240315_v1", 
    "traffico_comparison_study_20240315_v1"
]

# Include timestamp per versioning automatico
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"monthly_forecast_analysis_{timestamp}"
```

### 3. Gestione Errori Robusta

Implementa gestione errori completa:

```python
def generate_robust_report(model, **kwargs):
    """Genera report con gestione errori robusta."""
    
    try:
        # Verifica dipendenze
        from arima_forecaster.reporting import QuartoReportGenerator
        
        # Genera report
        report_path = model.generate_report(**kwargs)
        print(f"✅ Report generato: {report_path}")
        return report_path
        
    except ImportError as e:
        print("❌ Dipendenze mancanti:")
        print("   uv sync --extra reports")
        print("   Installa Quarto CLI: https://quarto.org/docs/get-started/")
        return None
        
    except FileNotFoundError as e:
        if "quarto" in str(e).lower():
            print("❌ Quarto CLI non trovato nel PATH")
            print("   Installa da: https://quarto.org/docs/get-started/")
        else:
            print(f"❌ File non trovato: {e}")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore rendering Quarto: {e}")
        print("   Controlla log dettagliato per debug")
        return None
        
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")
        import traceback
        traceback.print_exc()
        return None

# Uso robusto
report_path = generate_robust_report(
    model,
    report_title="Analisi Q4",
    format_type="html"
)
```

### 4. Automazione Reporting

Script per generazione automatica:

```python
#!/usr/bin/env python3
"""
Script automazione reporting per pipeline CI/CD
"""

import pandas as pd
from pathlib import Path
from arima_forecaster import ARIMAForecaster, SARIMAForecaster
from arima_forecaster.reporting import QuartoReportGenerator
from datetime import datetime

def automated_reporting_pipeline(data_path: str, output_dir: str):
    """Pipeline automatica per generazione report."""
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Carica dati
    data = pd.read_csv(data_path, index_col='date', parse_dates=True)
    
    # Modelli da testare
    models_config = {
        'ARIMA_conservative': {'order': (1, 1, 1)},
        'ARIMA_balanced': {'order': (2, 1, 2)},
        'SARIMA_seasonal': {
            'order': (1, 1, 1), 
            'seasonal_order': (1, 1, 1, 12)
        }
    }
    
    reports_generated = []
    
    # Genera report per ogni modello
    for model_name, config in models_config.items():
        try:
            # Addestra modello
            if 'SARIMA' in model_name:
                model = SARIMAForecaster(**config)
            else:
                model = ARIMAForecaster(**config)
            
            model.fit(data.iloc[:, 0])  # Prima colonna
            
            # Genera report
            report_path = model.generate_report(
                report_title=f"Automated Analysis - {model_name}",
                output_filename=f"{model_name.lower()}_{timestamp}",
                format_type="html",
                include_diagnostics=True,
                include_forecast=True
            )
            
            reports_generated.append({
                'model': model_name,
                'path': report_path,
                'status': 'success'
            })
            
        except Exception as e:
            reports_generated.append({
                'model': model_name,
                'error': str(e),
                'status': 'failed'
            })
    
    # Report sommario
    print(f"\\n📊 Report Automation Summary [{timestamp}]")
    print("=" * 50)
    
    for result in reports_generated:
        if result['status'] == 'success':
            print(f"✅ {result['model']}: {result['path']}")
        else:
            print(f"❌ {result['model']}: {result['error']}")
    
    return reports_generated

# Esecuzione
if __name__ == "__main__":
    reports = automated_reporting_pipeline(
        data_path="data/processed/monthly_sales.csv",
        output_dir="outputs/reports/automated"
    )
```

## Troubleshooting

### Problemi Comuni e Soluzioni

#### 1. "ImportError: QuartoReportGenerator not available"

```bash
# Soluzione: Installa dipendenze reporting
uv sync --extra reports
# oppure
pip install -e ".[reports]"
```

#### 2. "FileNotFoundError: quarto not found"

```bash
# Soluzione: Installa Quarto CLI
# Windows
winget install --id RStudio.quarto

# macOS
brew install --cask quarto

# Linux - scarica da https://quarto.org/docs/get-started/
```

#### 3. "LaTeX Error" (PDF Export)

```bash
# Soluzione: Installa distribuzione LaTeX
# Windows - MiKTeX
winget install MiKTeX.MiKTeX

# macOS - MacTeX
brew install --cask mactex

# Linux - TeX Live
sudo apt-get install texlive-full
```

#### 4. "Pandoc Error" (DOCX Export)

Pandoc è incluso con Quarto, ma per installazione separata:

```bash
# Windows
winget install --id JohnMacFarlane.Pandoc

# macOS
brew install pandoc

# Linux
sudo apt install pandoc
```

#### 5. Report Vuoti o Incompleti

```python
# Debug: Verifica contenuto modello
model_info = model.get_model_info()
print("Model Info:", model_info)

if model_info.get('status') != 'fitted':
    print("❌ Modello non addestrato correttamente")
    model.fit(data)  # Ri-addestra
```

#### 6. Errori Rendering Template

```python
# Debug: Genera report minimale
try:
    minimal_report = model.generate_report(
        report_title="Debug Test",
        output_filename="debug_test",
        format_type="html",
        include_diagnostics=False,  # Disabilita diagnostici
        include_forecast=False      # Disabilita forecast
    )
    print(f"✅ Report minimale OK: {minimal_report}")
except Exception as e:
    print(f"❌ Errore anche con report minimale: {e}")
```

### Debug Avanzato

Per problemi persistenti, abilita logging dettagliato:

```python
import logging
from arima_forecaster.utils import setup_logger

# Abilita logging debug
logger = setup_logger('reporting_debug', level='DEBUG')

# Test con logging verboso
model.generate_report(
    report_title="Debug Report",
    output_filename="debug_verbose",
    format_type="html"
)
```

### Supporto e Community

Per supporto aggiuntivo:

- 📖 **Documentazione**: Consulta i file in `docs/`
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/tuonome/arima-forecaster/issues)
- 💬 **Community**: [GitHub Discussions](https://github.com/tuonome/arima-forecaster/discussions)
- 🎓 **Esempi**: Vedi `examples/quarto_reporting.py`

---

*Questa guida copre tutti gli aspetti del sistema di reporting. Per esempi pratici completi, consulta `examples/quarto_reporting.py` nel repository.*