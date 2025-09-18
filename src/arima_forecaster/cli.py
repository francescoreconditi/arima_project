# ============================================
# FILE DI IMPLEMENTAZIONE CLI
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: CLI entry point per arima-forecaster
# ============================================

"""
CLI Module per ARIMA Forecaster

Fornisce interfaccia command line per operazioni principali:
- Training modelli
- Forecasting
- Valutazione modelli
- Report generation
- Model management
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from arima_forecaster import __version__
from arima_forecaster.core import ARIMAForecaster, SARIMAForecaster, VARForecaster
from arima_forecaster.data import DataLoader, TimeSeriesPreprocessor
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils.logger import setup_logger

console = Console()
logger = logging.getLogger(__name__)


def setup_cli_logger(verbose: bool = False) -> None:
    """Setup logger per CLI"""
    level = "DEBUG" if verbose else "INFO"
    setup_logger(name="arima_forecaster_cli", level=level)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Carica configurazione da file JSON"""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def train_command(args: argparse.Namespace) -> None:
    """Command per training modelli"""
    console.print(f"[green]🚀 Training modello {args.model}[/green]")

    try:
        # Carica dati
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            progress.add_task("Caricamento dati...", total=None)
            loader = DataLoader()
            data = loader.load_data(args.data)

        console.print(f"✅ Dati caricati: {len(data)} righe")

        # Preprocessing
        if args.preprocess:
            preprocessor = TimeSeriesPreprocessor()
            data = preprocessor.preprocess_pipeline(data)
            console.print("✅ Preprocessing completato")

        # Seleziona modello
        config = load_config(args.config)
        model_config = config.get(args.model, {})

        if args.model.lower() == "arima":
            order = model_config.get("order", (1, 1, 1))
            model = ARIMAForecaster(order=order)
        elif args.model.lower() == "sarima":
            order = model_config.get("order", (1, 1, 1))
            seasonal_order = model_config.get("seasonal_order", (1, 1, 1, 12))
            model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
        elif args.model.lower() == "var":
            model = VARForecaster()
        else:
            console.print(f"[red]❌ Modello {args.model} non supportato[/red]")
            return

        # Training
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            progress.add_task("Training in corso...", total=None)
            model.fit(data)

        # Salva modello
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_path))

        console.print(f"✅ Modello salvato: {output_path}")

        # Valutazione
        if args.evaluate:
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(model, data)

            table = Table(title="Metriche Valutazione")
            table.add_column("Metrica", style="cyan")
            table.add_column("Valore", style="green")

            for metric, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(metric, f"{value:.4f}")
                else:
                    table.add_row(metric, str(value))

            console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Errore durante training: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


def forecast_command(args: argparse.Namespace) -> None:
    """Command per forecasting"""
    console.print(f"[green]🔮 Forecasting {args.steps} periodi[/green]")

    try:
        # Carica modello
        if args.model.lower() == "arima":
            model = ARIMAForecaster.load_model(args.model_path)
        elif args.model.lower() == "sarima":
            model = SARIMAForecaster.load_model(args.model_path)
        elif args.model.lower() == "var":
            model = VARForecaster.load_model(args.model_path)
        else:
            console.print(f"[red]❌ Modello {args.model} non supportato[/red]")
            return

        # Genera forecast
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            progress.add_task("Generazione forecast...", total=None)
            forecast = model.forecast(steps=args.steps)

        # Salva risultati
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(forecast, pd.DataFrame):
                forecast.to_csv(output_path, index=True)
            else:
                pd.Series(forecast).to_csv(output_path, index=True)

            console.print(f"✅ Forecast salvato: {output_path}")

        # Mostra preview
        console.print("\n📊 Preview Forecast:")
        if isinstance(forecast, pd.DataFrame):
            console.print(forecast.head(10).to_string())
        else:
            console.print(pd.Series(forecast).head(10).to_string())

    except Exception as e:
        console.print(f"[red]❌ Errore durante forecasting: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


def evaluate_command(args: argparse.Namespace) -> None:
    """Command per valutazione modelli"""
    console.print("[green]📊 Valutazione modello[/green]")

    try:
        # Carica modello e dati
        data = DataLoader().load_data(args.data)

        if args.model.lower() == "arima":
            model = ARIMAForecaster.load_model(args.model_path)
        elif args.model.lower() == "sarima":
            model = SARIMAForecaster.load_model(args.model_path)
        elif args.model.lower() == "var":
            model = VARForecaster.load_model(args.model_path)
        else:
            console.print(f"[red]❌ Modello {args.model} non supportato[/red]")
            return

        # Valutazione
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, data)

        # Mostra risultati
        table = Table(title="Metriche Valutazione Modello")
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", style="green")

        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))

        console.print(table)

        # Salva risultati
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)

            console.print(f"✅ Metriche salvate: {output_path}")

    except Exception as e:
        console.print(f"[red]❌ Errore durante valutazione: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


def version_command(args: argparse.Namespace) -> None:
    """Mostra versione"""
    console.print(f"[green]ARIMA Forecaster v{__version__}[/green]")


def main() -> None:
    """Entry point principale CLI"""
    parser = argparse.ArgumentParser(
        prog="arima-forecast",
        description="ARIMA Forecaster CLI - Forecasting serie temporali",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  arima-forecast train --data data.csv --model sarima --output model.pkl
  arima-forecast forecast --model-path model.pkl --steps 30 --output forecast.csv
  arima-forecast evaluate --model-path model.pkl --data test.csv
  arima-forecast version
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Output verbose")

    parser.add_argument("--config", type=str, help="File configurazione JSON")

    subparsers = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Training modello")
    train_parser.add_argument("--data", "-d", required=True, help="File dati CSV")
    train_parser.add_argument(
        "--model", "-m", choices=["arima", "sarima", "var"], default="sarima", help="Tipo modello"
    )
    train_parser.add_argument("--output", "-o", required=True, help="Path output modello")
    train_parser.add_argument("--preprocess", action="store_true", help="Applica preprocessing")
    train_parser.add_argument(
        "--evaluate", action="store_true", help="Valuta modello dopo training"
    )

    # Command: forecast
    forecast_parser = subparsers.add_parser("forecast", help="Genera forecast")
    forecast_parser.add_argument("--model-path", "-p", required=True, help="Path modello salvato")
    forecast_parser.add_argument(
        "--model", "-m", choices=["arima", "sarima", "var"], default="sarima", help="Tipo modello"
    )
    forecast_parser.add_argument(
        "--steps", "-s", type=int, default=30, help="Numero periodi forecast"
    )
    forecast_parser.add_argument("--output", "-o", help="File output forecast CSV")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Valuta modello")
    eval_parser.add_argument("--model-path", "-p", required=True, help="Path modello salvato")
    eval_parser.add_argument(
        "--model", "-m", choices=["arima", "sarima", "var"], default="sarima", help="Tipo modello"
    )
    eval_parser.add_argument("--data", "-d", required=True, help="File dati test CSV")
    eval_parser.add_argument("--output", "-o", help="File output metriche JSON")

    # Command: version
    version_parser = subparsers.add_parser("version", help="Mostra versione")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_cli_logger(args.verbose)

    # Execute command
    if args.command == "train":
        train_command(args)
    elif args.command == "forecast":
        forecast_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "version":
        version_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
