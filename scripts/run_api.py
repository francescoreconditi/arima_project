"""
Script per eseguire il server API di ARIMA Forecaster.
"""

import argparse
import sys
from pathlib import Path

# Aggiungi src al percorso
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arima_forecaster.api.main import create_app
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Esegui API ARIMA Forecaster")
    parser.add_argument("--host", default="127.0.0.1", help="Indirizzo host")
    parser.add_argument("--port", type=int, default=8000, help="Numero porta")
    parser.add_argument("--models-path", default="models", help="Percorso per memorizzare i modelli")
    parser.add_argument("--reload", action="store_true", help="Abilita auto-ricaricamento")
    parser.add_argument("--workers", type=int, default=1, help="Numero di processi worker")
    
    args = parser.parse_args()
    
    # Crea app FastAPI
    app = create_app(model_storage_path=args.models_path)
    
    # Esegui server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()