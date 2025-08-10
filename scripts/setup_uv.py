#!/usr/bin/env python3
"""
Script di configurazione per inizializzare il progetto con UV.
Crea l'ambiente virtuale e installa tutte le dipendenze.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: str, description: str = None) -> bool:
    """Esegue un comando shell e gestisce errori."""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            cmd.split(),
            check=True, 
            capture_output=True,
            text=True
        )
        print(f"✅ {description or cmd} completato")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Funzione principale per configurazione progetto."""
    
    print("🚀 ARIMA Forecaster - Configurazione con UV")
    print("=" * 50)
    
    # Controlla se UV è installato
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV è installato")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV non trovato. Installalo con:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   oppure: winget install --id=astral-sh.uv")
        sys.exit(1)
    
    # Vai alla directory del progetto
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 Directory progetto: {project_root}")
    
    # Sincronizza dipendenze
    if not run_command("uv sync --all-extras", "Sincronizzazione dipendenze"):
        sys.exit(1)
    
    # Installa pre-commit hooks
    if not run_command("uv run pre-commit install", "Installazione pre-commit hooks"):
        print("⚠️  Pre-commit non installato, continuando...")
    
    # Crea directory necessarie
    dirs_to_create = ["logs", "outputs", "outputs/models", "outputs/plots"]
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Directory creata: {dir_name}")
    
    # Verifica installazione
    print("\n🔍 Verifica installazione...")
    
    if not run_command("uv run python -c \"import arima_forecaster; print('Package importato correttamente')\"", 
                      "Test import package"):
        print("❌ Verifica fallita")
        sys.exit(1)
    
    # Esegui test rapidi
    if not run_command("uv run pytest tests/ -x --tb=short", "Test rapidi"):
        print("⚠️  Alcuni test sono falliti, ma l'installazione base è completa")
    
    print("\n🎉 Configurazione completata con successo!")
    print("\nComandi utili:")
    print("  make help          - Mostra tutti i comandi disponibili")  
    print("  uv run pytest     - Esegui test")
    print("  make examples      - Esegui esempi")
    print("  make format        - Formatta codice")
    print("  make lint          - Controlli qualità")
    
    # Attivazione ambiente virtuale
    venv_path = project_root / ".venv"
    if venv_path.exists():
        if sys.platform == "win32":
            activate_cmd = str(venv_path / "Scripts" / "activate.bat")
        else:
            activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
        
        print(f"\n💡 Per attivare l'ambiente virtuale:")
        print(f"   {activate_cmd}")


if __name__ == "__main__":
    main()