#!/usr/bin/env python3
"""
Funzioni di utilità per gli esempi.
"""

from pathlib import Path
import os

def get_output_path(subdir: str, filename: str) -> Path:
    """
    Ottieni il percorso output corretto, che tu stia eseguendo da examples/ o dalla radice del progetto.
    
    Args:
        subdir: Sottodirectory sotto outputs/ ('plots', 'models', 'reports')
        filename: Nome del file
        
    Returns:
        Oggetto Path che punta alla posizione corretta
    """
    # Controlla se siamo nella directory examples/
    current_dir = Path.cwd()
    if current_dir.name == 'examples':
        # Siamo in examples/, quindi sali di un livello
        project_root = current_dir.parent
    else:
        # Siamo nella radice del progetto o altrove
        project_root = current_dir
    
    # Crea il percorso output
    output_path = project_root / 'outputs' / subdir / filename
    
    # Assicurati che la directory esista
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path

def get_plots_path(filename: str) -> Path:
    """Ottieni percorso per directory grafici."""
    return get_output_path('plots', filename)

def get_models_path(filename: str) -> Path:
    """Ottieni percorso per directory modelli."""
    return get_output_path('models', filename)

def get_reports_path(filename: str) -> Path:
    """Ottieni percorso per directory report."""
    return get_output_path('reports', filename)