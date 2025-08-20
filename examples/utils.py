#!/usr/bin/env python3
"""
Utility functions for examples.
"""

from pathlib import Path
import os

def get_output_path(subdir: str, filename: str) -> Path:
    """
    Get the correct output path, whether running from examples/ or project root.
    
    Args:
        subdir: Subdirectory under outputs/ ('plots', 'models', 'reports')
        filename: Name of the file
        
    Returns:
        Path object pointing to the correct location
    """
    # Check if we're in examples/ directory
    current_dir = Path.cwd()
    if current_dir.name == 'examples':
        # We're in examples/, so go up one level
        project_root = current_dir.parent
    else:
        # We're in project root or somewhere else
        project_root = current_dir
    
    # Create the output path
    output_path = project_root / 'outputs' / subdir / filename
    
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path

def get_plots_path(filename: str) -> Path:
    """Get path for plots directory."""
    return get_output_path('plots', filename)

def get_models_path(filename: str) -> Path:
    """Get path for models directory."""
    return get_output_path('models', filename)

def get_reports_path(filename: str) -> Path:
    """Get path for reports directory."""
    return get_output_path('reports', filename)