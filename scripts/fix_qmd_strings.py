#!/usr/bin/env python3
"""
Script per correggere automaticamente gli errori di stringhe nei file QMD generati.
Corregge le stringhe multi-linea non escapate che causano SyntaxError in Quarto.
"""

import os
import re
from pathlib import Path


def fix_qmd_string_literals(file_path: Path) -> bool:
    """
    Corregge gli errori di stringhe nei file QMD.

    Args:
        file_path: Path al file QMD da correggere

    Returns:
        bool: True se sono state fatte correzioni, False altrimenti
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Pattern problematici da correggere
        fixes = [
            # Corregge: print("
            # **Suggerimento per modelli SARIMAX:**")
            (
                r'print\("\s*\n\s*\*\*Suggerimento per modelli SARIMAX:\*\*"\)',
                'print("\\\\n**Suggerimento per modelli SARIMAX:**")',
            ),
            # Corregge: print(f"
            # **Numero totale di variabili esogene:** {len(exog_names)}")
            (
                r'print\(f"\s*\n\s*\*\*Numero totale di variabili esogene:\*\*',
                'print(f"\\\\n**Numero totale di variabili esogene:**',
            ),
            # Corregge: print("
            # **Importanza delle Variabili Esogene:**")
            (
                r'print\("\s*\n\s*\*\*Importanza delle Variabili Esogene:\*\*"\)',
                'print("\\\\n**Importanza delle Variabili Esogene:**")',
            ),
        ]

        changes_made = 0

        for pattern, replacement in fixes:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
                changes_made += len(matches)
                print(f"  Fixed {len(matches)} instances of pattern: {pattern[:50]}...")

        # Se sono state fatte modifiche, salva il file
        if changes_made > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[FIXED] Fixed {changes_made} issues in {file_path.name}")
            return True
        else:
            print(f"[OK] No issues found in {file_path.name}")
            return False

    except Exception as e:
        print(f"[ERROR] Error processing {file_path}: {e}")
        return False


def fix_all_qmd_files(reports_dir: str = "C:\\ZCS_PRG\\arima_project\\outputs\\reports") -> None:
    """
    Corregge tutti i file QMD nella directory dei report.

    Args:
        reports_dir: Path alla directory dei report
    """
    reports_path = Path(reports_dir)

    if not reports_path.exists():
        print(f"[ERROR] Directory non trovata: {reports_dir}")
        return

    print(f"[SEARCH] Cercando file QMD in {reports_dir}...")

    # Trova tutti i file QMD
    qmd_files = []
    for pattern in ["*_files/report.qmd", "*_files/*.qmd"]:
        qmd_files.extend(reports_path.glob(pattern))

    if not qmd_files:
        print("[INFO] Nessun file QMD trovato")
        return

    print(f"[FOUND] Trovati {len(qmd_files)} file QMD:")
    for qmd_file in qmd_files:
        print(f"  - {qmd_file.relative_to(reports_path)}")

    print("\n[START] Iniziando correzioni...")

    fixed_count = 0
    for qmd_file in qmd_files:
        print(f"\n[PROCESS] Processando {qmd_file.name}...")
        if fix_qmd_string_literals(qmd_file):
            fixed_count += 1

    print(f"\n[DONE] Completato! Corretti {fixed_count}/{len(qmd_files)} file")

    if fixed_count > 0:
        print("\n[TIP] Suggerimento: I prossimi report dovrebbero ora generarsi senza errori")


if __name__ == "__main__":
    print("=" * 60)
    print("QMD STRING LITERALS FIXER")
    print("=" * 60)
    print("Questo script corregge gli errori di stringhe nei file QMD")
    print("che causano SyntaxError durante il rendering di Quarto.")
    print()

    fix_all_qmd_files()
