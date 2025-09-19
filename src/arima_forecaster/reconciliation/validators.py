"""
Validatori e diagnostica per Forecast Reconciliation.

Verifica coerenza, calcola metriche di accuratezza e fornisce
diagnostica dettagliata per gerarchie e riconciliazioni.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from ..utils.logger import get_logger
from .structures import HierarchicalStructure


logger = get_logger(__name__)


@dataclass
class CoherenceReport:
    """Report di coerenza per previsioni riconciliate."""

    is_coherent: bool
    max_error: float
    mean_error: float
    n_incoherent_nodes: int
    incoherent_nodes: List[str]
    error_by_level: Dict[int, float]
    error_by_node: pd.DataFrame


@dataclass
class ReconciliationDiagnostics:
    """Diagnostica completa per riconciliazione."""

    coherence_report: CoherenceReport
    accuracy_metrics: Dict[str, float]
    improvement_metrics: Dict[str, float]
    computational_metrics: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]


class HierarchyValidator:
    """
    Validatore per strutture gerarchiche.
    """

    def __init__(self, hierarchy: HierarchicalStructure):
        """
        Inizializza il validatore.

        Args:
            hierarchy: Struttura gerarchica da validare
        """
        self.hierarchy = hierarchy

    def validate_structure(self) -> Tuple[bool, List[str], List[str]]:
        """
        Valida completamente la struttura gerarchica.

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Controlli base dalla gerarchia
        is_valid_base, base_errors = self.hierarchy.validate_hierarchy()
        errors.extend(base_errors)

        # Controlli aggiuntivi
        # 1. Verifica bilanciamento
        balance_check = self._check_hierarchy_balance()
        if not balance_check["is_balanced"]:
            warnings.append(f"Gerarchia sbilanciata: {balance_check['message']}")

        # 2. Verifica profondità
        depth_check = self._check_hierarchy_depth()
        if depth_check["max_depth"] > 10:
            warnings.append(f"Gerarchia molto profonda ({depth_check['max_depth']} livelli)")

        # 3. Verifica completezza
        completeness_check = self._check_completeness()
        if not completeness_check["is_complete"]:
            errors.append(f"Gerarchia incompleta: {completeness_check['message']}")

        # 4. Verifica ridondanza
        redundancy_check = self._check_redundancy()
        if redundancy_check["has_redundancy"]:
            warnings.append(f"Possibile ridondanza: {redundancy_check['message']}")

        is_valid = len(errors) == 0

        return is_valid, errors, warnings

    def _check_hierarchy_balance(self) -> Dict[str, Any]:
        """Verifica se la gerarchia è bilanciata."""
        leaves = self.hierarchy.get_leaves()
        leaf_depths = []

        for leaf in leaves:
            depth = leaf.level
            leaf_depths.append(depth)

        if not leaf_depths:
            return {"is_balanced": False, "message": "Nessun nodo foglia trovato"}

        min_depth = min(leaf_depths)
        max_depth = max(leaf_depths)

        is_balanced = (max_depth - min_depth) <= 1

        return {
            "is_balanced": is_balanced,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "message": f"Profondità foglie varia da {min_depth} a {max_depth}",
        }

    def _check_hierarchy_depth(self) -> Dict[str, Any]:
        """Calcola la profondità della gerarchia."""
        if not self.hierarchy.levels:
            return {"max_depth": 0, "avg_depth": 0}

        max_depth = max(self.hierarchy.levels.keys())

        # Calcola profondità media delle foglie
        leaves = self.hierarchy.get_leaves()
        if leaves:
            avg_depth = np.mean([leaf.level for leaf in leaves])
        else:
            avg_depth = 0

        return {"max_depth": max_depth, "avg_depth": avg_depth}

    def _check_completeness(self) -> Dict[str, Any]:
        """Verifica che ogni nodo non-foglia abbia almeno un figlio."""
        incomplete_nodes = []

        for node in self.hierarchy.nodes.values():
            if not node.is_leaf() and len(node.children_ids) == 0:
                incomplete_nodes.append(node.id)

        is_complete = len(incomplete_nodes) == 0

        return {
            "is_complete": is_complete,
            "incomplete_nodes": incomplete_nodes,
            "message": f"{len(incomplete_nodes)} nodi senza figli",
        }

    def _check_redundancy(self) -> Dict[str, Any]:
        """Verifica possibili ridondanze nella struttura."""
        redundant_paths = []

        # Cerca nodi con path multipli dal root
        roots = [n for n in self.hierarchy.nodes.values() if n.is_root()]
        if not roots:
            return {"has_redundancy": False, "message": "No root found"}

        root = roots[0]

        # Per ogni foglia, verifica unicità del path
        leaves = self.hierarchy.get_leaves()
        for leaf in leaves:
            paths = self._find_all_paths(root.id, leaf.id)
            if len(paths) > 1:
                redundant_paths.append({"leaf": leaf.id, "n_paths": len(paths)})

        has_redundancy = len(redundant_paths) > 0

        return {
            "has_redundancy": has_redundancy,
            "redundant_paths": redundant_paths,
            "message": f"{len(redundant_paths)} foglie con path multipli",
        }

    def _find_all_paths(self, start: str, end: str) -> List[List[str]]:
        """Trova tutti i path tra due nodi."""
        # Implementazione semplificata
        # In una vera gerarchia dovrebbe esserci un solo path
        if start == end:
            return [[start]]

        if start not in self.hierarchy.nodes:
            return []

        paths = []
        node = self.hierarchy.nodes[start]

        for child_id in node.children_ids:
            child_paths = self._find_all_paths(child_id, end)
            for path in child_paths:
                paths.append([start] + path)

        return paths

    def generate_validation_report(self) -> pd.DataFrame:
        """
        Genera report di validazione dettagliato.

        Returns:
            DataFrame con risultati validazione
        """
        is_valid, errors, warnings = self.validate_structure()

        report_data = {"Check": [], "Status": [], "Details": []}

        # Status generale
        report_data["Check"].append("Overall Validity")
        report_data["Status"].append("PASS" if is_valid else "FAIL")
        report_data["Details"].append(f"{len(errors)} errors, {len(warnings)} warnings")

        # Errori
        for error in errors:
            report_data["Check"].append("Error")
            report_data["Status"].append("FAIL")
            report_data["Details"].append(error)

        # Warning
        for warning in warnings:
            report_data["Check"].append("Warning")
            report_data["Status"].append("WARN")
            report_data["Details"].append(warning)

        # Statistiche struttura
        report_data["Check"].append("Nodes")
        report_data["Status"].append("INFO")
        report_data["Details"].append(f"Total: {len(self.hierarchy.nodes)}")

        report_data["Check"].append("Levels")
        report_data["Status"].append("INFO")
        report_data["Details"].append(f"Depth: {len(self.hierarchy.levels)}")

        report_data["Check"].append("Leaves")
        report_data["Status"].append("INFO")
        report_data["Details"].append(f"Count: {len(self.hierarchy.get_leaves())}")

        return pd.DataFrame(report_data)


class CoherenceChecker:
    """
    Verifica coerenza delle previsioni riconciliate.
    """

    def __init__(self, hierarchy: HierarchicalStructure, tolerance: float = 1e-6):
        """
        Inizializza il checker.

        Args:
            hierarchy: Struttura gerarchica
            tolerance: Tolleranza per errori numerici
        """
        self.hierarchy = hierarchy
        self.tolerance = tolerance

    def check_coherence(self, forecasts: Union[np.ndarray, pd.DataFrame]) -> CoherenceReport:
        """
        Verifica coerenza delle previsioni.

        Args:
            forecasts: Previsioni da verificare

        Returns:
            Report di coerenza
        """
        if isinstance(forecasts, pd.DataFrame):
            forecasts_np = forecasts.values
            node_names = list(forecasts.index)
        else:
            forecasts_np = forecasts
            node_names = [f"Node_{i}" for i in range(len(forecasts))]

        S = self.hierarchy.get_summing_matrix()
        n_total, n_bottom = S.shape

        # Assumendo che gli ultimi n_bottom siano i nodi foglia
        bottom_forecasts = forecasts_np[-n_bottom:, :]

        # Calcola valori aggregati attesi
        expected = S @ bottom_forecasts

        # Calcola errori
        errors = np.abs(forecasts_np - expected)
        max_errors = np.max(errors, axis=1)
        mean_errors = np.mean(errors, axis=1)

        # Identifica nodi incoerenti
        incoherent_mask = max_errors > self.tolerance
        incoherent_indices = np.where(incoherent_mask)[0]
        incoherent_nodes = [node_names[i] for i in incoherent_indices]

        # Errori per livello
        error_by_level = {}
        for level_id, level in self.hierarchy.levels.items():
            level_nodes_ids = [n.id for n in level.nodes]
            level_indices = [
                i
                for i, name in enumerate(node_names)
                if any(node_id in name for node_id in level_nodes_ids)
            ]
            if level_indices:
                level_errors = mean_errors[level_indices]
                error_by_level[level_id] = np.mean(level_errors)

        # DataFrame dettagliato
        error_df = pd.DataFrame(
            {
                "node": node_names,
                "max_error": max_errors,
                "mean_error": mean_errors,
                "is_coherent": ~incoherent_mask,
            }
        )

        return CoherenceReport(
            is_coherent=not np.any(incoherent_mask),
            max_error=float(np.max(max_errors)),
            mean_error=float(np.mean(mean_errors)),
            n_incoherent_nodes=len(incoherent_nodes),
            incoherent_nodes=incoherent_nodes,
            error_by_level=error_by_level,
            error_by_node=error_df,
        )

    def visualize_coherence(
        self, coherence_report: CoherenceReport, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualizza risultati di coerenza.

        Args:
            coherence_report: Report di coerenza
            figsize: Dimensioni figura

        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Distribuzione errori
        ax = axes[0, 0]
        error_data = coherence_report.error_by_node["mean_error"]
        ax.hist(error_data, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(
            self.tolerance, color="red", linestyle="--", label=f"Tolerance: {self.tolerance}"
        )
        ax.set_xlabel("Mean Error")
        ax.set_ylabel("Number of Nodes")
        ax.set_title("Error Distribution")
        ax.legend()

        # 2. Errori per livello
        ax = axes[0, 1]
        if coherence_report.error_by_level:
            levels = list(coherence_report.error_by_level.keys())
            errors = list(coherence_report.error_by_level.values())
            ax.bar(levels, errors, color="skyblue", edgecolor="black")
            ax.set_xlabel("Hierarchy Level")
            ax.set_ylabel("Mean Error")
            ax.set_title("Errors by Level")

        # 3. Top nodi incoerenti
        ax = axes[1, 0]
        top_incoherent = coherence_report.error_by_node.nlargest(10, "max_error")
        if not top_incoherent.empty:
            ax.barh(range(len(top_incoherent)), top_incoherent["max_error"])
            ax.set_yticks(range(len(top_incoherent)))
            ax.set_yticklabels(top_incoherent["node"])
            ax.set_xlabel("Max Error")
            ax.set_title("Top 10 Incoherent Nodes")

        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = f"""
        Coherence Check Summary
        ----------------------
        Status: {"COHERENT" if coherence_report.is_coherent else "INCOHERENT"}
        Max Error: {coherence_report.max_error:.2e}
        Mean Error: {coherence_report.mean_error:.2e}
        Incoherent Nodes: {coherence_report.n_incoherent_nodes}
        Tolerance: {self.tolerance:.2e}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=10, family="monospace", verticalalignment="center")

        plt.suptitle("Coherence Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig


class ReconciliationDiagnostics:
    """
    Diagnostica completa per processi di riconciliazione.
    """

    def __init__(self, hierarchy: HierarchicalStructure):
        """
        Inizializza la diagnostica.

        Args:
            hierarchy: Struttura gerarchica
        """
        self.hierarchy = hierarchy
        self.coherence_checker = CoherenceChecker(hierarchy)
        self.hierarchy_validator = HierarchyValidator(hierarchy)

    def diagnose(
        self,
        base_forecasts: Union[np.ndarray, pd.DataFrame],
        reconciled_forecasts: Union[np.ndarray, pd.DataFrame],
        actuals: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        computation_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Esegue diagnostica completa.

        Args:
            base_forecasts: Previsioni base (pre-riconciliazione)
            reconciled_forecasts: Previsioni riconciliate
            actuals: Valori reali (opzionale, per metriche accuratezza)
            computation_time: Tempo di calcolo in secondi

        Returns:
            Dizionario con diagnostica completa
        """
        diagnostics = {}

        # 1. Coerenza
        coherence_report = self.coherence_checker.check_coherence(reconciled_forecasts)
        diagnostics["coherence"] = coherence_report

        # 2. Metriche accuratezza (se disponibili valori reali)
        if actuals is not None:
            accuracy_metrics = self._compute_accuracy_metrics(
                base_forecasts, reconciled_forecasts, actuals
            )
            diagnostics["accuracy"] = accuracy_metrics

        # 3. Metriche miglioramento
        improvement_metrics = self._compute_improvement_metrics(
            base_forecasts, reconciled_forecasts
        )
        diagnostics["improvement"] = improvement_metrics

        # 4. Metriche computazionali
        computational_metrics = {
            "computation_time": computation_time,
            "n_nodes": len(self.hierarchy.nodes),
            "n_leaves": len(self.hierarchy.get_leaves()),
            "hierarchy_depth": max(self.hierarchy.levels.keys()) if self.hierarchy.levels else 0,
        }
        diagnostics["computational"] = computational_metrics

        # 5. Warning e raccomandazioni
        warnings = []
        recommendations = []

        if not coherence_report.is_coherent:
            warnings.append(f"{coherence_report.n_incoherent_nodes} nodi incoerenti trovati")
            recommendations.append("Verificare implementazione metodo riconciliazione")

        if coherence_report.max_error > 1e-3:
            warnings.append(f"Errore massimo elevato: {coherence_report.max_error:.2e}")
            recommendations.append("Considerare tolleranza numerica più alta")

        diagnostics["warnings"] = warnings
        diagnostics["recommendations"] = recommendations

        return diagnostics

    def _compute_accuracy_metrics(
        self,
        base_forecasts: Union[np.ndarray, pd.DataFrame],
        reconciled_forecasts: Union[np.ndarray, pd.DataFrame],
        actuals: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Dict[str, float]]:
        """Calcola metriche di accuratezza."""

        def to_numpy(x):
            return x.values if isinstance(x, pd.DataFrame) else x

        base_np = to_numpy(base_forecasts)
        reconciled_np = to_numpy(reconciled_forecasts)
        actuals_np = to_numpy(actuals)

        metrics = {}

        # MAE
        mae_base = np.mean(np.abs(base_np - actuals_np))
        mae_reconciled = np.mean(np.abs(reconciled_np - actuals_np))
        metrics["mae"] = {
            "base": float(mae_base),
            "reconciled": float(mae_reconciled),
            "improvement": float((mae_base - mae_reconciled) / mae_base * 100),
        }

        # RMSE
        rmse_base = np.sqrt(np.mean((base_np - actuals_np) ** 2))
        rmse_reconciled = np.sqrt(np.mean((reconciled_np - actuals_np) ** 2))
        metrics["rmse"] = {
            "base": float(rmse_base),
            "reconciled": float(rmse_reconciled),
            "improvement": float((rmse_base - rmse_reconciled) / rmse_base * 100),
        }

        # MAPE
        mape_base = np.mean(np.abs((base_np - actuals_np) / (actuals_np + 1e-10))) * 100
        mape_reconciled = np.mean(np.abs((reconciled_np - actuals_np) / (actuals_np + 1e-10))) * 100
        metrics["mape"] = {
            "base": float(mape_base),
            "reconciled": float(mape_reconciled),
            "improvement": float((mape_base - mape_reconciled) / mape_base * 100),
        }

        return metrics

    def _compute_improvement_metrics(
        self,
        base_forecasts: Union[np.ndarray, pd.DataFrame],
        reconciled_forecasts: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, float]:
        """Calcola metriche di miglioramento."""

        def to_numpy(x):
            return x.values if isinstance(x, pd.DataFrame) else x

        base_np = to_numpy(base_forecasts)
        reconciled_np = to_numpy(reconciled_forecasts)

        # Differenza media
        mean_diff = np.mean(np.abs(reconciled_np - base_np))

        # Percentuale nodi modificati
        changed_mask = np.any(np.abs(reconciled_np - base_np) > 1e-10, axis=1)
        pct_changed = np.mean(changed_mask) * 100

        # Varianza delle modifiche
        changes = reconciled_np - base_np
        change_variance = np.var(changes)

        return {
            "mean_adjustment": float(mean_diff),
            "pct_nodes_adjusted": float(pct_changed),
            "adjustment_variance": float(change_variance),
        }

    def generate_report(self, diagnostics: Dict[str, Any], output_format: str = "text") -> str:
        """
        Genera report testuale della diagnostica.

        Args:
            diagnostics: Risultati diagnostica
            output_format: Formato output ('text', 'html', 'markdown')

        Returns:
            Report formattato
        """
        if output_format == "text":
            return self._generate_text_report(diagnostics)
        elif output_format == "markdown":
            return self._generate_markdown_report(diagnostics)
        elif output_format == "html":
            return self._generate_html_report(diagnostics)
        else:
            raise ValueError(f"Formato non supportato: {output_format}")

    def _generate_text_report(self, diagnostics: Dict[str, Any]) -> str:
        """Genera report testuale."""
        lines = []
        lines.append("=" * 60)
        lines.append("FORECAST RECONCILIATION DIAGNOSTICS REPORT")
        lines.append("=" * 60)

        # Coerenza
        if "coherence" in diagnostics:
            coh = diagnostics["coherence"]
            lines.append("\nCOHERENCE CHECK:")
            lines.append(f"  Status: {'PASS' if coh.is_coherent else 'FAIL'}")
            lines.append(f"  Max Error: {coh.max_error:.2e}")
            lines.append(f"  Mean Error: {coh.mean_error:.2e}")
            lines.append(f"  Incoherent Nodes: {coh.n_incoherent_nodes}")

        # Accuratezza
        if "accuracy" in diagnostics:
            acc = diagnostics["accuracy"]
            lines.append("\nACCURACY METRICS:")
            for metric, values in acc.items():
                lines.append(f"  {metric.upper()}:")
                lines.append(f"    Base: {values['base']:.4f}")
                lines.append(f"    Reconciled: {values['reconciled']:.4f}")
                lines.append(f"    Improvement: {values['improvement']:.2f}%")

        # Computational
        if "computational" in diagnostics:
            comp = diagnostics["computational"]
            lines.append("\nCOMPUTATIONAL METRICS:")
            lines.append(f"  Nodes: {comp.get('n_nodes', 'N/A')}")
            lines.append(f"  Leaves: {comp.get('n_leaves', 'N/A')}")
            lines.append(f"  Depth: {comp.get('hierarchy_depth', 'N/A')}")
            if comp.get("computation_time"):
                lines.append(f"  Time: {comp['computation_time']:.3f}s")

        # Warnings
        if diagnostics.get("warnings"):
            lines.append("\nWARNINGS:")
            for warning in diagnostics["warnings"]:
                lines.append(f"  - {warning}")

        # Recommendations
        if diagnostics.get("recommendations"):
            lines.append("\nRECOMMENDATIONS:")
            for rec in diagnostics["recommendations"]:
                lines.append(f"  - {rec}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def _generate_markdown_report(self, diagnostics: Dict[str, Any]) -> str:
        """Genera report Markdown."""
        lines = []
        lines.append("# Forecast Reconciliation Diagnostics Report")
        lines.append("")

        # Coerenza
        if "coherence" in diagnostics:
            coh = diagnostics["coherence"]
            lines.append("## Coherence Check")
            lines.append("")
            lines.append(f"- **Status**: {'✅ PASS' if coh.is_coherent else '❌ FAIL'}")
            lines.append(f"- **Max Error**: {coh.max_error:.2e}")
            lines.append(f"- **Mean Error**: {coh.mean_error:.2e}")
            lines.append(f"- **Incoherent Nodes**: {coh.n_incoherent_nodes}")
            lines.append("")

        # Accuratezza
        if "accuracy" in diagnostics:
            acc = diagnostics["accuracy"]
            lines.append("## Accuracy Metrics")
            lines.append("")
            lines.append("| Metric | Base | Reconciled | Improvement |")
            lines.append("|--------|------|------------|-------------|")
            for metric, values in acc.items():
                lines.append(
                    f"| {metric.upper()} | {values['base']:.4f} | "
                    f"{values['reconciled']:.4f} | {values['improvement']:.2f}% |"
                )
            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, diagnostics: Dict[str, Any]) -> str:
        """Genera report HTML."""
        # Implementazione base HTML
        html = "<html><body>"
        html += "<h1>Forecast Reconciliation Diagnostics</h1>"

        # Converte markdown in HTML per semplicità
        markdown_report = self._generate_markdown_report(diagnostics)
        # In produzione useresti un markdown parser appropriato
        html += f"<pre>{markdown_report}</pre>"

        html += "</body></html>"
        return html
