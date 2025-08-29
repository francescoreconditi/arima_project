"""
Quarto report generator for ARIMA, SARIMA and SARIMAX models.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from ..utils.logger import get_logger
from ..utils.exceptions import ForecastError


class QuartoReportGenerator:
    """
    Generates comprehensive Quarto reports for ARIMA, SARIMA and SARIMAX model analysis.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "outputs/reports", template_name: str = "default"):
        """
        Initialize Quarto report generator.
        
        Args:
            output_dir: Directory where reports will be saved (relative to project root)
            template_name: Name of the template to use (default: "default")
        """
        # Always save to project root outputs/, regardless of current working directory
        if not os.path.isabs(output_dir):
            # Find project root by looking for pyproject.toml or other indicators
            current_path = Path(__file__).parent
            while current_path.parent != current_path:
                if (current_path / 'pyproject.toml').exists() or (current_path / 'CLAUDE.md').exists():
                    project_root = current_path
                    break
                current_path = current_path.parent
            else:
                # Fallback to 3 levels up from this file (src/arima_forecaster/reporting/generator.py)
                project_root = Path(__file__).parent.parent.parent.parent
            
            self.output_dir = project_root / output_dir
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup template directory
        self.template_name = template_name
        self.template_dir = Path(__file__).parent / "templates" / template_name
        if not self.template_dir.exists():
            raise ForecastError(f"Template '{template_name}' not found in {self.template_dir.parent}")
        
        # Load template metadata
        self.template_metadata = self._load_template_metadata()
        
        self.logger = get_logger(__name__)
    
    def _load_template_metadata(self) -> Dict[str, Any]:
        """Load template metadata from JSON file."""
        metadata_path = self.template_dir / "metadata.json"
        if not metadata_path.exists():
            self.logger.warning(f"Template metadata not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_template_file(self, filename: str) -> str:
        """Load a template file content."""
        file_path = self.template_dir / filename
        if not file_path.exists():
            raise ForecastError(f"Template file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_format_config(self, format_type: str) -> str:
        """Load format configuration from template."""
        format_configs_path = self.template_dir / "format_configs.json"
        if not format_configs_path.exists():
            # Fallback to default format YAML
            return self._get_default_format_yaml(format_type)
        
        with open(format_configs_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        if format_type in configs:
            return configs[format_type].get('format_yaml', self._get_default_format_yaml(format_type))
        else:
            return self._get_default_format_yaml(format_type)
    
    def _get_default_format_yaml(self, format_type: str) -> str:
        """Get default format YAML configuration."""
        if format_type == "html":
            return '''format:
  html:
    theme: cosmo
    toc: true
    toc-depth: 3
    toc-location: left
    code-fold: true
    code-summary: "Mostra codice"
    fig-width: 8
    fig-height: 5
    fig-align: center
    embed-resources: true'''
        elif format_type == "pdf":
            return '''format:
  pdf:
    geometry: margin=1in
    toc: true
    number-sections: true
    fig-width: 8
    fig-height: 6'''
        elif format_type == "docx":
            return '''format:
  docx:
    toc: true
    number-sections: true
    fig-width: 8
    fig-height: 6'''
        else:
            return '''format:
  html:
    theme: cosmo
    toc: true
    embed-resources: true'''
    
    def _copy_template_resources(self, output_dir: Path) -> Dict[str, str]:
        """
        Copy template CSS and other resources to output directory.
        
        Args:
            output_dir: Directory where resources will be copied
            
        Returns:
            Dict with paths to copied resources
        """
        resources = {}
        
        # Copy CSS file
        css_source = self.template_dir / "styles.css"
        if css_source.exists():
            css_dest = output_dir / "styles.css"
            shutil.copy2(css_source, css_dest)
            # Return relative path from HTML file to CSS
            resources['css_path'] = f"{output_dir.name}/styles.css"
        else:
            self.logger.warning(f"Template CSS file not found: {css_source}")
            resources['css_path'] = ""
        
        # Copy any additional JS files from assets directory if they exist
        assets_dir = Path(__file__).parent / "assets"
        js_source = assets_dir / "scripts.js"
        if js_source.exists():
            js_dest = output_dir / "scripts.js"
            shutil.copy2(js_source, js_dest)
            resources['js_path'] = f"{output_dir.name}/scripts.js"
        else:
            resources['js_path'] = ""
            
        return resources
        
    def generate_model_report(
        self,
        model_results: Dict[str, Any],
        plots_data: Optional[Dict[str, str]] = None,
        report_title: str = "Time Series Model Analysis Report",
        output_filename: str = None,
        format_type: str = "html"
    ) -> Path:
        """
        Generate comprehensive model analysis report using Quarto.
        
        Args:
            model_results: Dictionary containing model results and metadata
            plots_data: Dictionary with plot file paths or base64 encoded images
            report_title: Title for the report
            output_filename: Custom filename for the report
            format_type: Output format ('html', 'pdf', 'docx')
            
        Returns:
            Path to generated report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_filename is None:
                output_filename = f"arima_report_{timestamp}"
            
            # Create report directory
            report_dir = self.output_dir / f"{output_filename}_files"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate Quarto document
            qmd_path = self._create_quarto_document(
                model_results, plots_data, report_title, report_dir, format_type
            )
            
            # Render report
            output_path = self._render_report(qmd_path, format_type, output_filename)
            
            self.logger.info(f"Report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise ForecastError(f"Report generation failed: {str(e)}")
    
    def _create_quarto_document(
        self, 
        model_results: Dict[str, Any], 
        plots_data: Optional[Dict[str, str]], 
        title: str,
        report_dir: Path,
        format_type: str = "html"
    ) -> Path:
        """Create Quarto markdown document with model analysis."""
        
        # Save model results as JSON for data processing
        results_path = report_dir / "model_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(model_results)
            json.dump(serializable_results, f, indent=2)
        
        # Copy plot files to report directory
        plot_files = {}
        if plots_data:
            for plot_name, plot_path in plots_data.items():
                if os.path.exists(plot_path):
                    dest_path = report_dir / f"{plot_name}.png"
                    shutil.copy2(plot_path, dest_path)
                    # Just use the filename since Quarto will look in the same directory
                    plot_files[plot_name] = f"{plot_name}.png"
        
        # Generate Quarto document using template
        qmd_content = self._generate_qmd_content_from_template(
            title, model_results, plot_files, 'model_results.json', report_dir, format_type
        )
        
        qmd_path = report_dir / "report.qmd"
        with open(qmd_path, 'w', encoding='utf-8') as f:
            f.write(qmd_content)
        
        return qmd_path
    
    def _generate_qmd_content_from_template(
        self,
        title: str,
        model_results: Dict[str, Any],
        plot_files: Dict[str, str],
        results_path_str: str,
        output_dir: Path,
        format_type: str = "html"
    ) -> str:
        """Generate Quarto markdown content from template files."""
        
        # Copy template resources and get paths
        resources = self._copy_template_resources(output_dir)
        
        # Extract key information for placeholders
        model_type = model_results.get('model_type', 'ARIMA')
        order = model_results.get('order', 'N/A')
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Load format configuration
        format_yaml = self._load_format_config(format_type)
        
        # Load template sections
        header_content = self._load_template_file("header.qmd")
        body_content = self._load_template_file("body.qmd")
        footer_content = self._load_template_file("footer.qmd")
        config_section = self._load_template_file("config_section.qmd")
        
        # Generate plots section
        plots_section = self._generate_plots_section(plot_files)
        
        # Replace placeholders in header
        header_content = header_content.replace("{{title}}", title)
        header_content = header_content.replace("{{model_type}}", model_type)
        header_content = header_content.replace("{{date}}", date_str)
        header_content = header_content.replace("{{order}}", str(order))
        header_content = header_content.replace("{{format_yaml}}", format_yaml)
        header_content = header_content.replace("{{results_path}}", results_path_str)
        
        # Replace placeholders in body
        body_content = body_content.replace("{{plots_section}}", plots_section)
        body_content = body_content.replace("{{config_section}}", config_section)
        
        # Add custom CSS styles section if CSS exists
        css_section = ""
        if resources.get('css_path'):
            css_section = f'''
<style>
/* Limita dimensioni immagini esterne */
.figure img {{
    max-width: 800px !important;
    width: 100%;
    height: auto !important;
    display: block;
    margin: 0 auto;
}}
</style>
'''
        
        # Combine all sections
        qmd_content = header_content + "\n\n" + body_content + "\n\n" + css_section + "\n\n" + footer_content
        
        return qmd_content
    
    def _generate_plots_section(self, plot_files: Dict[str, str]) -> str:
        """Generate the plots section for the report."""
        if not plot_files:
            return ""
        
        plots_content = ""
        for plot_name, plot_path in plot_files.items():
            section_title = plot_name.replace('_', ' ').title()
            plots_content += f'''
### {section_title}

![{section_title}]({plot_path}){{.figure-img}}

'''
        
        return plots_content
    
    def _render_report(self, qmd_path: Path, format_type: str, output_filename: str) -> Path:
        """Render Quarto document to specified format."""
        try:
            # Quarto generates output in same directory as .qmd file
            # We'll render there and then move to final location
            temp_output_name = f"{output_filename}.{format_type}"
            temp_output_path = qmd_path.parent / temp_output_name
            final_output_path = self.output_dir / temp_output_name
            
            # Ensure we use the current Python environment  
            import sys
            
            # Use forward slashes for cross-platform compatibility
            qmd_path_posix = str(qmd_path.absolute()).replace('\\', '/')
            
            cmd = [
                "quarto", "render", qmd_path_posix,
                "--to", format_type,
                "--output", temp_output_name  # Just filename, not full path
            ]
            
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)
            # Also set QUARTO_PYTHON to point to our environment
            env['QUARTO_PYTHON'] = sys.executable
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                cwd=qmd_path.parent,
                env=env
            )
            
            # Debug del processo quarto
            self.logger.info(f"Quarto command: {' '.join(cmd)}")
            self.logger.info(f"Return code: {result.returncode}")
            if result.stdout:
                self.logger.info(f"Stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Stderr: {result.stderr}")
            
            if result.returncode != 0:
                self.logger.error(f"Quarto render failed: {result.stderr}")
                raise ForecastError(f"Quarto rendering failed: {result.stderr}")
            
            # Cerca file generati nella directory (fallback)
            generated_files = list(qmd_path.parent.glob("*.html"))
            self.logger.info(f"HTML files found in {qmd_path.parent}: {[f.name for f in generated_files]}")
            
            # Cerca il file atteso o qualsiasi HTML con nome simile
            if temp_output_path.exists():
                output_file = temp_output_path
            elif generated_files:
                # Usa il primo HTML trovato se il nome atteso non esiste
                output_file = generated_files[0]
                self.logger.info(f"Using fallback HTML file: {output_file.name}")
            else:
                raise ForecastError(f"No HTML output found in {qmd_path.parent}")
            
            # Move the generated file to final location
            import shutil
            # Ensure output directory exists
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output_file), str(final_output_path))
            
            # Fix image paths and inject magnifying lens functionality
            if format_type == "html":
                self._fix_html_resources(final_output_path, output_filename)
            
            return final_output_path
            
        except FileNotFoundError:
            raise ForecastError(
                "Quarto not found. Please install Quarto: https://quarto.org/docs/get-started/"
            )
        except Exception as e:
            raise ForecastError(f"Report rendering failed: {str(e)}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (int, float)):
            return obj  # Keep native Python numbers as-is
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif pd.isna(obj) if hasattr(pd, 'isna') else False:
            return None
        else:
            # Try to preserve numeric types before converting to string
            try:
                # Check if it's a numeric string that should remain a number
                if isinstance(obj, str):
                    try:
                        # Try to parse as int first, then float
                        if '.' not in obj:
                            return int(obj)
                        else:
                            return float(obj)
                    except ValueError:
                        return obj  # Keep as string if not numeric
                else:
                    return str(obj)
            except:
                return None
    
    def _fix_html_resources(self, html_path: Path, output_filename: str) -> None:
        """Fix all resource paths and inject magnifying lens functionality into HTML file."""
        try:
            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            files_dir = f"{output_filename}_files"
            
            import re
            
            # Fix all resource paths (CSS, JS, images)
            
            # 1. Fix CSS/JS paths: href="report_files/ -> href="output_filename_files/report_files/
            html_content = re.sub(
                r'href="report_files/',
                f'href="{files_dir}/report_files/',
                html_content
            )
            
            # 2. Fix script paths: src="report_files/ -> src="output_filename_files/report_files/
            html_content = re.sub(
                r'src="report_files/',
                f'src="{files_dir}/report_files/',
                html_content
            )
            
            # 3. Fix image paths that need the subdirectory prefix
            # Pattern: src="forecast_plot.png" -> src="output_filename_files/forecast_plot.png"
            # But only for simple image names, not paths that already have directories
            pattern = r'src="([^/\\]+\.png)"'
            
            def replace_img_path(match):
                filename = match.group(1)
                # Don't fix paths that already have the files_dir
                if files_dir in filename:
                    return match.group(0)
                # Don't fix paths that start with 'figure-html/' (Quarto internal images)
                if 'figure-html' in filename:
                    return match.group(0)
                # Fix the path
                return f'src="{files_dir}/{filename}"'
            
            html_content = re.sub(pattern, replace_img_path, html_content)
            
            # 4. Inject magnifying lens CSS and JavaScript if JS file exists
            report_dir = html_path.parent / files_dir
            js_path = report_dir / "scripts.js"
            css_path = report_dir / "styles.css"
            
            if js_path.exists() or css_path.exists():
                magnifier_code = ""
                if css_path.exists():
                    magnifier_code += f'<link rel="stylesheet" href="{files_dir}/styles.css" />\n'
                if js_path.exists():
                    magnifier_code += f'<script src="{files_dir}/scripts.js"></script>\n'
                
                # Inject the magnifier code before </head> tag
                head_pattern = r'</head>'
                if re.search(head_pattern, html_content, re.IGNORECASE):
                    html_content = re.sub(
                        head_pattern, 
                        magnifier_code + '</head>', 
                        html_content, 
                        flags=re.IGNORECASE
                    )
                    self.logger.info(f"Injected external resources into HTML report")
                else:
                    self.logger.warning("Could not find </head> tag to inject external resources")
            
            # Write back the fixed HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"Fixed all resource paths in HTML report: {html_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not fix HTML resources: {e}")
            # Don't fail the entire report generation for this issue
    
    def create_comparison_report(
        self,
        models_results: Dict[str, Dict[str, Any]],
        report_title: str = "ARIMA Models Comparison Report",
        output_filename: str = None,
        format_type: str = "html",
        template_name: str = None
    ) -> Path:
        """
        Generate comparative analysis report for multiple models.
        
        Args:
            models_results: Dictionary of model names and their results
            report_title: Title for the report
            output_filename: Custom filename for the report
            format_type: Output format ('html', 'pdf', 'docx')
            template_name: Template to use (default: use instance template)
            
        Returns:
            Path to generated report
        """
        try:
            # Use custom template if provided
            if template_name and template_name != self.template_name:
                old_template = self.template_name
                self.template_name = template_name
                self.template_dir = Path(__file__).parent / "templates" / template_name
                self.template_metadata = self._load_template_metadata()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_filename is None:
                output_filename = f"models_comparison_{timestamp}"
            
            report_dir = self.output_dir / f"{output_filename}_files"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Save models results
            results_path = report_dir / "models_comparison.json"
            with open(results_path, 'w') as f:
                serializable_results = {
                    model_name: self._make_json_serializable(results)
                    for model_name, results in models_results.items()
                }
                json.dump(serializable_results, f, indent=2)
            
            # Generate comparison document
            qmd_content = self._generate_comparison_qmd(report_title, models_results, results_path, format_type)
            
            qmd_path = report_dir / "comparison_report.qmd"
            with open(qmd_path, 'w', encoding='utf-8') as f:
                f.write(qmd_content)
            
            # Render report
            output_path = self._render_report(qmd_path, format_type, output_filename)
            
            # Restore original template if changed
            if template_name and template_name != old_template:
                self.template_name = old_template
                self.template_dir = Path(__file__).parent / "templates" / old_template
                self.template_metadata = self._load_template_metadata()
            
            self.logger.info(f"Comparison report generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {str(e)}")
            raise ForecastError(f"Comparison report generation failed: {str(e)}")
    
    def _generate_comparison_qmd(
        self, 
        title: str, 
        models_results: Dict[str, Dict[str, Any]], 
        results_path: Path,
        format_type: str = "html"
    ) -> str:
        """Generate Quarto document for models comparison."""
        
        # Get format configuration
        format_yaml = self._load_format_config(format_type)
        
        # For comparison reports, we still use the embedded template for now
        # In future, this could also be modularized
        model_type = "Multiple Models"
        
        qmd_content = f'''---
title: "{title}"
subtitle: "Analisi Comparativa Modelli"
author: "ARIMA Forecaster Library"
date: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
{format_yaml}
execute:
  echo: false
  warning: false
  message: false
jupyter: python3
---

```{{python}}
#| include: false
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Carica i risultati dei modelli
with open('{results_path}', 'r') as f:
    models_results = json.load(f)

plt.style.use('default')
sns.set_palette("Set2")
```

## Panoramica Confronto Modelli

```{{python}}
#| label: tbl-models-overview
#| tbl-cap: "Panoramica dei Modelli Confrontati"

models_data = []
for model_name, results in models_results.items():
    model_type = results.get('model_type', 'N/A')
    order = results.get('order', 'N/A')
    seasonal_order = results.get('seasonal_order', 'N/A')
    aic = results.get('metrics', {{}}).get('aic', 'N/A')
    
    models_data.append([
        model_name,
        model_type,
        str(order),
        str(seasonal_order) if seasonal_order != 'N/A' else 'N/A',
        f"{{aic:.2f}}" if isinstance(aic, (int, float)) else 'N/A'
    ])

models_df = pd.DataFrame(models_data, columns=[
    'Nome Modello', 'Tipo', 'Ordine (p,d,q)', 'Ordine Stagionale', 'AIC'
])
from IPython.display import display, Markdown, HTML
html_table = models_df.to_html(index=False, classes='table table-striped', escape=False)
display(HTML(html_table))
```

## Confronto Performance

```{{python}}
#| label: fig-performance-comparison
#| fig-cap: "Confronto delle Metriche di Performance"

# Estrai metriche per tutti i modelli
metrics_data = []
model_names = []

for model_name, results in models_results.items():
    metrics = results.get('metrics', {{}})
    model_names.append(model_name)
    metrics_data.append(metrics)

if metrics_data:
    # Identifica metriche comuni
    common_metrics = set(metrics_data[0].keys())
    for metrics in metrics_data[1:]:
        common_metrics &= set(metrics.keys())
    
    # Filtra metriche numeriche
    numeric_metrics = []
    for metric in common_metrics:
        if all(isinstance(m.get(metric), (int, float)) for m in metrics_data):
            numeric_metrics.append(metric)
    
    if numeric_metrics:
        # Crea DataFrame per il confronto
        comparison_data = []
        for i, model_name in enumerate(model_names):
            for metric in numeric_metrics:
                comparison_data.append([
                    model_name, 
                    metric.upper(), 
                    metrics_data[i][metric]
                ])
        
        comparison_df = pd.DataFrame(comparison_data, columns=['Modello', 'Metrica', 'Valore'])
        
        # Crea grafici di confronto
        n_metrics = len(numeric_metrics)
        fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(15, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(numeric_metrics[:3]):
            metric_data = comparison_df[comparison_df['Metrica'] == metric.upper()]
            ax = axes[i] if len(axes) > 1 else axes[0]
            
            bars = ax.bar(metric_data['Modello'], metric_data['Valore'], alpha=0.7)
            ax.set_title(f'Confronto {{metric.upper()}}', fontweight='bold')
            ax.set_ylabel('Valore')
            
            # Aggiungi valori sulle barre
            for bar, value in zip(bars, metric_data['Valore']):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + max(metric_data['Valore'])*0.01,
                       f'{{value:.4f}}', ha='center', va='bottom', fontsize=8)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    else:
        display(HTML("<p>Nessuna metrica numerica comune trovata per il confronto</p>"))
else:
    display(HTML("<p>Dati delle metriche non disponibili</p>"))
```

## Ranking dei Modelli

```{{python}}
#| label: tbl-model-ranking
#| tbl-cap: "Ranking dei Modelli per Performance"

ranking_data = []
for model_name, results in models_results.items():
    metrics = results.get('metrics', {{}})
    
    # Calcola score composito (lower is better per AIC, RMSE, MAE)
    score = 0
    score_count = 0
    
    # AIC (lower is better)
    if 'aic' in metrics and isinstance(metrics['aic'], (int, float)):
        score += metrics['aic']
        score_count += 1
    
    # RMSE (lower is better)  
    if 'rmse' in metrics and isinstance(metrics['rmse'], (int, float)):
        score += metrics['rmse'] * 100  # Scale up for better comparison
        score_count += 1
    
    # R² (higher is better, so we subtract from 1)
    if 'r2_score' in metrics and isinstance(metrics['r2_score'], (int, float)):
        score += (1 - metrics['r2_score']) * 100
        score_count += 1
    
    avg_score = score / score_count if score_count > 0 else float('inf')
    
    ranking_data.append([
        model_name,
        f"{{metrics.get('aic', 'N/A'):.2f}}" if isinstance(metrics.get('aic'), (int, float)) else 'N/A',
        f"{{metrics.get('rmse', 'N/A'):.4f}}" if isinstance(metrics.get('rmse'), (int, float)) else 'N/A',
        f"{{metrics.get('r2_score', 'N/A'):.4f}}" if isinstance(metrics.get('r2_score'), (int, float)) else 'N/A',
        f"{{avg_score:.2f}}" if avg_score != float('inf') else 'N/A'
    ])

# Ordina per score composito
ranking_data.sort(key=lambda x: float(x[4]) if x[4] != 'N/A' else float('inf'))

ranking_df = pd.DataFrame(ranking_data, columns=[
    'Modello', 'AIC', 'RMSE', 'R² Score', 'Score Composito'
])

# Aggiungi ranking
ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))

html_table = ranking_df.to_html(index=False, classes='table table-striped', escape=False)
display(HTML(html_table))
```

## Raccomandazioni Finali

```{{python}}
#| label: final-recommendations

if len(ranking_df) > 0:
    best_model = ranking_df.iloc[0]['Modello']
    display(HTML(f"<h3>Modello Raccomandato: {{best_model}}</h3>"))
    
    best_results = models_results.get(best_model, {{}})
    best_metrics = best_results.get('metrics', {{}})
    
    recommendations = [
        f"• **Modello migliore**: {{best_model}} - Mostra le performance complessive migliori",
        "• **Implementazione**: Procedere con l'implementazione del modello raccomandato",
        "• **Monitoraggio**: Implementare sistema di monitoraggio continuo delle performance",
        "• **Backup**: Mantenere il secondo miglior modello come backup"
    ]
    
    if len(ranking_df) > 1:
        second_best = ranking_df.iloc[1]['Modello']
        recommendations.append(f"• **Modello alternativo**: {{second_best}} - Considerare come alternativa")
    
    for rec in recommendations:
        display(HTML(f"<p>{{rec}}</p>"))
else:
    display(HTML("<p>Nessun modello disponibile per le raccomandazioni</p>"))
```

---

*Report comparativo generato automaticamente dalla libreria ARIMA Forecaster*
'''
        
        return qmd_content