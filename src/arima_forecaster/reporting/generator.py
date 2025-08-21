"""
Quarto report generator for ARIMA and SARIMA models.
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
    Generates comprehensive Quarto reports for ARIMA/SARIMA model analysis.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "outputs/reports"):
        """
        Initialize Quarto report generator.
        
        Args:
            output_dir: Directory where reports will be saved (relative to project root)
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
        self.logger = get_logger(__name__)
        
    def generate_model_report(
        self,
        model_results: Dict[str, Any],
        plots_data: Optional[Dict[str, str]] = None,
        report_title: str = "ARIMA Model Analysis Report",
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
        
        # Generate Quarto document - JSON file is in same directory as QMD
        qmd_content = self._generate_qmd_content(
            title, model_results, plot_files, 'model_results.json', format_type
        )
        
        qmd_path = report_dir / "report.qmd"
        with open(qmd_path, 'w', encoding='utf-8') as f:
            f.write(qmd_content)
        
        return qmd_path
    
    def _generate_qmd_content(
        self,
        title: str,
        model_results: Dict[str, Any],
        plot_files: Dict[str, str],
        results_path_str: str,
        format_type: str = "html"
    ) -> str:
        """Generate Quarto markdown content."""
        
        # Extract key information
        model_type = model_results.get('model_type', 'ARIMA')
        order = model_results.get('order', 'N/A')
        metrics = model_results.get('metrics', {})
        model_info = model_results.get('model_info', {})
        
        # Generate format-specific YAML header
        if format_type == "html":
            format_yaml = '''format:
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
    embed-resources: true
    include-in-header: |
      <style>
      /* TOC Navigation Improvements */
      #TOC {
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
      }
      
      #TOC li {
        margin: 3px 0;
        border-radius: 6px;
        transition: all 0.2s ease;
      }
      
      #TOC li:hover {
        background-color: #e9ecef;
      }
      
      #TOC li.active {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196f3 !important;
        padding-left: 8px;
      }
      
      #TOC li.active > a {
        color: #1976d2 !important;
        font-weight: 600 !important;
      }
      
      #TOC a {
        display: block;
        padding: 8px 12px;
        text-decoration: none;
        color: #495057;
        border-radius: 4px;
        transition: all 0.2s ease;
      }
      
      #TOC a:hover {
        background-color: rgba(33, 150, 243, 0.1);
        color: #1976d2;
      }
      
      /* Table Professional Styling */
      .table {
        border-collapse: separate;
        border-spacing: 0;
        margin: 20px 0;
        font-size: 0.95em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
        background: white;
      }
      
      .table thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.5px;
        padding: 15px 12px;
        border: none;
        position: sticky;
        top: 0;
        z-index: 10;
      }
      
      .table tbody tr {
        transition: all 0.2s ease;
        border-bottom: 1px solid #e9ecef;
      }
      
      .table tbody tr:nth-child(even) {
        background-color: #f8f9fa;
      }
      
      .table tbody tr:hover {
        background-color: #e3f2fd;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      
      .table tbody td {
        padding: 12px 12px;
        border: none;
        vertical-align: middle;
      }
      
      .table tbody td:first-child {
        font-weight: 500;
        color: #495057;
      }
      
      /* Responsive table wrapper */
      .table-responsive {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      
      /* Metrics badges */
      .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 500;
        margin-right: 5px;
      }
      
      .metric-good { background-color: #d4edda; color: #155724; }
      .metric-warning { background-color: #fff3cd; color: #856404; }
      .metric-danger { background-color: #f8d7da; color: #721c24; }
      
      /* Card styling for sections */
      .content-block {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
      }
      
      /* Header improvements */
      h2, h3 {
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
        margin-top: 30px;
      }
      
      h2 {
        font-size: 1.8em;
        color: #34495e;
      }
      
      h3 {
        font-size: 1.4em;
        color: #7f8c8d;
        border-bottom: 1px solid #bdc3c7;
      }
      
      /* Image zoom styles */
      .zoomable-image {
        cursor: crosshair;
        transition: transform 0.2s;
        display: block;
        max-width: 100%;
        height: auto;
      }
      
      .zoomable-image:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      
      /* Magnifying glass lens - migliorata per ridurre confusione visiva */
      .magnifier-lens {
        position: absolute !important;
        border: 1px solid #2196F3;
        border-radius: 50%;
        cursor: none;
        width: 80px;
        height: 80px;
        display: none;
        pointer-events: none;
        box-shadow: 0 0 5px rgba(33, 150, 243, 0.3);
        background: rgba(33, 150, 243, 0.05);
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 999998 !important;
      }
      
      /* Magnified result container - usando background-image per affidabilità */
      .magnifier-result {
        position: fixed !important;
        border: 2px solid #2196F3;
        border-radius: 8px;
        display: none;
        width: 300px;
        height: 300px;
        z-index: 999999 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        background-color: #f0f0f0;
        background-repeat: no-repeat;
        background-size: 900px 900px; /* 3x zoom di base */
        opacity: 0;
        transition: opacity 0.2s ease;
        /* Fallback pattern visibile quando l'immagine non è caricata */
        background-image: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,.5) 10px, rgba(255,255,255,.5) 20px);
      }
      
      /* Modal for zoomed image */
      .image-modal {
        display: none;
        position: fixed !important;
        z-index: 1000000 !important;
        padding-top: 60px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.9);
        cursor: zoom-out;
      }
      
      .modal-content {
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 90%;
        animation: zoom 0.3s;
      }
      
      @keyframes zoom {
        from {transform: scale(0.5)}
        to {transform: scale(1)}
      }
      
      .close-modal {
        position: absolute !important;
        top: 15px;
        right: 35px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        transition: 0.3s;
        cursor: pointer;
      }
      
      .close-modal:hover,
      .close-modal:focus {
        color: #bbb;
        text-decoration: none;
      }
      
      /* Caption for zoomed image */
      .image-caption {
        margin: auto;
        display: block;
        width: 80%;
        max-width: 700px;
        text-align: center;
        color: #ccc;
        padding: 10px 0;
      }
      
      /* Magnifying glass icon on hover */
      .image-container {
        position: relative;
        display: inline-block;
      }
      
      .image-container::after {
        content: "🔍";
        position: absolute !important;
        top: 10px;
        right: 10px;
        font-size: 24px;
        opacity: 0;
        transition: opacity 0.3s;
        pointer-events: none;
        background: rgba(255,255,255,0.9);
        padding: 5px;
        border-radius: 50%;
      }
      
      .image-container:hover::after {
        opacity: 1;
      }
      </style>
      <script>
      document.addEventListener('DOMContentLoaded', function() {
        // Image zoom functionality
        const images = document.querySelectorAll('img');
        
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'image-modal';
        modal.innerHTML = '<span class="close-modal">&times;</span><img class="modal-content"><div class="image-caption"></div>';
        document.body.appendChild(modal);
        
        const modalImg = modal.querySelector('.modal-content');
        const captionText = modal.querySelector('.image-caption');
        const closeBtn = modal.querySelector('.close-modal');
        
        images.forEach(img => {
          // Skip small images (icons, etc.)
          if (img.width > 100) {
            // Wrap image in container
            const container = document.createElement('div');
            container.className = 'image-container';
            container.style.position = 'relative';
            img.parentNode.insertBefore(container, img);
            container.appendChild(img);
            
            // Add zoomable class
            img.classList.add('zoomable-image');
            
            // Create magnifying lens
            const lens = document.createElement('div');
            lens.className = 'magnifier-lens';
            container.appendChild(lens);
            
            // Create magnified result container (usando background-image)
            const result = document.createElement('div');
            result.className = 'magnifier-result';
            // Imposta l'immagine come background invece di elemento img interno
            result.style.backgroundImage = `url('${img.src}')`;
            container.appendChild(result);
            
            // Calculate zoom level
            const zoom = 3;
            
            // Variabili per gestire il ritardo e il controllo della lente
            let lensTimeout = null;
            let isLensVisible = false;
            
            // Set up magnifier functionality con miglioramenti
            function magnify(e) {
              const rect = img.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const y = e.clientY - rect.top;
              
              // Debug coordinate e dimensioni
              console.log('Mouse coordinates:', { x, y, imgWidth: img.width, imgHeight: img.height });
              console.log('Image loading state:', { complete: img.complete, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight });
              
              // Check if mouse is within image bounds
              if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
                hideLens();
                return;
              }
              
              // Definisci margini per nascondere la lente vicino ai bordi
              const edgeMargin = 60; // pixels dal bordo
              const lensRadius = 40; // metà della dimensione della lente
              
              // Nascondi la lente se troppo vicina ai bordi
              if (x < edgeMargin || y < edgeMargin || 
                  x > rect.width - edgeMargin || y > rect.height - edgeMargin) {
                hideLens();
                return;
              }
              
              // Position lens al centro del cursore
              let lensX = x - lensRadius;
              let lensY = y - lensRadius;
              
              lens.style.left = lensX + 'px';
              lens.style.top = lensY + 'px';
              
              // Position result sempre fuori dall'immagine
              const viewportWidth = window.innerWidth;
              const viewportHeight = window.innerHeight;
              const imgRight = rect.right;
              const imgLeft = rect.left;
              
              let resultX, resultY;
              
              // Prova a posizionare a destra dell'immagine
              if (imgRight + result.offsetWidth + 20 < viewportWidth) {
                resultX = imgRight + 20;
                resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
              }
              // Altrimenti a sinistra dell'immagine
              else if (imgLeft - result.offsetWidth - 20 > 0) {
                resultX = imgLeft - result.offsetWidth - 20;
                resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
              }
              // Come fallback, posiziona sopra l'immagine se possibile
              else if (rect.top - result.offsetHeight - 20 > 0) {
                resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
                resultY = rect.top - result.offsetHeight - 20;
              }
              // Ultimo fallback: sotto l'immagine
              else {
                resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
                resultY = rect.bottom + 20;
              }
              
              // Assicura che il risultato rimanga all'interno del viewport
              resultX = Math.max(10, Math.min(resultX, viewportWidth - result.offsetWidth - 10));
              resultY = Math.max(10, Math.min(resultY, viewportHeight - result.offsetHeight - 10));
              
              result.style.left = resultX + 'px';
              result.style.top = resultY + 'px';
              
              // Assicurati che l'immagine sia completamente caricata prima di calcolare
              if (!img.complete || img.naturalWidth === 0) {
                console.log('Immagine non completamente caricata, aspettando...');
                img.onload = function() {
                  magnify(e); // Richiama la funzione quando l'immagine è caricata
                };
                return;
              }
              
              // Calculate magnified image position con correzioni
              const scaleX = img.naturalWidth / rect.width;
              const scaleY = img.naturalHeight / rect.height;
              
              const bgX = x * scaleX;
              const bgY = y * scaleY;
              
              console.log('Natural dimensions:', { 
                naturalWidth: img.naturalWidth, 
                naturalHeight: img.naturalHeight,
                scaleX, scaleY, bgX, bgY 
              });
              
              // Calcola posizionamento background con approccio più affidabile
              const magnifiedWidth = img.naturalWidth * zoom;
              const magnifiedHeight = img.naturalHeight * zoom;
              
              // Posizione del background per centrare l'area ingrandita
              const bgPosX = -(bgX * zoom - result.offsetWidth / 2);
              const bgPosY = -(bgY * zoom - result.offsetHeight / 2);
              
              // Applica background properties
              result.style.backgroundImage = `url('${img.src}')`;
              result.style.backgroundSize = `${magnifiedWidth}px ${magnifiedHeight}px`;
              result.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
              result.style.backgroundRepeat = 'no-repeat';
              
              console.log('Background positioning:', {
                backgroundSize: result.style.backgroundSize,
                backgroundPosition: result.style.backgroundPosition,
                magnifiedDimensions: { width: magnifiedWidth, height: magnifiedHeight },
                centerOffset: { x: bgPosX, y: bgPosY },
                mousePosition: { bgX, bgY }
              });
              
              // Mostra lente e risultato con ritardo se non già visibili
              if (!isLensVisible) {
                showLensWithDelay();
              }
            }
            
            function showLensWithDelay() {
              clearTimeout(lensTimeout);
              lensTimeout = setTimeout(() => {
                lens.style.display = 'block';
                result.style.display = 'block';
                // Forza reflow prima di applicare opacity
                lens.offsetHeight;
                result.offsetHeight;
                // Background-image è già impostato, non servono controlli aggiuntivi
                lens.style.opacity = '1';
                result.style.opacity = '1';
                isLensVisible = true;
                console.log('Lens and result shown, result dimensions:', { width: result.offsetWidth, height: result.offsetHeight });
              }, 200); // Ritardo di 200ms
            }
            
            function hideLens() {
              clearTimeout(lensTimeout);
              lens.style.opacity = '0';
              result.style.opacity = '0';
              setTimeout(() => {
                lens.style.display = 'none';
                result.style.display = 'none';
                isLensVisible = false;
              }, 200); // Tempo per l'animazione di fade out
            }
            
            // Add mouse events
            img.addEventListener('mousemove', magnify);
            img.addEventListener('mouseenter', magnify);
            img.addEventListener('mouseleave', function() {
              hideLens();
            });
            
            // Add click event for modal
            img.addEventListener('click', function() {
              modal.style.display = 'block';
              modalImg.src = this.src;
              captionText.innerHTML = this.alt || 'Clicca fuori dall\\'immagine o premi ESC per chiudere';
            });
          }
        });
        
        // Close modal on click
        closeBtn.addEventListener('click', function() {
          modal.style.display = 'none';
        });
        
        modal.addEventListener('click', function(e) {
          if (e.target === modal) {
            modal.style.display = 'none';
          }
        });
        
        // Close modal on ESC key
        document.addEventListener('keydown', function(e) {
          if (e.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = 'none';
          }
        });
        
        // Add active highlighting to TOC based on scroll position
        const sections = document.querySelectorAll('section[id]');
        const tocLinks = document.querySelectorAll('#TOC a[href^="#"]');
        
        function highlightTOC() {
          let current = '';
          const scrollY = window.pageYOffset;
          
          sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            
            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
              current = section.getAttribute('id');
            }
          });
          
          tocLinks.forEach(link => {
            const parent = link.parentElement;
            parent.classList.remove('active');
            
            if (link.getAttribute('href') === '#' + current) {
              parent.classList.add('active');
            }
          });
        }
        
        window.addEventListener('scroll', highlightTOC);
        highlightTOC(); // Initial call
      });
      </script>'''
        elif format_type == "pdf":
            format_yaml = '''format:
  pdf:
    geometry: margin=1in
    toc: true
    number-sections: true
    fig-width: 8
    fig-height: 6'''
        elif format_type == "docx":
            format_yaml = '''format:
  docx:
    toc: true
    number-sections: true
    fig-width: 8
    fig-height: 6'''
        else:
            # Default to HTML with custom styles
            format_yaml = format_yaml = '''format:
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
    embed-resources: true
    include-in-header: |
      <style>
      /* TOC Navigation Improvements */
      #TOC {
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
      }
      
      #TOC li {
        margin: 3px 0;
        border-radius: 6px;
        transition: all 0.2s ease;
      }
      
      #TOC li:hover {
        background-color: #e9ecef;
      }
      
      #TOC li.active {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196f3 !important;
        padding-left: 8px;
      }
      
      #TOC li.active > a {
        color: #1976d2 !important;
        font-weight: 600 !important;
      }
      
      #TOC a {
        display: block;
        padding: 8px 12px;
        text-decoration: none;
        color: #495057;
        border-radius: 4px;
        transition: all 0.2s ease;
      }
      
      #TOC a:hover {
        background-color: rgba(33, 150, 243, 0.1);
        color: #1976d2;
      }
      
      /* Table Professional Styling */
      .table {
        border-collapse: separate;
        border-spacing: 0;
        margin: 20px 0;
        font-size: 0.95em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
        background: white;
      }
      
      .table thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.5px;
        padding: 15px 12px;
        border: none;
        position: sticky;
        top: 0;
        z-index: 10;
      }
      
      .table tbody tr {
        transition: all 0.2s ease;
        border-bottom: 1px solid #e9ecef;
      }
      
      .table tbody tr:nth-child(even) {
        background-color: #f8f9fa;
      }
      
      .table tbody tr:hover {
        background-color: #e3f2fd;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      
      .table tbody td {
        padding: 12px 12px;
        border: none;
        vertical-align: middle;
      }
      
      .table tbody td:first-child {
        font-weight: 500;
        color: #495057;
      }
      
      /* Responsive table wrapper */
      .table-responsive {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      
      /* Metrics badges */
      .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 500;
        margin-right: 5px;
      }
      
      .metric-good { background-color: #d4edda; color: #155724; }
      .metric-warning { background-color: #fff3cd; color: #856404; }
      .metric-danger { background-color: #f8d7da; color: #721c24; }
      
      /* Card styling for sections */
      .content-block {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
      }
      
      /* Header improvements */
      h2, h3 {
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
        margin-top: 30px;
      }
      
      h2 {
        font-size: 1.8em;
        color: #34495e;
      }
      
      h3 {
        font-size: 1.4em;
        color: #7f8c8d;
        border-bottom: 1px solid #bdc3c7;
      }
      
      /* Image zoom styles */
      .zoomable-image {
        cursor: crosshair;
        transition: transform 0.2s;
        display: block;
        max-width: 100%;
        height: auto;
      }
      
      .zoomable-image:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      
      /* Magnifying glass lens - migliorata per ridurre confusione visiva */
      .magnifier-lens {
        position: absolute !important;
        border: 1px solid #2196F3;
        border-radius: 50%;
        cursor: none;
        width: 80px;
        height: 80px;
        display: none;
        pointer-events: none;
        box-shadow: 0 0 5px rgba(33, 150, 243, 0.3);
        background: rgba(33, 150, 243, 0.05);
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 999998 !important;
      }
      
      /* Magnified result container - usando background-image per affidabilità */
      .magnifier-result {
        position: fixed !important;
        border: 2px solid #2196F3;
        border-radius: 8px;
        display: none;
        width: 300px;
        height: 300px;
        z-index: 999999 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        background-color: #f0f0f0;
        background-repeat: no-repeat;
        background-size: 900px 900px; /* 3x zoom di base */
        opacity: 0;
        transition: opacity 0.2s ease;
        /* Fallback pattern visibile quando l'immagine non è caricata */
        background-image: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,.5) 10px, rgba(255,255,255,.5) 20px);
      }
      
      /* Modal for zoomed image */
      .image-modal {
        display: none;
        position: fixed !important;
        z-index: 1000000 !important;
        padding-top: 60px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.9);
        cursor: zoom-out;
      }
      
      .modal-content {
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 90%;
        animation: zoom 0.3s;
      }
      
      @keyframes zoom {
        from {transform: scale(0.5)}
        to {transform: scale(1)}
      }
      
      .close-modal {
        position: absolute !important;
        top: 15px;
        right: 35px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        transition: 0.3s;
        cursor: pointer;
      }
      
      .close-modal:hover,
      .close-modal:focus {
        color: #bbb;
        text-decoration: none;
      }
      
      /* Caption for zoomed image */
      .image-caption {
        margin: auto;
        display: block;
        width: 80%;
        max-width: 700px;
        text-align: center;
        color: #ccc;
        padding: 10px 0;
      }
      
      /* Magnifying glass icon on hover */
      .image-container {
        position: relative;
        display: inline-block;
      }
      
      .image-container::after {
        content: "🔍";
        position: absolute !important;
        top: 10px;
        right: 10px;
        font-size: 24px;
        opacity: 0;
        transition: opacity 0.3s;
        pointer-events: none;
        background: rgba(255,255,255,0.9);
        padding: 5px;
        border-radius: 50%;
      }
      
      .image-container:hover::after {
        opacity: 1;
      }
      </style>
      <script>
      document.addEventListener('DOMContentLoaded', function() {
        // Image zoom functionality
        const images = document.querySelectorAll('img');
        
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'image-modal';
        modal.innerHTML = '<span class="close-modal">&times;</span><img class="modal-content"><div class="image-caption"></div>';
        document.body.appendChild(modal);
        
        const modalImg = modal.querySelector('.modal-content');
        const captionText = modal.querySelector('.image-caption');
        const closeBtn = modal.querySelector('.close-modal');
        
        images.forEach(img => {
          // Skip small images (icons, etc.)
          if (img.width > 100) {
            // Wrap image in container
            const container = document.createElement('div');
            container.className = 'image-container';
            container.style.position = 'relative';
            img.parentNode.insertBefore(container, img);
            container.appendChild(img);
            
            // Add zoomable class
            img.classList.add('zoomable-image');
            
            // Create magnifying lens
            const lens = document.createElement('div');
            lens.className = 'magnifier-lens';
            container.appendChild(lens);
            
            // Create magnified result container (usando background-image)
            const result = document.createElement('div');
            result.className = 'magnifier-result';
            // Imposta l'immagine come background invece di elemento img interno
            result.style.backgroundImage = `url('${img.src}')`;
            container.appendChild(result);
            
            // Calculate zoom level
            const zoom = 3;
            
            // Variabili per gestire il ritardo e il controllo della lente
            let lensTimeout = null;
            let isLensVisible = false;
            
            // Set up magnifier functionality con miglioramenti
            function magnify(e) {
              const rect = img.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const y = e.clientY - rect.top;
              
              // Debug coordinate e dimensioni
              console.log('Mouse coordinates:', { x, y, imgWidth: img.width, imgHeight: img.height });
              console.log('Image loading state:', { complete: img.complete, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight });
              
              // Check if mouse is within image bounds
              if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
                hideLens();
                return;
              }
              
              // Definisci margini per nascondere la lente vicino ai bordi
              const edgeMargin = 60; // pixels dal bordo
              const lensRadius = 40; // metà della dimensione della lente
              
              // Nascondi la lente se troppo vicina ai bordi
              if (x < edgeMargin || y < edgeMargin || 
                  x > rect.width - edgeMargin || y > rect.height - edgeMargin) {
                hideLens();
                return;
              }
              
              // Position lens al centro del cursore
              let lensX = x - lensRadius;
              let lensY = y - lensRadius;
              
              lens.style.left = lensX + 'px';
              lens.style.top = lensY + 'px';
              
              // Position result sempre fuori dall'immagine
              const viewportWidth = window.innerWidth;
              const viewportHeight = window.innerHeight;
              const imgRight = rect.right;
              const imgLeft = rect.left;
              
              let resultX, resultY;
              
              // Prova a posizionare a destra dell'immagine
              if (imgRight + result.offsetWidth + 20 < viewportWidth) {
                resultX = imgRight + 20;
                resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
              }
              // Altrimenti a sinistra dell'immagine
              else if (imgLeft - result.offsetWidth - 20 > 0) {
                resultX = imgLeft - result.offsetWidth - 20;
                resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
              }
              // Come fallback, posiziona sopra l'immagine se possibile
              else if (rect.top - result.offsetHeight - 20 > 0) {
                resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
                resultY = rect.top - result.offsetHeight - 20;
              }
              // Ultimo fallback: sotto l'immagine
              else {
                resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
                resultY = rect.bottom + 20;
              }
              
              // Assicura che il risultato rimanga all'interno del viewport
              resultX = Math.max(10, Math.min(resultX, viewportWidth - result.offsetWidth - 10));
              resultY = Math.max(10, Math.min(resultY, viewportHeight - result.offsetHeight - 10));
              
              result.style.left = resultX + 'px';
              result.style.top = resultY + 'px';
              
              // Assicurati che l'immagine sia completamente caricata prima di calcolare
              if (!img.complete || img.naturalWidth === 0) {
                console.log('Immagine non completamente caricata, aspettando...');
                img.onload = function() {
                  magnify(e); // Richiama la funzione quando l'immagine è caricata
                };
                return;
              }
              
              // Calculate magnified image position con correzioni
              const scaleX = img.naturalWidth / rect.width;
              const scaleY = img.naturalHeight / rect.height;
              
              const bgX = x * scaleX;
              const bgY = y * scaleY;
              
              console.log('Natural dimensions:', { 
                naturalWidth: img.naturalWidth, 
                naturalHeight: img.naturalHeight,
                scaleX, scaleY, bgX, bgY 
              });
              
              // Calcola posizionamento background con approccio più affidabile
              const magnifiedWidth = img.naturalWidth * zoom;
              const magnifiedHeight = img.naturalHeight * zoom;
              
              // Posizione del background per centrare l'area ingrandita
              const bgPosX = -(bgX * zoom - result.offsetWidth / 2);
              const bgPosY = -(bgY * zoom - result.offsetHeight / 2);
              
              // Applica background properties
              result.style.backgroundImage = `url('${img.src}')`;
              result.style.backgroundSize = `${magnifiedWidth}px ${magnifiedHeight}px`;
              result.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
              result.style.backgroundRepeat = 'no-repeat';
              
              console.log('Background positioning:', {
                backgroundSize: result.style.backgroundSize,
                backgroundPosition: result.style.backgroundPosition,
                magnifiedDimensions: { width: magnifiedWidth, height: magnifiedHeight },
                centerOffset: { x: bgPosX, y: bgPosY },
                mousePosition: { bgX, bgY }
              });
              
              // Mostra lente e risultato con ritardo se non già visibili
              if (!isLensVisible) {
                showLensWithDelay();
              }
            }
            
            function showLensWithDelay() {
              clearTimeout(lensTimeout);
              lensTimeout = setTimeout(() => {
                lens.style.display = 'block';
                result.style.display = 'block';
                // Forza reflow prima di applicare opacity
                lens.offsetHeight;
                result.offsetHeight;
                // Background-image è già impostato, non servono controlli aggiuntivi
                lens.style.opacity = '1';
                result.style.opacity = '1';
                isLensVisible = true;
                console.log('Lens and result shown, result dimensions:', { width: result.offsetWidth, height: result.offsetHeight });
              }, 200); // Ritardo di 200ms
            }
            
            function hideLens() {
              clearTimeout(lensTimeout);
              lens.style.opacity = '0';
              result.style.opacity = '0';
              setTimeout(() => {
                lens.style.display = 'none';
                result.style.display = 'none';
                isLensVisible = false;
              }, 200); // Tempo per l'animazione di fade out
            }
            
            // Add mouse events
            img.addEventListener('mousemove', magnify);
            img.addEventListener('mouseenter', magnify);
            img.addEventListener('mouseleave', function() {
              hideLens();
            });
            
            // Add click event for modal
            img.addEventListener('click', function() {
              modal.style.display = 'block';
              modalImg.src = this.src;
              captionText.innerHTML = this.alt || 'Clicca fuori dall\\'immagine o premi ESC per chiudere';
            });
          }
        });
        
        // Close modal on click
        closeBtn.addEventListener('click', function() {
          modal.style.display = 'none';
        });
        
        modal.addEventListener('click', function(e) {
          if (e.target === modal) {
            modal.style.display = 'none';
          }
        });
        
        // Close modal on ESC key
        document.addEventListener('keydown', function(e) {
          if (e.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = 'none';
          }
        });
        
        // Add active highlighting to TOC based on scroll position
        const sections = document.querySelectorAll('section[id]');
        const tocLinks = document.querySelectorAll('#TOC a[href^="#"]');
        
        function highlightTOC() {
          let current = '';
          const scrollY = window.pageYOffset;
          
          sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            
            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
              current = section.getAttribute('id');
            }
          });
          
          tocLinks.forEach(link => {
            const parent = link.parentElement;
            parent.classList.remove('active');
            
            if (link.getAttribute('href') === '#' + current) {
              parent.classList.add('active');
            }
          });
        }
        
        window.addEventListener('scroll', highlightTOC);
        highlightTOC(); // Initial call
      });
      </script>'''
        
        qmd_content = f'''---
title: "{title}"
subtitle: "Analisi Completa del Modello {model_type}"
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
from pathlib import Path

# Carica i risultati del modello (using Path for cross-platform compatibility)
results_file = Path('{results_path_str}')
with open(results_file, 'r') as f:
    model_results = json.load(f)

# Configura stile grafici
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for report generation
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (8, 5)
```

## Executive Summary {{#sec-summary}}

### Panoramica del Modello

**Tipo di Modello:** {model_type}  
**Ordine:** {order}  
**Data Analisi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Risultati Chiave

```{{python}}
#| label: tbl-key-metrics
#| tbl-cap: "Metriche Principali del Modello"

metrics = model_results.get('metrics', {{}})
if metrics:
    metrics_df = pd.DataFrame([
        ['MAE (Mean Absolute Error)', f"{{metrics.get('mae', 'N/A'):.4f}}" if isinstance(metrics.get('mae'), (int, float)) else 'N/A'],
        ['RMSE (Root Mean Square Error)', f"{{metrics.get('rmse', 'N/A'):.4f}}" if isinstance(metrics.get('rmse'), (int, float)) else 'N/A'],
        ['MAPE (Mean Absolute Percentage Error)', f"{{metrics.get('mape', 'N/A'):.2f}}%" if isinstance(metrics.get('mape'), (int, float)) else 'N/A'],
        ['R² Score', f"{{metrics.get('r2_score', 'N/A'):.4f}}" if isinstance(metrics.get('r2_score'), (int, float)) else 'N/A'],
        ['AIC (Akaike Information Criterion)', f"{{metrics.get('aic', 'N/A'):.2f}}" if isinstance(metrics.get('aic'), (int, float)) else 'N/A'],
        ['BIC (Bayesian Information Criterion)', f"{{metrics.get('bic', 'N/A'):.2f}}" if isinstance(metrics.get('bic'), (int, float)) else 'N/A'],
    ], columns=['Metrica', 'Valore'])
    
    # Display as rendered HTML table
    from IPython.display import HTML, display
    html_table = metrics_df.to_html(index=False, classes='table table-striped', escape=False, table_id='metrics-table')
    display(HTML(html_table))
else:
    print("Nessuna metrica disponibile")
```

## Metodologia e Approccio {{#sec-methodology}}

### Processo di Selezione del Modello

Il modello è stato selezionato attraverso un processo sistematico che include:

1. **Analisi Esplorativa dei Dati**
   - Verifica della stazionarietà della serie temporale
   - Identificazione di trend e stagionalità
   - Analisi di autocorrelazione (ACF) e autocorrelazione parziale (PACF)

2. **Preprocessing dei Dati**
   - Gestione dei valori mancanti
   - Rimozione di outlier se necessario
   - Trasformazioni per raggiungere la stazionarietà

3. **Selezione dei Parametri**
   - Grid search sui parametri (p, d, q)
   - Ottimizzazione basata su criteri informativi (AIC, BIC)
   - Validazione incrociata temporale

### Parametri del Modello

```{{python}}
#| label: tbl-model-params
#| tbl-cap: "Parametri del Modello Selezionato"

model_info = model_results.get('model_info', {{}})
if model_info:
    params_data = []
    
    # Parametri base
    if 'order' in model_results:
        order = model_results['order']
        if isinstance(order, (list, tuple)) and len(order) >= 3:
            params_data.extend([
                ['p (Ordine Autoregressivo)', str(order[0])],
                ['d (Ordine di Differenziazione)', str(order[1])],
                ['q (Ordine Media Mobile)', str(order[2])]
            ])
    
    # Parametri stagionali se presenti
    if 'seasonal_order' in model_results:
        seasonal_order = model_results['seasonal_order']
        if isinstance(seasonal_order, (list, tuple)) and len(seasonal_order) >= 4:
            params_data.extend([
                ['P (Ordine Autoregressivo Stagionale)', str(seasonal_order[0])],
                ['D (Ordine Differenziazione Stagionale)', str(seasonal_order[1])],
                ['Q (Ordine Media Mobile Stagionale)', str(seasonal_order[2])],
                ['S (Periodicità Stagionale)', str(seasonal_order[3])]
            ])
    
    # Altri parametri del modello
    for key, value in model_info.items():
        if key not in ['order', 'seasonal_order'] and isinstance(value, (str, int, float)):
            params_data.append([key.replace('_', ' ').title(), str(value)])
    
    if params_data:
        params_df = pd.DataFrame(params_data, columns=['Parametro', 'Valore'])
        from IPython.display import HTML, display
        html_table = params_df.to_html(index=False, classes='table table-striped', escape=False, table_id='params-table')
        display(HTML(html_table))
    else:
        print("Parametri del modello non disponibili")
else:
    print("Informazioni del modello non disponibili")
```

## Analisi dei Risultati {{#sec-results}}

### Performance del Modello

```{{python}}
#| label: fig-metrics-comparison
#| fig-cap: "Confronto delle Metriche di Performance"

metrics = model_results.get('metrics', {{}})
if metrics:
    # Crea grafico delle metriche principali
    metric_names = []
    metric_values = []
    
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            metric_names.append(metric.upper())
            metric_values.append(value)
    
    if metric_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        ax.set_title('Metriche di Performance del Modello', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valore')
        
        # Aggiungi valori sulle barre
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                   f'{{value:.4f}}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Nessuna metrica numerica disponibile per la visualizzazione")
else:
    print("Metriche non disponibili")
```

### Diagnostica del Modello

La validazione del modello include diversi test diagnostici:

```{{python}}
#| label: diagnostics-summary
#| tbl-cap: "Riassunto Diagnostici del Modello"

diagnostics = model_results.get('diagnostics', {{}})
if diagnostics:
    diag_data = []
    for test, result in diagnostics.items():
        if isinstance(result, dict):
            status = "✓ Passato" if result.get('passed', False) else "✗ Fallito"
            p_value = result.get('p_value', 'N/A')
            if isinstance(p_value, (int, float)):
                p_value = f"{{p_value:.4f}}"
            diag_data.append([test.replace('_', ' ').title(), status, str(p_value)])
        else:
            diag_data.append([test.replace('_', ' ').title(), str(result), 'N/A'])
    
    if diag_data:
        diag_df = pd.DataFrame(diag_data, columns=['Test', 'Risultato', 'P-Value'])
        from IPython.display import HTML, display
        html_table = diag_df.to_html(index=False, classes='table table-striped', escape=False, table_id='diagnostics-table')
        display(HTML(html_table))
    else:
        print("Nessun risultato diagnostico disponibile")
else:
    print("Diagnostici non disponibili")
```

## Visualizzazioni {{#sec-plots}}

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

        # Add plot sections
        if plot_files:
            for plot_name, plot_path in plot_files.items():
                section_title = plot_name.replace('_', ' ').title()
                qmd_content += f'''
### {section_title}

![{section_title}]({plot_path}){{.figure-img}}

'''
        
        qmd_content += '''
## Previsioni e Raccomandazioni {#sec-recommendations}

### Interpretazione dei Risultati

```{python}
#| label: interpretation

# Analisi delle performance
metrics = model_results.get('metrics', {})
interpretation = []

if 'mae' in metrics and isinstance(metrics['mae'], (int, float)):
    mae_value = metrics['mae']
    interpretation.append(f"• **MAE ({mae_value:.4f})**: {'Ottimo' if mae_value < 0.1 else 'Buono' if mae_value < 0.5 else 'Accettabile' if mae_value < 1.0 else 'Da migliorare'} - L'errore medio assoluto indica la precisione delle previsioni.")

if 'mape' in metrics and isinstance(metrics['mape'], (int, float)):
    mape_value = metrics['mape']
    interpretation.append(f"• **MAPE ({mape_value:.2f}%)**: {'Eccellente' if mape_value < 5 else 'Buono' if mape_value < 10 else 'Accettabile' if mape_value < 20 else 'Da migliorare'} - Errore percentuale medio.")

if 'r2_score' in metrics and isinstance(metrics['r2_score'], (int, float)):
    r2_value = metrics['r2_score']
    interpretation.append(f"• **R² Score ({r2_value:.4f})**: {'Ottimo' if r2_value > 0.9 else 'Buono' if r2_value > 0.7 else 'Accettabile' if r2_value > 0.5 else 'Insufficiente'} - Varianza spiegata dal modello.")

if interpretation:
    for item in interpretation:
        print(item)
else:
    print("Non sono disponibili metriche per l'interpretazione automatica.")
```

### Raccomandazioni Operative

```{python}
#| label: recommendations

recommendations = []

# Raccomandazioni basate sulle performance
metrics = model_results.get('metrics', {})
if 'aic' in metrics and 'bic' in metrics:
    aic_val = metrics['aic']
    bic_val = metrics['bic']
    if isinstance(aic_val, (int, float)) and isinstance(bic_val, (int, float)):
        if abs(aic_val - bic_val) < 10:
            recommendations.append("✓ I criteri AIC e BIC sono allineati, indicando un buon bilanciamento complessità/performance")
        else:
            recommendations.append("⚠ Differenza significativa tra AIC e BIC - considerare modelli alternativi")

# Raccomandazioni basate sui residui
diagnostics = model_results.get('diagnostics', {})
if diagnostics:
    if diagnostics.get('ljung_box', {}).get('passed', False):
        recommendations.append("✓ I residui mostrano caratteristiche di rumore bianco")
    else:
        recommendations.append("⚠ I residui mostrano autocorrelazione - considerare modelli più complessi")

# Raccomandazioni generali
recommendations.extend([
    "• Monitorare la performance su nuovi dati per rilevare eventuali degradi",
    "• Considerare re-training periodico del modello",
    "• Valutare l'aggiunta di variabili esogene se disponibili",
    "• Implementare sistema di alerting per anomalie nelle previsioni"
])

for rec in recommendations:
    print(rec)
```

## Dettagli Tecnici {#sec-technical}

### Informazioni sul Processo

```{python}
#| label: tbl-process-info
#| tbl-cap: "Dettagli Tecnici del Processo"

process_info = [
    ['Data Generazione Report', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    ['Libreria', 'ARIMA Forecaster'],
    ['Versione Python', model_results.get('python_version', 'N/A')],
    ['Ambiente', model_results.get('environment', 'N/A')]
]

# Aggiungi informazioni sui dati se disponibili
if 'data_info' in model_results:
    data_info = model_results['data_info']
    for key, value in data_info.items():
        process_info.append([key.replace('_', ' ').title(), str(value)])

process_df = pd.DataFrame(process_info, columns=['Parametro', 'Valore'])
from IPython.display import HTML, display
html_table = process_df.to_html(index=False, classes='table table-striped', escape=False, table_id='process-table')
display(HTML(html_table))
```

### Configurazione Completa

```{python}
#| label: full-config
#| code-fold: false
#| echo: false

import pandas as pd
from IPython.display import HTML, display

def flatten_dict(d, parent_key='', sep='.'):
    """Appiattisce un dizionario nested in formato chiave-valore"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Formatta la chiave in modo più leggibile
        formatted_key = new_key.replace('_', ' ').title()
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Se è una lista, mostrala come stringa formattata
            if len(v) > 0 and isinstance(v[0], (int, float)):
                # Se è una lista di numeri, mostra solo i primi elementi
                if len(v) > 5:
                    v_str = f"[{', '.join(map(str, v[:5]))}, ... ({len(v)} valori totali)]"
                else:
                    v_str = f"[{', '.join(map(str, v))}]"
            else:
                v_str = str(v)
            items.append((formatted_key, v_str))
        else:
            # Formatta i valori numerici
            if isinstance(v, float):
                if abs(v) > 1000:
                    v_str = f"{v:,.2f}"
                else:
                    v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            items.append((formatted_key, v_str))
    return dict(items)

# Crea tabella con configurazione completa
config_data = flatten_dict(model_results)

# Organizza i dati in categorie
categories = {
    'Informazioni Modello': [],
    'Parametri': [],
    'Metriche Performance': [],
    'Diagnostica': [],
    'Training Data': [],
    'Forecast': [],
    'Altro': []
}

for key, value in config_data.items():
    key_lower = key.lower()
    if 'model' in key_lower or 'type' in key_lower or 'status' in key_lower:
        categories['Informazioni Modello'].append((key, value))
    elif 'order' in key_lower or 'seasonal' in key_lower or 'trend' in key_lower:
        categories['Parametri'].append((key, value))
    elif any(metric in key_lower for metric in ['mae', 'rmse', 'mape', 'r2', 'aic', 'bic', 'accuracy']):
        categories['Metriche Performance'].append((key, value))
    elif any(diag in key_lower for diag in ['ljung', 'jarque', 'heteroscedasticity', 'stationarity', 'residual']):
        categories['Diagnostica'].append((key, value))
    elif 'training' in key_lower or 'observations' in key_lower:
        categories['Training Data'].append((key, value))
    elif 'forecast' in key_lower or 'prediction' in key_lower:
        categories['Forecast'].append((key, value))
    else:
        categories['Altro'].append((key, value))

# Crea HTML con tabelle separate per categoria
html_output = ""
for category, items in categories.items():
    if items:
        html_output += f"<h4 style='color: #495057; margin-top: 20px;'>{category}</h4>"
        df = pd.DataFrame(items, columns=['Parametro', 'Valore'])
        html_table = df.to_html(
            index=False, 
            classes='table table-striped table-hover', 
            escape=False,
            table_id=f'config-{category.lower().replace(" ", "-")}'
        )
        html_output += html_table

display(HTML(html_output))
```

---

*Report generato automaticamente dalla libreria ARIMA Forecaster*
'''
        
        return qmd_content
    
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
            
            if result.returncode != 0:
                self.logger.error(f"Quarto render failed: {result.stderr}")
                raise ForecastError(f"Quarto rendering failed: {result.stderr}")
            
            # Move the generated file to final location
            if temp_output_path.exists():
                import shutil
                # Ensure output directory exists
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_output_path), str(final_output_path))
                
                # Fix image paths and inject magnifying lens functionality
                if format_type == "html":
                    self._fix_html_resources(final_output_path, output_filename)
                
                return final_output_path
            else:
                raise ForecastError(f"Expected output file not found: {temp_output_path}")
            
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
            # Try to convert to basic types
            try:
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
            
            # 4. Inject magnifying lens CSS and JavaScript before </head>
            magnifier_code = '''
<style>
/* Image zoom styles */
.zoomable-image {
  cursor: crosshair;
  transition: transform 0.2s;
  display: block;
  max-width: 100%;
  height: auto;
}

.zoomable-image:hover {
  transform: scale(1.02);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Magnifying glass lens - migliorata per ridurre confusione visiva */
.magnifier-lens {
  position: absolute !important;
  border: 1px solid #2196F3;
  border-radius: 50%;
  cursor: none;
  width: 80px;
  height: 80px;
  display: none;
  pointer-events: none;
  box-shadow: 0 0 5px rgba(33, 150, 243, 0.3);
  background: rgba(33, 150, 243, 0.05);
  opacity: 0;
  transition: opacity 0.3s ease;
  z-index: 999998 !important;
}

/* Magnified result container - usando background-image per affidabilit\u00e0 */
.magnifier-result {
  position: fixed !important;
  border: 2px solid #2196F3;
  border-radius: 8px;
  display: none;
  width: 300px;
  height: 300px;
  z-index: 999999 !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  background-color: #f0f0f0;
  background-repeat: no-repeat;
  background-size: 900px 900px; /* 3x zoom di base */
  opacity: 0;
  transition: opacity 0.2s ease;
  /* Fallback pattern visibile quando l'immagine non \u00e8 caricata */
  background-image: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,.5) 10px, rgba(255,255,255,.5) 20px);
}

/* Modal for zoomed image */
.image-modal {
  display: none;
  position: fixed !important;
  z-index: 1000000 !important;
  padding-top: 60px;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.9);
  cursor: zoom-out;
}

.modal-content {
  margin: auto;
  display: block;
  max-width: 90%;
  max-height: 90%;
  animation: zoom 0.3s;
}

@keyframes zoom {
  from {transform: scale(0.5)}
  to {transform: scale(1)}
}

.close-modal {
  position: absolute !important;
  top: 15px;
  right: 35px;
  color: #f1f1f1;
  font-size: 40px;
  font-weight: bold;
  transition: 0.3s;
  cursor: pointer;
}

.close-modal:hover,
.close-modal:focus {
  color: #bbb;
  text-decoration: none;
}

/* Caption for zoomed image */
.image-caption {
  margin: auto;
  display: block;
  width: 80%;
  max-width: 700px;
  text-align: center;
  color: #ccc;
  padding: 10px 0;
}

/* Magnifying glass icon on hover */
.image-container {
  position: relative;
  display: inline-block;
}

.image-container::after {
  content: "🔍";
  position: absolute !important;
  top: 10px;
  right: 10px;
  font-size: 24px;
  opacity: 0;
  transition: opacity 0.3s;
  pointer-events: none;
  background: rgba(255,255,255,0.9);
  padding: 5px;
  border-radius: 50%;
}

.image-container:hover::after {
  opacity: 1;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Image zoom functionality
  const images = document.querySelectorAll('img');
  
  // Create modal container
  const modal = document.createElement('div');
  modal.className = 'image-modal';
  modal.innerHTML = '<span class="close-modal">&times;</span><img class="modal-content"><div class="image-caption"></div>';
  document.body.appendChild(modal);
  
  const modalImg = modal.querySelector('.modal-content');
  const captionText = modal.querySelector('.image-caption');
  const closeBtn = modal.querySelector('.close-modal');
  
  images.forEach(img => {
    // Skip small images (icons, etc.)
    if (img.width > 100) {
      // Wrap image in container
      const container = document.createElement('div');
      container.className = 'image-container';
      container.style.position = 'relative';
      img.parentNode.insertBefore(container, img);
      container.appendChild(img);
      
      // Add zoomable class
      img.classList.add('zoomable-image');
      
      // Create magnifying lens
      const lens = document.createElement('div');
      lens.className = 'magnifier-lens';
      container.appendChild(lens);
      
      // Create magnified result container (usando background-image)
      const result = document.createElement('div');
      result.className = 'magnifier-result';
      // Imposta l'immagine come background invece di elemento img interno
      result.style.backgroundImage = `url('${img.src}')`;
      container.appendChild(result);
      
      // Calculate zoom level
      const zoom = 3;
      
      // Variabili per gestire il ritardo e il controllo della lente
      let lensTimeout = null;
      let isLensVisible = false;
      
      // Set up magnifier functionality con miglioramenti
      function magnify(e) {
        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Debug coordinate
        console.log('Mouse coordinates:', { x, y, imgWidth: img.width, imgHeight: img.height });
        
        // Check if mouse is within image bounds
        if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
          hideLens();
          return;
        }
        
        // Definisci margini per nascondere la lente vicino ai bordi
        const edgeMargin = 60; // pixels dal bordo
        const lensRadius = 40; // metà della dimensione della lente
        
        // Nascondi la lente se troppo vicina ai bordi
        if (x < edgeMargin || y < edgeMargin || 
            x > rect.width - edgeMargin || y > rect.height - edgeMargin) {
          hideLens();
          return;
        }
        
        // Position lens al centro del cursore
        let lensX = x - lensRadius;
        let lensY = y - lensRadius;
        
        lens.style.left = lensX + 'px';
        lens.style.top = lensY + 'px';
        
        // Position result sempre fuori dall'immagine
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const imgRight = rect.right;
        const imgLeft = rect.left;
        
        let resultX, resultY;
        
        // Prova a posizionare a destra dell'immagine
        if (imgRight + result.offsetWidth + 20 < viewportWidth) {
          resultX = imgRight + 20;
          resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
        }
        // Altrimenti a sinistra dell'immagine
        else if (imgLeft - result.offsetWidth - 20 > 0) {
          resultX = imgLeft - result.offsetWidth - 20;
          resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
        }
        // Come fallback, posiziona sopra l'immagine se possibile
        else if (rect.top - result.offsetHeight - 20 > 0) {
          resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
          resultY = rect.top - result.offsetHeight - 20;
        }
        // Ultimo fallback: sotto l'immagine
        else {
          resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
          resultY = rect.bottom + 20;
        }
        
        // Assicura che il risultato rimanga all'interno del viewport
        resultX = Math.max(10, Math.min(resultX, viewportWidth - result.offsetWidth - 10));
        resultY = Math.max(10, Math.min(resultY, viewportHeight - result.offsetHeight - 10));
        
        result.style.left = resultX + 'px';
        result.style.top = resultY + 'px';
        
        // Assicurati che l'immagine sia completamente caricata prima di calcolare
        if (!img.complete || img.naturalWidth === 0) {
          console.log('Immagine non completamente caricata, aspettando...');
          img.onload = function() {
            magnify(e); // Richiama la funzione quando l'immagine è caricata
          };
          return;
        }
        
        // Calculate magnified image position con correzioni
        const scaleX = img.naturalWidth / rect.width;
        const scaleY = img.naturalHeight / rect.height;
        
        const bgX = x * scaleX;
        const bgY = y * scaleY;
        
        console.log('Natural dimensions:', { 
          naturalWidth: img.naturalWidth, 
          naturalHeight: img.naturalHeight,
          scaleX, scaleY, bgX, bgY 
        });
        
        // Applica background image con posizionamento calcolato
        const magnifiedWidth = img.naturalWidth * zoom;
        const magnifiedHeight = img.naturalHeight * zoom;
        const bgPosX = -(bgX * zoom - result.offsetWidth / 2);
        const bgPosY = -(bgY * zoom - result.offsetHeight / 2);
        
        result.style.backgroundImage = `url('${img.src}')`;
        result.style.backgroundSize = `${magnifiedWidth}px ${magnifiedHeight}px`;
        result.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
        result.style.backgroundRepeat = 'no-repeat';
        
        console.log('Background positioning:', {
          backgroundSize: result.style.backgroundSize,
          backgroundPosition: result.style.backgroundPosition
        });
        
        // Mostra lente e risultato con ritardo se non già visibili
        if (!isLensVisible) {
          showLensWithDelay();
        }
      }
      
      function showLensWithDelay() {
        clearTimeout(lensTimeout);
        lensTimeout = setTimeout(() => {
          lens.style.display = 'block';
          result.style.display = 'block';
          // Forza reflow prima di applicare opacity
          lens.offsetHeight;
          result.offsetHeight;
          lens.style.opacity = '1';
          result.style.opacity = '1';
          isLensVisible = true;
        }, 200); // Ritardo di 200ms
      }
      
      function hideLens() {
        clearTimeout(lensTimeout);
        lens.style.opacity = '0';
        result.style.opacity = '0';
        setTimeout(() => {
          lens.style.display = 'none';
          result.style.display = 'none';
          isLensVisible = false;
        }, 200); // Tempo per l'animazione di fade out
      }
      
      // Add mouse events
      img.addEventListener('mousemove', magnify);
      img.addEventListener('mouseenter', magnify);
      img.addEventListener('mouseleave', function() {
        hideLens();
      });
      
      // Add click event for modal
      img.addEventListener('click', function() {
        modal.style.display = 'block';
        modalImg.src = this.src;
        captionText.innerHTML = this.alt || 'Clicca fuori dall\\'immagine o premi ESC per chiudere';
      });
    }
  });
  
  // Close modal on click
  closeBtn.addEventListener('click', function() {
    modal.style.display = 'none';
  });
  
  modal.addEventListener('click', function(e) {
    if (e.target === modal) {
      modal.style.display = 'none';
    }
  });
  
  // Close modal on ESC key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && modal.style.display === 'block') {
      modal.style.display = 'none';
    }
  });
});
</script>
'''
            
            # Inject the magnifier code before </head> tag
            head_pattern = r'</head>'
            if re.search(head_pattern, html_content, re.IGNORECASE):
                html_content = re.sub(
                    head_pattern, 
                    magnifier_code + '\n</head>', 
                    html_content, 
                    flags=re.IGNORECASE
                )
                self.logger.info(f"Injected magnifying lens functionality into HTML report")
            else:
                self.logger.warning("Could not find </head> tag to inject magnifying lens code")
            
            # Write back the fixed HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"Fixed all resource paths and injected magnifying lens in HTML report: {html_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not fix HTML resources or inject magnifying lens: {e}")
            # Don't fail the entire report generation for this issue
    
    def create_comparison_report(
        self,
        models_results: Dict[str, Dict[str, Any]],
        report_title: str = "ARIMA Models Comparison Report",
        output_filename: str = None,
        format_type: str = "html"
    ) -> Path:
        """
        Generate comparative analysis report for multiple models.
        
        Args:
            models_results: Dictionary of model names and their results
            report_title: Title for the report
            output_filename: Custom filename for the report
            format_type: Output format ('html', 'pdf', 'docx')
            
        Returns:
            Path to generated report
        """
        try:
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
            qmd_content = self._generate_comparison_qmd(report_title, models_results, results_path)
            
            qmd_path = report_dir / "comparison_report.qmd"
            with open(qmd_path, 'w', encoding='utf-8') as f:
                f.write(qmd_content)
            
            # Render report
            output_path = self._render_report(qmd_path, format_type, output_filename)
            
            self.logger.info(f"Comparison report generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {str(e)}")
            raise ForecastError(f"Comparison report generation failed: {str(e)}")
    
    def _generate_comparison_qmd(
        self, 
        title: str, 
        models_results: Dict[str, Dict[str, Any]], 
        results_path: Path
    ) -> str:
        """Generate Quarto document for models comparison."""
        
        qmd_content = f'''---
title: "{title}"
subtitle: "Confronto Comparativo dei Modelli ARIMA/SARIMA"
author: "ARIMA Forecaster Library"
date: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
format:
  html:
    theme: cosmo
    toc: true
    toc-depth: 3
    code-fold: true
    code-summary: "Mostra codice"
    fig-width: 12
    fig-height: 8
  pdf:
    geometry: margin=1in
    toc: true
    number-sections: true
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
print(models_df.to_markdown(index=False))
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
        print("Nessuna metrica numerica comune trovata per il confronto")
else:
    print("Dati delle metriche non disponibili")
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

print(ranking_df.to_markdown(index=False))
```

## Raccomandazioni Finali

```{{python}}
#| label: final-recommendations

if len(ranking_df) > 0:
    best_model = ranking_df.iloc[0]['Modello']
    print(f"### Modello Raccomandato: {{best_model}}")
    print()
    
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
        print(rec)
else:
    print("Nessun modello disponibile per le raccomandazioni")
```

---

*Report comparativo generato automaticamente dalla libreria ARIMA Forecaster*
'''
        
        return qmd_content