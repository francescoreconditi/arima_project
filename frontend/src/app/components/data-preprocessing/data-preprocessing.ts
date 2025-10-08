// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-08
// Scopo: Componente Angular per preprocessing dataset
// ============================================

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ArimaApiService } from '../../services/arima-api.service';

@Component({
  selector: 'app-data-preprocessing',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './data-preprocessing.html',
  styleUrl: './data-preprocessing.scss'
})
export class DataPreprocessing implements OnInit {
  // Dataset selection
  availableDatasets: any[] = [];
  selectedDatasetId = '';
  selectedDataset: any = null;

  // Preprocessing steps configuration
  handleMissing = true;
  missingMethod: 'interpolate' | 'forward_fill' | 'backward_fill' | 'drop' = 'interpolate';

  removeOutliers = false;
  outlierMethod: 'iqr' | 'zscore' | 'modified_zscore' = 'iqr';

  makeStationary = false;
  stationarityMethod: 'difference' | 'log_difference' = 'difference';

  normalize = false;
  normalizeMethod: 'minmax' | 'standard' = 'standard';

  // State
  isProcessing = false;
  processingJobId = '';
  processingProgress = 0;
  processingSuccess = false;
  processedDatasetId = '';
  errorMessage = '';

  constructor(private apiService: ArimaApiService) {}

  ngOnInit(): void {
    this.loadAvailableDatasets();
  }

  loadAvailableDatasets(): void {
    this.apiService.listDatasets().subscribe({
      next: (datasets) => {
        this.availableDatasets = datasets;
      },
      error: (error) => {
        this.errorMessage = 'Errore caricamento dataset';
      }
    });
  }

  onDatasetSelected(): void {
    if (!this.selectedDatasetId) {
      this.selectedDataset = null;
      return;
    }
    const dataset = this.availableDatasets.find(d => d.dataset_id === this.selectedDatasetId);
    if (dataset) {
      this.selectedDataset = dataset;
    }
  }

  startPreprocessing(): void {
    if (!this.selectedDatasetId) {
      this.errorMessage = 'Seleziona un dataset';
      return;
    }

    const pipeline: any[] = [];
    if (this.handleMissing) pipeline.push({ type: 'handle_missing', method: this.missingMethod });
    if (this.removeOutliers) pipeline.push({ type: 'remove_outliers', method: this.outlierMethod });
    // makeStationary non supportato dal backend, uso 'difference' se era 'difference', altrimenti 'log_transform'
    if (this.makeStationary) {
      if (this.stationarityMethod === 'difference') {
        pipeline.push({ type: 'difference', lag: 1 });
      } else {
        pipeline.push({ type: 'log_transform' });
      }
    }
    if (this.normalize) pipeline.push({ type: 'normalize', method: this.normalizeMethod });

    if (pipeline.length === 0) {
      this.errorMessage = 'Seleziona almeno uno step';
      return;
    }

    this.isProcessing = true;
    const request = {
      dataset_id: this.selectedDatasetId,
      preprocessing_steps: pipeline,
      output_dataset_name: `${this.selectedDataset.name}_processed`
    };

    console.log('Preprocessing request:', JSON.stringify(request, null, 2));

    this.apiService.preprocessDataset(request).subscribe({
      next: (response) => {
        this.processingJobId = response.job_id;
        this.pollProcessingStatus();
      },
      error: (error) => {
        this.errorMessage = `Errore: ${error.error?.detail || error.message}`;
        this.isProcessing = false;
      }
    });
  }

  private pollProcessingStatus(): void {
    const intervalId = setInterval(() => {
      this.apiService.getDataJobStatus(this.processingJobId).subscribe({
        next: (status) => {
          this.processingProgress = (status.progress || 0) * 100;
          if (status.status === 'completed') {
            clearInterval(intervalId);
            this.isProcessing = false;
            this.processingSuccess = true;
            this.processedDatasetId = status.dataset_id;
            setTimeout(() => {
              this.loadAvailableDatasets();
              this.resetForm();
            }, 3000);
          } else if (status.status === 'failed') {
            clearInterval(intervalId);
            this.isProcessing = false;
            this.errorMessage = status.error_message || 'Errore';
          }
        },
        error: () => {
          clearInterval(intervalId);
          this.isProcessing = false;
        }
      });
    }, 1000);
  }

  resetForm(): void {
    this.selectedDatasetId = '';
    this.selectedDataset = null;
    this.processingSuccess = false;
    this.processedDatasetId = '';
  }
}
