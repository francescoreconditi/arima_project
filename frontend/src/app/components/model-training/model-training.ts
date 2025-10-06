// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Componente Angular per training modelli ARIMA
// ============================================

import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ArimaApiService } from '../../services/arima-api.service';
import { ModelTrainingRequest, ModelInfo, TimeSeriesData } from '../../models/api.models';

@Component({
  selector: 'app-model-training',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './model-training.html',
  styleUrl: './model-training.scss'
})
export class ModelTraining {
  // Form data
  modelType: 'arima' | 'sarima' | 'sarimax' = 'arima';
  csvData = '';
  p = 1;
  d = 1;
  q = 1;

  // Seasonal parameters (per SARIMA)
  P = 1;
  D = 1;
  Q = 1;
  s = 12;

  // State
  isTraining = false;
  trainedModel: ModelInfo | null = null;
  errorMessage = '';

  // Esponi Object per template
  Object = Object;

  constructor(private apiService: ArimaApiService) {}

  /**
   * Parse CSV data dal textarea
   */
  private parseCSVData(): TimeSeriesData | null {
    try {
      const lines = this.csvData.trim().split('\n');
      const timestamps: string[] = [];
      const values: number[] = [];

      for (const line of lines) {
        const [timestamp, value] = line.split(',').map(s => s.trim());
        if (timestamp && value) {
          timestamps.push(timestamp);
          values.push(parseFloat(value));
        }
      }

      if (timestamps.length === 0) {
        throw new Error('Nessun dato valido trovato');
      }

      return { timestamps, values };
    } catch (error) {
      this.errorMessage = `Errore parsing CSV: ${error}`;
      return null;
    }
  }

  /**
   * Avvia il training del modello
   */
  onTrainModel(): void {
    this.errorMessage = '';
    this.trainedModel = null;

    const data = this.parseCSVData();
    if (!data) return;

    const request: ModelTrainingRequest = {
      model_type: this.modelType,
      data: data,
      order: { p: this.p, d: this.d, q: this.q }
    };

    // Aggiungi parametri stagionali per SARIMA/SARIMAX
    if (this.modelType === 'sarima' || this.modelType === 'sarimax') {
      request.seasonal_order = {
        P: this.P,
        D: this.D,
        Q: this.Q,
        s: this.s
      };
    }

    this.isTraining = true;

    this.apiService.trainModel(request).subscribe({
      next: (result) => {
        this.trainedModel = result;
        this.isTraining = false;
        console.log('Modello addestrato:', result);

        // Polling per verificare completamento
        this.pollModelStatus(result.model_id);
      },
      error: (error) => {
        this.errorMessage = `Errore training: ${error.error?.detail || error.message}`;
        this.isTraining = false;
      }
    });
  }

  /**
   * Verifica periodicamente lo stato del modello
   */
  private pollModelStatus(modelId: string, maxAttempts = 30, interval = 2000): void {
    let attempts = 0;

    const checkStatus = () => {
      if (attempts >= maxAttempts) {
        this.errorMessage = 'Timeout: training non completato';
        return;
      }

      this.apiService.checkModelStatus(modelId).subscribe({
        next: (model) => {
          this.trainedModel = model;

          if (model.status === 'completed') {
            console.log('Training completato!', model);
          } else if (model.status === 'failed') {
            this.errorMessage = 'Training fallito';
          } else {
            // Continua polling
            attempts++;
            setTimeout(checkStatus, interval);
          }
        },
        error: (error) => {
          console.error('Errore verifica stato:', error);
          attempts++;
          setTimeout(checkStatus, interval);
        }
      });
    };

    checkStatus();
  }

  /**
   * Esempio di dati CSV per testing
   */
  loadSampleData(): void {
    this.csvData = `2024-01-01,100
2024-01-02,105
2024-01-03,103
2024-01-04,108
2024-01-05,107
2024-01-06,110
2024-01-07,112
2024-01-08,109
2024-01-09,115
2024-01-10,118`;
  }
}
