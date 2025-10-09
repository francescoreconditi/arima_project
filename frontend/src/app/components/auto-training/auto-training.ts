// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-07
// Scopo: Componente Angular per auto-training modelli ARIMA/SARIMA/SARIMAX
// ============================================

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ArimaApiService } from '../../services/arima-api.service';
import { AutoSelectionRequest, ModelInfo, TimeSeriesData } from '../../models/api.models';

interface ModelCandidate {
  order: number[];
  seasonal_order?: number[];
  aic?: number;
  bic?: number;
  selected: boolean;
}

@Component({
  selector: 'app-auto-training',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './auto-training.html',
  styleUrl: './auto-training.scss'
})
export class AutoTraining implements OnInit {
  // Dataset selection
  availableDatasets: any[] = [];
  selectedDatasetId = '';
  selectedDataset: any = null;
  dataInputMode: 'dataset' | 'manual' = 'dataset';

  // Form data
  modelType: 'auto-arima' | 'auto-sarima' | 'auto-sarimax' = 'auto-arima';
  csvData = '';

  // Parametri ricerca automatica
  maxP = 3;
  maxD = 2;
  maxQ = 3;
  seasonalPeriod = 12;
  criterion: 'aic' | 'bic' = 'aic';
  maxModels = 20;

  // State
  isSearching = false;
  modelCandidates: ModelCandidate[] = [];
  selectedModelIndex: number | null = null;
  searchResults: any = null;
  errorMessage = '';
  trainedModel: ModelInfo | null = null;
  isTraining = false;

  // Esponi Object per template
  Object = Object;

  constructor(private apiService: ArimaApiService) {}

  ngOnInit(): void {
    this.loadAvailableDatasets();
  }

  /**
   * Carica lista dataset disponibili
   */
  loadAvailableDatasets(): void {
    this.apiService.listDatasets().subscribe({
      next: (datasets) => {
        this.availableDatasets = datasets;
        console.log('Dataset disponibili:', datasets);
      },
      error: (error) => {
        console.error('Errore caricamento dataset:', error);
      }
    });
  }

  /**
   * Gestisce selezione dataset
   */
  onDatasetSelected(): void {
    if (!this.selectedDatasetId) {
      this.selectedDataset = null;
      return;
    }

    const dataset = this.availableDatasets.find(d => d.dataset_id === this.selectedDatasetId);
    if (dataset) {
      this.selectedDataset = dataset;
      console.log('Dataset selezionato:', dataset);
    }
  }

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
   * Avvia la ricerca automatica dei modelli
   */
  onSearchModels(): void {
    this.errorMessage = '';
    this.modelCandidates = [];
    this.searchResults = null;
    this.selectedModelIndex = null;

    // NOTA: L'endpoint /models/auto-select richiede obbligatoriamente il campo 'data'
    // e non supporta 'dataset_id'. Quindi funziona solo in modalità manuale.
    if (this.dataInputMode === 'dataset') {
      this.errorMessage = 'Auto-selection con dataset non supportata dal backend. Usa modalità manuale o vai in Training standard.';
      return;
    }

    // Ottieni dati da input manuale
    const data = this.parseCSVData();
    if (!data) return;

    const request: AutoSelectionRequest = {
      data: data,
      model_type: this.modelType.replace('auto-', ''), // 'auto-arima' -> 'arima'
      max_p: this.maxP,
      max_d: this.maxD,
      max_q: this.maxQ,
      seasonal: this.modelType !== 'auto-arima',
      seasonal_period: this.seasonalPeriod,
      criterion: this.criterion
    };

    this.isSearching = true;

    this.apiService.autoSelectModel(request).subscribe({
      next: (result) => {
        this.searchResults = result;
        this.isSearching = false;

        console.log('Ricerca completata:', result);

        // Prepara lista modelli candidati
        this.modelCandidates = result.all_results.slice(0, this.maxModels).map((model: any, index: number) => ({
          order: model.order,
          seasonal_order: model.seasonal_order,
          aic: model.aic,
          bic: model.bic,
          selected: index === 0 // Primo modello selezionato per default
        }));

        if (this.modelCandidates.length > 0) {
          this.selectedModelIndex = 0;
        }
      },
      error: (error) => {
        this.errorMessage = `Errore ricerca: ${error.error?.detail || error.message}`;
        this.isSearching = false;
      }
    });
  }

  /**
   * Seleziona un modello dalla lista
   */
  selectModel(index: number): void {
    this.selectedModelIndex = index;
    this.modelCandidates.forEach((m, i) => m.selected = i === index);
  }

  /**
   * Addestra il modello selezionato
   */
  onTrainSelectedModel(): void {
    if (this.selectedModelIndex === null) {
      this.errorMessage = 'Seleziona un modello dalla lista';
      return;
    }

    const selectedCandidate = this.modelCandidates[this.selectedModelIndex];
    const data = this.parseCSVData();
    if (!data) return;

    // Determina il tipo di modello basandosi su seasonal_order
    let modelType: 'arima' | 'sarima' | 'sarimax' = 'arima';
    if (selectedCandidate.seasonal_order && selectedCandidate.seasonal_order.length > 0) {
      modelType = this.modelType === 'auto-sarimax' ? 'sarimax' : 'sarima';
    }

    const order = selectedCandidate.order;
    const request: any = {
      model_type: modelType,
      data: data,
      order: {
        p: order[0],
        d: order[1],
        q: order[2]
      }
    };

    // Aggiungi parametri stagionali se presenti
    if (selectedCandidate.seasonal_order) {
      const sOrder = selectedCandidate.seasonal_order;
      request.seasonal_order = {
        p: order[0],
        d: order[1],
        q: order[2],
        P: sOrder[0],
        D: sOrder[1],
        Q: sOrder[2],
        s: sOrder[3]
      };
    }

    this.isTraining = true;
    this.errorMessage = '';

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
2024-01-10,118
2024-01-11,120
2024-01-12,122
2024-01-13,119
2024-01-14,125
2024-01-15,128
2024-01-16,126
2024-01-17,130
2024-01-18,132
2024-01-19,129
2024-01-20,135`;
  }

  /**
   * Reset completo form
   */
  resetForm(): void {
    this.modelCandidates = [];
    this.selectedModelIndex = null;
    this.searchResults = null;
    this.errorMessage = '';
    this.trainedModel = null;
  }

  /**
   * Formatta l'ordine del modello per visualizzazione
   */
  formatOrder(order: number[], seasonal?: number[]): string {
    if (!seasonal || seasonal.length === 0) {
      return `ARIMA(${order[0]},${order[1]},${order[2]})`;
    }
    return `SARIMA(${order[0]},${order[1]},${order[2]})(${seasonal[0]},${seasonal[1]},${seasonal[2]},${seasonal[3]})`;
  }
}
