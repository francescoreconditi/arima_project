// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Componente Angular per generazione forecast
// ============================================

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ArimaApiService } from '../../services/arima-api.service';
import { ForecastRequest, ForecastResponse, ModelInfo } from '../../models/api.models';

@Component({
  selector: 'app-model-forecast',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './model-forecast.html',
  styleUrl: './model-forecast.scss'
})
export class ModelForecast implements OnInit {
  // Available models
  availableModels: ModelInfo[] = [];
  selectedModelId = '';

  // Forecast parameters
  forecastSteps = 30;
  confidenceLevel = 0.95;
  includeConfidenceIntervals = true;

  // State
  isForecasting = false;
  forecastResult: ForecastResponse | null = null;
  errorMessage = '';

  // Chart data
  chartLabels: string[] = [];
  forecastData: number[] = [];
  lowerBound: number[] = [];
  upperBound: number[] = [];

  // Plotly chart
  plotlyChartUrl: SafeHtml | null = null;
  isLoadingChart = false;

  // Report generation
  isGeneratingReport = false;
  reportJobId: string | null = null;
  reportProgress = 0;
  reportGenerationError = '';
  generatedReportUrls: string[] = [];
  private downloadTriggered = false; // Flag per evitare download multipli

  constructor(
    private apiService: ArimaApiService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {
    this.loadAvailableModels();
  }

  /**
   * Carica la lista di modelli disponibili
   */
  loadAvailableModels(): void {
    this.apiService.listModels().subscribe({
      next: (models) => {
        // Filtra solo modelli completati
        this.availableModels = models.filter(m => m.status === 'completed');
        if (this.availableModels.length > 0) {
          this.selectedModelId = this.availableModels[0].model_id;
        }
      },
      error: (error) => {
        console.error('Errore caricamento modelli:', error);
        this.errorMessage = 'Impossibile caricare i modelli';
      }
    });
  }

  /**
   * Genera le previsioni
   */
  onGenerateForecast(): void {
    if (!this.selectedModelId) {
      this.errorMessage = 'Seleziona un modello';
      return;
    }

    this.errorMessage = '';
    this.forecastResult = null;

    const request: ForecastRequest = {
      steps: this.forecastSteps,
      confidence_level: this.confidenceLevel,
      return_intervals: this.includeConfidenceIntervals
    };

    this.isForecasting = true;

    this.apiService.generateForecast(this.selectedModelId, request).subscribe({
      next: (result) => {
        this.forecastResult = result;
        this.isForecasting = false;
        console.log('Forecast generato:', result);

        // Prepara dati per il grafico
        this.prepareChartData(result);

        // Genera il grafico Plotly interattivo
        this.generatePlotlyChart();
      },
      error: (error) => {
        this.errorMessage = `Errore forecast: ${error.error?.detail || error.message}`;
        this.isForecasting = false;
      }
    });
  }

  /**
   * Genera il grafico Plotly interattivo usando URL diretto in iframe
   */
  generatePlotlyChart(): void {
    if (!this.selectedModelId) return;

    this.isLoadingChart = true;
    this.plotlyChartUrl = null;

    // Costruisci URL diretto all'endpoint di visualizzazione
    const baseUrl = this.apiService['API_BASE_URL']; // Accesso alla proprietà privata
    const chartUrl = `${baseUrl}/visualization/forecast-plot/${this.selectedModelId}/${this.forecastSteps}?confidence_level=${this.confidenceLevel}&include_intervals=${this.includeConfidenceIntervals}&theme=plotly_white`;

    // Sanitizza l'URL per l'iframe
    this.plotlyChartUrl = this.sanitizer.bypassSecurityTrustResourceUrl(chartUrl);
    this.isLoadingChart = false;
  }

  /**
   * Prepara i dati per la visualizzazione nel grafico
   */
  private prepareChartData(forecast: ForecastResponse): void {
    this.chartLabels = forecast.timestamps;
    this.forecastData = forecast.forecast;

    if (forecast.confidence_intervals) {
      this.lowerBound = forecast.confidence_intervals.lower;
      this.upperBound = forecast.confidence_intervals.upper;
    } else {
      this.lowerBound = [];
      this.upperBound = [];
    }
  }

  /**
   * Esporta i risultati in CSV
   */
  exportToCSV(): void {
    if (!this.forecastResult) return;

    let csv = 'Timestamp,Forecast';
    if (this.includeConfidenceIntervals) {
      csv += ',Lower Bound,Upper Bound';
    }
    csv += '\n';

    for (let i = 0; i < this.forecastResult.timestamps.length; i++) {
      csv += `${this.forecastResult.timestamps[i]},${this.forecastResult.forecast[i]}`;
      if (this.includeConfidenceIntervals && this.forecastResult.confidence_intervals) {
        csv += `,${this.forecastResult.confidence_intervals.lower[i]},${this.forecastResult.confidence_intervals.upper[i]}`;
      }
      csv += '\n';
    }

    // Download del file
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecast_${this.selectedModelId}_${Date.now()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  /**
   * Formatta numeri per visualizzazione
   */
  formatNumber(value: number): string {
    return value.toFixed(2);
  }

  /**
   * Calcola il valore massimo per scaling del grafico
   */
  getMaxValue(): number {
    if (this.forecastData.length === 0) return 1;
    return Math.max(...this.forecastData);
  }

  /**
   * Calcola la media
   */
  calculateMean(data: number[]): number {
    if (data.length === 0) return 0;
    return data.reduce((sum, val) => sum + val, 0) / data.length;
  }

  /**
   * Calcola il valore minimo
   */
  calculateMin(data: number[]): number {
    if (data.length === 0) return 0;
    return Math.min(...data);
  }

  /**
   * Calcola il valore massimo
   */
  calculateMax(data: number[]): number {
    if (data.length === 0) return 0;
    return Math.max(...data);
  }

  /**
   * Calcola la deviazione standard
   */
  calculateStdDev(data: number[]): number {
    if (data.length === 0) return 0;
    const mean = this.calculateMean(data);
    const squaredDiffs = data.map(val => Math.pow(val - mean, 2));
    const variance = this.calculateMean(squaredDiffs);
    return Math.sqrt(variance);
  }

  /**
   * Genera report professionale usando Quarto (metodo asincrono con job)
   */
  generateQuartoReport(formats: ('pdf' | 'html' | 'docx')[] = ['html']): void {
    if (!this.selectedModelId) {
      this.errorMessage = 'Seleziona un modello prima di generare il report';
      return;
    }

    this.isGeneratingReport = true;
    this.reportGenerationError = '';
    this.reportProgress = 0;
    this.generatedReportUrls = [];
    this.downloadTriggered = false; // Reset flag per nuovo report

    // Configurazione report
    const reportConfig = {
      model_ids: [this.selectedModelId],
      report_type: 'comprehensive' as const,
      include_sections: ['summary', 'methodology', 'results', 'diagnostics', 'recommendations'],
      export_formats: formats,
      template_style: 'corporate' as const,
      language: 'it'
    };

    // Avvia la generazione del report
    this.apiService.generateReport(reportConfig).subscribe({
      next: (response) => {
        this.reportJobId = response.job_id;
        console.log('Report job avviato:', response.job_id);

        // Polling dello stato del job ogni 2 secondi
        this.pollReportStatus();
      },
      error: (error) => {
        this.reportGenerationError = `Errore avvio generazione report: ${error.error?.detail || error.message}`;
        this.isGeneratingReport = false;
        console.error('Errore generazione report:', error);
      }
    });
  }

  /**
   * Polling dello stato del job di generazione report
   */
  private pollReportStatus(): void {
    if (!this.reportJobId) return;

    let pollingCompleted = false; // Flag per evitare esecuzioni multiple

    const intervalId = setInterval(() => {
      if (!this.reportJobId || pollingCompleted) {
        clearInterval(intervalId);
        return;
      }

      this.apiService.checkJobStatus(this.reportJobId).subscribe({
        next: (status) => {
          this.reportProgress = (status.progress || 0) * 100;

          if (status.status === 'completed') {
            // Job completato - imposta flag e ferma polling
            pollingCompleted = true;
            clearInterval(intervalId);
            this.isGeneratingReport = false;
            this.generatedReportUrls = status.results_urls || [];
            console.log('Report generati:', this.generatedReportUrls);

            // Scarica automaticamente tutti i file
            this.downloadAllGeneratedReports();

            // Reset jobId per evitare richieste duplicate
            this.reportJobId = null;
          } else if (status.status === 'failed') {
            // Job fallito
            pollingCompleted = true;
            clearInterval(intervalId);
            this.isGeneratingReport = false;
            this.reportGenerationError = status.error || 'Errore sconosciuto durante la generazione';
            console.error('Report generation failed:', status.error);
            this.reportJobId = null;
          }
          // Altrimenti continua il polling (status = 'queued' o 'running')
        },
        error: (error) => {
          pollingCompleted = true;
          clearInterval(intervalId);
          this.isGeneratingReport = false;
          this.reportGenerationError = `Errore controllo stato job: ${error.message}`;
          console.error('Errore polling status:', error);
          this.reportJobId = null;
        }
      });
    }, 2000); // Polling ogni 2 secondi
  }

  /**
   * Scarica tutti i report generati
   */
  private downloadAllGeneratedReports(): void {
    // Protezione contro chiamate multiple
    if (this.downloadTriggered) {
      console.log('Download già avviato, skip');
      return;
    }

    if (this.generatedReportUrls.length === 0) {
      this.reportGenerationError = 'Nessun file generato';
      return;
    }

    this.downloadTriggered = true; // Imposta flag

    // Scarica ogni file con un piccolo delay tra uno e l'altro
    this.generatedReportUrls.forEach((filePath, index) => {
      setTimeout(() => {
        const downloadUrl = this.apiService.getDownloadReportUrl(filePath);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = '';  // Il server fornisce il filename
        link.target = '_blank';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }, index * 500); // Delay di 500ms tra i download
    });
  }

  /**
   * Wrapper per compatibilità con i pulsanti esistenti
   * Ora usa il sistema Quarto invece dell'endpoint semplice
   */
  downloadReport(format: 'pdf' | 'html'): void {
    this.generateQuartoReport([format]);
  }
}
