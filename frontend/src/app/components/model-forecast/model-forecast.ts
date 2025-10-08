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
}
