// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Servizio Angular per chiamate API ARIMA Forecaster
// ============================================

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import {
  ModelTrainingRequest,
  ModelInfo,
  ModelListResponse,
  ForecastRequest,
  ForecastResponse,
  AutoSelectionRequest,
  AutoSelectionResult,
  HealthResponse
} from '../models/api.models';

/**
 * Servizio per interfacciarsi con l'API FastAPI di ARIMA Forecaster
 */
@Injectable({
  providedIn: 'root'
})
export class ArimaApiService {
  private readonly API_BASE_URL = 'http://localhost:8000';

  private readonly httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json'
    })
  };

  constructor(private http: HttpClient) {}

  // ===== HEALTH ENDPOINTS =====

  /**
   * Verifica lo stato dell'API
   */
  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.API_BASE_URL}/health`);
  }

  // ===== TRAINING ENDPOINTS =====

  /**
   * Addestra un nuovo modello ARIMA/SARIMA/SARIMAX
   */
  trainModel(request: ModelTrainingRequest): Observable<ModelInfo> {
    return this.http.post<ModelInfo>(
      `${this.API_BASE_URL}/models/train`,
      request,
      this.httpOptions
    );
  }

  /**
   * Selezione automatica dei parametri ottimali
   */
  autoSelectModel(request: AutoSelectionRequest): Observable<AutoSelectionResult> {
    return this.http.post<AutoSelectionResult>(
      `${this.API_BASE_URL}/models/auto-select`,
      request,
      this.httpOptions
    );
  }

  // ===== FORECASTING ENDPOINTS =====

  /**
   * Genera previsioni con un modello addestrato
   */
  generateForecast(modelId: string, request: ForecastRequest): Observable<ForecastResponse> {
    return this.http.post<ForecastResponse>(
      `${this.API_BASE_URL}/models/${modelId}/forecast`,
      request,
      this.httpOptions
    );
  }

  // ===== MODEL MANAGEMENT ENDPOINTS =====

  /**
   * Ottiene la lista di tutti i modelli
   */
  listModels(): Observable<ModelInfo[]> {
    return this.http.get<ModelListResponse>(`${this.API_BASE_URL}/models`).pipe(
      map(response => response.models)
    );
  }

  /**
   * Ottiene i dettagli di un modello specifico
   */
  getModelInfo(modelId: string): Observable<ModelInfo> {
    return this.http.get<ModelInfo>(`${this.API_BASE_URL}/models/${modelId}`);
  }

  /**
   * Elimina un modello
   */
  deleteModel(modelId: string): Observable<void> {
    return this.http.delete<void>(`${this.API_BASE_URL}/models/${modelId}`);
  }

  /**
   * Verifica lo stato di un modello in training
   */
  checkModelStatus(modelId: string): Observable<ModelInfo> {
    // Usa endpoint corretto: /models/{model_id} invece di /models/{model_id}/status
    return this.http.get<ModelInfo>(`${this.API_BASE_URL}/models/${modelId}`);
  }

  /**
   * Aggiorna la descrizione di un modello
   */
  updateModelDescription(modelId: string, description: string): Observable<any> {
    return this.http.patch(
      `${this.API_BASE_URL}/models/${modelId}/description`,
      null,
      {
        ...this.httpOptions,
        params: { description }
      }
    );
  }

  /**
   * Ottiene TUTTI i dettagli completi di un modello dal file .pkl
   */
  getModelFullDetails(modelId: string): Observable<any> {
    return this.http.get<any>(`${this.API_BASE_URL}/models/${modelId}/details`);
  }

  /**
   * Genera grafico interattivo Plotly con Serie Temporale + Forecast
   * Restituisce HTML completo del grafico
   */
  getForecastPlotHtml(
    modelId: string,
    forecastSteps: number,
    confidenceLevel: number = 0.95,
    includeIntervals: boolean = true,
    theme: string = 'plotly_white'
  ): Observable<string> {
    const params = {
      confidence_level: confidenceLevel.toString(),
      include_intervals: includeIntervals.toString(),
      theme: theme
    };

    return this.http.get(
      `${this.API_BASE_URL}/visualization/forecast-plot/${modelId}/${forecastSteps}`,
      {
        params: params,
        responseType: 'text'
      }
    );
  }

  // ===== VISUALIZATION & REPORTING ENDPOINTS =====

  /**
   * Genera report professionale usando Quarto (asincrono - job based)
   * Restituisce job_id per monitorare il progresso
   */
  generateReport(config: ReportGenerationRequest): Observable<VisualizationJobResponse> {
    return this.http.post<VisualizationJobResponse>(
      `${this.API_BASE_URL}/visualization/generate-report`,
      config,
      this.httpOptions
    );
  }

  /**
   * Controlla lo stato di un job di visualizzazione/report
   */
  checkJobStatus(jobId: string): Observable<VisualizationJobResponse> {
    return this.http.get<VisualizationJobResponse>(
      `${this.API_BASE_URL}/visualization/job-status/${jobId}`
    );
  }

  /**
   * Costruisce l'URL per scaricare un file report generato
   */
  getDownloadReportUrl(filePath: string): string {
    return `${this.API_BASE_URL}/visualization/download-report/${filePath}`;
  }

  // ===== DATA MANAGEMENT ENDPOINTS =====

  /**
   * Carica un file CSV con configurazione
   */
  uploadDataset(file: File, config: any): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    // Config deve essere un query parameter, non FormData
    const configParam = encodeURIComponent(JSON.stringify(config));
    const url = `${this.API_BASE_URL}/data/upload?config=${configParam}`;

    return this.http.post(url, formData);
  }

  /**
   * Ottiene lo stato di un job di data management
   */
  getDataJobStatus(jobId: string): Observable<any> {
    return this.http.get(`${this.API_BASE_URL}/data/job-status/${jobId}`);
  }

  /**
   * Ottiene lista di tutti i dataset caricati
   */
  listDatasets(): Observable<any[]> {
    return this.http.get<any[]>(`${this.API_BASE_URL}/data/datasets`);
  }

  /**
   * Esegue preprocessing su un dataset
   */
  preprocessDataset(request: any): Observable<any> {
    return this.http.post(
      `${this.API_BASE_URL}/data/preprocess`,
      request,
      this.httpOptions
    );
  }
}

// ===== NUOVI MODELLI PER REPORT GENERATION =====

export interface ReportGenerationRequest {
  model_ids: string[];
  report_type: 'executive' | 'technical' | 'comprehensive' | 'regulatory';
  include_sections: string[];
  export_formats: ('pdf' | 'html' | 'docx')[];
  template_style: 'corporate' | 'minimal' | 'academic' | 'dashboard';
  language: string;
}

export interface VisualizationJobResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress?: number;
  estimated_completion?: string;
  results_urls?: string[];
  error?: string;
}
