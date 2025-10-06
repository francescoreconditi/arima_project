// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Servizio Angular per chiamate API ARIMA Forecaster
// ============================================

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  ModelTrainingRequest,
  ModelInfo,
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
    return this.http.get<ModelInfo[]>(`${this.API_BASE_URL}/models`);
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
    return this.http.get<ModelInfo>(`${this.API_BASE_URL}/models/${modelId}/status`);
  }
}
