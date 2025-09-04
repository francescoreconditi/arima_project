// ============================================
// SERVIZIO FORECAST
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Gestione forecast e previsioni
// ============================================

import { Injectable } from '@angular/core';
import { Observable, BehaviorSubject, of } from 'rxjs';
import { map, tap, catchError } from 'rxjs/operators';
import { ApiService } from './api.service';
import { 
  ForecastRequest, 
  ForecastResult, 
  HistoricalData,
  ForecastComparison,
  TimeSeriesPoint,
  DataStatistics
} from '../models/forecast.model';

@Injectable({
  providedIn: 'root'
})
export class ForecastService {
  private forecastCache = new Map<string, ForecastResult>();
  private currentForecast$ = new BehaviorSubject<ForecastResult | null>(null);
  private historicalData$ = new BehaviorSubject<HistoricalData[]>([]);
  private isLoading$ = new BehaviorSubject<boolean>(false);

  constructor(private api: ApiService) {}

  // Observable per forecast corrente
  getCurrentForecast(): Observable<ForecastResult | null> {
    return this.currentForecast$.asObservable();
  }

  // Observable per dati storici
  getHistoricalData(): Observable<HistoricalData[]> {
    return this.historicalData$.asObservable();
  }

  // Observable per stato loading
  getLoadingState(): Observable<boolean> {
    return this.isLoading$.asObservable();
  }

  // Genera forecast
  generateForecast(request: ForecastRequest): Observable<ForecastResult> {
    this.isLoading$.next(true);
    
    const cacheKey = this.getCacheKey(request);
    if (this.forecastCache.has(cacheKey)) {
      const cached = this.forecastCache.get(cacheKey)!;
      this.currentForecast$.next(cached);
      this.isLoading$.next(false);
      return of(cached);
    }

    // Usa dati mock per evitare errori API durante sviluppo
    const mockResult = this.generateMockForecast(request);
    this.forecastCache.set(cacheKey, mockResult);
    this.currentForecast$.next(mockResult);
    this.isLoading$.next(false);

    // DISABLE API call for demo - using mock data only
    // Future integration: this.api.post<ForecastResult>(`models/{model_id}/forecast`, request)
    console.log('🎯 Using mock forecast data for demo purposes');

    return of(mockResult);
  }

  // Ottieni dati storici per prodotto
  getProductHistory(productId: string, startDate?: string, endDate?: string): Observable<HistoricalData> {
    const params: any = { productId };
    if (startDate) params.startDate = startDate;
    if (endDate) params.endDate = endDate;

    // For demo, return mock data without API call
    const mockHistory = this.generateMockHistoryData(productId);
    return of(mockHistory).pipe(
      tap(data => {
        const current = this.historicalData$.value;
        const index = current.findIndex(h => h.productId === productId);
        if (index >= 0) {
          current[index] = data;
        } else {
          current.push(data);
        }
        this.historicalData$.next([...current]);
      })
    );
  }

  // Confronta modelli diversi
  compareModels(productId: string, models: string[], steps: number): Observable<ForecastComparison> {
    return this.api.post<ForecastComparison>('forecast/compare', {
      productId,
      models,
      steps
    });
  }

  // Batch forecast per multipli prodotti
  batchForecast(productIds: string[], steps: number, modelType: string = 'arima'): Observable<ForecastResult[]> {
    return this.api.post<ForecastResult[]>('forecast/batch', {
      productIds,
      steps,
      modelType
    });
  }

  // Ottieni statistiche sui dati
  getDataStatistics(data: TimeSeriesPoint[]): DataStatistics {
    const values = data.map(d => d.value);
    const n = values.length;
    
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const sorted = [...values].sort((a, b) => a - b);
    const median = n % 2 === 0 
      ? (sorted[n/2 - 1] + sorted[n/2]) / 2 
      : sorted[Math.floor(n/2)];
    
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Trend detection semplice
    const firstQuarter = values.slice(0, Math.floor(n/4));
    const lastQuarter = values.slice(Math.floor(3*n/4));
    const firstMean = firstQuarter.reduce((a, b) => a + b, 0) / firstQuarter.length;
    const lastMean = lastQuarter.reduce((a, b) => a + b, 0) / lastQuarter.length;
    
    let trend: 'increasing' | 'decreasing' | 'stable';
    if (lastMean > firstMean * 1.1) {
      trend = 'increasing';
    } else if (lastMean < firstMean * 0.9) {
      trend = 'decreasing';
    } else {
      trend = 'stable';
    }

    return {
      mean,
      median,
      std,
      min,
      max,
      trend,
      seasonality: this.detectSeasonality(values),
      seasonalPeriod: 12 // Default mensile
    };
  }

  // Rileva stagionalità semplice
  private detectSeasonality(values: number[]): boolean {
    if (values.length < 24) return false;
    
    // Calcola autocorrelazione per lag 12 (mensile)
    const lag = 12;
    const n = values.length;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    
    let numerator = 0;
    let denominator = 0;
    
    for (let i = lag; i < n; i++) {
      numerator += (values[i] - mean) * (values[i - lag] - mean);
    }
    
    for (let i = 0; i < n; i++) {
      denominator += Math.pow(values[i] - mean, 2);
    }
    
    const autocorr = numerator / denominator;
    return autocorr > 0.5; // Soglia per stagionalità significativa
  }

  // Esporta forecast in CSV
  exportForecastCSV(forecast: ForecastResult): void {
    const csv = this.convertToCSV(forecast);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `forecast_${forecast.productId}_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }

  private convertToCSV(forecast: ForecastResult): string {
    const headers = ['Date', 'Predicted Value', 'Lower Bound', 'Upper Bound'];
    const rows = forecast.predictions.map((point, index) => {
      const lower = forecast.confidenceInterval?.lower[index]?.value || '';
      const upper = forecast.confidenceInterval?.upper[index]?.value || '';
      return [point.date, point.value, lower, upper].join(',');
    });
    return [headers.join(','), ...rows].join('\n');
  }

  // Cache key generator
  private getCacheKey(request: ForecastRequest): string {
    return `${request.productId}_${request.modelType}_${request.steps}_${request.startDate}_${request.endDate}`;
  }

  // Genera forecast mock per sviluppo
  private generateMockForecast(request: ForecastRequest): ForecastResult {
    const predictions: TimeSeriesPoint[] = [];
    const confidenceIntervalUpper: TimeSeriesPoint[] = [];
    const confidenceIntervalLower: TimeSeriesPoint[] = [];
    
    const startDate = new Date();
    const baseValue = this.getProductBaseValue(request.productId || 'default');
    
    for (let i = 1; i <= request.steps; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Genera valore previsto con trend e rumore
      const trendFactor = 1 + (Math.random() - 0.5) * 0.1; // ±5% variazione
      const seasonalFactor = 1 + Math.sin((i * 2 * Math.PI) / 30) * 0.05; // Stagionalità mensile
      const noiseeFactor = 1 + (Math.random() - 0.5) * 0.2; // ±10% rumore
      
      const predictedValue = Math.round(baseValue * trendFactor * seasonalFactor * noiseeFactor);
      const lowerBound = Math.round(predictedValue * 0.85);
      const upperBound = Math.round(predictedValue * 1.15);
      
      predictions.push({
        date: date.toISOString(),
        value: predictedValue
      });
      
      confidenceIntervalLower.push({
        date: date.toISOString(),
        value: lowerBound
      });
      
      confidenceIntervalUpper.push({
        date: date.toISOString(),
        value: upperBound
      });
    }
    
    return {
      productId: request.productId || 'default',
      productName: this.getProductName(request.productId || 'default'),
      predictions,
      confidenceInterval: {
        lower: confidenceIntervalLower,
        upper: confidenceIntervalUpper
      },
      metrics: {
        mape: Math.round((5 + Math.random() * 10) * 100) / 100, // 5-15% MAPE
        rmse: Math.round((baseValue * 0.1 + Math.random() * baseValue * 0.1) * 100) / 100,
        mae: Math.round((baseValue * 0.05 + Math.random() * baseValue * 0.05) * 100) / 100
      },
      modelInfo: {
        type: request.modelType || 'arima',
        parameters: {
          order: [1, 1, 1],
          seasonalOrder: request.modelType === 'sarima' ? [1, 1, 1, 12] : undefined
        },
        trainingPeriod: {
          start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(),
          end: new Date().toISOString()
        }
      },
      timestamp: new Date()
    } as ForecastResult;
  }

  // Genera dati storici mock
  private generateMockHistoryData(productId: string): HistoricalData {
    const data: TimeSeriesPoint[] = [];
    const baseValue = this.getProductBaseValue(productId);
    const endDate = new Date();
    
    // Genera 12 mesi di dati storici
    for (let i = 365; i > 0; i--) {
      const date = new Date(endDate);
      date.setDate(date.getDate() - i);
      
      const seasonalFactor = 1 + Math.sin((i * 2 * Math.PI) / 365) * 0.1;
      const noiseFactor = 1 + (Math.random() - 0.5) * 0.3;
      const trendFactor = 1 + (365 - i) / 365 * 0.1; // Trend crescente leggero
      
      const value = Math.round(baseValue * seasonalFactor * noiseFactor * trendFactor);
      
      data.push({
        date: date.toISOString(),
        value: Math.max(1, value) // Minimo 1
      });
    }
    
    return {
      productId,
      productName: this.getProductName(productId),
      data,
      statistics: this.getDataStatistics(data)
    };
  }

  // Ottieni valore base per prodotto (per mock)
  private getProductBaseValue(productId: string): number {
    const productValues: { [key: string]: number } = {
      'CRZ001': 25, // Carrozzina
      'MAT001': 30, // Materasso
      'ELT001': 15, // Saturimetro
      'DEA001': 8,  // Deambulatore
      'TER001': 20, // Termometro
      'SER001': 3,  // Sollevatore
      'LET001': 5,  // Letto
      'MON001': 2,  // Monitor
      'VEN001': 1,  // Ventilatore
      'DEF001': 1   // Defibrillatore
    };
    
    return productValues[productId] || 10; // Valore default
  }

  // Ottieni nome prodotto (per mock)
  private getProductName(productId: string): string {
    const productNames: { [key: string]: string } = {
      'CRZ001': 'Carrozzina Standard',
      'MAT001': 'Materasso Antidecubito',
      'ELT001': 'Saturimetro',
      'DEA001': 'Deambulatore',
      'TER001': 'Termometro Digitale',
      'SER001': 'Sollevatore Pazienti',
      'LET001': 'Letto Ospedaliero',
      'MON001': 'Monitor Parametri Vitali',
      'VEN001': 'Ventilatore Polmonare',
      'DEF001': 'Defibrillatore'
    };
    
    return productNames[productId] || 'Prodotto Generico';
  }

  // Clear cache
  clearCache(): void {
    this.forecastCache.clear();
    this.currentForecast$.next(null);
    this.historicalData$.next([]);
  }

  // Valida i dati di input
  validateTimeSeriesData(data: TimeSeriesPoint[]): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    if (!data || data.length === 0) {
      errors.push('No data provided');
    }
    
    if (data.length < 10) {
      errors.push('Insufficient data points (minimum 10 required)');
    }
    
    const hasNullValues = data.some(d => d.value === null || d.value === undefined);
    if (hasNullValues) {
      errors.push('Data contains null values');
    }
    
    const hasNegativeValues = data.some(d => d.value < 0);
    if (hasNegativeValues) {
      errors.push('Data contains negative values');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}