/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Servizio API per comunicazione con FastAPI
 * ============================================
 */

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Product, ForecastData } from '../models/product.model';
import { DataService } from './data.service';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://localhost:8000';
  
  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json'
    })
  };

  constructor(private http: HttpClient, private dataService: DataService) { }

  // Metodo per fare forecast con fallback
  forecast(modelId: string, steps: number = 30): Observable<ForecastData[]> {
    const forecastRequest = {
      steps: steps,
      confidence_level: 0.95,
      return_confidence_intervals: true
    };

    return this.http.post<any>(`${this.baseUrl}/models/${modelId}/forecast`, forecastRequest, this.httpOptions)
      .pipe(
        catchError(error => {
          console.warn('API forecast non disponibile, uso dati mock reattivi', error);
          // Usa DataService per dati mock reattivi ai parametri
          const productCode = modelId.replace('arima_', '');
          const mockData = this.dataService.generateMockForecast(productCode, steps);
          return of(mockData.map(item => ({
            date: item.data.toISOString().split('T')[0],
            forecast: item.valore,
            lower_ci: item.lower_bound,
            upper_ci: item.upper_bound
          })));
        })
      );
  }

  // Metodo per ottenere modelli disponibili con fallback
  getModels(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/models`, this.httpOptions)
      .pipe(
        catchError(error => {
          console.warn('API models non disponibile, uso dati mock', error);
          return of([]);
        })
      );
  }

  // Metodo per training modello con fallback
  trainModel(data: any): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/models/train/arima`, data, this.httpOptions)
      .pipe(
        catchError(error => {
          console.warn('API training non disponibile', error);
          return of({ message: 'Training non disponibile - modalità demo' });
        })
      );
  }

  // Metodi per inventory management con fallback
  getInventoryAnalysis(productCode: string): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/inventory/analysis/${productCode}`, this.httpOptions)
      .pipe(
        catchError(error => {
          console.warn('API inventory non disponibile, uso dati mock', error);
          return this.getMockInventoryData(productCode);
        })
      );
  }

  // Dati mock per fallback
  private getMockForecastData(steps: number): Observable<ForecastData[]> {
    const mockData: ForecastData[] = [];
    const baseValue = 25;
    
    for (let i = 0; i < steps; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i + 1);
      
      mockData.push({
        date: date.toISOString().split('T')[0],
        forecast: baseValue + Math.random() * 10 - 5,
        lower_ci: baseValue - 3 + Math.random() * 2,
        upper_ci: baseValue + 3 + Math.random() * 2
      });
    }
    
    return of(mockData);
  }

  private getMockInventoryData(productCode: string): Observable<any> {
    const mockData = {
      product_code: productCode,
      current_stock: 150,
      safety_stock: 45,
      reorder_point: 85,
      economic_order_quantity: 200,
      days_of_supply: 6.2,
      stockout_risk: 0.15
    };
    
    return of(mockData);
  }
}