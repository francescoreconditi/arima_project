// ============================================
// SERVIZIO API BASE
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Comunicazione con FastAPI backend
// ============================================

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry, timeout } from 'rxjs/operators';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl: string = environment.apiUrl || 'http://localhost:8000/api';
  private defaultHeaders = new HttpHeaders({
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  });

  constructor(private http: HttpClient) {}

  // GET request generico
  get<T>(endpoint: string, params?: any): Observable<T> {
    const httpParams = this.buildParams(params);
    return this.http.get<T>(`${this.baseUrl}/${endpoint}`, {
      headers: this.defaultHeaders,
      params: httpParams
    }).pipe(
      timeout(30000),
      retry(2),
      catchError(this.handleError)
    );
  }

  // POST request generico
  post<T>(endpoint: string, body: any): Observable<T> {
    return this.http.post<T>(`${this.baseUrl}/${endpoint}`, body, {
      headers: this.defaultHeaders
    }).pipe(
      timeout(30000),
      catchError(this.handleError)
    );
  }

  // PUT request generico
  put<T>(endpoint: string, body: any): Observable<T> {
    return this.http.put<T>(`${this.baseUrl}/${endpoint}`, body, {
      headers: this.defaultHeaders
    }).pipe(
      timeout(30000),
      catchError(this.handleError)
    );
  }

  // DELETE request generico
  delete<T>(endpoint: string): Observable<T> {
    return this.http.delete<T>(`${this.baseUrl}/${endpoint}`, {
      headers: this.defaultHeaders
    }).pipe(
      timeout(30000),
      catchError(this.handleError)
    );
  }

  // PATCH request generico
  patch<T>(endpoint: string, body: any): Observable<T> {
    return this.http.patch<T>(`${this.baseUrl}/${endpoint}`, body, {
      headers: this.defaultHeaders
    }).pipe(
      timeout(30000),
      catchError(this.handleError)
    );
  }

  // Upload file
  uploadFile(endpoint: string, file: File, additionalData?: any): Observable<any> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    
    if (additionalData) {
      Object.keys(additionalData).forEach(key => {
        formData.append(key, additionalData[key]);
      });
    }

    return this.http.post(`${this.baseUrl}/${endpoint}`, formData).pipe(
      timeout(60000),
      catchError(this.handleError)
    );
  }

  // Download file
  downloadFile(endpoint: string): Observable<Blob> {
    return this.http.get(`${this.baseUrl}/${endpoint}`, {
      responseType: 'blob'
    }).pipe(
      timeout(60000),
      catchError(this.handleError)
    );
  }

  // Build HTTP params from object
  private buildParams(params: any): HttpParams {
    let httpParams = new HttpParams();
    if (params) {
      Object.keys(params).forEach(key => {
        if (params[key] !== null && params[key] !== undefined) {
          if (Array.isArray(params[key])) {
            params[key].forEach((item: any) => {
              httpParams = httpParams.append(key, item.toString());
            });
          } else {
            httpParams = httpParams.set(key, params[key].toString());
          }
        }
      });
    }
    return httpParams;
  }

  // Error handler
  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';

    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = this.getServerErrorMessage(error);
    }

    console.error('API Error:', errorMessage);
    return throwError(() => ({
      status: error.status,
      message: errorMessage,
      details: error.error
    }));
  }

  private getServerErrorMessage(error: HttpErrorResponse): string {
    switch (error.status) {
      case 0:
        return 'Unable to connect to server. Please check your connection.';
      case 400:
        return error.error?.detail || 'Bad request. Please check your input.';
      case 401:
        return 'Authentication required. Please log in.';
      case 403:
        return 'Access denied. You do not have permission.';
      case 404:
        return 'Resource not found.';
      case 422:
        return error.error?.detail || 'Validation error. Please check your data.';
      case 500:
        return 'Server error. Please try again later.';
      case 502:
        return 'Bad gateway. Server is temporarily unavailable.';
      case 503:
        return 'Service unavailable. Please try again later.';
      default:
        return error.error?.detail || `Server error: ${error.status}`;
    }
  }

  // Set custom headers
  setHeaders(headers: { [key: string]: string }): void {
    Object.keys(headers).forEach(key => {
      this.defaultHeaders = this.defaultHeaders.set(key, headers[key]);
    });
  }

  // Get full URL for endpoint
  getFullUrl(endpoint: string): string {
    return `${this.baseUrl}/${endpoint}`;
  }

  // Health check
  healthCheck(): Observable<any> {
    return this.get('health');
  }
}