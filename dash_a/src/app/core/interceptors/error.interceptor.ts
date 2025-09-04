// ============================================
// INTERCEPTOR GESTIONE ERRORI
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Gestione centralizzata errori HTTP
// ============================================

import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable()
export class ErrorInterceptor implements HttpInterceptor {
  constructor(private snackBar: MatSnackBar) {}

  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<any> {
    return next.handle(req).pipe(
      retry(1), // Retry once for failed requests
      catchError((error: HttpErrorResponse) => {
        let errorMessage = 'Si è verificato un errore';

        if (error.error instanceof ErrorEvent) {
          // Client-side error
          errorMessage = `Errore client: ${error.error.message}`;
        } else {
          // Server-side error
          switch (error.status) {
            case 0:
              errorMessage = 'Impossibile connettersi al server';
              break;
            case 400:
              errorMessage = 'Richiesta non valida';
              break;
            case 401:
              errorMessage = 'Accesso non autorizzato';
              break;
            case 403:
              errorMessage = 'Accesso negato';
              break;
            case 404:
              errorMessage = 'Risorsa non trovata';
              break;
            case 422:
              errorMessage = error.error?.detail || 'Dati non validi';
              break;
            case 500:
              errorMessage = 'Errore interno del server';
              break;
            default:
              errorMessage = error.error?.detail || `Errore: ${error.status}`;
          }
        }

        // Show error message
        this.snackBar.open(errorMessage, 'Chiudi', {
          duration: 5000,
          panelClass: ['error-snackbar']
        });

        return throwError(() => error);
      })
    );
  }
}