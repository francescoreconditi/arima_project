// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Componente Angular per lista modelli
// ============================================

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ArimaApiService } from '../../services/arima-api.service';
import { ModelInfo } from '../../models/api.models';

@Component({
  selector: 'app-model-list',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './model-list.html',
  styleUrl: './model-list.scss'
})
export class ModelList implements OnInit {
  models: ModelInfo[] = [];
  isLoading = false;
  errorMessage = '';

  constructor(private apiService: ArimaApiService) {}

  ngOnInit(): void {
    this.loadModels();
  }

  loadModels(): void {
    this.isLoading = true;
    this.errorMessage = '';

    this.apiService.listModels().subscribe({
      next: (models) => {
        this.models = models;
        this.isLoading = false;
      },
      error: (error) => {
        this.errorMessage = `Errore caricamento: ${error.message}`;
        this.isLoading = false;
      }
    });
  }

  deleteModel(modelId: string): void {
    if (!confirm('Sei sicuro di voler eliminare questo modello?')) {
      return;
    }

    this.apiService.deleteModel(modelId).subscribe({
      next: () => {
        this.models = this.models.filter(m => m.model_id !== modelId);
      },
      error: (error) => {
        this.errorMessage = `Errore eliminazione: ${error.message}`;
      }
    });
  }
}
