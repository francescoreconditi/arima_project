// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Componente Angular per lista modelli
// ============================================

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ArimaApiService } from '../../services/arima-api.service';
import { ModelInfo } from '../../models/api.models';

@Component({
  selector: 'app-model-list',
  standalone: true,
  imports: [CommonModule, RouterLink, FormsModule],
  templateUrl: './model-list.html',
  styleUrl: './model-list.scss'
})
export class ModelList implements OnInit {
  models: ModelInfo[] = [];
  isLoading = false;
  errorMessage = '';

  // Modale modifica descrizione
  showEditModal = false;
  selectedModel: ModelInfo | null = null;
  editingDescription = '';

  // Modale visualizza dettagli
  showDetailsModal = false;
  fullModelDetails: any = null;
  isLoadingDetails = false;

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

  editDescription(model: ModelInfo): void {
    this.selectedModel = model;
    this.editingDescription = model.descrizione || '';
    this.showEditModal = true;
  }

  closeEditModal(): void {
    this.showEditModal = false;
    this.selectedModel = null;
    this.editingDescription = '';
  }

  saveDescription(): void {
    if (!this.selectedModel) return;

    this.apiService.updateModelDescription(this.selectedModel.model_id, this.editingDescription).subscribe({
      next: () => {
        // Aggiorna il modello nella lista
        const index = this.models.findIndex(m => m.model_id === this.selectedModel!.model_id);
        if (index !== -1) {
          this.models[index].descrizione = this.editingDescription;
        }
        this.closeEditModal();
      },
      error: (error) => {
        this.errorMessage = `Errore aggiornamento descrizione: ${error.message}`;
      }
    });
  }

  viewModelDetails(model: ModelInfo): void {
    this.selectedModel = model;
    this.showDetailsModal = true;
    this.isLoadingDetails = true;
    this.fullModelDetails = null;

    // Carica i dettagli completi dal file .pkl
    this.apiService.getModelFullDetails(model.model_id).subscribe({
      next: (details) => {
        this.fullModelDetails = details;
        this.isLoadingDetails = false;
      },
      error: (error) => {
        this.errorMessage = `Errore caricamento dettagli: ${error.message}`;
        this.isLoadingDetails = false;
      }
    });
  }

  closeDetailsModal(): void {
    this.showDetailsModal = false;
    this.selectedModel = null;
    this.fullModelDetails = null;
    this.isLoadingDetails = false;
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
