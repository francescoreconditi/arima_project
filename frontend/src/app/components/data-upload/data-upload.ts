// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-08
// Scopo: Componente Angular per upload dataset CSV
// ============================================

import { Component, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ArimaApiService } from '../../services/arima-api.service';

@Component({
  selector: 'app-data-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './data-upload.html',
  styleUrl: './data-upload.scss'
})
export class DataUpload {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  // File selection
  selectedFile: File | null = null;
  fileName = '';

  // Configuration
  datasetName = '';
  dateColumn = '';
  separator = ',';
  encoding = 'utf-8';
  skipRows = 0;
  validateData = true;

  // CSV preview
  csvHeaders: string[] = [];
  csvPreview: string[][] = [];
  showPreview = false;

  // Upload state
  isUploading = false;
  uploadProgress = 0;
  uploadJobId: string | null = null;
  uploadError = '';
  uploadSuccess = false;
  uploadedDatasetId = '';

  // Value columns input
  valueColumnsInput = '';

  constructor(private apiService: ArimaApiService) {}

  /**
   * Gestisce selezione file
   */
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.fileName = file.name;
      this.datasetName = file.name.replace('.csv', '');

      // Preview CSV
      this.previewCSV(file);
    }
  }

  /**
   * Preview del file CSV
   */
  private previewCSV(file: File): void {
    const reader = new FileReader();
    reader.onload = (e: any) => {
      const text = e.target.result;
      const lines = text.split('\n').slice(0, 6); // Prime 6 righe

      if (lines.length > 0) {
        // Header
        this.csvHeaders = lines[0].split(this.separator).map((h: string) => h.trim());

        // Data rows
        this.csvPreview = lines.slice(1, 6).map((line: string) =>
          line.split(this.separator).map((cell: string) => cell.trim())
        );

        this.showPreview = true;
      }
    };
    reader.readAsText(file);
  }

  /**
   * Carica dataset
   */
  uploadDataset(): void {
    if (!this.selectedFile) {
      this.uploadError = 'Seleziona un file CSV';
      return;
    }

    if (!this.datasetName) {
      this.uploadError = 'Inserisci un nome per il dataset';
      return;
    }

    // Parse value columns
    const valueColsArray = this.valueColumnsInput
      .split(',')
      .map(c => c.trim())
      .filter(c => c.length > 0);

    if (valueColsArray.length === 0) {
      this.uploadError = 'Inserisci almeno una colonna valore';
      return;
    }

    this.isUploading = true;
    this.uploadError = '';
    this.uploadProgress = 0;

    const config = {
      dataset_name: this.datasetName,
      date_column: this.dateColumn || undefined,
      value_columns: valueColsArray,
      separator: this.separator,
      encoding: this.encoding,
      skip_rows: this.skipRows,
      validate_data: this.validateData
    };

    this.apiService.uploadDataset(this.selectedFile, config).subscribe({
      next: (response) => {
        this.uploadJobId = response.job_id;
        console.log('Upload job avviato:', response.job_id);

        // Polling dello stato
        this.pollUploadStatus();
      },
      error: (error) => {
        this.uploadError = `Errore upload: ${error.error?.detail || error.message}`;
        this.isUploading = false;
        console.error('Errore upload:', error);
      }
    });
  }

  /**
   * Polling dello stato upload
   */
  private pollUploadStatus(): void {
    if (!this.uploadJobId) return;

    const intervalId = setInterval(() => {
      if (!this.uploadJobId) {
        clearInterval(intervalId);
        return;
      }

      this.apiService.getDataJobStatus(this.uploadJobId).subscribe({
        next: (status) => {
          this.uploadProgress = (status.progress || 0) * 100;

          if (status.status === 'completed') {
            clearInterval(intervalId);
            this.isUploading = false;
            this.uploadSuccess = true;
            this.uploadedDatasetId = status.dataset_id;
            console.log('Upload completato:', status);

            // Reset form dopo 3 secondi
            setTimeout(() => {
              this.resetForm();
            }, 3000);
          } else if (status.status === 'failed') {
            clearInterval(intervalId);
            this.isUploading = false;
            this.uploadError = status.error_message || 'Errore sconosciuto';
            console.error('Upload fallito:', status);
          }
        },
        error: (error) => {
          clearInterval(intervalId);
          this.isUploading = false;
          this.uploadError = `Errore controllo stato: ${error.message}`;
          console.error('Errore polling:', error);
        }
      });
    }, 1000); // Polling ogni secondo
  }

  /**
   * Reset form
   */
  resetForm(): void {
    this.selectedFile = null;
    this.fileName = '';
    this.datasetName = '';
    this.dateColumn = '';
    this.valueColumnsInput = '';
    this.csvHeaders = [];
    this.csvPreview = [];
    this.showPreview = false;
    this.uploadSuccess = false;
    this.uploadedDatasetId = '';

    // Reset file input element
    if (this.fileInput && this.fileInput.nativeElement) {
      this.fileInput.nativeElement.value = '';
    }
  }
}
