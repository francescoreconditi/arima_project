/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Wrapper component per widget dashboard
 * ============================================
 */

import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DashboardWidget } from '../../models/dashboard-widget.model';

// Import componenti esistenti
import { MetricCardComponent } from '../metric-card/metric-card.component';
import { AlertBoxComponent } from '../alert-box/alert-box.component';
import { ForecastChartComponent } from '../forecast-chart/forecast-chart.component';

@Component({
  selector: 'app-dashboard-widget',
  standalone: true,
  imports: [
    CommonModule,
    MetricCardComponent,
    AlertBoxComponent,
    ForecastChartComponent
  ],
  template: `
    <div class="widget-container" [ngClass]="getWidgetClass()">
      <!-- Widget Header -->
      <div class="widget-header" *ngIf="showHeader">
        <span class="widget-title">{{ widget.title }}</span>
        <div class="widget-controls">
          <button class="widget-btn" (click)="onEditClick()" title="Modifica widget">
            <span class="widget-icon">⚙️</span>
          </button>
          <button class="widget-btn" (click)="onRemoveClick()" title="Rimuovi widget" *ngIf="allowRemove">
            <span class="widget-icon">❌</span>
          </button>
        </div>
      </div>

      <!-- Widget Content -->
      <div class="widget-content">
        <!-- Metric Widget -->
        <app-metric-card 
          *ngIf="widget.type === 'metric'" 
          [metric]="widget.data">
        </app-metric-card>

        <!-- Chart Widget -->
        <app-forecast-chart
          *ngIf="widget.type === 'chart'"
          [forecastData]="chartData.forecastData || []"
          [historicalData]="chartData.historicalData || []"
          [title]="widget.data?.title || widget.title">
        </app-forecast-chart>

        <!-- Alerts Widget -->
        <div *ngIf="widget.type === 'alert'" class="alerts-widget">
          <app-alert-box 
            *ngFor="let alert of getAlertsData()" 
            [alert]="alert">
          </app-alert-box>
          <div *ngIf="getAlertsData().length === 0" class="no-alerts">
            [OK] Nessun avviso attivo
          </div>
        </div>

        <!-- Product Info Widget -->
        <div *ngIf="widget.type === 'product-info'" class="product-info-widget">
          <div *ngIf="productData; else noProductSelected" class="product-details">
            <h3>{{ productData.nome }} ({{ productData.codice }})</h3>
            <div class="detail-grid">
              <div class="detail-item">
                <span class="label">Categoria:</span>
                <span class="value">{{ productData.categoria }}</span>
              </div>
              <div class="detail-item">
                <span class="label">Fornitore:</span>
                <span class="value">{{ productData.fornitore }}</span>
              </div>
              <div class="detail-item">
                <span class="label">Prezzo Unitario:</span>
                <span class="value">€{{ productData.prezzo_unitario.toLocaleString() }}</span>
              </div>
              <div class="detail-item">
                <span class="label">Lead Time:</span>
                <span class="value">{{ productData.lead_time }} giorni</span>
              </div>
              <div class="detail-item">
                <span class="label">Giacenza Attuale:</span>
                <span class="value">{{ productData.giacenza_attuale }} unità</span>
              </div>
              <div class="detail-item">
                <span class="label">Punto Riordino:</span>
                <span class="value">{{ productData.punto_riordino }} unità</span>
              </div>
            </div>
          </div>
          <ng-template #noProductSelected>
            <div class="no-product">
              <p>🏥 Seleziona un prodotto per visualizzare i dettagli</p>
            </div>
          </ng-template>
        </div>

        <!-- Placeholder per widget non implementati -->
        <div *ngIf="!isKnownWidgetType()" class="unknown-widget">
          <p>Widget tipo "{{ widget.type }}" non implementato</p>
          <pre>{{ widget | json }}</pre>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .widget-container {
      height: 100%;
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border: 1px solid #e0e0e0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .widget-header {
      background: #f8f9fa;
      border-bottom: 1px solid #e0e0e0;
      padding: 8px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      min-height: 40px;
    }

    .widget-title {
      font-weight: 600;
      font-size: 14px;
      color: #333;
      margin: 0;
    }

    .widget-controls {
      display: flex;
      gap: 4px;
    }

    .widget-btn {
      background: none;
      border: none;
      padding: 4px;
      cursor: pointer;
      border-radius: 4px;
      transition: background-color 0.2s;
    }

    .widget-btn:hover {
      background: rgba(0,0,0,0.05);
    }

    .widget-icon {
      font-size: 12px;
    }

    .widget-content {
      flex: 1;
      padding: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .alerts-widget {
      height: 100%;
      overflow-y: auto;
    }

    .no-alerts {
      text-align: center;
      padding: 20px;
      color: #28a745;
      font-weight: 500;
    }

    .product-info-widget {
      height: 100%;
      overflow-y: auto;
    }

    .product-details h3 {
      margin: 0 0 15px 0;
      color: #333;
      font-size: 16px;
      font-weight: 600;
      border-bottom: 2px solid #667eea;
      padding-bottom: 8px;
    }

    .detail-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }

    .detail-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid #f0f0f0;
    }

    .detail-item:last-child {
      border-bottom: none;
    }

    .detail-item .label {
      font-weight: 500;
      color: #666;
      font-size: 13px;
    }

    .detail-item .value {
      color: #333;
      font-weight: 600;
      font-size: 13px;
    }

    .no-product {
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: #666;
      font-style: italic;
    }

    .unknown-widget {
      padding: 20px;
      text-align: center;
      color: #666;
    }

    .unknown-widget pre {
      background: #f8f9fa;
      padding: 10px;
      border-radius: 4px;
      font-size: 11px;
      text-align: left;
      overflow: auto;
    }

    /* Widget type specific styles */
    .widget-type-metric .widget-content {
      padding: 0;
    }

    .widget-type-chart .widget-content {
      padding: 0;
    }
  `]
})
export class DashboardWidgetComponent implements OnInit {
  @Input() widget!: DashboardWidget;
  @Input() showHeader: boolean = true;
  @Input() allowRemove: boolean = true;
  @Input() chartData: any = {};
  @Input() productData: any = null;
  @Input() alertsData: any[] = [];

  @Output() editWidget = new EventEmitter<DashboardWidget>();
  @Output() removeWidget = new EventEmitter<string>();

  ngOnInit() {
    // Inizializzazione del widget se necessaria
  }

  getWidgetClass(): string {
    return `widget-type-${this.widget.type}`;
  }

  isKnownWidgetType(): boolean {
    return ['metric', 'chart', 'alert', 'product-info'].includes(this.widget.type);
  }

  getAlertsData(): any[] {
    return this.alertsData || [];
  }

  onEditClick(): void {
    this.editWidget.emit(this.widget);
  }

  onRemoveClick(): void {
    if (confirm(`Rimuovere il widget "${this.widget.title}"?`)) {
      this.removeWidget.emit(this.widget.id);
    }
  }
}