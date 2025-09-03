/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Componente card metriche per dashboard
 * ============================================
 */

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MetricData } from '../../models/product.model';

@Component({
  selector: 'app-metric-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="metric-card" [ngClass]="getCardClass()">
      <div class="metric-title">{{ metric.title }}</div>
      <div class="metric-value">{{ formatValue() }}</div>
      <div class="metric-delta" *ngIf="metric.delta !== undefined" [ngClass]="getDeltaClass()">
        <span class="delta-icon">{{ getDeltaIcon() }}</span>
        {{ Math.abs(metric.delta) }}{{ metric.format === 'percentage' ? '%' : '' }}
      </div>
    </div>
  `,
  styles: [`
    .metric-card {
      background: #ffffff;
      border: 2px solid #444444;
      border-radius: 10px;
      padding: 20px;
      text-align: center;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      transition: transform 0.2s;
    }
    
    .metric-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-title {
      font-size: 14px;
      color: #666;
      font-weight: 500;
      margin-bottom: 8px;
    }
    
    .metric-value {
      font-size: 24px;
      font-weight: bold;
      color: #333;
      margin-bottom: 8px;
    }
    
    .metric-delta {
      font-size: 12px;
      font-weight: 500;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }
    
    .delta-positive {
      color: #28a745;
    }
    
    .delta-negative {
      color: #dc3545;
    }
    
    .delta-neutral {
      color: #6c757d;
    }
    
    .delta-icon {
      font-size: 16px;
    }
  `]
})
export class MetricCardComponent {
  @Input() metric!: MetricData;
  
  Math = Math; // Expose Math to template

  formatValue(): string {
    if (typeof this.metric.value === 'string') {
      return this.metric.value;
    }
    
    switch (this.metric.format) {
      case 'currency':
        return `€${this.metric.value.toLocaleString()}`;
      case 'percentage':
        return `${this.metric.value}%`;
      default:
        return this.metric.value.toString();
    }
  }

  getCardClass(): string {
    if (this.metric.delta === undefined) return '';
    
    if (this.metric.delta > 0) {
      return 'positive-trend';
    } else if (this.metric.delta < 0) {
      return 'negative-trend';
    }
    return 'neutral-trend';
  }

  getDeltaClass(): string {
    if (this.metric.delta === undefined) return 'delta-neutral';
    
    if (this.metric.delta > 0) {
      return 'delta-positive';
    } else if (this.metric.delta < 0) {
      return 'delta-negative';
    }
    return 'delta-neutral';
  }

  getDeltaIcon(): string {
    if (this.metric.delta === undefined) return '';
    
    if (this.metric.delta > 0) {
      return '↗';
    } else if (this.metric.delta < 0) {
      return '↘';
    }
    return '→';
  }
}