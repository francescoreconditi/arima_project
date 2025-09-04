// ============================================
// COMPONENTE KPI CARD
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Card per visualizzazione KPI
// ============================================

import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';

export interface KpiData {
  title: string;
  value: number | string;
  unit?: string;
  icon?: string;
  trend?: 'up' | 'down' | 'stable';
  changePercent?: number;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info';
  subtitle?: string;
  target?: number;
}

@Component({
  selector: 'app-kpi-card',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatTooltipModule
  ],
  templateUrl: './kpi-card.component.html',
  styleUrls: ['./kpi-card.component.scss']
})
export class KpiCardComponent implements OnInit {
  @Input() data!: KpiData;
  @Input() loading: boolean = false;
  @Input() clickable: boolean = false;

  formattedValue: string = '';
  trendIcon: string = '';
  progressPercent: number = 0;

  ngOnInit(): void {
    this.formatValue();
    this.setTrendIndicator();
    this.calculateProgress();
  }

  private formatValue(): void {
    if (typeof this.data.value === 'number') {
      if (this.data.unit === '€' || this.data.unit === '$') {
        this.formattedValue = this.formatCurrency(this.data.value);
      } else if (this.data.unit === '%') {
        this.formattedValue = `${this.data.value.toFixed(1)}%`;
      } else {
        this.formattedValue = this.formatNumber(this.data.value);
      }
    } else {
      this.formattedValue = this.data.value;
    }

    if (this.data.unit && this.data.unit !== '%' && this.data.unit !== '€' && this.data.unit !== '$') {
      this.formattedValue += ` ${this.data.unit}`;
    }
  }

  private formatNumber(value: number): string {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toFixed(0);
  }

  private formatCurrency(value: number): string {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  }

  private setTrendIndicator(): void {
    if (!this.data.trend) return;

    switch (this.data.trend) {
      case 'up':
        this.trendIcon = 'trending_up';
        break;
      case 'down':
        this.trendIcon = 'trending_down';
        break;
      case 'stable':
        this.trendIcon = 'trending_flat';
        break;
    }
  }

  private calculateProgress(): void {
    if (this.data.target && typeof this.data.value === 'number') {
      this.progressPercent = Math.min(100, (this.data.value / this.data.target) * 100);
    }
  }

  getColorClass(): string {
    return `kpi-${this.data.color || 'primary'}`;
  }

  getIconName(): string {
    return this.data.icon || 'dashboard';
  }
}