import { Component, Input, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MorettiDataService } from '../../../core/services/moretti-data';
import { KPIMetric } from '../../../core/models/product.model';

@Component({
  selector: 'app-kpi-cards',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './kpi-cards.html',
  styleUrl: './kpi-cards.scss'
})
export class KpiCardsComponent implements OnInit, OnDestroy {
  @Input() selectedProduct = '';
  @Input() selectedCategory = '';

  private destroy$ = new Subject<void>();
  kpis: KPIMetric[] = [];

  constructor(private dataService: MorettiDataService) {}

  ngOnInit(): void {
    this.loadKPIs();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private loadKPIs(): void {
    this.dataService.getKPIs()
      .pipe(takeUntil(this.destroy$))
      .subscribe(kpis => {
        this.kpis = kpis;
      });
  }

  getTrendIcon(trend?: string): string {
    switch (trend) {
      case 'up': return '↑';
      case 'down': return '↓';
      default: return '→';
    }
  }

  getTrendClass(trend?: string): string {
    switch (trend) {
      case 'up': return 'trend-up';
      case 'down': return 'trend-down';
      default: return 'trend-neutral';
    }
  }

  getVariationText(variazione?: number): string {
    if (!variazione) return '';
    const sign = variazione > 0 ? '+' : '';
    return `${sign}${variazione}%`;
  }
}
