/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Servizio gestione layout dashboard drag & drop
 * ============================================
 */

import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { DashboardWidget, DashboardLayout } from '../models/dashboard-widget.model';

@Injectable({
  providedIn: 'root'
})
export class DashboardLayoutService {
  private readonly STORAGE_KEY = 'moretti_dashboard_layout';
  private layoutSubject = new BehaviorSubject<DashboardWidget[]>(this.getDefaultLayout());

  constructor() {
    // Forza reset al nuovo layout per risolvere problemi di allineamento
    // TODO: Rimuovere dopo verifica funzionamento
    const forceReset = true;
    if (forceReset) {
      localStorage.removeItem(this.STORAGE_KEY);
    }
    this.loadLayout();
  }

  get layout$(): Observable<DashboardWidget[]> {
    return this.layoutSubject.asObservable();
  }

  get currentLayout(): DashboardWidget[] {
    return this.layoutSubject.value;
  }

  updateLayout(widgets: DashboardWidget[]): void {
    this.layoutSubject.next(widgets);
    this.saveLayout(widgets);
  }

  addWidget(widget: DashboardWidget): void {
    const currentLayout = [...this.currentLayout];
    currentLayout.push(widget);
    this.updateLayout(currentLayout);
  }

  removeWidget(widgetId: string): void {
    const currentLayout = this.currentLayout.filter(w => w.id !== widgetId);
    this.updateLayout(currentLayout);
  }

  resetToDefault(): void {
    const defaultLayout = this.getDefaultLayout();
    this.updateLayout(defaultLayout);
  }

  saveLayoutAsTemplate(name: string): void {
    const template: DashboardLayout = {
      widgets: this.currentLayout,
      lastModified: new Date(),
      name: name
    };
    
    const templates = this.getSavedTemplates();
    templates[name] = template;
    localStorage.setItem(`${this.STORAGE_KEY}_templates`, JSON.stringify(templates));
  }

  loadTemplate(name: string): void {
    const templates = this.getSavedTemplates();
    if (templates[name]) {
      this.updateLayout(templates[name].widgets);
    }
  }

  getSavedTemplates(): { [key: string]: DashboardLayout } {
    const templatesJson = localStorage.getItem(`${this.STORAGE_KEY}_templates`);
    return templatesJson ? JSON.parse(templatesJson) : {};
  }

  private saveLayout(widgets: DashboardWidget[]): void {
    const layout: DashboardLayout = {
      widgets: widgets,
      lastModified: new Date(),
      name: 'current'
    };
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(layout));
  }

  private loadLayout(): void {
    const savedLayoutJson = localStorage.getItem(this.STORAGE_KEY);
    if (savedLayoutJson) {
      try {
        const savedLayout: DashboardLayout = JSON.parse(savedLayoutJson);
        this.layoutSubject.next(savedLayout.widgets);
      } catch (error) {
        console.warn('Errore caricamento layout salvato, uso default:', error);
        this.layoutSubject.next(this.getDefaultLayout());
      }
    }
  }

  private getDefaultLayout(): DashboardWidget[] {
    return [
      // Row 1: KPI Metrics - Griglia 2x2 con più spazio
      {
        id: 'metric-revenue',
        type: 'metric',
        title: 'Fatturato Totale',
        cols: 2,
        rows: 3,
        x: 0,
        y: 0,
        data: { title: 'Fatturato Totale', value: '€325,450', delta: 12.5, format: 'currency' }
      },
      {
        id: 'metric-products',
        type: 'metric',
        title: 'Prodotti Attivi',
        cols: 2,
        rows: 3,
        x: 2,
        y: 0,
        data: { title: 'Prodotti Attivi', value: 3, delta: 0, format: 'number' }
      },
      {
        id: 'metric-coverage',
        type: 'metric',
        title: 'Giorni Copertura Media',
        cols: 2,
        rows: 3,
        x: 0,
        y: 3,
        data: { title: 'Giorni Copertura Media', value: '6.8', delta: -0.5, format: 'number' }
      },
      {
        id: 'metric-stockout',
        type: 'metric',
        title: 'Rischio Stockout',
        cols: 2,
        rows: 3,
        x: 2,
        y: 3,
        data: { title: 'Rischio Stockout', value: '15.2%', delta: -2.1, format: 'percentage' }
      },
      
      // Row 2: Product Info e Alerts
      {
        id: 'product-info',
        type: 'product-info',
        title: 'Dettagli Prodotto',
        cols: 2,
        rows: 3,
        x: 0,
        y: 6
      },
      
      {
        id: 'alerts',
        type: 'alert',
        title: 'Avvisi Sistema',
        cols: 2,
        rows: 3,
        x: 2,
        y: 6
      },
      
      // Row 3: Chart
      {
        id: 'forecast-chart',
        type: 'chart',
        title: 'Grafico Previsioni',
        cols: 4,
        rows: 6,
        x: 0,
        y: 9,
        data: {
          title: 'Previsioni Domanda',
          chartData: {},
          dataSource: 'combined'
        }
      }
    ];
  }
}