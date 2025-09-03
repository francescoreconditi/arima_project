/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Componente principale dashboard Moretti Angular
 * ============================================
 */

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

// Importa i servizi
import { DataService } from './services/data.service';
import { ApiService } from './services/api.service';
import { DashboardLayoutService } from './services/dashboard-layout.service';

// Importa i componenti
import { DashboardWidgetComponent } from './components/dashboard-widget/dashboard-widget.component';

// Importa i modelli
import { Product, MetricData, AlertData, ForecastData, SalesData } from './models/product.model';
import { DashboardWidget } from './models/dashboard-widget.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    DashboardWidgetComponent
  ],
  providers: [DataService, ApiService, DashboardLayoutService],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App implements OnInit {
  title = 'moretti-dashboard';
  
  // Dati principali
  products: Product[] = [];
  categories: string[] = [];
  metrics: MetricData[] = [];
  alerts: AlertData[] = [];
  forecastData: ForecastData[] = [];
  historicalData: SalesData[] = [];
  
  // Dashboard widgets (layout CSS Grid nativo)
  dashboardWidgets: DashboardWidget[] = [];
  
  // Selezioni attive
  selectedLanguage: string = 'it';
  selectedCategory: string = 'Tutti';
  selectedProductCode: string = '';
  selectedProduct?: Product;
  filteredProducts: Product[] = [];
  
  // Parametri controllo
  forecastDays: number = 30;
  leadTimeModifier: number = 100;
  demandModifier: number = 100;

  constructor(
    private dataService: DataService,
    private apiService: ApiService,
    private layoutService: DashboardLayoutService
  ) {}

  ngOnInit() {
    this.loadInitialData();
    this.loadDashboardLayout();
  }

  private loadDashboardLayout(): void {
    this.layoutService.layout$.subscribe(widgets => {
      this.dashboardWidgets = widgets;
      this.updateWidgetsData();
    });
  }

  private loadInitialData() {
    // Carica categorie
    this.dataService.getCategories().subscribe(categories => {
      this.categories = categories;
    });

    // Carica prodotti
    this.dataService.getProducts().subscribe(products => {
      this.products = products;
      this.filteredProducts = products;
      
      // Seleziona il primo prodotto per default
      if (products.length > 0) {
        this.selectedProductCode = products[0].codice;
        this.onProductChange();
      }
    });

    // Carica metriche
    this.dataService.getMetrics().subscribe(metrics => {
      this.metrics = metrics;
    });

    // Carica avvisi
    this.dataService.getAlerts().subscribe(alerts => {
      this.alerts = alerts;
    });
  }

  onLanguageChange() {
    console.log('Lingua cambiata:', this.selectedLanguage);
    // Qui si potrebbero ricaricare le traduzioni
  }

  onCategoryChange() {
    this.dataService.getProductsByCategory(this.selectedCategory).subscribe(products => {
      this.filteredProducts = products;
      
      // Reset selezione prodotto
      this.selectedProductCode = '';
      this.selectedProduct = undefined;
      
      // Se c'è un solo prodotto, selezionalo automaticamente
      if (products.length === 1) {
        this.selectedProductCode = products[0].codice;
        this.onProductChange();
      }
    });
  }

  onProductChange() {
    if (!this.selectedProductCode) {
      this.selectedProduct = undefined;
      return;
    }

    this.dataService.getProductByCode(this.selectedProductCode).subscribe(product => {
      this.selectedProduct = product;
      
      if (product) {
        // Carica dati storici
        this.dataService.getSalesHistory(product.codice).subscribe(sales => {
          this.historicalData = sales;
        });
        
        // Carica previsioni via API (con fallback)
        this.apiService.forecast(`arima_${product.codice}`, this.forecastDays).subscribe(
          forecast => {
            this.forecastData = forecast;
          },
          error => {
            console.warn('Errore caricamento forecast:', error);
          }
        );
      }
      
      // Aggiorna i dati dei widget
      this.updateWidgetsData();
    });
  }

  onForecastDaysChange() {
    // Aggiorna i giorni di previsione nel DataService
    this.dataService.updateForecastDays(this.forecastDays);
    
    if (this.selectedProduct) {
      // Ricarica previsioni con nuovo periodo
      this.apiService.forecast(`arima_${this.selectedProduct.codice}`, this.forecastDays).subscribe(
        forecast => {
          this.forecastData = forecast;
          this.updateWidgetsData();
        },
        error => {
          console.warn('Errore aggiornamento forecast:', error);
        }
      );
    }
  }

  onScenarioChange() {
    console.log('Scenario changed:', {
      leadTime: this.leadTimeModifier,
      demand: this.demandModifier
    });
    
    // Aggiorna i parametri nel DataService
    this.dataService.updateScenarioParams(this.leadTimeModifier, this.demandModifier);
    
    // Ricarica le metriche con i nuovi parametri
    this.dataService.getMetrics().subscribe(metrics => {
      this.metrics = metrics;
    });
    
    // Se c'è un prodotto selezionato, ricarica anche le previsioni
    if (this.selectedProduct) {
      this.apiService.forecast(`arima_${this.selectedProduct.codice}`, this.forecastDays).subscribe(
        forecast => {
          this.forecastData = forecast;
          this.updateWidgetsData();
        },
        error => {
          console.warn('Errore aggiornamento forecast con nuovi parametri:', error);
        }
      );
    } else {
      this.updateWidgetsData();
    }
  }

  // Dashboard Widget Management
  private updateWidgetsData(): void {
    // Aggiorna i dati dei widget con i dati correnti
    this.dashboardWidgets.forEach(widget => {
      switch (widget.type) {
        case 'metric':
          // Trova la metrica corrispondente
          const metric = this.metrics.find(m => m.title === widget.title);
          if (metric) {
            widget.data = metric;
          }
          break;
        case 'chart':
          widget.data = {
            title: widget.title,
            forecastData: this.forecastData,
            historicalData: this.historicalData,
            dataSource: 'combined'
          };
          break;
        case 'alert':
          widget.data = {
            alerts: this.alerts
          };
          break;
        case 'product-info':
          widget.data = this.selectedProduct;
          break;
      }
    });
  }

  onRemoveWidget(widgetId: string): void {
    this.layoutService.removeWidget(widgetId);
  }

  onEditWidget(widget: DashboardWidget): void {
    // TODO: Implementare dialog di modifica widget
    console.log('Edit widget:', widget);
  }

  resetDashboardLayout(): void {
    if (confirm('Ripristinare il layout di default? Tutte le personalizzazioni andranno perse.')) {
      this.layoutService.resetToDefault();
    }
  }

  saveDashboardTemplate(): void {
    const templateName = prompt('Nome del template:');
    if (templateName) {
      this.layoutService.saveLayoutAsTemplate(templateName);
      alert(`Template "${templateName}" salvato con successo!`);
    }
  }

  getChartDataForWidget(widget: DashboardWidget): any {
    return {
      forecastData: this.forecastData,
      historicalData: this.historicalData
    };
  }

  // Helper functions per il nuovo layout nativo
  getMetricWidgets(): DashboardWidget[] {
    return this.dashboardWidgets.filter(w => w.type === 'metric');
  }

  getProductInfoWidget(): DashboardWidget {
    return this.dashboardWidgets.find(w => w.type === 'product-info') || this.dashboardWidgets[0];
  }

  getAlertsWidget(): DashboardWidget {
    return this.dashboardWidgets.find(w => w.type === 'alert') || this.dashboardWidgets[0];
  }

  getChartWidget(): DashboardWidget {
    return this.dashboardWidgets.find(w => w.type === 'chart') || this.dashboardWidgets[0];
  }
}