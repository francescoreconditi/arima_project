/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Servizio dati mock per dashboard Moretti
 * ============================================
 */

import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { Product, SalesData, MetricData, AlertData } from '../models/product.model';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  
  // Parametri scenario per calcoli dinamici
  private leadTimeModifier: number = 100;
  private demandModifier: number = 100;
  private forecastDays: number = 30;
  
  private mockProducts: Product[] = [
    {
      codice: 'CRZ001',
      nome: 'Carrozzina Standard',
      categoria: 'Mobilità',
      fornitore: 'MedSupply Italia',
      prezzo_unitario: 1250.00,
      lead_time: 5,
      domanda_media_giornaliera: 27.2,
      giacenza_attuale: 150,
      punto_riordino: 136,
      stock_sicurezza: 45
    },
    {
      codice: 'MAT001',
      nome: 'Materasso Antidecubito',
      categoria: 'Comfort',
      fornitore: 'AntiDecubito Pro',
      prezzo_unitario: 1534.00,
      lead_time: 7,
      domanda_media_giornaliera: 26.7,
      giacenza_attuale: 89,
      punto_riordino: 187,
      stock_sicurezza: 52
    },
    {
      codice: 'ELT001',
      nome: 'Saturimetro',
      categoria: 'Elettromedicali',
      fornitore: 'DiagnosticPro',
      prezzo_unitario: 1035.00,
      lead_time: 3,
      domanda_media_giornaliera: 19.2,
      giacenza_attuale: 245,
      punto_riordino: 58,
      stock_sicurezza: 25
    }
  ];

  constructor() { }

  getProducts(): Observable<Product[]> {
    return of(this.mockProducts);
  }

  getProductByCode(code: string): Observable<Product | undefined> {
    const product = this.mockProducts.find(p => p.codice === code);
    return of(product);
  }

  getProductsByCategory(category: string): Observable<Product[]> {
    if (category === 'Tutti') {
      return of(this.mockProducts);
    }
    const products = this.mockProducts.filter(p => p.categoria === category);
    return of(products);
  }

  getCategories(): Observable<string[]> {
    const categories = ['Tutti', ...new Set(this.mockProducts.map(p => p.categoria))];
    return of(categories);
  }


  getAlerts(): Observable<AlertData[]> {
    return of([
      {
        level: 'critica',
        message: 'MAT001: Giacenza sotto punto riordino (89 < 187)',
        prodotto_codice: 'MAT001'
      },
      {
        level: 'alta',
        message: 'CRZ001: Previsione alta domanda prossimi 7 giorni',
        prodotto_codice: 'CRZ001'
      },
      {
        level: 'media',
        message: 'ELT001: Lead time fornitore aumentato a 4 giorni',
        prodotto_codice: 'ELT001'
      }
    ]);
  }

  getSalesHistory(productCode: string, days: number = 90): Observable<SalesData[]> {
    const salesData: SalesData[] = [];
    const product = this.mockProducts.find(p => p.codice === productCode);
    
    if (!product) {
      return of(salesData);
    }

    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      // Simula dati di vendita con variazione casuale
      const baseAmount = product.domanda_media_giornaliera;
      const variation = (Math.random() - 0.5) * 0.4 * baseAmount;
      const quantity = Math.max(0, Math.round(baseAmount + variation));
      
      salesData.push({
        data: date,
        quantita: quantity,
        prodotto_codice: productCode
      });
    }
    
    return of(salesData);
  }

  // Metodi per aggiornare parametri scenario
  updateScenarioParams(leadTimeModifier: number, demandModifier: number): void {
    this.leadTimeModifier = leadTimeModifier;
    this.demandModifier = demandModifier;
  }

  updateForecastDays(days: number): void {
    this.forecastDays = days;
  }

  // Genera dati mock forecast reattivi ai parametri
  generateMockForecast(productCode: string, days: number): any[] {
    const product = this.mockProducts.find(p => p.codice === productCode);
    if (!product) return [];

    const forecast = [];
    const baseDemand = product.domanda_media_giornaliera * (this.demandModifier / 100);
    
    for (let i = 1; i <= days; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i);
      
      // Simula trend con influenza dei parametri scenario
      const seasonality = 1 + 0.1 * Math.sin((i / days) * 2 * Math.PI);
      const trend = 1 + (i / days) * 0.05;
      const noise = 1 + (Math.random() - 0.5) * 0.2;
      
      const value = baseDemand * seasonality * trend * noise;
      
      forecast.push({
        data: date,
        valore: Math.max(0, Math.round(value)),
        lower_bound: Math.round(value * 0.8),
        upper_bound: Math.round(value * 1.2),
        prodotto_codice: productCode
      });
    }
    
    return forecast;
  }

  // Aggiorna metriche basate sui parametri
  getMetrics(): Observable<MetricData[]> {
    const demandImpact = this.demandModifier / 100;
    const leadTimeImpact = this.leadTimeModifier / 100;
    
    const metrics: MetricData[] = [
      {
        title: 'Fatturato Totale',
        value: `€${(325450 * demandImpact).toLocaleString()}`,
        delta: (12.5 * demandImpact) - 12.5,
        format: 'currency'
      },
      {
        title: 'Prodotti Attivi',
        value: 3,
        delta: 0,
        format: 'number'
      },
      {
        title: 'Giorni Copertura Media',
        value: (6.8 / leadTimeImpact).toFixed(1),
        delta: (6.8 / leadTimeImpact) - 6.8,
        format: 'number'
      },
      {
        title: 'Rischio Stockout',
        value: `${(15.2 * leadTimeImpact / demandImpact).toFixed(1)}%`,
        delta: (15.2 * leadTimeImpact / demandImpact) - 15.2,
        format: 'percentage'
      }
    ];
    
    return of(metrics);
  }
}