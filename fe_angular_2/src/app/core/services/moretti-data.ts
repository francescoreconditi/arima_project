import { Injectable } from '@angular/core';
import { Observable, of, BehaviorSubject } from 'rxjs';
import { delay, map } from 'rxjs/operators';
import { 
  Product, 
  ProductInventory, 
  ForecastResult, 
  KPIMetric, 
  SupplierInfo, 
  OrderRecommendation 
} from '../models/product.model';

@Injectable({
  providedIn: 'root'
})
export class MorettiDataService {
  private productsSubject = new BehaviorSubject<Product[]>([]);
  private inventorySubject = new BehaviorSubject<ProductInventory[]>([]);
  private kpiSubject = new BehaviorSubject<KPIMetric[]>([]);

  // Observables pubblici
  products$ = this.productsSubject.asObservable();
  inventory$ = this.inventorySubject.asObservable();
  kpi$ = this.kpiSubject.asObservable();

  constructor() {
    this.initializeMockData();
  }

  private initializeMockData(): void {
    // Dati prodotti mock basati su dashboard Moretti
    const mockProducts: Product[] = [
      {
        codice: 'CRZ001',
        nome: 'Carrozzina Standard',
        categoria: 'Carrozzine',
        prezzo_unitario: 450.00,
        lead_time_giorni: 7,
        costo_stockout_giornaliero: 150.00,
        costo_mantenimento_scorta: 2.25,
        scorta_minima: 10,
        scorta_massima: 100,
        quantita_riordino_standard: 50,
        fornitore_principale: 'MedSupply Italia',
        fornitori_alternativi: ['Healthcare Solutions', 'Mobility Plus']
      },
      {
        codice: 'MAT001',
        nome: 'Materasso Antidecubito',
        categoria: 'Materassi Antidecubito',
        prezzo_unitario: 680.00,
        lead_time_giorni: 10,
        costo_stockout_giornaliero: 220.00,
        costo_mantenimento_scorta: 3.40,
        scorta_minima: 8,
        scorta_massima: 60,
        quantita_riordino_standard: 25,
        fornitore_principale: 'AntiDecubito Pro',
        fornitori_alternativi: ['MedComfort', 'TheraTech']
      },
      {
        codice: 'ELT001',
        nome: 'Saturimetro Digitale',
        categoria: 'Elettromedicali',
        prezzo_unitario: 89.90,
        lead_time_giorni: 5,
        costo_stockout_giornaliero: 45.00,
        costo_mantenimento_scorta: 0.45,
        scorta_minima: 25,
        scorta_massima: 200,
        quantita_riordino_standard: 75,
        fornitore_principale: 'DiagnosticPro',
        fornitori_alternativi: ['MedDevice', 'ElectroMed']
      }
    ];

    const mockInventory: ProductInventory[] = [
      {
        codice: 'CRZ001',
        nome: 'Carrozzina Standard',
        scorta_attuale: 15,
        scorta_minima: 10,
        scorta_massima: 100,
        punto_riordino: 25,
        quantita_riordino: 50,
        domanda_media_giornaliera: 3.2,
        deviazione_standard: 1.1,
        lead_time_giorni: 7,
        livello_servizio_target: 0.95,
        giorni_copertura: 4.7,
        stato: 'attenzione'
      },
      {
        codice: 'MAT001',
        nome: 'Materasso Antidecubito',
        scorta_attuale: 5,
        scorta_minima: 8,
        scorta_massima: 60,
        punto_riordino: 18,
        quantita_riordino: 25,
        domanda_media_giornaliera: 2.8,
        deviazione_standard: 0.9,
        lead_time_giorni: 10,
        livello_servizio_target: 0.95,
        giorni_copertura: 1.8,
        stato: 'critico'
      },
      {
        codice: 'ELT001',
        nome: 'Saturimetro Digitale',
        scorta_attuale: 45,
        scorta_minima: 25,
        scorta_massima: 200,
        punto_riordino: 50,
        quantita_riordino: 75,
        domanda_media_giornaliera: 4.5,
        deviazione_standard: 1.8,
        lead_time_giorni: 5,
        livello_servizio_target: 0.90,
        giorni_copertura: 10.0,
        stato: 'ottimale'
      }
    ];

    const mockKPIs: KPIMetric[] = [
      {
        nome: 'Livello Servizio',
        valore: '94.2%',
        variazione: 2.1,
        trend: 'up',
        descrizione: 'Percentuale ordini evasi senza stockout'
      },
      {
        nome: 'Rotazione Scorte',
        valore: '6.8x',
        variazione: -0.3,
        trend: 'down',
        descrizione: 'Numero di volte scorte ruotano annualmente'
      },
      {
        nome: 'Copertura Media',
        valore: '12.5',
        variazione: 1.8,
        trend: 'up',
        unita: 'giorni',
        descrizione: 'Giorni copertura media scorte'
      },
      {
        nome: 'Investimento Scorte',
        valore: '€125.8K',
        variazione: -5.2,
        trend: 'down',
        descrizione: 'Valore totale scorte magazzino'
      }
    ];

    this.productsSubject.next(mockProducts);
    this.inventorySubject.next(mockInventory);
    this.kpiSubject.next(mockKPIs);
  }

  // Metodi per ottenere dati
  getProducts(): Observable<Product[]> {
    return this.products$.pipe(delay(200));
  }

  getProduct(codice: string): Observable<Product | undefined> {
    return this.products$.pipe(
      map(products => products.find(p => p.codice === codice)),
      delay(100)
    );
  }

  getInventory(): Observable<ProductInventory[]> {
    return this.inventory$.pipe(delay(200));
  }

  getInventoryByProduct(codice: string): Observable<ProductInventory | undefined> {
    return this.inventory$.pipe(
      map(inventory => inventory.find(i => i.codice === codice)),
      delay(100)
    );
  }

  getKPIs(): Observable<KPIMetric[]> {
    return this.kpi$.pipe(delay(100));
  }

  getCategories(): Observable<string[]> {
    return this.products$.pipe(
      map(products => [...new Set(products.map(p => p.categoria))]),
      delay(100)
    );
  }

  getProductsByCategory(category: string): Observable<Product[]> {
    return this.products$.pipe(
      map(products => products.filter(p => p.categoria === category)),
      delay(100)
    );
  }

  // Simulazione forecast
  getForecast(productCode: string, days: number = 30): Observable<ForecastResult[]> {
    const startDate = new Date();
    const forecast: ForecastResult[] = [];
    
    for (let i = 0; i < days; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Simulazione dati forecast con trend e rumore
      const baseValue = 25 + Math.sin(i * 0.2) * 5;
      const noise = (Math.random() - 0.5) * 8;
      const prediction = Math.max(0, baseValue + noise);
      
      forecast.push({
        prodotto: productCode,
        data: date,
        previsione: Math.round(prediction * 100) / 100,
        limite_inferiore: Math.round((prediction - 5) * 100) / 100,
        limite_superiore: Math.round((prediction + 5) * 100) / 100,
        intervallo_confidenza: 0.95
      });
    }
    
    return of(forecast).pipe(delay(500));
  }

  // Aggiornamento dati
  updateInventory(inventory: ProductInventory[]): Observable<boolean> {
    this.inventorySubject.next(inventory);
    return of(true).pipe(delay(200));
  }

  refreshData(): Observable<boolean> {
    this.initializeMockData();
    return of(true).pipe(delay(300));
  }
}
