// ============================================
// SERVIZIO INVENTARIO
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Gestione inventario e ottimizzazione
// ============================================

import { Injectable } from '@angular/core';
import { Observable, BehaviorSubject, combineLatest, timer, of } from 'rxjs';
import { map, tap, switchMap, shareReplay, catchError } from 'rxjs/operators';
import { ApiService } from './api.service';
import {
  Product,
  ProductCategory,
  Supplier,
  InventoryOptimization,
  ReorderPlan,
  StockMovement,
  InventoryAlert,
  InventoryDashboardData,
  ABCClassification,
  XYZAnalysis,
  InventoryKPI
} from '../models/inventory.model';

@Injectable({
  providedIn: 'root'
})
export class InventoryService {
  private products$ = new BehaviorSubject<Product[]>([]);
  private categories$ = new BehaviorSubject<ProductCategory[]>([]);
  private suppliers$ = new BehaviorSubject<Supplier[]>([]);
  private alerts$ = new BehaviorSubject<InventoryAlert[]>([]);
  private selectedProduct$ = new BehaviorSubject<Product | null>(null);
  private dashboardData$ = new BehaviorSubject<InventoryDashboardData | null>(null);
  private refreshInterval = 60000; // 1 minuto

  constructor(private api: ApiService) {
    this.initializeAutoRefresh();
  }

  // Observable per prodotti
  getProducts(): Observable<Product[]> {
    return this.products$.asObservable();
  }

  // Observable per categorie
  getCategories(): Observable<ProductCategory[]> {
    return this.categories$.asObservable();
  }

  // Observable per fornitori
  getSuppliers(): Observable<Supplier[]> {
    return this.suppliers$.asObservable();
  }

  // Observable per alerts
  getAlerts(): Observable<InventoryAlert[]> {
    return this.alerts$.asObservable();
  }

  // Observable per prodotto selezionato
  getSelectedProduct(): Observable<Product | null> {
    return this.selectedProduct$.asObservable();
  }

  // Observable per dashboard data
  getDashboardData(): Observable<InventoryDashboardData | null> {
    return this.dashboardData$.asObservable();
  }

  // Carica tutti i prodotti
  loadProducts(): Observable<Product[]> {
    // Usa sempre dati mock per evitare errori API durante lo sviluppo
    const mockProducts = this.getMockProducts();
    this.products$.next(mockProducts);
    return of(mockProducts).pipe(
      shareReplay(1),
      // Prova comunque la chiamata API in background (opzionale)
      tap(() => {
        this.api.get<Product[]>('inventory/products').pipe(
          catchError(() => of([]))
        ).subscribe(apiProducts => {
          if (apiProducts && apiProducts.length > 0) {
            this.products$.next(apiProducts);
          }
        });
      })
    );
  }

  // Carica categorie
  loadCategories(): Observable<ProductCategory[]> {
    return this.api.get<ProductCategory[]>('inventory/categories').pipe(
      tap(categories => this.categories$.next(categories)),
      shareReplay(1)
    );
  }

  // Carica fornitori
  loadSuppliers(): Observable<Supplier[]> {
    return this.api.get<Supplier[]>('inventory/suppliers').pipe(
      tap(suppliers => this.suppliers$.next(suppliers)),
      shareReplay(1)
    );
  }

  // Carica dati dashboard
  loadDashboardData(): Observable<InventoryDashboardData> {
    // Usa sempre dati mock per evitare errori API durante lo sviluppo
    const mockData = this.getMockDashboardData();
    this.dashboardData$.next(mockData);
    if (mockData.alerts) {
      this.alerts$.next(mockData.alerts);
    }
    
    // Prova comunque la chiamata API in background (opzionale)
    this.api.get<InventoryDashboardData>('inventory/dashboard').pipe(
      catchError(() => of(null))
    ).subscribe(apiData => {
      if (apiData) {
        this.dashboardData$.next(apiData);
        if (apiData.alerts) {
          this.alerts$.next(apiData.alerts);
        }
      }
    });
    
    return of(mockData).pipe(shareReplay(1));
  }

  // Ottieni singolo prodotto
  getProductById(id: string): Observable<Product> {
    return this.api.get<Product>(`inventory/products/${id}`).pipe(
      tap(product => this.selectedProduct$.next(product))
    );
  }

  // Crea nuovo prodotto
  createProduct(product: Partial<Product>): Observable<Product> {
    return this.api.post<Product>('inventory/products', product).pipe(
      tap(newProduct => {
        const current = this.products$.value;
        this.products$.next([...current, newProduct]);
      })
    );
  }

  // Aggiorna prodotto
  updateProduct(id: string, updates: Partial<Product>): Observable<Product> {
    return this.api.put<Product>(`inventory/products/${id}`, updates).pipe(
      tap(updated => {
        const current = this.products$.value;
        const index = current.findIndex(p => p.id === id);
        if (index >= 0) {
          current[index] = updated;
          this.products$.next([...current]);
        }
        if (this.selectedProduct$.value?.id === id) {
          this.selectedProduct$.next(updated);
        }
      })
    );
  }

  // Elimina prodotto
  deleteProduct(id: string): Observable<void> {
    return this.api.delete<void>(`inventory/products/${id}`).pipe(
      tap(() => {
        const current = this.products$.value;
        this.products$.next(current.filter(p => p.id !== id));
        if (this.selectedProduct$.value?.id === id) {
          this.selectedProduct$.next(null);
        }
      })
    );
  }

  // Ottieni ottimizzazione inventario
  getOptimization(productId: string): Observable<InventoryOptimization> {
    return this.api.get<InventoryOptimization>(`inventory/optimize/${productId}`);
  }

  // Ottimizzazione batch
  batchOptimize(productIds: string[]): Observable<InventoryOptimization[]> {
    return this.api.post<InventoryOptimization[]>('inventory/optimize/batch', { productIds });
  }

  // Genera piano di riordino
  generateReorderPlan(optimization: InventoryOptimization): Observable<ReorderPlan> {
    return this.api.post<ReorderPlan>('inventory/reorder-plan', optimization);
  }

  // Ottieni movimenti stock
  getStockMovements(productId: string, startDate?: string, endDate?: string): Observable<StockMovement[]> {
    const params: any = { productId };
    if (startDate) params.startDate = startDate;
    if (endDate) params.endDate = endDate;
    
    return this.api.get<StockMovement[]>('inventory/movements', params);
  }

  // Registra movimento stock
  recordStockMovement(movement: Partial<StockMovement>): Observable<StockMovement> {
    return this.api.post<StockMovement>('inventory/movements', movement).pipe(
      tap(() => this.loadProducts().subscribe())
    );
  }

  // ABC Classification
  getABCClassification(): Observable<ABCClassification[]> {
    return this.api.get<ABCClassification[]>('inventory/analysis/abc');
  }

  // XYZ Analysis
  getXYZAnalysis(): Observable<XYZAnalysis[]> {
    return this.api.get<XYZAnalysis[]>('inventory/analysis/xyz');
  }

  // Combinazione ABC-XYZ
  getABCXYZMatrix(): Observable<any> {
    return combineLatest([
      this.getABCClassification(),
      this.getXYZAnalysis()
    ]).pipe(
      map(([abc, xyz]) => this.combineABCXYZ(abc, xyz))
    );
  }

  private combineABCXYZ(abc: ABCClassification[], xyz: XYZAnalysis[]): any {
    const matrix: any = {};
    
    abc.forEach(abcItem => {
      const xyzItem = xyz.find(x => x.productId === abcItem.productId);
      if (xyzItem) {
        const key = `${abcItem.class}${xyzItem.class}`;
        if (!matrix[key]) {
          matrix[key] = [];
        }
        matrix[key].push({
          productId: abcItem.productId,
          abcClass: abcItem.class,
          xyzClass: xyzItem.class,
          value: abcItem.value,
          variability: xyzItem.variability
        });
      }
    });
    
    return matrix;
  }

  // KPI Calculations
  calculateKPIs(products: Product[]): InventoryKPI[] {
    const totalValue = products.reduce((sum, p) => sum + (p.currentStock * p.unitPrice), 0);
    const lowStockCount = products.filter(p => p.currentStock < p.minStock).length;
    const overStockCount = products.filter(p => p.currentStock > p.maxStock).length;
    const stockoutCount = products.filter(p => p.currentStock === 0).length;
    
    return [
      {
        name: 'Total Inventory Value',
        value: totalValue,
        unit: '€',
        status: 'good'
      },
      {
        name: 'Low Stock Items',
        value: lowStockCount,
        status: lowStockCount > 5 ? 'warning' : 'good'
      },
      {
        name: 'Overstock Items',
        value: overStockCount,
        status: overStockCount > 10 ? 'warning' : 'good'
      },
      {
        name: 'Stockout Items',
        value: stockoutCount,
        status: stockoutCount > 0 ? 'critical' : 'good'
      },
      {
        name: 'Service Level',
        value: ((products.length - stockoutCount) / products.length) * 100,
        unit: '%',
        target: 95,
        status: ((products.length - stockoutCount) / products.length) * 100 >= 95 ? 'good' : 'warning'
      }
    ];
  }

  // Mark alert as read
  markAlertAsRead(alertId: string): Observable<void> {
    return this.api.patch<void>(`inventory/alerts/${alertId}`, { isRead: true }).pipe(
      tap(() => {
        const current = this.alerts$.value;
        const alert = current.find(a => a.id === alertId);
        if (alert) {
          alert.isRead = true;
          this.alerts$.next([...current]);
        }
      })
    );
  }

  // Clear all alerts
  clearAllAlerts(): void {
    this.alerts$.next([]);
  }

  // Export inventory report
  exportInventoryReport(format: 'csv' | 'excel' | 'pdf' = 'csv'): Observable<Blob> {
    return this.api.downloadFile(`inventory/export?format=${format}`);
  }

  // Auto-refresh initialization
  private initializeAutoRefresh(): void {
    // Disabilitato temporaneamente per evitare troppe chiamate API durante lo sviluppo
    // timer(0, this.refreshInterval).pipe(
    //   switchMap(() => this.loadDashboardData())
    // ).subscribe();
  }

  // Set refresh interval
  setRefreshInterval(milliseconds: number): void {
    this.refreshInterval = milliseconds;
    this.initializeAutoRefresh();
  }

  // Search products
  searchProducts(query: string): Observable<Product[]> {
    if (!query || query.trim().length === 0) {
      return this.products$;
    }
    
    const lowerQuery = query.toLowerCase();
    return this.products$.pipe(
      map(products => products.filter(p => 
        p.name.toLowerCase().includes(lowerQuery) ||
        p.code.toLowerCase().includes(lowerQuery) ||
        p.description?.toLowerCase().includes(lowerQuery)
      ))
    );
  }

  // Genera prodotti mock per sviluppo e fallback
  private getMockProducts(): Product[] {
    return [
      {
        id: 'CRZ001',
        code: 'CRZ001',
        name: 'Carrozzina Standard',
        description: 'Carrozzina pieghevole standard',
        currentStock: 45,
        minStock: 20,
        maxStock: 100,
        unitPrice: 350,
        category: { id: 'wheelchairs', name: 'Carrozzine' },
        leadTimeDays: 14,
        isActive: true
      },
      {
        id: 'MAT001',
        code: 'MAT001',
        name: 'Materasso Antidecubito',
        description: 'Materasso antidecubito con compressore',
        currentStock: 23,
        minStock: 10,
        maxStock: 50,
        unitPrice: 450,
        category: { id: 'mattresses', name: 'Materassi' },
        leadTimeDays: 7,
        isActive: true
      },
      {
        id: 'ELT001',
        code: 'ELT001',
        name: 'Saturimetro',
        description: 'Saturimetro da dito con display',
        currentStock: 120,
        minStock: 50,
        maxStock: 200,
        unitPrice: 75,
        category: { id: 'medical', name: 'Dispositivi Medici' },
        leadTimeDays: 5,
        isActive: true
      },
      {
        id: 'DEA001',
        code: 'DEA001',
        name: 'Deambulatore',
        description: 'Deambulatore 4 ruote con seduta',
        currentStock: 15,
        minStock: 10,
        maxStock: 40,
        unitPrice: 120,
        category: { id: 'wheelchairs', name: 'Carrozzine' },
        leadTimeDays: 10,
        isActive: true
      },
      {
        id: 'TER001',
        code: 'TER001',
        name: 'Termometro Digitale',
        description: 'Termometro digitale infrarossi',
        currentStock: 80,
        minStock: 30,
        maxStock: 150,
        unitPrice: 45,
        category: { id: 'medical', name: 'Dispositivi Medici' },
        leadTimeDays: 5,
        isActive: true
      },
      {
        id: 'SER001',
        code: 'SER001',
        name: 'Sollevatore Pazienti',
        description: 'Sollevatore elettrico per pazienti',
        currentStock: 8,
        minStock: 3,
        maxStock: 15,
        unitPrice: 2500,
        category: { id: 'lifting', name: 'Sollevatori' },
        leadTimeDays: 21,
        isActive: true
      },
      {
        id: 'LET001',
        code: 'LET001',
        name: 'Letto Ospedaliero',
        description: 'Letto articolato elettrico 4 sezioni',
        currentStock: 12,
        minStock: 5,
        maxStock: 25,
        unitPrice: 1800,
        category: { id: 'beds', name: 'Letti' },
        leadTimeDays: 28,
        isActive: true
      },
      {
        id: 'MON001',
        code: 'MON001',
        name: 'Monitor Parametri Vitali',
        description: 'Monitor multiparametrico per terapia intensiva',
        currentStock: 6,
        minStock: 2,
        maxStock: 10,
        unitPrice: 8500,
        category: { id: 'monitoring', name: 'Monitoraggio' },
        leadTimeDays: 35,
        isActive: true
      },
      {
        id: 'VEN001',
        code: 'VEN001',
        name: 'Ventilatore Polmonare',
        description: 'Ventilatore per terapia intensiva',
        currentStock: 4,
        minStock: 2,
        maxStock: 8,
        unitPrice: 15000,
        category: { id: 'respiratory', name: 'Respiratori' },
        leadTimeDays: 42,
        isActive: true
      },
      {
        id: 'DEF001',
        code: 'DEF001',
        name: 'Defibrillatore',
        description: 'Defibrillatore semiautomatico esterno',
        currentStock: 0,
        minStock: 1,
        maxStock: 5,
        unitPrice: 3500,
        category: { id: 'emergency', name: 'Emergenza' },
        leadTimeDays: 14,
        isActive: true
      }
    ];
  }

  // Genera dashboard data mock
  private getMockDashboardData(): InventoryDashboardData {
    const mockProducts = this.getMockProducts();
    const totalValue = mockProducts.reduce((sum, p) => sum + (p.currentStock * p.unitPrice), 0);
    const lowStockItems = mockProducts.filter(p => p.currentStock < p.minStock).length;
    const stockoutItems = mockProducts.filter(p => p.currentStock === 0).length;
    
    return {
      summary: {
        totalProducts: mockProducts.length,
        totalValue: totalValue,
        lowStockItems: lowStockItems,
        overstockItems: 0,
        pendingOrders: lowStockItems + stockoutItems,
        turnoverRate: 4.2,
        stockAccuracy: 98.5
      },
      kpis: [
        {
          name: 'Valore Inventario',
          value: totalValue,
          unit: '€',
          status: 'good',
          trend: 'up'
        },
        {
          name: 'Prodotti in Stock',
          value: mockProducts.length,
          status: 'good'
        },
        {
          name: 'Ordini Pendenti',
          value: lowStockItems + stockoutItems,
          status: stockoutItems > 0 ? 'critical' : lowStockItems > 0 ? 'warning' : 'good'
        },
        {
          name: 'Livello Servizio',
          value: ((mockProducts.length - stockoutItems) / mockProducts.length) * 100,
          unit: '%',
          status: stockoutItems === 0 ? 'good' : 'warning',
          target: 95,
          trend: 'up'
        }
      ],
      alerts: [
        {
          id: '1',
          productId: 'DEF001',
          productName: 'Defibrillatore',
          type: 'stockout',
          severity: 'critical',
          message: 'Prodotto esaurito - Riordino urgente',
          isRead: false,
          createdAt: new Date(),
          actionRequired: true
        },
        {
          id: '2',
          productId: 'VEN001',
          productName: 'Ventilatore Polmonare',
          type: 'lowstock',
          severity: 'warning',
          message: 'Stock sotto il minimo - 4/2',
          isRead: false,
          createdAt: new Date(),
          actionRequired: true
        },
        {
          id: '3',
          productId: 'MON001',
          productName: 'Monitor Parametri Vitali',
          type: 'lowstock',
          severity: 'warning',
          message: 'Stock sotto il minimo - 6/2',
          isRead: false,
          createdAt: new Date(),
          actionRequired: true
        },
        {
          id: '4',
          productId: 'SER001',
          productName: 'Sollevatore Pazienti',
          type: 'lowstock',
          severity: 'info',
          message: 'Stock vicino al minimo - 8/3',
          isRead: true,
          createdAt: new Date(Date.now() - 86400000),
          actionRequired: false
        }
      ],
      recentMovements: [],
      topProducts: [],
      reorderPlans: []
    };
  }

  // Filter products by category
  filterByCategory(categoryId: string): Observable<Product[]> {
    if (!categoryId) {
      return this.products$;
    }
    
    return this.products$.pipe(
      map(products => products.filter(p => p.category.id === categoryId))
    );
  }

  // Get products needing reorder
  getProductsNeedingReorder(): Observable<Product[]> {
    return this.products$.pipe(
      map(products => products.filter(p => p.currentStock <= p.minStock))
    );
  }
}