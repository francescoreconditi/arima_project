import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { GridsterModule, GridsterConfig, GridsterItem } from 'angular-gridster2';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

// Import widget components
import { KpiCardsComponent } from '../widgets/kpi-cards/kpi-cards';
import { ForecastChartComponent } from '../widgets/forecast-chart/forecast-chart';
import { InventoryTableComponent } from '../widgets/inventory-table/inventory-table';
import { SupplierOptimizationComponent } from '../widgets/supplier-optimization/supplier-optimization';
import { AlertsPanelComponent } from '../widgets/alerts-panel/alerts-panel';
import { WhatIfAnalysisComponent } from '../widgets/what-if-analysis/what-if-analysis';

// Import services
import { MorettiDataService } from '../../core/services/moretti-data';

@Component({
  selector: 'app-dashboard-main',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    GridsterModule,
    KpiCardsComponent,
    ForecastChartComponent,
    InventoryTableComponent,
    SupplierOptimizationComponent,
    AlertsPanelComponent,
    WhatIfAnalysisComponent
  ],
  templateUrl: './dashboard-main.html',
  styleUrl: './dashboard-main.scss'
})
export class DashboardMainComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();
  
  options: GridsterConfig = {};
  dashboard: Array<GridsterItem & { component: string }> = [];

  selectedProduct = '';
  selectedCategory = '';
  now = new Date();
  
  constructor(public dataService: MorettiDataService) {}

  ngOnInit(): void {
    this.setupGridster();
    this.initializeDashboard();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private setupGridster(): void {
    this.options = {
      gridType: 'fit',
      compactType: 'none',
      margin: 16,
      outerMargin: true,
      outerMarginTop: null,
      outerMarginRight: null,
      outerMarginBottom: null,
      outerMarginLeft: null,
      useTransformPositioning: true,
      mobileBreakpoint: 640,
      minCols: 12,
      maxCols: 12,
      minRows: 6,
      maxRows: 100,
      maxItemCols: 100,
      minItemCols: 1,
      maxItemRows: 100,
      minItemRows: 1,
      maxItemArea: 2500,
      minItemArea: 1,
      defaultItemCols: 3,
      defaultItemRows: 2,
      fixedColWidth: 105,
      fixedRowHeight: 105,
      keepFixedHeightInMobile: false,
      keepFixedWidthInMobile: false,
      scrollSensitivity: 10,
      scrollSpeed: 20,
      enableEmptyCellClick: false,
      enableEmptyCellContextMenu: false,
      enableEmptyCellDrop: false,
      enableEmptyCellDrag: false,
      enableOccupiedCellDrop: false,
      emptyCellDragMaxCols: 50,
      emptyCellDragMaxRows: 50,
      ignoreMarginInRow: false,
      draggable: {
        enabled: true,
      },
      resizable: {
        enabled: true,
      },
      swap: false,
      pushItems: true,
      disablePushOnDrag: false,
      disablePushOnResize: false,
      pushResizeItems: false,
      displayGrid: 'onDrag&Resize',
      disableWindowResize: false,
      disableWarnings: false,
      scrollToNewItems: false
    };
  }

  private initializeDashboard(): void {
    this.dashboard = [
      {
        cols: 6,
        rows: 2,
        y: 0,
        x: 0,
        component: 'kpi-cards'
      },
      {
        cols: 6,
        rows: 2,
        y: 0,
        x: 6,
        component: 'alerts-panel'
      },
      {
        cols: 8,
        rows: 4,
        y: 2,
        x: 0,
        component: 'forecast-chart'
      },
      {
        cols: 4,
        rows: 4,
        y: 2,
        x: 8,
        component: 'supplier-optimization'
      },
      {
        cols: 12,
        rows: 3,
        y: 6,
        x: 0,
        component: 'inventory-table'
      },
      {
        cols: 12,
        rows: 3,
        y: 9,
        x: 0,
        component: 'what-if-analysis'
      }
    ];
  }

  onProductSelect(productCode: string): void {
    this.selectedProduct = productCode;
  }

  onCategorySelect(category: string): void {
    this.selectedCategory = category;
    this.selectedProduct = ''; // Reset product when category changes
  }

  changedOptions(): void {
    if (this.options.api && this.options.api.optionsChanged) {
      this.options.api.optionsChanged();
    }
  }

  removeItem($event: MouseEvent | TouchEvent, item: any): void {
    $event.preventDefault();
    $event.stopPropagation();
    this.dashboard.splice(this.dashboard.indexOf(item), 1);
  }

  addItem(): void {
    this.dashboard.push({
      x: 0,
      y: 0,
      cols: 4,
      rows: 2,
      component: 'kpi-cards'
    });
  }

  trackByFn(index: number, item: any): any {
    return item.component + index;
  }
}
