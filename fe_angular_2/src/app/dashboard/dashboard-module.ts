import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { GridsterModule } from 'angular-gridster2';

import { DashboardRoutingModule } from './dashboard-routing-module';
import { DashboardMainComponent } from './dashboard-main/dashboard-main';

// Widget components
import { KpiCardsComponent } from './widgets/kpi-cards/kpi-cards';
import { ForecastChartComponent } from './widgets/forecast-chart/forecast-chart';
import { InventoryTableComponent } from './widgets/inventory-table/inventory-table';
import { SupplierOptimizationComponent } from './widgets/supplier-optimization/supplier-optimization';
import { AlertsPanelComponent } from './widgets/alerts-panel/alerts-panel';
import { WhatIfAnalysisComponent } from './widgets/what-if-analysis/what-if-analysis';

@NgModule({
  declarations: [
    // Non servono dichiarazioni per componenti standalone
  ],
  imports: [
    CommonModule,
    FormsModule,
    GridsterModule,
    DashboardRoutingModule,
    // Importa i componenti standalone
    DashboardMainComponent,
    KpiCardsComponent,
    ForecastChartComponent,
    InventoryTableComponent,
    SupplierOptimizationComponent,
    AlertsPanelComponent,
    WhatIfAnalysisComponent
  ]
})
export class DashboardModule { }
