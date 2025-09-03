/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Modelli per dashboard drag & drop
 * ============================================
 */

import { GridsterItem } from 'angular-gridster2';

export interface DashboardWidget extends GridsterItem {
  id: string;
  type: 'metric' | 'chart' | 'alert' | 'product-info';
  title: string;
  data?: any;
  config?: WidgetConfig;
}

export interface WidgetConfig {
  showTitle?: boolean;
  backgroundColor?: string;
  borderColor?: string;
  textColor?: string;
  chartType?: string;
  refreshInterval?: number;
}

export interface DashboardLayout {
  widgets: DashboardWidget[];
  lastModified: Date;
  userId?: string;
  name: string;
}

export interface MetricWidgetData {
  title: string;
  value: string | number;
  delta?: number;
  format?: 'currency' | 'percentage' | 'number';
  icon?: string;
}

export interface ChartWidgetData {
  title: string;
  chartData: any;
  chartOptions?: any;
  dataSource: 'historical' | 'forecast' | 'combined';
}

export interface AlertWidgetData {
  alerts: Array<{
    level: 'critica' | 'alta' | 'media';
    message: string;
    timestamp?: Date;
    productCode?: string;
  }>;
  maxItems?: number;
  showTimestamp?: boolean;
}