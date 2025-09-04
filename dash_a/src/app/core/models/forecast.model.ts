// ============================================
// MODELLI DATI FORECAST
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Definizione tipi per forecasting
// ============================================

export interface TimeSeriesData {
  date: Date | string;
  value: number;
  label?: string;
}

export interface ForecastRequest {
  productId?: string;
  productCode?: string;
  startDate?: string;
  endDate?: string;
  steps: number;
  modelType: 'arima' | 'sarima' | 'var';
  parameters?: ArimaParameters | SarimaParameters;
  includeConfidenceInterval?: boolean;
}

export interface ArimaParameters {
  order: [number, number, number]; // [p, d, q]
  trend?: 'n' | 'c' | 't' | 'ct';
}

export interface SarimaParameters extends ArimaParameters {
  seasonalOrder: [number, number, number, number]; // [P, D, Q, s]
}

export interface ForecastResult {
  productId: string;
  productName: string;
  predictions: TimeSeriesPoint[];
  confidenceInterval?: {
    lower: TimeSeriesPoint[];
    upper: TimeSeriesPoint[];
  };
  metrics?: ForecastMetrics;
  modelInfo: ModelInfo;
  timestamp: Date;
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
}

export interface ForecastMetrics {
  mape: number;
  rmse: number;
  mae: number;
  r2?: number;
  accuracy?: number;
}

export interface ModelInfo {
  type: string;
  parameters: any;
  trainingPeriod: {
    start: string;
    end: string;
  };
}

export interface HistoricalData {
  productId: string;
  productName: string;
  data: TimeSeriesPoint[];
  statistics?: DataStatistics;
}

export interface DataStatistics {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  seasonality?: boolean;
  seasonalPeriod?: number;
}

export interface ForecastComparison {
  productId: string;
  models: {
    [modelName: string]: ForecastResult;
  };
  bestModel: string;
  comparisonMetrics: {
    [modelName: string]: ForecastMetrics;
  };
}