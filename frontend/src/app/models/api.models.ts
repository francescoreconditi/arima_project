// ============================================
// FILE DI PRODUZIONE
// Creato da: Claude Code
// Data: 2025-10-06
// Scopo: Modelli TypeScript per API ARIMA Forecaster
// ============================================

/**
 * Modelli TypeScript per interfacciare l'API ARIMA Forecaster
 */

// ===== REQUEST MODELS =====

export interface TimeSeriesData {
  timestamps: string[];
  values: number[];
}

export interface ModelOrder {
  p: number;
  d: number;
  q: number;
}

export interface SeasonalOrder {
  P: number;
  D: number;
  Q: number;
  s: number;
}

export interface ModelTrainingRequest {
  model_type: 'arima' | 'sarima' | 'sarimax';
  data: TimeSeriesData;
  order: ModelOrder;
  seasonal_order?: SeasonalOrder;
}

export interface ForecastRequest {
  steps: number;
  confidence_level?: number;
  return_confidence_intervals?: boolean;
}

export interface AutoSelectionRequest {
  data: TimeSeriesData;
  max_p?: number;
  max_d?: number;
  max_q?: number;
  seasonal?: boolean;
  seasonal_period?: number;
  criterion?: 'aic' | 'bic';
}

// ===== RESPONSE MODELS =====

export interface ModelInfo {
  model_id: string;
  model_type: string;
  status: 'training' | 'completed' | 'failed';
  created_at: string;
  training_observations: number;
  parameters: Record<string, any>;
  metrics: Record<string, number>;
}

export interface ConfidenceIntervals {
  lower: number[];
  upper: number[];
}

export interface ForecastResponse {
  forecast: number[];
  timestamps: string[];
  confidence_intervals?: ConfidenceIntervals;
  model_id: string;
  forecast_steps: number;
}

export interface AutoSelectionResult {
  best_model: {
    order: number[];
    seasonal_order?: number[];
    aic?: number;
    bic?: number;
  };
  all_models: Array<{
    order: number[];
    seasonal_order?: number[];
    aic?: number;
    bic?: number;
  }>;
  search_time_seconds: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

// ===== CHART DATA MODELS =====

export interface ChartDataPoint {
  x: string | number;
  y: number;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
}

export interface ForecastChartData {
  actual?: ChartSeries;
  forecast: ChartSeries;
  lower_bound?: ChartSeries;
  upper_bound?: ChartSeries;
}
