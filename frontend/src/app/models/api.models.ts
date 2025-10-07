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
  p: number;  // Non-seasonal AR order (required by backend)
  d: number;  // Non-seasonal differencing (required by backend)
  q: number;  // Non-seasonal MA order (required by backend)
  P: number;  // Seasonal AR order
  D: number;  // Seasonal differencing
  Q: number;  // Seasonal MA order
  s: number;  // Seasonal period
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
  return_intervals?: boolean;
}

export interface AutoSelectionRequest {
  data: TimeSeriesData;
  model_type: string;  // Required: 'arima', 'sarima', 'sarimax'
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

export interface ModelListResponse {
  models: ModelInfo[];
  total: number;
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
  all_results: Array<{
    order: number[];
    seasonal_order?: number[];
    aic?: number;
    bic?: number;
  }>;
  models_tested: number;
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
