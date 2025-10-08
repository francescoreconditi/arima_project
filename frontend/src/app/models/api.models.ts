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
  descrizione?: string;
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

// ===== DATA MANAGEMENT MODELS =====

export interface DataUploadConfig {
  dataset_name: string;
  date_column?: string;
  value_columns: string[];
  separator?: string;
  date_format?: string;
  encoding?: string;
  skip_rows?: number;
  validate_data?: boolean;
}

export interface DatasetMetadata {
  dataset_id: string;
  name: string;
  rows: number;
  columns: number;
  size_bytes: number;
  upload_timestamp: string;
  file_format: string;
  column_info: {
    date_column?: string;
    value_columns: string[];
    inferred_types: { [key: string]: string };
    all_columns?: string[];
  };
  data_quality_score: number;
  missing_values_ratio: number;
}

export interface DataJobResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  dataset_id?: string;
  results?: any;
  error_message?: string;
}

export interface PreprocessingStep {
  type: 'handle_missing' | 'remove_outliers' | 'make_stationary' | 'normalize';
  method?: string;
  threshold?: number;
  feature_range?: [number, number];
}

export interface PreprocessingRequest {
  dataset_id: string;
  preprocessing_steps: PreprocessingStep[];
  output_dataset_name: string;
  preserve_original?: boolean;
}
