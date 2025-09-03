/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Modelli TypeScript per dashboard Moretti
 * ============================================
 */

export interface Product {
  codice: string;
  nome: string;
  categoria: string;
  fornitore: string;
  prezzo_unitario: number;
  lead_time: number;
  domanda_media_giornaliera: number;
  giacenza_attuale: number;
  punto_riordino: number;
  stock_sicurezza: number;
}

export interface SalesData {
  data: Date;
  quantita: number;
  prodotto_codice: string;
}

export interface ForecastData {
  date: string;
  forecast: number;
  lower_ci?: number;
  upper_ci?: number;
}

export interface MetricData {
  title: string;
  value: string | number;
  delta?: number;
  format?: 'currency' | 'percentage' | 'number';
}

export interface AlertData {
  level: 'critica' | 'alta' | 'media';
  message: string;
  prodotto_codice?: string;
}

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string;
    borderWidth?: number;
    fill?: boolean;
  }[];
}