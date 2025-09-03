// Product Model - Basato su dashboard Moretti

export interface Product {
  codice: string;
  nome: string;
  categoria: string;
  prezzo_unitario: number;
  lead_time_giorni: number;
  costo_stockout_giornaliero: number;
  costo_mantenimento_scorta: number;
  scorta_minima: number;
  scorta_massima: number;
  quantita_riordino_standard: number;
  fornitore_principale: string;
  fornitori_alternativi: string[];
}

export interface ProductInventory {
  codice: string;
  nome: string;
  scorta_attuale: number;
  scorta_minima: number;
  scorta_massima: number;
  punto_riordino: number;
  quantita_riordino: number;
  domanda_media_giornaliera: number;
  deviazione_standard: number;
  lead_time_giorni: number;
  livello_servizio_target: number;
  giorni_copertura: number;
  stato: 'critico' | 'attenzione' | 'ottimale' | 'eccesso';
}

export interface ForecastResult {
  prodotto: string;
  data: Date;
  previsione: number;
  limite_inferiore?: number;
  limite_superiore?: number;
  intervallo_confidenza?: number;
}

export interface KPIMetric {
  nome: string;
  valore: number | string;
  variazione?: number;
  trend?: 'up' | 'down' | 'stable';
  unita?: string;
  descrizione?: string;
}

export interface SupplierInfo {
  codice: string;
  nome: string;
  affidabilita: number;
  tempo_consegna_medio: number;
  prezzo_base: number;
  sconto_quantita?: {
    quantita_minima: number;
    sconto_percentuale: number;
  }[];
  valutazione: number;
  prodotti_forniti: string[];
}

export interface OrderRecommendation {
  prodotto_codice: string;
  prodotto_nome: string;
  fornitore_consigliato: string;
  quantita_consigliata: number;
  costo_totale: number;
  data_ordine_suggerita: Date;
  data_arrivo_prevista: Date;
  urgenza: 'alta' | 'media' | 'bassa';
  note?: string;
}