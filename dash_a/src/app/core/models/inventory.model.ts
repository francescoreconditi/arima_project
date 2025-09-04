// ============================================
// MODELLI DATI INVENTARIO
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Definizione tipi per gestione inventario
// ============================================

export interface Product {
  id: string;
  code: string;
  name: string;
  description?: string;
  category: ProductCategory;
  unitPrice: number;
  currentStock: number;
  minStock: number;
  maxStock: number;
  leadTimeDays: number;
  supplier?: Supplier;
  metrics?: ProductMetrics;
  isActive: boolean;
  tags?: string[];
}

export interface ProductCategory {
  id: string;
  name: string;
  description?: string;
  parentId?: string;
}

export interface Supplier {
  id: string;
  name: string;
  code: string;
  contactInfo?: ContactInfo;
  leadTime: number;
  reliability: number; // 0-100
  pricingTiers?: PricingTier[];
  isPreferred: boolean;
}

export interface ContactInfo {
  email?: string;
  phone?: string;
  address?: string;
  contactPerson?: string;
}

export interface PricingTier {
  minQuantity: number;
  maxQuantity?: number;
  unitPrice: number;
  discount?: number;
}

export interface ProductMetrics {
  turnoverRate: number;
  stockoutRate: number;
  averageDemand: number;
  demandVariability: number;
  serviceLevel: number;
}

export interface InventoryOptimization {
  productId: string;
  recommendations: OptimizationRecommendation[];
  economicOrderQuantity: number;
  reorderPoint: number;
  safetyStock: number;
  totalCost: number;
  savings: number;
}

export interface OptimizationRecommendation {
  type: 'reorder' | 'excess' | 'obsolete' | 'seasonal' | 'promotion';
  priority: 'high' | 'medium' | 'low';
  action: string;
  impact: string;
  value?: number;
  deadline?: Date;
}

export interface ReorderPlan {
  id: string;
  productId: string;
  productName: string;
  supplierId: string;
  supplierName: string;
  quantity: number;
  unitPrice: number;
  totalCost: number;
  orderDate: Date;
  expectedDelivery: Date;
  status: 'pending' | 'ordered' | 'shipped' | 'delivered' | 'cancelled';
  priority: number;
}

export interface StockMovement {
  id: string;
  productId: string;
  type: 'in' | 'out' | 'adjustment';
  quantity: number;
  date: Date;
  reason?: string;
  referenceDoc?: string;
  performedBy?: string;
}

export interface InventoryAlert {
  id: string;
  type: 'stockout' | 'lowstock' | 'overstock' | 'expiry' | 'slowmoving';
  severity: 'critical' | 'warning' | 'info';
  productId: string;
  productName: string;
  message: string;
  value?: number;
  threshold?: number;
  createdAt: Date;
  isRead: boolean;
  actionRequired: boolean;
}

export interface InventoryDashboardData {
  summary: InventorySummary;
  alerts: InventoryAlert[];
  topProducts: Product[];
  recentMovements: StockMovement[];
  reorderPlans: ReorderPlan[];
  kpis: InventoryKPI[];
}

export interface InventorySummary {
  totalProducts: number;
  totalValue: number;
  lowStockItems: number;
  overstockItems: number;
  pendingOrders: number;
  turnoverRate: number;
  stockAccuracy: number;
}

export interface InventoryKPI {
  name: string;
  value: number;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  changePercent?: number;
  target?: number;
  status?: 'good' | 'warning' | 'critical';
}

export interface ABCClassification {
  productId: string;
  class: 'A' | 'B' | 'C';
  value: number;
  volume: number;
  cumulativePercent: number;
}

export interface XYZAnalysis {
  productId: string;
  class: 'X' | 'Y' | 'Z';
  variability: number;
  predictability: 'high' | 'medium' | 'low';
}