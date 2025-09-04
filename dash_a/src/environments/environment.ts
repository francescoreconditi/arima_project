// ============================================
// CONFIGURAZIONE ENVIRONMENT
// Creato da: Claude Code
// Data: 2025-09-04
// Scopo: Configurazione ambiente sviluppo
// ============================================

export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000/api',
  wsUrl: 'ws://localhost:8000/ws',
  appVersion: '1.0.0',
  defaultLanguage: 'it',
  supportedLanguages: ['it', 'en', 'es', 'fr', 'zh'],
  refreshInterval: 60000, // 1 minuto
  cacheTimeout: 300000, // 5 minuti
  features: {
    darkMode: true,
    multiLanguage: true,
    realTimeUpdates: true,
    exportData: true,
    advancedAnalytics: true
  }
};