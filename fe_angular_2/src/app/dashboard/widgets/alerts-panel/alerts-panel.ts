import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-alerts-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './alerts-panel.html',
  styleUrl: './alerts-panel.scss'
})
export class AlertsPanelComponent {
  @Input() selectedProduct = '';
  @Input() selectedCategory = '';

  alerts = [
    { tipo: 'critico', messaggio: 'Materasso Antidecubito: scorta critica (5 unità)' },
    { tipo: 'attenzione', messaggio: 'Carrozzina Standard: sotto punto riordino' },
    { tipo: 'info', messaggio: 'Previsione domanda aumentata per Marzo' }
  ];

  getAlertBackground(tipo: string): string {
    switch (tipo) {
      case 'critico': return '#ffebee';
      case 'attenzione': return '#fff3e0';
      case 'info': return '#e3f2fd';
      default: return '#f5f5f5';
    }
  }

  getAlertColor(tipo: string): string {
    switch (tipo) {
      case 'critico': return '#f44336';
      case 'attenzione': return '#ff9800';
      case 'info': return '#2196f3';
      default: return '#666';
    }
  }

  getAlertTextColor(tipo: string): string {
    switch (tipo) {
      case 'critico': return '#c62828';
      case 'attenzione': return '#e65100';
      case 'info': return '#1976d2';
      default: return '#333';
    }
  }
}
