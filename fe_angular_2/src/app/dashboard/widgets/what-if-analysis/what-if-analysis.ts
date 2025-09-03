import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-what-if-analysis',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './what-if-analysis.html',
  styleUrl: './what-if-analysis.scss'
})
export class WhatIfAnalysisComponent {
  @Input() selectedProduct = '';
  @Input() selectedCategory = '';

  scenarios = [
    { nome: 'Scenario Base', previsione: 100, impatto: 'neutro' },
    { nome: 'Aumento Domanda +20%', previsione: 120, impatto: 'positivo' },
    { nome: 'Ritardi Fornitori', previsione: 85, impatto: 'negativo' }
  ];

  getImpactColor(impatto: string): string {
    switch (impatto) {
      case 'positivo': return '#4caf50';
      case 'negativo': return '#f44336';
      case 'neutro': return '#666';
      default: return '#666';
    }
  }
}
