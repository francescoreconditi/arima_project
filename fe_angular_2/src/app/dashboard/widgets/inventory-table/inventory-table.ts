import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-inventory-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './inventory-table.html',
  styleUrl: './inventory-table.scss'
})
export class InventoryTableComponent {
  @Input() selectedProduct = '';
  @Input() selectedCategory = '';

  inventoryData = [
    { codice: 'CRZ001', nome: 'Carrozzina Standard', scorta: 15, stato: 'attenzione' },
    { codice: 'MAT001', nome: 'Materasso Antidecubito', scorta: 5, stato: 'critico' },
    { codice: 'ELT001', nome: 'Saturimetro Digitale', scorta: 45, stato: 'ottimale' }
  ];

  getStatusColor(stato: string): string {
    switch (stato) {
      case 'ottimale': return 'linear-gradient(135deg, #4caf50, #388e3c)';
      case 'attenzione': return 'linear-gradient(135deg, #ff9800, #f57c00)';
      case 'critico': return 'linear-gradient(135deg, #f44336, #d32f2f)';
      default: return 'linear-gradient(135deg, #9e9e9e, #757575)';
    }
  }

  getStatusTextColor(stato: string): string {
    return 'white';
  }
}