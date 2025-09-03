import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-supplier-optimization',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './supplier-optimization.html',
  styleUrl: './supplier-optimization.scss'
})
export class SupplierOptimizationComponent {
  @Input() selectedProduct = '';
  @Input() selectedCategory = '';

  supplierData = [
    { nome: 'MedSupply Italia', affidabilita: 95, prezzo: 450 },
    { nome: 'Healthcare Solutions', affidabilita: 88, prezzo: 465 },
    { nome: 'Mobility Plus', affidabilita: 92, prezzo: 455 }
  ];
}
