/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Componente alert per dashboard
 * ============================================
 */

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AlertData } from '../../models/product.model';

@Component({
  selector: 'app-alert-box',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="alert-box" [ngClass]="getAlertClass()">
      <div class="alert-content">
        <span class="alert-icon">{{ getAlertIcon() }}</span>
        <span class="alert-message">{{ alert.message }}</span>
      </div>
    </div>
  `,
  styles: [`
    .alert-box {
      padding: 15px;
      border-radius: 8px;
      margin: 10px 0;
      font-weight: 500;
      background-color: #ffffff;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-content {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    .alert-icon {
      font-size: 20px;
      font-weight: bold;
    }
    
    .alert-message {
      flex: 1;
      color: #000000;
    }
    
    .alert-critica {
      border: 3px solid #dc3545;
      box-shadow: 0 2px 6px rgba(220, 53, 69, 0.3);
    }
    
    .alert-critica .alert-icon {
      color: #dc3545;
    }
    
    .alert-alta {
      border: 3px solid #fd7e14;
      box-shadow: 0 2px 6px rgba(253, 126, 20, 0.3);
    }
    
    .alert-alta .alert-icon {
      color: #fd7e14;
    }
    
    .alert-media {
      border: 3px solid #ffc107;
      box-shadow: 0 2px 6px rgba(255, 193, 7, 0.3);
    }
    
    .alert-media .alert-icon {
      color: #ffc107;
    }
  `]
})
export class AlertBoxComponent {
  @Input() alert!: AlertData;

  getAlertClass(): string {
    return `alert-${this.alert.level}`;
  }

  getAlertIcon(): string {
    switch (this.alert.level) {
      case 'critica':
        return '🔴';
      case 'alta':
        return '🟠';
      case 'media':
        return '🟡';
      default:
        return '🔵';
    }
  }
}