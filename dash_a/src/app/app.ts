// ============================================
// COMPONENTE ROOT
// Creato da: Claude Code  
// Data: 2025-09-04
// Scopo: Componente principale applicazione
// ============================================

import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('ARIMA Dashboard');
}
