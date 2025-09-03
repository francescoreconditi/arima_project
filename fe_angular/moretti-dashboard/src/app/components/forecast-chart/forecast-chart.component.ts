/*
 * ============================================
 * FILE DI TEST/DEBUG - NON PER PRODUZIONE
 * Creato da: Claude Code
 * Data: 2025-09-03
 * Scopo: Componente grafico forecasting per dashboard
 * ============================================
 */

import { Component, Input, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, ChartType, registerables } from 'chart.js';
import { ForecastData, SalesData } from '../../models/product.model';

Chart.register(...registerables);

@Component({
  selector: 'app-forecast-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container">
      <canvas #chartCanvas width="400" height="200"></canvas>
    </div>
  `,
  styles: [`
    .chart-container {
      position: relative;
      height: 400px;
      width: 100%;
      padding: 20px;
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    canvas {
      width: 100% !important;
      height: 100% !important;
    }
  `]
})
export class ForecastChartComponent implements OnInit, OnDestroy {
  @Input() forecastData: ForecastData[] = [];
  @Input() historicalData: SalesData[] = [];
  @Input() title: string = 'Previsioni Domanda';
  
  @ViewChild('chartCanvas', { static: true }) chartCanvas!: ElementRef<HTMLCanvasElement>;
  
  private chart?: Chart;

  ngOnInit() {
    this.createChart();
  }

  ngOnDestroy() {
    if (this.chart) {
      this.chart.destroy();
    }
  }

  private createChart() {
    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    // Prepara dati storici
    const historicalLabels = this.historicalData.slice(-30).map(d => 
      d.data.toLocaleDateString('it-IT', { month: 'short', day: 'numeric' })
    );
    const historicalValues = this.historicalData.slice(-30).map(d => d.quantita);

    // Prepara dati previsionali
    const forecastLabels = this.forecastData.map(d => {
      const date = new Date(d.date);
      return date.toLocaleDateString('it-IT', { month: 'short', day: 'numeric' });
    });
    const forecastValues = this.forecastData.map(d => d.forecast);
    const lowerCI = this.forecastData.map(d => d.lower_ci || 0);
    const upperCI = this.forecastData.map(d => d.upper_ci || 0);

    // Combina le etichette
    const allLabels = [...historicalLabels, ...forecastLabels];
    
    // Crea dataset con separazione tra storico e previsioni
    const historicalDataset = [...historicalValues, ...new Array(forecastValues.length).fill(null)];
    const forecastDataset = [...new Array(historicalValues.length).fill(null), ...forecastValues];
    const lowerCIDataset = [...new Array(historicalValues.length).fill(null), ...lowerCI];
    const upperCIDataset = [...new Array(historicalValues.length).fill(null), ...upperCI];

    const config: ChartConfiguration = {
      type: 'line' as ChartType,
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Vendite Storiche',
            data: historicalDataset,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.1
          },
          {
            label: 'Previsioni',
            data: forecastDataset,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1
          },
          {
            label: 'IC Superiore',
            data: upperCIDataset,
            borderColor: 'rgba(255, 99, 132, 0.3)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 1,
            fill: '+1',
            pointRadius: 0
          },
          {
            label: 'IC Inferiore',
            data: lowerCIDataset,
            borderColor: 'rgba(255, 99, 132, 0.3)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 1,
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: this.title,
            font: {
              size: 16
            }
          },
          legend: {
            display: true,
            position: 'top',
            labels: {
              filter: (legendItem) => {
                return !legendItem.text?.includes('IC'); // Nasconde le linee degli intervalli di confidenza dalla legenda
              }
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Data'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Quantità'
            },
            beginAtZero: true
          }
        },
        elements: {
          point: {
            radius: 4,
            hoverRadius: 8
          }
        }
      }
    };

    this.chart = new Chart(ctx, config);
  }

  updateChart(newForecastData: ForecastData[], newHistoricalData: SalesData[]) {
    this.forecastData = newForecastData;
    this.historicalData = newHistoricalData;
    
    if (this.chart) {
      this.chart.destroy();
    }
    
    this.createChart();
  }
}