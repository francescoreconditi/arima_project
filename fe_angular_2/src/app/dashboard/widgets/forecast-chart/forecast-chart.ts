import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-forecast-chart',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './forecast-chart.html',
  styleUrl: './forecast-chart.scss'
})
export class ForecastChartComponent {
  @Input() selectedProduct = '';
  @Input() selectedCategory = '';

  chartData = {
    labels: ['Gen', 'Feb', 'Mar', 'Apr', 'Mag'],
    values: [25, 30, 28, 35, 32]
  };
}
