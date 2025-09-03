import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ForecastChart } from './forecast-chart';

describe('ForecastChart', () => {
  let component: ForecastChart;
  let fixture: ComponentFixture<ForecastChart>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ForecastChart]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ForecastChart);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
