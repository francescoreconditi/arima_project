import { TestBed } from '@angular/core/testing';

import { Forecasting } from './forecasting';

describe('Forecasting', () => {
  let service: Forecasting;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Forecasting);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
