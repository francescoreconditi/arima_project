import { TestBed } from '@angular/core/testing';

import { MorettiData } from './moretti-data';

describe('MorettiData', () => {
  let service: MorettiData;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(MorettiData);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
