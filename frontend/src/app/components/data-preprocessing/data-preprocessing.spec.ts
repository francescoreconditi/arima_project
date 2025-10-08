import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataPreprocessing } from './data-preprocessing';

describe('DataPreprocessing', () => {
  let component: DataPreprocessing;
  let fixture: ComponentFixture<DataPreprocessing>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DataPreprocessing]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DataPreprocessing);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
