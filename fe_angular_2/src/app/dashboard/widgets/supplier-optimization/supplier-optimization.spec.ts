import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SupplierOptimization } from './supplier-optimization';

describe('SupplierOptimization', () => {
  let component: SupplierOptimization;
  let fixture: ComponentFixture<SupplierOptimization>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SupplierOptimization]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SupplierOptimization);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
