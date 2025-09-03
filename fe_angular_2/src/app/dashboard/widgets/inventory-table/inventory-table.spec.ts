import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InventoryTable } from './inventory-table';

describe('InventoryTable', () => {
  let component: InventoryTable;
  let fixture: ComponentFixture<InventoryTable>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InventoryTable]
    })
    .compileComponents();

    fixture = TestBed.createComponent(InventoryTable);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
