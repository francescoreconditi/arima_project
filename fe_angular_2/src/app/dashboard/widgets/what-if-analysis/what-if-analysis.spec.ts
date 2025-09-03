import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WhatIfAnalysis } from './what-if-analysis';

describe('WhatIfAnalysis', () => {
  let component: WhatIfAnalysis;
  let fixture: ComponentFixture<WhatIfAnalysis>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WhatIfAnalysis]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WhatIfAnalysis);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
