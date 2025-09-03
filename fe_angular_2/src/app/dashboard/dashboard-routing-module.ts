import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardMainComponent } from './dashboard-main/dashboard-main';

const routes: Routes = [
  {
    path: '',
    component: DashboardMainComponent
  },
  {
    path: 'widget/:type/:id',
    component: DashboardMainComponent,
    data: { 
      title: 'Widget Detail' 
    }
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class DashboardRoutingModule { }
