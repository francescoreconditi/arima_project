import { Routes } from '@angular/router';
import { ModelTraining } from './components/model-training/model-training';
import { ModelForecast } from './components/model-forecast/model-forecast';
import { ModelList } from './components/model-list/model-list';
import { AutoTraining } from './components/auto-training/auto-training';

export const routes: Routes = [
  { path: '', redirectTo: '/training', pathMatch: 'full' },
  { path: 'training', component: ModelTraining },
  { path: 'auto-training', component: AutoTraining },
  { path: 'forecast', component: ModelForecast },
  { path: 'models', component: ModelList }
];
