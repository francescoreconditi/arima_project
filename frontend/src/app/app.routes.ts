import { Routes } from '@angular/router';
import { ModelTraining } from './components/model-training/model-training';
import { ModelForecast } from './components/model-forecast/model-forecast';
import { ModelList } from './components/model-list/model-list';
import { AutoTraining } from './components/auto-training/auto-training';
import { DataUpload } from './components/data-upload/data-upload';
import { DataPreprocessing } from './components/data-preprocessing/data-preprocessing';

export const routes: Routes = [
  { path: '', redirectTo: '/training', pathMatch: 'full' },
  { path: 'training', component: ModelTraining },
  { path: 'auto-training', component: AutoTraining },
  { path: 'forecast', component: ModelForecast },
  { path: 'models', component: ModelList },
  { path: 'data-upload', component: DataUpload },
  { path: 'data-preprocessing', component: DataPreprocessing }
];
