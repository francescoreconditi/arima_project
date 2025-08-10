from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd
from arima_model.evaluation import plot_forecast

model = ARIMAResults.load("model_arima.pkl")

forecast_steps = 10
forecast = model.forecast(steps=forecast_steps)

last_index = model.data.row_labels[-1]
forecast_index = pd.date_range(start=last_index, periods=forecast_steps+1, freq='M')[1:]

forecast = pd.Series(forecast, index=forecast_index)

plot_forecast(model.data.endog, forecast)