from statsmodels.tsa.arima.model import ARIMA

def train_arima(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=10):
    forecast = model_fit.forecast(steps=steps)
    return forecast