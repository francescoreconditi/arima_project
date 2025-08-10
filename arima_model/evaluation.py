import matplotlib.pyplot as plt

def plot_forecast(series, forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Historical")
    plt.plot(forecast.index, forecast, label="Forecast", color="red")
    plt.legend()
    plt.show()