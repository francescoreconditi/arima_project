import matplotlib.pyplot as plt

def plot_forecast(series, forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Storico")
    plt.plot(forecast.index, forecast, label="Previsione", color="red")
    plt.legend()
    plt.show()