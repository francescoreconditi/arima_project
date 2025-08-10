from arima_model.preprocessing import load_and_clean_data, make_stationary
from arima_model.model import train_arima

df = load_and_clean_data("data/processed/your_data.csv")
series = make_stationary(df['sales'])

model = train_arima(series, order=(1,1,1))
model.save("model_arima.pkl")