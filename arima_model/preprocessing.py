import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    return df

def make_stationary(series):
    return series.diff().dropna()