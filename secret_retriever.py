import pandas as pd

secrets_df = pd.read_csv('secrets.csv')

def retrieve_secret(key: str):
    return secrets_df.loc[secrets_df['key'] == key, 'value'].values[0]