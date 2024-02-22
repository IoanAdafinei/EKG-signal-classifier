import pandas as pd

df = pd.read_csv('merged.csv', header=None)
ds = df.sample(frac = 1)
ds.to_csv('new_file.csv')