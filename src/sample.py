import pandas as pd

df = pd.read_csv('../../lstm_data/training.csv', encoding = 'ISO-8859-1')

sampled_df = df.sample(n=100000)


sampled_df.to_csv('../../lstm_data/sampleTrain.csv')



