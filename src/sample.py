import pandas as pd

df = pd.read_csv('../../lstm_data/training.csv', encoding = 'ISO-8859-1')

train_sampled_df = df.sample(n=500000)
test_sampled_df = df.sample(n=3000)

train_sampled_df.to_csv('../data/sampleTrain.csv')
test_sampled_df.to_csv('../data/sampleTest.csv')




