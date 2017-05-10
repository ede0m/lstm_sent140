import pandas as pd

df = pd.read_csv('../../lstm_data/training_mod.csv', encoding = 'ISO-8859-1')

train_sampled_df = df.sample(n=400000)
test_sampled_df = df.sample(n=2000)

train_sampled_df.to_csv('../../lstm_data/sampleTrain.csv')
test_sampled_df.to_csv('../../lstm_data/sampleTest.csv')




