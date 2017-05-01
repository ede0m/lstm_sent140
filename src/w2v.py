# Used for transforming our test and training dataset into a word2Vec vocabulary 

import gensim
import pandas as pd


#tweets = []


class loadData(object):

	def __init__(self, path):
		self.df = pd.read_csv(path, encoding = 'ISO-8859-1')

	# Memory friendly iterator. Process one tweet in memory then forget it.
	def __iter__(self):
		for index, row in self.df.iterrows():
			tweet = row[5].split(" ")
			#print(tweet)
			yield tweet


data = loadData('../data/concat.csv')
## Create Model ##
model = gensim.models.Word2Vec(data, min_count = 5, size = 50, workers = 4)

print('\nMost Similar pos/neg for "weed", "rip", "stick"\n')
print(model.most_similar(positive=['weed', 'rip', 'stick'], negative=['weed', 'rip', 'stick']))
print('\nSample raw feature vector for "plant"\n')
print(model['plant'])

model.save('../vocab')
print('\nmodel saved')
