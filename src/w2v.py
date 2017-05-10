from gensim.models import Word2Vec
import pandas as pd
import re
import numpy as np


#        Used for transforming our test and training dataset into a word2Vec vocabulary              #
#																								     #
#  Words are preprocessed if they are usernames, urls, or contain elongated concescutive characters  #
#																									 #

class loadData(object):

	def __init__(self, path):
		self.df = pd.read_csv(path, encoding = 'ISO-8859-1')

	# Memory friendly iterator. Process one tweet in memory then forget it.
	def __iter__(self):
		# process the tweet
		for index, row in self.df.iterrows():
			tweet = row[5].split(" ")
			# Process data.
			processed = []
			for word in tweet:
				if word.startswith('@'):
					word = 'USERNAME'
				elif word.startswith('http:'):
					word = 'URL'
				elif re.search(r'(.)\1\1', word):
					word = self.__reduce_word(word)
				processed.append(word)
			yield processed

	def __reduce_word(self, word):
		
		w_processed = ''
		ll = ''
		lcount = 1
		for letter in word:
			if ll is letter:
				lcount += 1
			else:
				lcount = 1
			
			if lcount < 3:
				w_processed = w_processed + letter

			ll = letter
		return w_processed




data = loadData('../data/concat.csv')
# ## Create Model ##
model = Word2Vec(data, min_count = 5, size = 50, workers = 4)
weights = model.wv.syn0
np.save(open('../vocab_weights', 'wb'), weights)

print('\n Most Similar pos/neg for "weed", "rip", "stick"\n')
print(model.most_similar(positive=['weed', 'rip', 'stick'], negative=['weed', 'rip', 'stick']))
print('\n\n Sample raw feature vector for "plant"\n')
print(model['plant'])

model.save('../vocab')
print('\n ---- model saved ----\n')
