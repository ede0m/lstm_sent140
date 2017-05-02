import dataset as ds
import gensim 
from gensim.utils import simple_preprocess
import numpy as np
import math
import pandas as pd
# import pydot
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import plot_model
from keras.preprocessing import sequence


# Load W2V Model #
w2v = gensim.models.Word2Vec.load('../vocab')
numWords = len(w2v.wv.vocab)
dimension = w2v.vector_size

# Load in text data and transform to vectors #
df = pd.read_csv('../../lstm_data/sampleTrain.csv', encoding = 'ISO-8859-1')

tweet_vecs = []
outputs = []
lengths = []

for index,row in df.iterrows():
	#tweets.append(row[5])
	tokens = row[5].split(" ")
	lengths.append(len(tokens))
	vec = []
	for word in tokens:
		try:
			vec.extend(w2v[word])						
		# zero padding
		except TypeError:
			vec.extend(np.zeros(dimension))
		# word not it dict
		except KeyError:
			vec.extend(np.zeros(dimension))
	
	outputs.append(row[0])
	tweet_vecs.append(vec)

mean = sum(lengths)/len(lengths)
std = np.std(lengths) 
time_sequence = mean + (2*std)

# print(tweet_vecs)


# train and test data format #
# X_train = sequence.pad_sequences(tweet_vecs, maxlen=time_sequence)
#X_test = sequence.pad_sequences(, maxlen=time_sequence)



weights = np.load(open('../vocab_weights', 'rb'))

# create Model #
model = Sequential()
model.add(Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights]))
model.add(LSTM(100, input_shape=(input_len, feats), dropout=0.2, recurrent_dropout=0.2)) # 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit and Train Model #
model.fit(tweet_vecs, outputs, epochs=10, batch_size=60)









