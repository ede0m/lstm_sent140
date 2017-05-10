# Main Classification model script with Configuration Class - Garritt Moede #

# utilities
import gensim 
import numpy as np
import math
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor

# keras shit
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
#from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# threading bug.. 
import tensorflow as tf



# 	- Configuration -
#
# takes epochs and other model global params
# takes layer array of tuples that specifies the order of layers in the network
#
#	example: layers = [(Embedding, embedding_weights, input_length),
#					   (LSTM, total_cells_in_layer, optional:dropout_rate), 
#					   (Dense, 50), (Dense, 1)]  

class Configuration(object):

	def __init__(self, name, batchsize, epochs, layers):
		self.name = name
		self.batchsize = batchsize
		self.epochs = epochs
		self.layers = layers
		self.model = Sequential()
		for idx,layer in enumerate(self.layers):
			nextelem = self.layers[(idx + 1) % len(self.layers)][0]
			prevelem = self.layers[(idx - 1) % len(self.layers)][0]
			if layer[0] == 'Embedding':
				weights = layer[1]
				self.model.add(Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], input_length=layer[2], weights=[weights]))

			units = layer[1]
			try:
				dropout = layer[2]
			except IndexError:
				dropout = 0	

			if layer[0] == 'LSTM':
				if nextelem == 'LSTM': #or prevelem == 'LSTM':
					self.model.add(LSTM(units, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)) 
				else:
					self.model.add(LSTM(units, dropout=dropout, recurrent_dropout=dropout)) 
			elif layer[0] == 'Dense':
				self.model.add(Dense(units, activation='sigmoid'))
			## Other Layers ?? ##

		self.graph = tf.get_default_graph() 	# used for multithreading hack. see https://github.com/fchollet/keras/issues/2397
	

	def run(self, lossf, optimizef, TrX, TrY):
		# Do early stopping and ModelCheckpoint callback
		checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')
		early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=0, mode='auto')
		callbacks = [checkpoint, early]
		self.model.compile(loss=lossf, optimizer=optimizef, metrics=['accuracy'])
		print(self.model.summary())
		# train
		self.model.fit(TrX, TrY, epochs=self.epochs, batch_size=self.batchsize, callbacks=callbacks, validation_split=.25)
		

	def metrics(self, TsX,  TsY):

		self.model.load_weights('best_weights.hdf5')
		test_tweet = 'this is a neutral tweet, yuh.'
		test_vec = tweet_covert(test_tweet.split(" "))
		test_vec = sequence.pad_sequences([test_vec], maxlen=time_sequence)
		pred_tw = c.model.predict(test_vec)
		scores = self.model.evaluate(TsX, TsY, verbose=1)
	    
	    # Confusion Matrix (based on binary outputs)
		cf = [[0,0],[0,0]]
		for idx, example in enumerate(TsX):
			output = self.model.predict(np.array([example]))
			if output[0][0] < .50:
				if TsY[idx] is 0:
					cf[0][0] += 1
				else:
					cf[1][0] += 1
			else:
				if TsY[idx] is 1:
					cf[1][1] += 1
				else:
					cf[0][1] += 1
		cf_acc = (cf[0][0] + cf[1][1]) / (cf[0][0] + cf[1][1] + cf[0][1] + cf[1][0])

		print('\n\n ---- ', self.name, ' ----\n')
		print(' -- epochs: ', self.epochs)
		print(' -- batchsize: ', self.batchsize)
		print(' -- layers: ', end='')
		for layer in self.layers:
			if layer[0] is 'Embedding':
				print('Embedding (Input) -> ', end='')
			else:
				print(layer, ' -> ', end='')
		print('Output')
		print(' -- test accuracy: ', scores[1])
		print(' -- confusion accuracy: ', cf_acc)
		print(' -- output for "this is a neutral tweet, yuh.": ', pred_tw)
		print(' -- confusion matrix: \n')
		for row in cf:
			print('\t', row[0], '\t' , row[1])
		print()

	# Not currently working with keras and tensorflow
	def run_worker(self, Xt, Yt, Xts, Yts):
		with tf.Session(graph=self.graph):
			self.run('mean_squared_error', 'adam', Xt, Yt)
			self.metrics(Xts, Yts)


# preps dataframe as embedded data
def dataprep(df):
	lengths = []
	tweet_vecs = []
	outputs = []
	#check skew vars
	pos = 0
	neg = 0
	for index,row in df.iterrows():
		tokens = row[6].split(" ")
		lengths.append(len(tokens))
		vec = tweet_covert(tokens)

		if (row[1] == 4):
			outputs.append(1)
			pos += 1
		else:
			outputs.append(row[1])
			neg += 1
		tweet_vecs.append(vec)
	
	#print(pos, ' ',neg)
	return (tweet_vecs, outputs, lengths)

# converts a tweet into a an embedded vector #
def tweet_covert(tweet):
	vec = []
	for word in tweet:
		
		# preprocess word
		if word.startswith('@'):
			word = 'USERNAME'
		elif word.startswith('http:'):
			word = 'URL'
		elif re.search(r'(.)\1\1', word):
			word = reduce_word(word)

		try:
			idx = w2v.wv.vocab[word].index
			vec.extend([idx])							
		# zero padding
		except TypeError:								
			pass															
		# word not it dict, simply ignore
		except KeyError:
			pass
	return vec

def reduce_word(word):
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



# Load W2V Model #
w2v = gensim.models.Word2Vec.load('../vocab')
numWords = len(w2v.wv.vocab)
dimension = w2v.vector_size

# Load in text data and transform to vectors. TRAIN AND TEST#
df_train = pd.read_csv('../data/sampleTrain.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('../data/sampleTest.csv', encoding='ISO-8859-1')


# Testing data prep #
data = dataprep(df_test)
tweet_vecs_test = data[0]
outputs_test = data[1]
lengths = data[2]

# Training data prep
data = dataprep(df_train)
tweet_vecs_train = data[0]
outputs_train = data[1]
lengths = data[2]



# train and test data format #
mean = sum(lengths)/len(lengths)
std = np.std(lengths) 
time_sequence = math.ceil(mean + (3*std))										# Constant for both train and test sets. Evaluated from training set examples
tweet_vecs_train = np.array(tweet_vecs_train)

X_train = sequence.pad_sequences(tweet_vecs_train, maxlen=time_sequence) 		# Fixed by downgrading to numpy 1.11.2 
X_test = sequence.pad_sequences(tweet_vecs_test, maxlen=time_sequence)
weights = np.load(open('../vocab_weights', 'rb'))

# create different configs based on LSTM Units #
confs = []
confs.append(Configuration('CONF1', 60, 5, [('Embedding', weights, time_sequence), ('LSTM', 100, .2), ('Dense', 1)]))
confs.append(Configuration('CONF2', 60, 5, [('Embedding', weights, time_sequence), ('LSTM', 50, .2), ('Dense', 1)]))
confs.append(Configuration('CONF3', 60, 5, [('Embedding', weights, time_sequence), ('LSTM', 1, .2), ('Dense', 1)]))
# create different configs based on Dense Units
confs.append(Configuration('CONF_DENSE10', 60, 5, [('Embedding', weights, time_sequence), ('LSTM', 50, .2), ('Dense', 10), ('Dense', 1)]))
# stacked LSTM layers
confs.append(Configuration('CONF_LSTM_LAYER', 60, 5, [('Embedding', weights, time_sequence), ('LSTM', 50, .2), ('LSTM', 25, .2), ('Dense', 1)]))

# Run inline 
for c in confs:
	c.run('mean_squared_error', 'adam', X_train, outputs_train)
	c.metrics(X_test, outputs_test)

# Threading #
# -- does not currently work

# with ThreadPoolExecutor(max_workers=4) as e:
# 	for c in confs:
# 		result = e.submit(c.run_worker, X_test, outputs_test, X_test, outputs_test)





