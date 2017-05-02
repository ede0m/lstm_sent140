import dataset as ds
import gensim 
import numpy
import math
# import pydot
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import plot_model

# Load W2V Model #
w2v = gensim.models.Word2Vec.load('../vocab')
numWords = len(w2v.wv.vocab)
dimension = w2v.vector_size

print(w2v)


# Set up embeddings #
embedding_weights = numpy.zeros((numWords, dimension))
count = 0
for word in w2v.wv.vocab.items():
	word = word[0]
	embedding_weights[count,:] = w2v[word]
	count += 1


# Set up Training Data # 
d = ds.DataSet('../../lstm_data/training.csv', 2) ## 2 = GRAM VALUE HERE 
data = d.data_padded
targets = d.outputs
input_len = len(data[0])
feats = len(data[0][0])
# print(input_len, " ", feats)
# print(data)
# print(targets)


# Set up Classifier Model #
model = Sequential()
# embedding (one-hot) layer #

model.add(Embedding(input_dim=numWords, output_dim=dimension, input_length=d.time_series, mask_zero=True, weights=[embedding_weights])) # might need to be numWords + 1
model.add(LSTM(100, input_shape=(input_len, feats), dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Fit and Train Model #
model.fit(data, targets, epochs=10, batch_size=60)


# Set up Testing Data #
test = ds.DataSet('../../lstm_data/test.csv', 2)
test_d = test.data_padded
test_targets = test.outputs

# Test! #
testPredict = model.predict(test_d)
print(testPredict)
# testScore = math.sqrt(mean_squared_error(test_targets, testPredict[:,0]))


# plot_model(model, to_file='../model.png')