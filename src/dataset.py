## used for constructing different configurations of datasets for our classification model

import pandas as pd
import numpy
import math

def qgram(tweet, gram_size):
	ret = []
	tweet = tweet.split(" ")
	window_start = 0 - gram_size + 1
	window_end = 0
	while window_start < (len(tweet)):
		gram = []
		temp_s = window_start
		temp_e = window_end
		# Go through window #
		while temp_s <= window_end:
			if (temp_s < 0) or (temp_s > (len(tweet)-1)):
				gram.append(0)
			else:
				gram.append(tweet[temp_s])
			temp_s += 1

		ret.append(gram)
		window_start += 1
		window_end += 1
	return ret

class DataSet(object):

	# Slow AF with sent140
	def __init__(self, path, window_size ):
		data = []
		self.data_padded = []
		self.outputs = []
		df = pd.read_csv(path, encoding = 'ISO-8859-1')
		lens = []
		print('\n- transforming dataset - ')
		for index, row in df.iterrows():
			tweet = row[5]
			grams = qgram(tweet, window_size)
			self.outputs.append(row[0])
			data.append(grams)
			length = len(grams)
			lens.append(length)	
		
		# mean  + 1 std either way is used to set the dimension size of the "time_series" for our LSTM cells. 68% of all tweets will be fully captured. 
		std = numpy.std(lens)
		mean = sum(lens)/len(lens)
		grams_dimenson = math.ceil(mean + (2*std))		# Time series for LSTM layer
		
		print('\n- constructing dataset padding -\n')
		for twt_grms in data:
			n_grams = len(twt_grms)
			diff = (grams_dimenson - n_grams)
			# resize grams vector to fixed size based on datasets mean and standard deviation. Dependent on window_size. #
			
			# print('\n diff: ', diff)
			if diff > 0:
				rp = math.ceil(diff/2)
				lp = diff - rp
				self.data_padded.append(numpy.pad(twt_grms, ((lp, rp),(0,0)), mode='constant', constant_values=0)) 
				# print(twt_grms, "\n")
			else:
				self.data_padded.append(numpy.resize(twt_grms, grams_dimenson))




# class DataSet(object):

# 	# Slow AF with sent140
# 	def __init__(self, path, model, window_size, embedding_features):
# 		data = []
# 		self.data_padded = []
# 		self.outputs = []
# 		df = pd.read_csv(path, encoding = 'ISO-8859-1')
# 		lens = []
# 		print('\n- transforming data - ')
# 		for index, row in df.iterrows():
# 			tweet = row[5]
# 			grams = qgram(tweet, window_size)
# 			grams_vec = []
# 			# covnert word to vector with model
# 			for gram in grams:
# 				vec = []
# 				# should be number of words in a gram. (2 for bigram, 3 for trigram, ect)
# 				for word in gram:
# 					try:
# 						vec.append(model[word])						
# 					# zero padding
# 					except TypeError:
# 						vec.append(numpy.zeros(embedding_features)) 
# 					# word not it dict
# 					except KeyError:
# 						vec.append(numpy.zeros(embedding_features)) 
# 				grams_vec.append(vec)

# 			self.outputs.append(row[0])
# 			data.append(grams_vec)
# 			length = len(grams_vec)
# 			lens.append(length)	

# 		print(data)
		
# 		# mean  + 1 std either way is used to set the dimension size of the "time_series" for our LSTM cells. 68% of all tweets will be fully captured. 
# 		std = numpy.std(lens)
# 		mean = sum(lens)/len(lens)
# 		grams_dimenson = math.ceil(mean + (2*std))
		
# 		print('\n- constructing dataset padding -\n')
# 		for twt_grms in self.data:
# 			n_grams = len(twt_grms)
# 			diff = (grams_dimenson - n_grams)
# 			# resize grams vector to fixed size based on datasets mean and standard deviation. Dependent on window_size. #
			
# 			# print('\n diff: ', diff)
# 			if diff > 0:
# 				rp = math.ceil(diff/2)
# 				lp = diff - rp
# 				self.data_padded.append(numpy.pad(twt_grms, ((lp, rp),(0,0)), mode='constant', constant_values=0)) 
# 				# print(twt_grms, "\n")
# 			else:
# 				self.data_padded.append(numpy.resize(twt_grms, grams_dimenson))
				# print(twt_grms, "\n")
		








