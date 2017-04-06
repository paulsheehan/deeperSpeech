import csv
import os
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

testOutput = open('data/cluster_test_results.csv', 'w')
outputWriter = csv.writer(testOutput, delimiter=',')

headers = ["id", "freq_min", "freq_max", "freq_std", "mfcc_power", "mfcc_mean", "mfcc_std", "mfcc_delta_mean", "mfcc_delta_std", "min_zero_crossing_rate", "result", "label"]

def import_model():
        dataframe = pd.read_csv('data/cluster_output.csv')
        return dataframe.as_matrix()

def create_test_input():
	dataframe = pd.read_csv('data/test_features.csv')
	return dataframe.as_matrix()

def extract_features(i):
	sample = data[i*frame:frame+i*frame]
#		stft = np.abs(librosa.stft(data))
		#data = librosa.util.normalize(data[0:2080])
      	mfcc = np.mean(librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=26).T, axis=0)
	return np.abs(mfcc)


def distribute_samples(samples, data):
        # Finds the nearest centroid for each sample
        #START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
        expanded_vectors = tf.expand_dims(samples, 0)
        expanded_centroids = tf.expand_dims(data, 1)
        distances = tf.square(
        tf.subtract(samples, data))
        mins = tf.argmin(distances, 0)
        # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
        nearest_indices = mins
        return nearest_indices



model = import_model()
data = create_test_input()

outputWriter.writerow(headers)
testOutput.flush()

X = tf.placeholder(tf.float64, shape=model.shape, name="input")
Y = tf.placeholder(tf.float64, shape=data.shape, name="result")

with tf.Session() as session:
	for i in range(len(data[:])):
		Y = session.run(distribute_samples(model, data[i][:]))
		if i % 100 == 0:
			print Y, data[i][10]
		outputWriter.writerow([Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7], Y[8], Y[9], Y[10], data[i][10]])
		testOutput.flush()
