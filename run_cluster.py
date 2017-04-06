import pandas as pd
import tensorflow as tf
import numpy as np
import functions as kmeans
import csv
from tensorflow.python import debug as tf_debug

def create_samples():
        dataframe = pd.read_csv('data/training_features.csv')
        return dataframe.as_matrix()

def drop_features(dataframe):
	return dataframe

samples = create_samples()

n_clusters = 3
n_features = len(samples[0][:])
training_steps = 100

initial_centroids = kmeans.create_initial_centroids(n_clusters, n_features, samples)
nearest_indices = kmeans.distribute_samples(samples, initial_centroids)
updated_centroids = kmeans.update_centroids(samples, nearest_indices, n_clusters)

#tensorflow placeholders
X = tf.placeholder(tf.float32, shape=samples.shape, name="feature_input")
Y = tf.placeholder(tf.float32, (11, 2), name="cluster_data")
print tf.shape(X), tf.shape(Y)

outCSV = open('data/cluster_output.csv', 'w')
output = csv.writer(outCSV, delimiter=',')

model = tf.global_variables_initializer()
with tf.Session() as session:
        samples = session.run(model, feed_dict={X: samples})
	for i in range(training_steps):
		nearest_indices_value = session.run(nearest_indices)
        	Y = session.run(updated_centroids)
		if(i%10 == 0):	
			print i, ": ", "10 Training Steps"
	print Y
	for j in Y:
        	output.writerow(j)
        	outCSV.flush()

