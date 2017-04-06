import pandas as pd
import tensorflow as tf
import numpy as np

def create_initial_centroids(n_clusters, k, samples):
	#convert centroid python list to tensor
	n_samples = tf.shape(samples)[0]
	random_indices = tf.random_shuffle(tf.range(0, n_samples))
	begin = [0,]
	size = [n_clusters,]
	size[0] = n_clusters
	centroid_indices = tf.slice(random_indices, begin, size)
	initial_centroids = tf.gather(samples, centroid_indices)
	return initial_centroids

def distribute_samples(samples, centroids):
	# Finds the nearest centroid for each sample

   	#START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
	expanded_vectors = tf.expand_dims(samples, 0)
	expanded_centroids = tf.expand_dims(centroids, 1)
	distances = tf.reduce_sum( tf.square(
	tf.subtract(expanded_vectors, expanded_centroids)), 2)
	mins = tf.argmin(distances, 0)
    	# END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
	nearest_indices = mins
	return nearest_indices

def update_centroids(samples, nearest_indices, n_clusters):
	# Updates the centroid to be the mean of all samples associated with it.
	nearest_indices = nearest_indices
	partitions = tf.dynamic_partition(samples, tf.to_int32(nearest_indices), n_clusters)
	new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
	return new_centroids

