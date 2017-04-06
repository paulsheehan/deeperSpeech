import pandas as pd
import numpy as np
import tensorflow as tf

def create_samples():
        dataframe = pd.read_csv('data/training_features.csv')
	dataframe.drop(['id'], axis=1)
        #return dataframe.as_matrix()
	return dataframe

data = create_samples()

#Prepare data
#input features
inputX = data.loc[:, ['freq_min', 'freq_max', 'mfcc_mean']].as_matrix()

#labels
inputY = data.loc[:, ['result']].as_matrix()

#Write out the hyperparameters
learning_rate = 0.00001
training_epochs = 10000
display_step = 1000
n_samples = inputY.size

#write the computation graph
x = tf.placeholder(tf.float32, [None, inputX[0][:].size])

#create weights
W = tf.Variable(tf.zeros([inputX[0][:].size, 1]))

#add biases
B = tf.Variable(tf.zeros([1]))

#multiply inputs by weigths and add biases
y_values = tf.add(tf.matmul(x, W), B)

#apply sigmoid activatin function to the outputs
y = tf.nn.sigmoid(y_values)

#feed in a matrix of labels
y_ = tf.placeholder(tf.float32, [None, 1])

#perform training
#the cost function uses mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
#gradient decent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#start the training session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#training loop
for i in range(training_epochs):
	sess.run(optimizer, feed_dict={x: inputX, y_: inputY})

	#write out logs of training
	if i % display_step == 0:
		cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
		print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)

print "Optimization Finished!"
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print "\nTraining cost = ", training_cost, "W=", sess.run(W), "B=", sess.run(B)

Prediction = sess.run(y, feed_dict = {x: inputX})
prediction = np.asarray(Prediction)
for val in range(0, 10):
	print "Prediction: ", prediction[val], " Actual Value:", inputY[val] 

for val in range(5123, 5133):
        print "Prediction: ", prediction[val], " Actual Value:", inputY[val]

