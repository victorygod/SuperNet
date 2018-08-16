import tensorflow as tf
import numpy as np
import os
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.relu

batch_size = 100
class_num = 100

def Normal(input_tensor, istraining = True):
	x = tf.layers.conv2d(input_tensor, 16, 3, strides = (2,2), padding = "same", activation = activation)
	# x = tf.layers.conv2d(x, 8, 3, padding = "same")
	# x = tf.layers.batch_normalization(x, training = True)
	# x = activation(x)	
	x = tf.layers.conv2d(x, 32, 3, strides = (2,2), padding = "same")
	x = tf.layers.batch_normalization(x, training = True)
	x = activation(x)
	# x = tf.layers.conv2d(x, 8, 3, padding = "same")
	# x = tf.layers.batch_normalization(x, training = True)
	# x = activation(x)	
	x = tf.layers.conv2d(x, 64, 3, strides = (2,2), padding = "same")
	x = tf.layers.batch_normalization(x, training = True)
	x = activation(x)
	# x = tf.layers.conv2d(x, 8, 3, padding = "same")
	# x = tf.layers.batch_normalization(x, training = True)
	# x = activation(x)	
	x = tf.layers.conv2d(x, 128, 3, strides = (2,2), padding = "same")
	x = tf.layers.batch_normalization(x, training = True)
	x = activation(x)
	x = tf.layers.average_pooling2d(x, 2, strides = 2)
	x = tf.layers.flatten(x)
	x = tf.layers.dense(x, class_num)
	return x

if __name__ == "__main__":
	print("start")
	input_tensor = tf.placeholder(tf.float32, [None, 32, 32, 3])
	labels = tf.placeholder(tf.float32, shape = (None, class_num))

	output_ = Normal(input_tensor)
	output = tf.nn.softmax(output_)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = output_)
	loss = tf.reduce_mean(loss)

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		optim = tf.train.AdamOptimizer(1e-3).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# restore_path = 'normal_ckpt/'
	# saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)

	with tf.Session() as sess:
		print("start session")
		sess.run(tf.global_variables_initializer())

		# checkpoint = tf.train.latest_checkpoint(restore_path)
		# if checkpoint:
		# 	print("restore from: " + checkpoint)
		# 	saver.restore(sess, checkpoint)

		data, label, _ = utils.get_data(True)
		test_data, test_label, _ = utils.get_data(False)
		l = len(data)
		l_test = len(test_data)
		for epoch in range(600):
			i = 0
			while i<l:
				batch_data = []
				batch_label = []
				for j in range(batch_size):
					batch_data.append(data[i])
					lll = np.zeros((class_num,))
					lll[label[i]] = 1
					batch_label.append(lll)
					i+=1
				_, lo, la = sess.run([optim, loss, output], feed_dict = {input_tensor: batch_data, labels: batch_label})
				# if i%5000==0:
				# 	print(lo)
					# saver.save(sess, restore_path+"ckpt")

			i = 0
			acc=0
			while i < l_test:
				batch_data = []
				batch_label = []
				for j in range(batch_size):
					batch_data.append(data[i])
					lll = np.zeros((class_num,))
					lll[label[i]] = 1
					batch_label.append(lll)
					i+=1
				acc += sess.run(accuracy, feed_dict={input_tensor: batch_data, labels: batch_label})

			print(epoch, " acc: ", acc/(l_test//batch_size))