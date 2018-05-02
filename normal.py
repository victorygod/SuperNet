import tensorflow as tf
import numpy as np
import os
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.relu

batch_size = 50

filters = [128, 128, 128, 128]

img_in = tf.placeholder("float", shape = (None, 32, 32, 3))


x = tf.layers.conv2d(img_in, filters = filters[0], kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)
x = tf.layers.conv2d(x, filters = filters[0], strides = (2,2), kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)

x = tf.layers.conv2d(x, filters = filters[1], kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)
x = tf.layers.conv2d(x, filters = filters[1], strides = (2, 2), kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)

x = tf.layers.conv2d(x, filters = filters[2], kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)
x = tf.layers.conv2d(x, filters = filters[2], strides = (2, 2), kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)

x = tf.layers.conv2d(x, filters = filters[3], kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)
x = tf.layers.conv2d(x, filters = filters[3], strides = (2, 2), kernel_size = 5, padding = "same", activation = activation)
x = tf.layers.batch_normalization(x, training = True)

x = tf.reshape(x, (-1, 2*2*filters[3]))
x = tf.layers.dense(x, units = 500, activation = activation)
x = tf.layers.dense(x, units = 500, activation = activation)

# x_array = []
# for i in range(10):
# 	x_t = tf.layers.dense(x, units = 100, activation = activation)
# 	x_t = tf.layers.dense(x_t, units = 100, activation = activation)
# 	x_t = tf.layers.dense(x_t, units = 10)
# 	x_array.append(x_t)

# output = tf.concat(x_array, axis = -1)

output = tf.layers.dense(x, units = 100)
output = tf.nn.softmax(output)

labels = tf.placeholder("float", shape = (None, 100))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = output)
loss = tf.reduce_mean(loss)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	optim = tf.train.RMSPropOptimizer(0.0001).minimize(loss)

s1 = tf.summary.scalar("loss", loss)

restore_path = 'normal_checkpoint/'
saver = tf.train.Saver(max_to_keep = 1)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	writer = tf.summary.FileWriter("normal_summary/", sess.graph)

	checkpoint = tf.train.latest_checkpoint(restore_path)
	if checkpoint:
		print("restore from: " + checkpoint)
		saver.restore(sess, checkpoint)

	data, label, coarse_label = utils.get_data()

	l = len(data)

	for epoch in range(100):
		i = 0
		while i<l:
			batch_data = []
			batch_label = []
			for j in range(batch_size):
				batch_data.append(np.reshape(data[i], (32, 32, 3)))
				lll = np.zeros((100,))
				lll[label[i]] = 1
				batch_label.append(lll)
				i+=1
			_, lo, la, s = sess.run([optim, loss, output, s1], feed_dict = {img_in: batch_data, labels: batch_label})

			writer.add_summary(s, i+epoch*l)
			if i%1000==0:
				print(lo)
				# print(np.argmax(la, 1)==np.argmax(batch_label, 1))
				print((np.argmax(la, 1)==np.argmax(batch_label, 1)).sum()/batch_size)
				print("=============", epoch, i)
				saver.save(sess, restore_path, global_step = epoch*l+i)