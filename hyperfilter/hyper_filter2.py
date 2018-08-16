import tensorflow as tf
import numpy as np
import os
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.relu

batch_size = 100
class_num = 10


def conv(emb, x, Nin, Nout, in_size, out_size, strides = [1,2,2,1]):
	filters = tf.layers.dense(emb, Nin*Nout*3*3)
	filters = tf.reshape(filters, (batch_size, 3, 3, Nin, Nout))
	bias = tf.layers.dense(emb, Nout)
	bias = tf.reshape(bias, (batch_size*Nout,))

	filters = tf.transpose(filters, [1, 2, 0, 3, 4])
	filters = tf.reshape(filters, [3, 3, Nin*batch_size, Nout])
	x = tf.transpose(x, [1, 2, 0, 3])
	x = tf.reshape(x, [1, in_size, in_size, batch_size*Nin])
	x = tf.nn.depthwise_conv2d(x, filter = filters, strides = strides, padding = "SAME")
	x = tf.reshape(x, [out_size, out_size, batch_size, Nin, Nout])
	x = tf.transpose(x, [2, 0, 1, 3, 4])
	x = tf.reduce_sum(x, axis = 3)

	x = tf.transpose(x, [1, 2, 0, 3])
	x = tf.reshape(x, (out_size*out_size, batch_size*Nout))
	x = tf.nn.bias_add(x, bias)
	x = tf.reshape(x, (out_size, out_size, batch_size, Nout))
	x = tf.transpose(x, [2, 0, 1, 3])
	x = tf.layers.batch_normalization(x, training = True)
	x = activation(x)
	return x

def Hyper_filter(input_tensor):
	d = 8
	bottleneck = 8
	x = tf.layers.conv2d(input_tensor, 16, 3, strides = (2,2), padding = "same", activation = activation)

	# x = tf.layers.conv2d(x, 32, 3, strides = (2,2), padding = "same", activation = activation)
	emb = tf.layers.conv2d(x, 32, 3, strides = (2,2), padding = "same", activation = activation)
	emb = tf.layers.average_pooling2d(x, 8, strides = 8)
	emb = tf.layers.flatten(emb)
	emb = tf.layers.dense(emb, d)
	x = conv(emb, x, 16, bottleneck, 16, 16, strides = [1,1,1,1])
	x = conv(emb, x, bottleneck, 32, 16, 8)

	# x = tf.layers.conv2d(x, 64, 3, strides = (2,2), padding = "same", activation = activation)
	emb = tf.layers.conv2d(x, 32, 3, strides = (2,2), padding = "same", activation = activation)
	emb = tf.layers.average_pooling2d(x, 4, strides = 4)
	emb = tf.layers.flatten(emb)
	emb = tf.layers.dense(emb, d)
	x = conv(emb, x, 32, bottleneck, 8, 8, strides = [1,1,1,1])
	x = conv(emb, x, bottleneck, 64, 8, 4)

	emb = tf.layers.conv2d(x, 32, 3, strides = (2,2), padding = "same", activation = activation)
	emb = tf.layers.average_pooling2d(x, 2, strides = 2)
	emb = tf.layers.flatten(emb)
	emb = tf.layers.dense(emb, d)	
	x = conv(emb, x, 64, bottleneck, 4, 4, strides = [1,1,1,1])
	x = conv(emb, x, bottleneck, 128, 4, 2)

	x = tf.layers.average_pooling2d(x, 2, strides = 2)
	x = tf.layers.flatten(x)
	x = tf.layers.dense(x, class_num)
	return x	

if __name__ == "__main__":
	print("start")
	input_tensor = tf.placeholder(tf.float32, [None, 32, 32, 3])
	labels = tf.placeholder(tf.float32, shape = (None, class_num))

	output_ = Hyper_filter(input_tensor)
	output = tf.nn.softmax(output_)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = output_)
	loss = tf.reduce_mean(loss)

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		optim = tf.train.AdamOptimizer(1e-3).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# restore_path = 'hyper_filter_2_ckpt/'
	# saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)

	with tf.Session() as sess:
		print("start session")
		sess.run(tf.global_variables_initializer())

		# checkpoint = tf.train.latest_checkpoint(restore_path)
		# if checkpoint:
		# 	print("restore from: " + checkpoint)
		# 	saver.restore(sess, checkpoint)

		data, label = utils.get_data_10(True)
		test_data, test_label = utils.get_data_10(False)
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