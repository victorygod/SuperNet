import tensorflow as tf
import numpy as np
import os
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.leaky_relu

batch_size = 20

def generator(data_in, units, reuse = False, trainable = True):
	ksize = 5
	with tf.variable_scope('generator', reuse=reuse):
		x = data_in
		# x = tf.layers.dense(x, 100, activation = activation)
		# x = tf.layers.dense(x, 30, activation = activation)
		# x = tf.layers.batch_normalization(x, training = True)
		# x = tf.layers.dense(x, 256, activation = activation)
		# x = tf.layers.batch_normalization(x, training = True)
		# x = tf.layers.dense(x, 256, activation = activation)
		# x = tf.layers.batch_normalization(x, training = True)
		

		fc1_w = tf.layers.dense(x, ksize*ksize*3*units[0], trainable = trainable)
		fc1_w = tf.reshape(fc1_w, (ksize, ksize, 3, units[0]))
		fc1_b = tf.layers.dense(x, units[0], trainable = trainable)
		fc1_b = tf.reshape(fc1_b, (units[0],))
		fc2_w = tf.layers.dense(x, ksize*ksize*units[0]*units[1], trainable = trainable)
		fc2_w = tf.reshape(fc2_w, (ksize,ksize,units[0], units[1]))
		fc2_b = tf.layers.dense(x, units[1], trainable = trainable)
		fc2_b = tf.reshape(fc2_b, (units[1],))
		fc3_w = tf.layers.dense(x, ksize*ksize*units[1]*units[2], trainable = trainable)
		fc3_w = tf.reshape(fc3_w, (ksize,ksize,units[1], units[2]))
		fc3_b = tf.layers.dense(x, units[2], trainable = trainable)
		fc3_b = tf.reshape(fc3_b, (units[2],))		
		fc4_w = tf.layers.dense(x, ksize*ksize*units[2]*units[3], trainable = trainable)
		fc4_w = tf.reshape(fc4_w, (ksize,ksize,units[2], units[3]))
		fc4_b = tf.layers.dense(x, units[3], trainable = trainable)
		fc4_b = tf.reshape(fc4_b, (units[3],))		

		fc_fc1_w = tf.layers.dense(x, 2*2*units[3]*units[4], trainable = trainable)
		fc_fc1_w = tf.reshape(fc_fc1_w, (2*2*units[3], units[4]))
		fc_fc1_b = tf.layers.dense(x, units[4], trainable = trainable)
		fc_fc1_b = tf.reshape(fc_fc1_b, (units[4],))	

		fc_fc2_w = tf.layers.dense(x, units[4]*units[5], trainable = trainable)
		fc_fc2_w = tf.reshape(fc_fc2_w, (units[4], units[5]))
		fc_fc2_b = tf.layers.dense(x, units[5], trainable = trainable)
		fc_fc2_b = tf.reshape(fc_fc2_b, (units[5],))	

		output_w = tf.layers.dense(x, units[5]*1, trainable = trainable)
		output_w = tf.reshape(output_w, (units[5], 1))
		output_b = tf.layers.dense(x, 1, trainable = trainable)
		output_b = tf.reshape(output_b, (1,))	

		weights = {
			"conv1": [tf.nn.tanh(fc1_w), tf.nn.tanh(fc1_b)],
			"conv2": [tf.nn.tanh(fc2_w), tf.nn.tanh(fc2_b)],
			"conv3": [tf.nn.tanh(fc3_w), tf.nn.tanh(fc3_b)],
			"conv4": [tf.nn.tanh(fc4_w), tf.nn.tanh(fc4_b)],
			"fc1": [tf.nn.tanh(fc_fc1_w), tf.nn.tanh(fc_fc1_b)],
			"fc2": [tf.nn.tanh(fc_fc2_w), tf.nn.tanh(fc_fc2_b)],
			"output": [tf.nn.tanh(output_w), tf.nn.tanh(output_b)],
		}
		return weights

def classifier(data_in, weights, units, name, reuse = False):
	with tf.variable_scope('classifer_' + name, reuse=reuse):
		x = tf.nn.conv2d(data_in, weights["conv1"][0], strides = [1, 2, 2, 1], padding = 'SAME')
		x = tf.nn.bias_add(x, weights["conv1"][1])
		x = activation(x)
		# x = tf.layers.batch_normalization(x, training = True)

		x = tf.nn.conv2d(x, weights["conv2"][0], strides = [1, 2, 2, 1], padding = 'SAME')
		x = tf.nn.bias_add(x, weights["conv2"][1])
		x = activation(x)
		# x = tf.layers.batch_normalization(x, training = True)

		x = tf.nn.conv2d(x, weights["conv3"][0], strides = [1, 2, 2, 1], padding = 'SAME')
		x = tf.nn.bias_add(x, weights["conv3"][1])
		x = activation(x)
		# x = tf.layers.batch_normalization(x, training = True)

		x = tf.nn.conv2d(x, weights["conv4"][0], strides = [1, 2, 2, 1], padding = 'SAME')
		x = tf.nn.bias_add(x, weights["conv4"][1])
		x = activation(x)
		# x = tf.layers.batch_normalization(x, training = True)

		x = tf.reshape(x, (-1, 2*2*units[3]))

		x = tf.matmul(x, weights["fc1"][0])
		x = tf.nn.bias_add(x, weights["fc1"][1])

		x = tf.matmul(x, weights["fc2"][0])
		x = tf.nn.bias_add(x, weights["fc2"][1])

		output = tf.matmul(x, weights["output"][0])
		output = tf.nn.bias_add(output, weights["output"][1])
		

		return output


img_in = tf.placeholder("float", shape = (None, 32, 32, 3))
labels = tf.placeholder("float", shape = (None, 1))
z = tf.placeholder("float", shape = (1, 100))

filters = [16, 16, 16, 16, 100, 100]
weights = generator(z, filters)

output = classifier(img_in, weights, filters, "c1")
out_label = tf.nn.sigmoid(output)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels))




source = tf.ones((1,1))
z_generate = tf.layers.dense(source, units=100)
weights_generate = generator(z_generate, filters, reuse = True, trainable = False)
output_generate = classifier(img_in, weights_generate, filters, "c1")
out_label_generate = tf.nn.sigmoid(output_generate)
loss_generate = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_generate, labels=labels))



# weights_t = generator(z, filters)
# output_t = classifier(img_in, weights_t, filters, "c_t")
# output_label_t = tf.nn.sigmoid(output_t)
# loss_t = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_t, labels=labels))


# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
optim = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
optim_g = tf.train.RMSPropOptimizer(0.0001).minimize(loss_generate)

# optim_t = tf.train.RMSPropOptimizer(0.0001).minimize(loss_t)


restore_path = 'superNet_conv_checkpoint/'
saver = tf.train.Saver(max_to_keep = 1)

s1 = tf.summary.scalar("loss", loss)
s2 = tf.summary.scalar("loss_g", loss_generate)
# s3 = tf.summary.scalar("loss_t", loss_t)


with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	checkpoint = tf.train.latest_checkpoint(restore_path)
	if checkpoint:
		print("restore from: " + checkpoint)
		saver.restore(sess, checkpoint)

	writer = tf.summary.FileWriter("superNet_conv_summary/", sess.graph)

	data, label, coarse_label = utils.get_data()
	dataset = utils.organize_data3(data, label)
	

	for epoch in range(20):
		for step in range(50000//batch_size):
			k = np.random.randint(len(dataset)-1)
			l = len(dataset[k])
			a = np.random.choice(l, batch_size)
			batch_data = []
			batch_label = []
			batch_z = np.zeros((1, 100))
			batch_z[0,k] = 1
			for ai in a:
				batch_data.append(np.reshape(dataset[k][ai]["data"], (32, 32, 3)))
				batch_label.append([dataset[k][ai]["label"]])
			_, lo, la, s = sess.run([optim, loss, out_label, s1], feed_dict = {img_in: batch_data, labels: batch_label, z: batch_z})

			writer.add_summary(s, step+epoch*50000//batch_size)
			if step % 100 == 0:
				print(lo)
				print((np.round(la)==np.array(batch_label)).sum()/batch_size)
				print("=============", epoch, step)

				saver.save(sess, restore_path, global_step = step+epoch*50000//batch_size)

	k = len(dataset)-1
	l = len(dataset[k])
	for epoch in range(20):
		for step in range(l//batch_size):
			a = np.random.choice(l, batch_size)
			batch_data = []
			batch_label = []
			for ai in a:
				batch_data.append(np.reshape(dataset[k][ai]["data"], (32, 32, 3)))
				batch_label.append([dataset[k][ai]["label"]])
			_, lo, la, s = sess.run([optim_g, loss_generate, out_label_generate, s2], feed_dict = {img_in: batch_data, labels: batch_label})

			writer.add_summary(s, step+epoch*l//batch_size)
			if step % 100 == 0:
				print(lo)
				print((np.round(la)==np.array(batch_label)).sum()/batch_size)
				print("=============", epoch, step)

				saver.save(sess, restore_path, global_step = step+epoch*l//batch_size)



	# print("============== test ===========")
	# i=200
	# while i<1000:
	# 	batch_data = []
	# 	batch_label = []
	# 	for j in range(batch_size):
	# 		batch_data.append(np.reshape(dataset[k][i]["data"], (32, 32, 3)))
	# 		batch_label.append([dataset[k][i]["label"]])
	# 		i+=1
	# 	lo, la = sess.run([loss_generate, out_label_generate], feed_dict = {img_in: batch_data, labels: batch_label})

	# 	# writer.add_summary(s22, i+epoch*l)

	# 	print(lo)
	# 	print((np.round(la)==np.array(batch_label)).sum()/batch_size)
	# 	print("=============", i)





# with tf.Session() as sess:
# 	init = tf.global_variables_initializer()
# 	sess.run(init)

# 	data, label, coarse_label = utils.get_data()
# 	dataset = utils.organize_data3(data, label)

# 	k = 49
# 	l = len(dataset[k])
# 	for epoch in range(27):
# 		i = 0
# 		while i<200:
# 			batch_data = []
# 			batch_label = []
# 			batch_z = np.zeros((1, 100))
# 			batch_z[0,k] = 1
# 			for j in range(batch_size):
# 				batch_data.append(np.reshape(dataset[k][i]["data"], (32, 32, 3)))
# 				batch_label.append([dataset[k][i]["label"]])
# 				i+=1
# 			_, lo, la, s = sess.run([optim_t, loss_t, output_label_t, s3], feed_dict = {img_in: batch_data, labels: batch_label, z: batch_z})

# 			# writer.add_summary(s, i+k*l+epoch*(len(dataset)-2)*l)

# 			print(lo)
# 			print((np.round(la)==np.array(batch_label)).sum()/batch_size)
# 			print("=============", epoch, i)
		
# 		# saver.save(sess, restore_path, global_step = epoch)

# 	print("============== test ===========")
# 	i=200
# 	while i<1000:
# 		batch_data = []
# 		batch_label = []
# 		batch_z = np.zeros((1, 100))
# 		batch_z[0,k] = 1
# 		for j in range(batch_size):
# 			batch_data.append(np.reshape(dataset[k][i]["data"], (32, 32, 3)))
# 			batch_label.append([dataset[k][i]["label"]])
# 			i+=1
# 		lo, la = sess.run([loss_t, output_label_t], feed_dict = {img_in: batch_data, labels: batch_label, z:batch_z})

# 		# writer.add_summary(s22, i+epoch*l)

# 		print(lo)
# 		print((np.round(la)==np.array(batch_label)).sum()/batch_size)
# 		print("=============", i)



# with tf.Session() as sess:
# 	init = tf.global_variables_initializer()
# 	sess.run(init)

# 	checkpoint = tf.train.latest_checkpoint(restore_path)
# 	if checkpoint:
# 		print("restore from: " + checkpoint)
# 		saver.restore(sess, checkpoint)

# 	writer = tf.summary.FileWriter("superNet_conv_summary/", sess.graph)

# 	data, label, coarse_label = utils.get_data()
# 	dataset = utils.organize_data3(data, label)
	

# 	for epoch in range(7):
# 		# for step in range(50000)
# 		for k in range(len(dataset)-1):
# 			l = len(dataset[k])
# 			i = 0
# 			while i<l:
# 				batch_data = []
# 				batch_label = []
# 				batch_z = np.zeros((1, 100))
# 				batch_z[0,k] = 1
# 				for j in range(batch_size):
# 					batch_data.append(np.reshape(dataset[k][i]["data"], (32, 32, 3)))
# 					batch_label.append([dataset[k][i]["label"]])
# 					i+=1
# 				_, lo, la, s = sess.run([optim, loss, out_label, s1], feed_dict = {img_in: batch_data, labels: batch_label, z: batch_z})

# 				writer.add_summary(s, i+k*l+epoch*(len(dataset)-2)*l)
# 			print(lo)
# 			print((np.round(la)==np.array(batch_label)).sum()/batch_size)
# 			print("=============", epoch, k)

			

# 		saver.save(sess, restore_path, global_step = epoch)
# 	k = 49
# 	l = len(dataset[k])
# 	for epoch in range(20):
# 		i = 0
# 		while i<200:
# 			batch_data = []
# 			batch_label = []
# 			for j in range(batch_size):
# 				batch_data.append(np.reshape(dataset[k][i]["data"], (32, 32, 3)))
# 				batch_label.append([dataset[k][i]["label"]])
# 				i+=1
# 			_, lo, la, s22 = sess.run([optim_g, loss_generate, out_label_generate, s2], feed_dict = {img_in: batch_data, labels: batch_label})

# 			writer.add_summary(s22, i+epoch*l)

# 			print(lo)
# 			print((np.round(la)==np.array(batch_label)).sum()/batch_size)
# 			print("=============", epoch, i)
		
# 		saver.save(sess, restore_path, global_step = epoch)

# 	print("============== test ===========")
# 	i=200
# 	while i<1000:
# 		batch_data = []
# 		batch_label = []
# 		for j in range(batch_size):
# 			batch_data.append(np.reshape(dataset[k][i]["data"], (32, 32, 3)))
# 			batch_label.append([dataset[k][i]["label"]])
# 			i+=1
# 		lo, la = sess.run([loss_generate, out_label_generate], feed_dict = {img_in: batch_data, labels: batch_label})

# 		# writer.add_summary(s22, i+epoch*l)

# 		print(lo)
# 		print((np.round(la)==np.array(batch_label)).sum()/batch_size)
# 		print("=============", i)

