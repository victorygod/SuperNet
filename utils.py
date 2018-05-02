import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data():
	path = "cifar-100-python/train"
	d = unpickle(path)

	label = d[b'fine_labels']
	coarse_label = d[b'coarse_labels']
	data = []
	for i in range(len(d[b'data'])):
		img = np.array([d[b'data'][i][0:1024], d[b'data'][i][1024:2048], d[b'data'][i][2048:3072]])
		data.append(img.transpose())
	
	return data, label, coarse_label

def organize_data(data, label):
	dataset = [[] for i in range(99)]
	for i in range(len(data)):
		if label[i]==99:
			for j in range(99):
				dataset[j].append({
						'data': data[i],
						'label': 1
					})
		else:
			dataset[label[i]].append({
					'data': data[i],
					'label': 0
				})

	return dataset

def organize_data2(data, label, coarse_label):
	dataset = [[] for i in range(5)] #multi-calss
	# for i in range(len(coarse_label)):

	return dataset

def organize_data3(data, label):
	dataset = [[] for i in range(100)]
	for i in range(len(data)):
		dataset[label[i]//2].append({
				'data' : data[i],
				'label' : label[i]&1
			})
		dataset[50+(label[i]//2)].append({
				'data' : data[i],
				'label' : 1- (label[i]&1)
			})
	return dataset

def organize_data4(data, label):
	dataset = [[] for i in range(5)]
	