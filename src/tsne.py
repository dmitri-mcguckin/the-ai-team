import matplotlib.pyplot as plt, numpy as np, sys, csv

def euclidean_distance(p1, p2):
	if (len(p1) != len(p2)): return -1
	sum = 0
	for i in range(len(p1)): sum += np.square(p1[i] - p2[i])
	return np.sqrt(sum)

def load_data(filename):
	labels = []
	data = []
	file = open(filename)
	input = csv.reader(file)
	for i, row in enumerate(input):
		row = list(map(lambda x: float(int(x) / 255), row))
		labels.append(row[0])
		data.append(row[1:])
	file.close()
	return labels, data

def start(args):
	if(len(args) < 2):
		print("usage: ... <dimensional reduction> <data file>")
		sys.exit(-1)

	dim = int(args[0])
	labels, data = load_data(args[1])
	print("Reducing to " + str(dim) + " dimensions...\nLoaded: (" + str(len(labels)) + "|" + str(len(data)) + ") label/data points with " + str(len(data[0])) + " features!")

	p1 = [0, 1, 2]
	p2 = [5, 23, 7]
	d = round(euclidean_distance(p1, p2), 2)
	print("Distance between p1(" + str(p1) + ") and p2(" + str(p2) + "): " + str(d))
