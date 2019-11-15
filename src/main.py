from core import NeuralNetwork
from core import Graph
import numpy as np
from utils import get_data

if __name__ == "__main__":
	dir_cat = '../data/train/cat/*.png'
	dir_noncat = '../data/train/noncat/*.png'

	X_cat, Y_cat = get_data(dir_cat, "cat")
	X_noncat, Y_noncat = get_data(dir_noncat, "noncat")

	X = X_cat + X_noncat
	Y = Y_cat + Y_noncat

	X = np.asarray(X)
	Y = np.asarray(Y)

	X = X/255
	
	# X = np.insert(X, obj=0, values=1, axis=1)
	Y = np.expand_dims(Y, axis=1)

	Y = Y.T
	X = X.T
	print(X.shape)
	print(Y.shape)

	lr = 0.01
	epochs = 1000
	amount_data = X.shape[0]
	hidden_neurons = 20
	output_neurons = 1

	nn = NeuralNetwork(
		lr = lr,
		epochs = epochs,
		amount_data = amount_data,
		hidden_neurons = hidden_neurons,            
		output_neurons = output_neurons)

	loss, acc = nn.fit(X, Y)

	print(nn.W1.shape)

	# graph = Graph(
	# 	file = "nn1",
	# 	amount_data = amount_data,
	# 	hidden_neurons = hidden_neurons,
	# 	output_neurons = output_neurons,
	# 	W1 = nn.W1,
	# 	W2 = nn.W2,
	# 	b1 = nn.b1,
	# 	b2 = nn.b2)

	nn.plot(loss, "loss", "Epochs", r"$J(\theta)$", show = True)
	nn.plot(acc, "acc", "Epochs", "Accuracy", kind = "accuracy", show = True)
	# graph.build_graph(show = False)