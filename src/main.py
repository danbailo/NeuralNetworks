from core import NeuralNetwork
from core import Graph
import numpy as np
import cupy as cp
from utils import get_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
	dir_cat = '../data/train/cat/*.png'
	dir_noncat = '../data/train/noncat/*.png'

	X_cat, Y_cat = get_data(dir_cat, "cat")
	X_noncat, Y_noncat = get_data(dir_noncat, "noncat")

	X = X_cat + X_noncat
	Y = Y_cat + Y_noncat

	X = np.transpose(np.asarray(X))
	Y = np.transpose(np.asarray(Y))

	Y = np.expand_dims(Y, axis=0)
	X = X/255 #normalization

	lr = 0.1
	epochs = 4000
	amount_data = X.shape[0]
	hidden_neurons = 200
	output_neurons = 1

	nn = NeuralNetwork(
		lr = lr,
		epochs = epochs,
		amount_data = amount_data,
		hidden_neurons = hidden_neurons,            
		output_neurons = output_neurons)

	X_train, X_validate, Y_train, Y_validate = nn.split_data(X, Y, 0.3)

	loss_train, acc_train, loss_validate, acc_validate =\
	nn.train(
		X_train = X_train,
		Y_train = Y_train,
		X_validate = X_validate,
		Y_validate = Y_validate)

	# print(nn.W1.shape)

	graph = Graph(
		file = "nn1",
		amount_data = amount_data,
		hidden_neurons = hidden_neurons,
		output_neurons = output_neurons,
		W1 = nn.W1,
		W2 = nn.W2,
		b1 = nn.b1,
		b2 = nn.b2)

	plt.figure()
	plt.plot(loss_train, label="train")
	plt.plot(loss_validate, label="validate")
	plt.xlabel('epochs')
	plt.ylabel(r'J($\theta$)')	
	plt.legend()

	plt.figure()
	plt.plot(acc_train, label="train")
	plt.plot(acc_validate, label="validate")
	plt.yticks(np.arange(0.0, 1.1, 0.1))
	plt.xlabel('epochs')
	plt.ylabel(r'acc')		
	plt.legend()
	
	plt.show()

	print(f"\nMAX train = {nn.MAX_acc_train*100:.2f}%")
	print(f"MAX validate = {nn.MAX_acc_validate*100:.2f}%")


	# graph.build_graph(show = False)