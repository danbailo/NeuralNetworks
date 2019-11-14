from core import NeuralNetwork
from core import Graph
import numpy as np

if __name__ == "__main__":
	X=np.array([
		[0,0,1,1],
		[0,1,1,0]])
	m=X.shape[1]
	Y=np.array([[0,1,0,1]])

	amount_data = 2
	hidden_neurons = 10
	output_neurons = 1

	nn = NeuralNetwork(
		X = X, 
		Y = Y, 
		lr = 5,
		m = m, 
		epochs = 10000,
		amount_data = amount_data,
		hidden_neurons = hidden_neurons,            
		output_neurons = output_neurons)

	loss, acc = nn.fit()

	for y,output in zip(Y[0], nn.A2[0]):
		print(f"Expected: {y}, Output: {output:.4}")

	graph = Graph(
		file = "nn1",
		amount_data = amount_data,
		hidden_neurons = hidden_neurons,
		output_neurons = output_neurons,
		W1 = nn.W1,
		W2 = nn.W2,
		b1 = nn.b1,
		b2 = nn.b2)

	nn.plot(loss, "loss", "Epochs", r"$J(\theta)$", show = False)
	nn.plot(acc, "acc", "Epochs", "Accuracy", kind = "accuracy", show = False)
	graph.build_graph(show = False)