import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

np.random.seed(1)

class NeuralNetwork:
	def __init__(self, X, Y, lr, m, epochs, amount_data, hidden_neurons, output_neurons):
		self.X = X
		self.Y = Y
		self.lr = lr
		self.m = m
		self.epochs = epochs
		self.amount_data = amount_data
		self.hidden_neurons = hidden_neurons
		self.output_neurons = output_neurons

		p = 0.01
		self.W1=np.random.rand(hidden_neurons, amount_data)*p
		self.b1=np.random.rand(hidden_neurons,1)*p
		
		self.W2=np.random.rand(output_neurons, hidden_neurons)*p
		self.b2=np.random.rand(output_neurons,1)*p

	def plot(self, obj, name, xlabel, ylabel, kind = None, show = False):
		plt.figure()
		plt.plot(obj)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if kind == "accuracy":
			plt.yticks(np.arange(0,1.1,0.1))
		plt.savefig("../imgs/"+name+".pdf")
		if show:
			plt.show()

	
	def sigmoid(self, Z, derivative=False):
		if derivative: 
			return (Z * (1 - Z))
		return 1 / (1 + np.exp(-Z))

	def loss(self):
		return (1 / self.m) * np.sum(-self.Y * np.log(self.A2) - (1 - self.Y) * (np.log(1 - self.A2)))

	def propagation(self):
		Z1 = (self.W1.dot(self.X)) + self.b1
		self.A1 = self.sigmoid(Z1)
		Z2 = (self.W2.dot(self.A1)) + self.b2
		self.A2 = self.sigmoid(Z2)

	def backpropagation(self):
		loss_A2 = self.A2 - self.Y
		loss_W2 = (1/  self.m) * (loss_A2.dot(self.A1.T))
		loss_b2 = (1 / self.m) * np.sum(loss_A2, axis=1, keepdims=True)
	
		loss_A1 = self.W2.T.dot(loss_A2) * self.sigmoid(self.A1, True)
		loss_W1 = (1 / self.m) * (loss_A1.dot(self.X.T))
		loss_b1 = (1 / self.m) * np.sum(loss_A1, axis=1, keepdims=True)
	
		self.W2 -= self.lr * loss_W2
		self.b2 -= self.lr * loss_b2
		self.W1 -= self.lr * loss_W1
		self.b1 -= self.lr * loss_b1

	def fit(self):
		loss_total = []
		acc = []
		for _ in range(self.epochs):
			self.propagation()
			self.backpropagation()
			loss_total.append(self.loss())
			acc.append(np.sum(self.Y == (self.A2 >= 0.5)) / self.m)
		return loss_total, acc
 

