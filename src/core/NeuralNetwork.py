from tqdm import trange
import random
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from matplotlib.animation import FuncAnimation

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

random.seed(1)

class NeuralNetwork:
	def __init__(self, lr, epochs, amount_data, hidden_neurons, output_neurons, activation = "sigmoid"):
		self.lr = lr
		self.epochs = epochs
		self.amount_data = amount_data
		self.hidden_neurons = hidden_neurons
		self.output_neurons = output_neurons
		self.activation = activation

		p = 0.001
		self.W1 = np.random.rand(hidden_neurons, amount_data)*p
		self.b1 = np.random.rand(hidden_neurons,1)*p
		
		self.W2 = np.random.rand(output_neurons, hidden_neurons)*p
		self.b2 = np.random.rand(output_neurons,1)*p

		self.__loss_train = []
		self.__acc_train = []
		self.__loss_validate = []
		self.__acc_validate = []

		self.MAX_acc_train = 0
		self.MAX_acc_validate = 0

	def split_data(self, X, Y, ratio=0.3, shuffle=False):
		smaller = min(len(Y[Y==0]), len(Y[Y==1]))
		ratio_data = int(smaller*ratio)
		X_train = []
		Y_train = []
		X_validate = []
		Y_validate = []
		indexes = []

		values = list(zip(X.T, Y.T))

		if shuffle:
			random.shuffle(values)

		data = {key:values[key] for key in range(len(values))}

		for k,v in data.items():
			if v[1][0]==1 and len(X_train) < (smaller - ratio_data) and k not in indexes: #2 class, cat and noncat
				X_train.append(v[0])
				Y_train.append(v[1][0])
				indexes.append(k)
			if v[1][0]==0 and len(X_train) < (smaller - ratio_data)*2 and k not in indexes: #*2 pq ja tem metade disso na lista
				X_train.append(v[0])
				Y_train.append(v[1][0])
				indexes.append(k)

		for k,v in data.items():
			if v[1][0]==1 and len(X_validate) < ratio_data and k not in indexes: #2 class, cat and noncat
				X_validate.append(v[0])
				Y_validate.append(v[1][0])
				indexes.append(k)
			if v[1][0]==0 and len(X_validate) < ratio_data*2 and k not in indexes: #*2 pq ja tem metade disso na lista
				X_validate.append(v[0])
				Y_validate.append(v[1][0])
				indexes.append(k)

		X_train = np.asarray(X_train)
		Y_train = np.asarray(Y_train)
		Y_train = np.expand_dims(Y_train, axis=1)

		X_validate = np.asarray(X_validate)
		Y_validate = np.asarray(Y_validate)
		Y_validate = np.expand_dims(Y_validate, axis=1)		
	
		X_train = X_train.T
		X_validate = X_validate.T
		Y_train = Y_train.T
		Y_validate = Y_validate.T
		return X_train, X_validate, Y_train, Y_validate

	def g(self, Z, derivative = False):
		if self.activation == "sigmoid":
			if derivative:
				return self.g(Z) * (1 - self.g(Z))
			return 1 / (1 + np.exp(-Z))
		if self.activation == "relu":
			return Z * (Z > 0)

	def cost(self, Y_predicted, Y, m):		
		if self.activation == "sigmoid":
			return (1 / m) * np.sum(-Y * np.log(Y_predicted) - (1 - Y) * (np.log(1 - Y_predicted)))
		if self.activation == "relu":
			return np.sqrt(np.mean((Y - Y_predicted) ** 2))		

	def fit(self, m, X_train, Y_train):
		#propagation
		Z1 = (self.W1.dot(X_train)) + self.b1
		A1 = self.g(Z1)
		Z2 = (self.W2.dot(A1)) + self.b2
		A2 = self.g(Z2)

		#backpropagation
		loss_A2 = A2 - Y_train
		loss_W2 = (1/  m) * (loss_A2.dot(A1.T))
		loss_b2 = (1 / m) * np.sum(loss_A2, axis=1, keepdims=True)
		loss_A1 = self.W2.T.dot(loss_A2) * (self.g(Z1, True))
		loss_W1 = (1 / m) * (loss_A1.dot(X_train.T))
		loss_b1 = (1 / m) * np.sum(loss_A1, axis=1, keepdims=True)
		self.W2 -= self.lr * loss_W2
		self.b2 -= self.lr * loss_b2
		self.W1 -= self.lr * loss_W1
		self.b1 -= self.lr * loss_b1

		self.__loss_train.append(self.cost(A2, Y_train, m))
		self.__acc_train.append(np.sum((A2 >= 0.5) == Y_train) / m)
		print(f"train = {self.__acc_train[-1]*100:.2f}%")
		if self.__acc_train[-1] > self.MAX_acc_train:
			self.MAX_acc_train = self.__acc_train[-1]

	def predict(self, m, X_validate, Y_validate):
		#propagation
		Z1 = (self.W1.dot(X_validate)) + self.b1
		A1 = self.g(Z1)
		Z2 = (self.W2.dot(A1)) + self.b2
		A2 = self.g(Z2)
		self.__loss_validate.append(self.cost(A2, Y_validate, m))
		self.__acc_validate.append(np.sum((A2 >= 0.5) == Y_validate) / m)
		print(f"validate = {self.__acc_validate[-1]*100:.2f}%")
		if self.__acc_validate[-1] > self.MAX_acc_validate:
			self.MAX_acc_validate = self.__acc_validate[-1]		

	def train(self, X_train, X_validate, Y_train, Y_validate):
		m_train = X_train.shape[1]
		m_validate = X_validate.shape[1]

		for _ in trange(self.epochs):
			self.fit(m_train, X_train, Y_train)
			self.predict(m_validate, X_validate, Y_validate)
			
		return self.__loss_train, self.__acc_train, self.__loss_validate, self.__acc_validate		