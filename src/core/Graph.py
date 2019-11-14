from graphviz import Digraph
import numpy as np
import os

class Graph:
	def __init__(self, file, amount_data, hidden_neurons, output_neurons, W1, W2, b1, b2):
		self.dot = Digraph(engine='dot')
		self.dot.attr(rankdir='LR', splines="true")
		self.dot.attr("node", shape="circle")
		self.dot.attr("node", fixedsize="true")
		self.dot.attr("node", height="0.8")

		self.file = file
		self.X = ["X"+str(x) for x in range(amount_data)]
		self.hidden_neurons = ["N"+str(n)+ f"\nb = {b1[n][0]:.2}" for n in range(hidden_neurons)]
		self.output_neurons = ["O"+str(n)+ f"\nb = {b2[n][0]:.3}" for n in range(output_neurons)]

		self.W1 = np.mean(W1, axis=1)
		self.W2 = W2[0]
	
	def build_graph(self, show = False):
		for i in range(len(self.X)):
			self.dot.node(self.X[i])

		for i in range(len(self.hidden_neurons)):
			self.dot.node(self.hidden_neurons[i])	

		for i in range(len(self.output_neurons)):
			self.dot.node(self.output_neurons[i])
		
		for i in range(len(self.X)):
			for j in range(len(self.hidden_neurons)):
				self.dot.edge(self.X[i], self.hidden_neurons[j], label = f"{self.W1[j]:.2f}", minlen="3")

		for i in range(len(self.hidden_neurons)):
			for j in range(len(self.output_neurons)):
				self.dot.edge(self.hidden_neurons[i], self.output_neurons[j], label = f"{self.W2[i]:.2f}", minlen="3")

		self.dot.render(os.path.join("..","imgs",self.file), view = show)  