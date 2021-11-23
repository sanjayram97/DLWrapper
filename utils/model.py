import numpy as np

class Perceptron:
    def __init__(self, lr, epochs):
        np.random.seed(42)
        self.lr =lr
        self.epochs = epochs
        self.weights = np.random.randn(3) * 1e-4
        print("Weights before model training: ", self.weights)
        
    def fit(self, x, y):
        self.x = x
        self.y = y
        x_with_bias = np.c_[x, -1*np.ones((len(x), 1))]
        
        for epoch in range(self.epochs):
            print("--"*20)
            print("Epoch: ", epoch)
            print("--"*20)
            yhat = self.activation(x_with_bias, self.weights)
            self.error = y - yhat
            self.weights = self.weights + self.lr * np.dot(x_with_bias.T, self.error)
            print("Updated weights: ", self.weights)
            
    def predict(self, x):
        x_with_bias = np.c_[x, -1*np.ones((len(x), 1))]
        return self.activation(x_with_bias, self.weights)
    
    def activation(self, inputs, weights):
        z = np.dot(inputs, weights)
        return np.where(z>0, 1, 0)
    
    def total_loss(self):
        return np.sum(self.error)