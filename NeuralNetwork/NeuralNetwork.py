import numpy as np
import matplotlib.pyplot as plt
src = "http://iamtrask.github.io/2015/07/12/basic-python-network/"
video = "https://www.youtube.com/watch?v=5u0jaA3qAGk"

X = np.array(([3,5],[5,1],[10,2]), dtype=float)
y = np.array(([75],[82],[93]), dtype=float)


#Normalize
X = X/np.amax(X, axis=0)
y = y/100

#Neural Netwok Class

class Neural_Network(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1-np.exp(-z))

    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrim(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params


    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))


NN = Neural_Network()

yHat = NN.forward(X)

plt.bar([0,1,2], y, width = 0.35, alpha=0.8)
plt.bar([0.35,1.35,2.35],yHat, width = 0.35, color='r', alpha=0.8)
plt.grid(1)
plt.legend(['y', 'yHat'])
plt.show()