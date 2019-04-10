import numpy as np

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self, y, t):
        # t is one-hot
        delta = 1e-7
        
        return -np.sum(t * np.log(y + delta))

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
