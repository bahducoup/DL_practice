import json
import numpy as np

from collections import OrderedDict

from activation import *
from affine import Affine
from softmax import SoftmaxWithLoss

class NeuralNetwork:
    def __init__(self, NN_conf=None):
        if NN_conf:
            print('INITIALIZE FROM CONFIG FILE')
        else:
            self.input_dimension = int(input('Input dimension: '))
            self.num_hidden_layers = int(input('Number of hidden layers: '))
            self.hidden_dimensions = []
            for i in range(self.num_hidden_layers):
                self.hidden_dimensions.append(int(input('Dimension of hidden layer %d (%d/%d): ' % (i+1, i+1, self.num_hidden_layers))))
            self.output_dimension = int(input('Output dimension: '))
            self.weight_init_std = float(input('weight initial standard deviation: '))
        
        self._build_network()

    def _build_network(self):
        # initialize parameters
        self.params = {}
        dimensions = [self.input_dimension] + [dim for dim in self.hidden_dimensions] + [self.output_dimension]
        for i in range(len(dimensions)-1):
            self.params['W%d' % (i+1)] = self.weight_init_std * \
                    np.random.randn(dimensions[i], dimensions[i+1])
            self.params['b%d' % (i+1)] = self.weight_init_std * \
                    np.random.randn(dimensions[i], dimensions[i+1])

        for k, v in self.params.items():
            print(f'{k}: {v.shape}')

        # create layers
        self.layers = OrderedDict()
        for i in range(len(dimensions) - 1):
            self.layers['Affine%d' % (i+1)] = \
                    Affine(self.params['W%d' % (i+1)], self.params['b%d' % (i+1)])
            if i != len(dimensions) - 2:
                self.layers['Relu%d' % (i+1)] = Relu()
        self.lastLayer = SoftmaxWithLoss()

        print('-->'.join([layer for layer in self.layers.keys()]))


if __name__=='__main__':
    NN = NeuralNetwork()
