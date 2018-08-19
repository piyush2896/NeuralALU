import numpy as np
from utils import x_generator

def relu(range_l=-10, range_u=10, is_int=False, m=10000):
    x = x_generator(range_l, range_u, is_int, m, 1)
    return x, np.maximum(x, 0)

def sigmoid(range_l=-10, range_u=10, is_int=False, m=10000):
    x = x_generator(range_l, range_u, is_int, m, 1)
    return x, 1 / (1 + np.exp(- (x + 1e-10)))

def tanh(range_l=-10, range_u=10, is_int=False, m=10000):
    x = x_generator(range_l, range_u, is_int, m, 1)
    return x,  np.tanh(x+1e-10)

def leaky_relu(alpha=0.01, range_l=-10, range_u=10, is_int=False, m=10000):
    x, relu_x = relu(range_l, range_u, is_int, m, 1)
    relu_x = alpha * x[relu_x == 0]
    return x, relu_x
