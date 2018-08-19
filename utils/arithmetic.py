import numpy as np
from utils import x_generator
from utils import ab_generator

def xab_generator(range_l=-10, range_u=10, is_int=False, m=10000, dims=100):
    X = x_generator(range_l, range_u, is_int, m, dims)
    a, b, set_range = ab_generator(X)
    return X, a, b, set_range

def add_ab(range_l=-10, range_u=10, is_int=False, m=10000, dims=100):
    X, a, b, set_range = xab_generator(range_l, range_u, is_int, m, dims)
    return X, a + b, set_range

def sub_ab(range_l=-10, range_u=10, is_int=False, m=10000, dims=100):
    X, a, b, set_range = xab_generator(range_l, range_u, is_int, m, dims)
    return X, a - b, set_range

def mul_ab(range_l=-10, range_u=10, is_int=False, m=10000, dims=100):
    X, a, b, set_range = xab_generator(range_l, range_u, is_int, m, dims)
    return X, a * b, set_range

def div_ab(range_l=-10, range_u=10, is_int=False, m=10000, dims=100):
    X, a, b, set_range = xy_generator(range_l, range_u, is_int, m, dims)
    return X, a / (b+1e-10), set_range
