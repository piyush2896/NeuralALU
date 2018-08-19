import numpy as np

def x_generator(range_l=-1000, range_u=1000, is_int=False, m=10000, dims=100):
    if not is_int:
        return (range_u - range_l) * np.random.rand(m, dims) + range_l
    return np.random.randint(range_l, range_u+1, (m, dims))

def ab_generator(x):
    subsection = np.random.randint(0, x.shape[1], 4)
    subsection.sort()
    m, p, n, q = subsection

    a = np.sum(x[:, m:n], axis=1)
    b = np.sum(x[:, p:q], axis=1)
    return a, b, (m, n, p, q)

def xy_generator(range_l=-1000, range_u=1000, is_int=False, m=10000):
    x = np.expand_dims(x_generator(range_l, range_u, is_int, m), 1)
    y = np.expand_dims(x_generator(range_l, range_u, is_int, m), 1)
    return x, y
