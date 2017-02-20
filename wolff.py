import numpy as np
from matplotlib.mlab import amap
from numba import jit, njit

@jit
def neighbourhood(n): #,dtype='int'):
    I = np.eye(n, dtype=np.byte)
    res = np.vstack([-I, I])
    return res

@jit
def idx(a, ix):
    return a[tuple(ix.T)]

@njit
def expand(ngb, prob, state):
    candidates = np.logical_xor(state, ngb)
    return np.logical_and(candidates,
                        np.random.rand(candidates.size) < prob)

@jit
def update(new, state, lattice, S):
    lattice[tuple(new.T)] = state
    S.extend(new)
    return S

@jit
def cluster(lattice, prob):
    nn = neighbourhood(lattice.ndim)
    dims = np.array(lattice.shape)
    n=0
    S=[]
    start = amap(np.random.randint,dims)
    state = not lattice[tuple(start)]
    new = start[np.newaxis,:]
    while update(new, state, lattice, S):
        n+=1
        current = S.pop()
        neighbours = ((current + nn) % dims)
        flips = expand(idx(lattice, neighbours), prob, state)
        new = neighbours[flips]
    return n*(2*state-1)

#@jit
def run(lattice, prob):
    yield lattice.sum() - lattice.size/2
    while True:
        yield cluster(lattice, prob)
