import numpy as np
import pdb 

def expand_ndarray(inputdata,target_shape,expand_dim):
    """
    inputdata = [1,2]
    target_shape = (1,2,3)
    expand_dim = 1 # only can choose 1!!! 
    output: 
        np.ndarray([
            [1,1,1],
            [2,2,,2]
        ])
    """
    assert(target_shape[expand_dim] == len(inputdata))
    out = np.ones(target_shape)
    for coord in np.ndindex(target_shape):
        i = coord[expand_dim]
        out[coord] *= inputdata[i]
    return out

def reduction_ndarray(ndarray, reduction_index):
    sum_dims = [j for j in range(ndarray.ndim) if not j == reduction_index]
    sum_dims.sort(reverse=True)
    collapsing_marginal = ndarray
    for j in sum_dims:
        collapsing_marginal = collapsing_marginal.sum(j)  # lose 1 dim
    
    return collapsing_marginal

def ndarray_denominator(ndarray):
    ndarray[np.isclose(ndarray, 0)] = np.inf
    return ndarray

def batch_normal_angle(m):
    dx = np.cos(m)
    dy = np.sin(m)
    return np.arctan2(dy, dx)