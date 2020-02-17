import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter

def diagnoal_potential(d_1: int, d_2: int, rng: np.random.RandomState) -> np.ndarray:
    # TODO, cnp will have trouble for zero potential case
    factor_potential = rng.randint(5, 15, size=(d_1, d_2)) * 1.0
    d = np.min([d_1,d_2])
    I = np.eye(d)
    if rng.normal(size=1) > 0:
        I = np.flip(I,axis=0)
    if d_2 > d_1:
        diagonal = np.concatenate([I,np.zeros(d,d_2-d)],axis=1)
    elif d_2 < d_1:
        diagonal = np.concatenate([I,np.zeros(d_1-d,d)],axis=0)
    else:
        diagonal = I

    return np.exp(factor_potential) + diagonal * 50000
    
def identity_potential(d_1: int, d_2: int, rng: np.random.RandomState) -> np.ndarray:
    d = np.min([d_1,d_2])
    I = np.eye(d)
    if d_2 > d_1:
        diagonal = np.concatenate([I,np.zeros(d,d_2-d)],axis=1)
    elif d_2 < d_1:
        diagonal = np.concatenate([I, np.zeros(d_1 - d, d)], axis=0)
    else:
        diagonal = I
    
    return diagonal





