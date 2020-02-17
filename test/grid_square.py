import os

import numpy as np
from scipy.ndimage import gaussian_filter

from cbp.factor_node import FactorNode
from cbp.graph_model import GraphModel
from cbp.np_utils import batch_normal_angle
from cbp.var_node import VarNode

from .base_grid import BaseGrid


class GridSquare(BaseGrid):
    def __init__(self, length, d, policy, sigma_val=1.0, rand_seed=1):
        assert isinstance(d, int)
        self.d_grid = d
        self.observation_sigma = sigma_val
        super().__init__(length, d, d, d * d, policy, rand_seed=rand_seed)
        self._produce_observation_potential()

    def _produce_observation_potential(self):
        potential = []
        for i in range(self.d_grid):
            for j in range(self.d_grid):
                empty_matrix = np.zeros((self.d_grid, self.d_grid))
                empty_matrix[i, j] = 100.0
                result = gaussian_filter(
                    empty_matrix, sigma=self.observation_sigma)
                potential.append(result.flatten() / np.sum(result))

        normalizedrow_potential = np.array(potential).reshape(self.d, self.d)
        self.observation_potential = normalizedrow_potential / \
            np.mean(normalizedrow_potential)
