import numpy as np
from scipy.stats import poisson
from cbp.graph_model import GraphModel
from cbp.np_utils import expand_ndarray, ndarray_denominator
from cbp.coef_policy import *

class CGModel(GraphModel):
    def __init__(self, silent=True, epsilon=1, coef_policy=bp_policy):
        super().__init__(silent=silent, epsilon=epsilon, coef_policy=coef_policy)

    def get_factor_exponent(self, factorNode):
        assert len(factorNode.connections) == 2
        i_item = self.get_factor_derivative(factorNode, 0)
        j_item = self.get_factor_derivative(factorNode, 1)
        return expand_ndarray(i_item,
                              factorNode.potential.shape,
                              1) + expand_ndarray(j_item,
                                                  factorNode.potential.shape,
                                                  0)

    def get_factor_derivative(self, factorNode, dim):
        varnode_name = factorNode.connections[dim]
        varnode = self.varnode_recorder[varnode_name]
        observation = varnode.observation

        alpha = 0.4
        estimated = varnode.marginal() * self.num_sample
        if hasattr(varnode, 'prev_estimated'):
            estimated = alpha * estimated + (1-alpha) * varnode.prev_estimated
        else:
            varnode.prev_estimated = estimated
        probability = poisson.pmf(observation, estimated)
        # print(f"{varnode.name} has real \n {estimated}, \n observed \n {observation}")
        p_derivative = poisson.pmf(observation - 1, estimated) - probability
        item_mean = np.mean(probability)
        probability = probability / item_mean
        p_derivative /= item_mean
        l_derivative = p_derivative / ndarray_denominator(probability)

        return l_derivative / varnode.node_degree 

    def update_potential(self):
        for _, factorNode in self.factornode_recorder.items():
            factorNode.potential = factorNode.original_potential * \
                np.exp(self.get_factor_exponent(factorNode))

    def parallel_message(self, run_constrained=True):
        self.engine_loop(super().parallel_message)
        self.update_potential()
        # if not self.silent:
        print(f'check difference: {self.compared_gt()}')

    def compared_gt(self):
        estimated_marginal = dict([(node.name, node.marginal())
                                   for node in self.varnode_recorder.values()])
        if not hasattr(self, 'gt_distribution'):
            self.gt_distribution = dict(
                [(node.name, node.gt_dist) for node in self.varnode_recorder.values()])
        return self.compare_marginals(self.gt_distribution, estimated_marginal)
