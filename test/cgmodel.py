import numpy as np
from scipy.stats import poisson
from cbp.graph_model import GraphModel
from cbp.np_utils import expand_ndarray, ndarray_denominator
from cbp.coef_policy import *

class CGModel(GraphModel):
    def __init__(self, silent=True, epsilon=1, coef_policy=bp_policy):
        super().__init__(silent=silent, epsilon=epsilon, coef_policy=coef_policy)
        self.update_p_cnt = 0

    def get_factor_exponent(self, factorNode):
        assert len(factorNode.connections) == 2
        i_item = self.get_factor_derivative(factorNode, 0)
        j_item = self.get_factor_derivative(factorNode, 1)
        return expand_ndarray(i_item,
                              factorNode.potential.shape,
                              0) + expand_ndarray(j_item,
                                                  factorNode.potential.shape,
                                                  1)

    def get_factor_derivative(self, factorNode, dim):
        varnode_name = factorNode.connections[dim]
        varnode = self.varnode_recorder[varnode_name]
        observation = varnode.observation

        alpha = 0.08
        estimated = varnode.marginal() * self.num_sample
        if hasattr(varnode, 'prev_estimated'):
            estimated = alpha * estimated + (1-alpha) * varnode.prev_estimated
            varnode.prev_estimated = estimated
        else:
            varnode.prev_estimated = estimated

        needed_item = observation * 1.0 / estimated -1
        
        return needed_item

    def update_potential(self):
        for _, factorNode in self.factornode_recorder.items():
            factorNode.potential = factorNode.original_potential * \
                np.exp(self.get_factor_exponent(factorNode))
            factorNode.potential /= np.mean(factorNode.potential)
            factorNode.potential *= np.max(factorNode.potential)

    def parallel_message(self, run_constrained=True):
        for node in self.nodes:
            node.reset()
        self.first_belief_propagation()
        self.two_pass()
        self.update_potential()
        self.update_p_cnt += 1
        # print(f'step {self.update_p_cnt:4.5f} check difference: {self.compared_gt()}')

    def compared_gt(self):
        estimated_marginal = dict([(node.name, node.marginal())
                                   for node in self.varnode_recorder.values()])
        if not hasattr(self, 'gt_distribution'):
            self.gt_distribution = dict(
                [(node.name, node.gt_dist) for node in self.varnode_recorder.values()])
        return self.compare_marginals(self.gt_distribution, estimated_marginal)
