import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/test')

from .factor_potential import diagnoal_potential,identity_potential
from cbp.coef_policy import *
from cbp.var_node import VarNode
from cbp.factor_node import FactorNode
from cbp.graph_model import GraphModel
import numpy as np

from numpy.random import RandomState

class BaseBuilder(object):
    def __init__(self,d,policy,rand_seed=1):
        self.graph = GraphModel(True, coef_policy=policy)
        self.d = d
        self.rng = RandomState(rand_seed)
    
    def __call__(self):
        self.init_graph()
        return self.graph

    def add_constrained_node(self, probability=None):
        if probability is None:
            log_probability = self.rng.normal(size=self.d)
            probability = np.exp(log_probability)
        else:
            probability = np.array(probability)

        dim = probability.shape[0]
        varnode = VarNode(
            dim,
            constrainedMarginal=probability /
            np.sum(probability))
        self.graph.add_varnode(varnode)
        return varnode

    def add_trivial_node(self,dim=None):
        if dim is None:
            dim = self.d
        varnode = VarNode(dim)
        self.graph.add_varnode(varnode)
        return varnode

    def add_factor(self, name_list):
        factor_potential = diagnoal_potential(self.d,self.d,self.rng)
        factornode = FactorNode(name_list, factor_potential)
        self.graph.add_factornode(factornode)
        return factornode

    def init_graph(self):
        raise NotImplementedError()

class BuildTree(BaseBuilder):
    def __init__(self, tree_depth, d, policy,rand_seed=1):
        self.tree_depth = tree_depth
        super().__init__(d, policy,rand_seed)

    def init_graph(self):
        varnode_namelist = ["VarNode_000"]
        self.add_trivial_node()

        # add lays
        for i in range(1, self.tree_depth):
            for node_id in range(2**(i) - 1, 2**(i + 1) - 1):
                varnode_namelist.append(f"VarNode_{node_id:03d}")
                self.add_trivial_node()

        # add one lay for fix marginals
        for node_id in range(2**(self.tree_depth) - 1, 2 **
                             (self.tree_depth + 1) - 1):
            varnode_namelist.append(f"VarNode_{node_id:03d}")
            self.add_constrained_node()

        for node_index in range(1, 2 ** (self.tree_depth + 1) - 1):
            connnected_var = [varnode_namelist[node_index],
                              varnode_namelist[int((node_index + 1) / 2 - 1)]]
            self.add_factor(connnected_var)

class BuildLine(BaseBuilder):
    def __init__(self, num_node, d, policy,rand_seed=1):
        self.num_node = num_node
        super().__init__(d, policy,rand_seed)

    def init_graph(self):
        varnode_namelist = ["VarNode_000"]
        self.add_constrained_node()

        for i in range(1,self.num_node-1):
            varnode_namelist.append(f"VarNode_{i:03d}")
            self.add_trivial_node()

        varnode_namelist.append(f"VarNode_{self.num_node - 1:03d}")
        self.add_constrained_node()

        for i in range(self.num_node - 1):
            edge_connection = varnode_namelist[i: i + 2]
            self.add_factor(edge_connection)

class BuildStar(BaseBuilder):
    def __init__(self,num_node,d,policy,rand_seed):
        self.num_node = num_node
        super().__init__(d,policy,rand_seed)

    def init_graph(self):
        varnode_namelist = ["VarNode_000"]
        self.add_trivial_node()

        for i in range(1,self.num_node):
            varnode_namelist.append(f"VarNode_{i:03d}")
            self.add_constrained_node()

        for i in range(1,self.num_node):
            edge_connection = [varnode_namelist[0],varnode_namelist[i]]
            self.add_factor(edge_connection)

class BuildHMM(BaseBuilder):
    def __init__(self,length,d,policy,rand_seed=1):
        self.hmm_length = length
        super().__init__(d,policy,rand_seed)

    def init_graph(self):
        varnode_namelist = []

        for i in range(0,self.hmm_length):
            varnode_namelist.append(f"VarNode_{i:03d}")
            varnode_namelist.append(f"VarNode_{2 * i + 1:03d}")
            self.add_trivial_node()
            self.add_constrained_node()

        for i in range(0,self.hmm_length -1):
            edge_connection = [f"VarNode_{2 * i:03d}",f"VarNode_{2 * i + 1:03d}"]
            self.add_factor(edge_connection)
            edge_connection = [f"VarNode_{2 * i:03d}",f"VarNode_{2 * i+2:03d}"]
            self.add_factor(edge_connection)

        edge_connection = [f"VarNode_{2 * self.hmm_length - 2:03d}",f"VarNode_{2*self.hmm_length - 1:03d}"]
        self.add_factor(edge_connection)
        
class BuildZeroCaseHMM(BuildHMM):
    def __init__(self, length, d, policy, rand_seed=1):
        super().__init__(length, d, policy, rand_seed=rand_seed)
    
    def add_factor(self, name_list):
        factor_potential = identity_potential(self.d, self.d, self.rng)
        factornode = FactorNode(name_list, factor_potential)
        self.graph.add_factornode(factornode)

    def add_constrained_node(self):
        probability = np.zeros(self.d)
        probability[0] = 1
        probability[-1] = 1
        varnode = VarNode(
            self.d,
            constrainedMarginal=probability /
            np.sum(probability))
        self.graph.add_varnode(varnode)
