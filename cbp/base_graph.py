import json
import numpy as np
from functools import partial
from .var_node import VarNode
from .factor_node import FactorNode
from .message import Message
from .np_utils import expand_ndarray, reduction_ndarray, ndarray_denominator
from .coef_policy import *

try:
    import pygraphviz
    import tempfile
    import matplotlib
    import matplotlib.pyplot as plt
except BaseException:
    pygraphviz = None


class BaseGraph(object):
    def __init__(self, silent=True, epsilon=1, coef_policy=bp_policy):
        self.varnode_recorder = {}
        self.constrained_recorder = []
        self.factornode_recorder = {}
        self.epsilon = epsilon
        self.coef_policy = coef_policy

        # debug utils
        self.silent = silent

    def add_varnode(self, node):
        assert(isinstance(node, VarNode))
        num_varnode = len(self.varnode_recorder)
        varnode_name = f"VarNode_{num_varnode:03d}"
        node.format_name(varnode_name)
        self.varnode_recorder[varnode_name] = node
        if node.isconstrained:
            self.constrained_recorder.append(varnode_name)

    def add_factornode(self, factornode):
        factornode.check_potential(self.varnode_recorder)
        num_factornode = len(self.factornode_recorder)
        factornode_name = f"FactorNode_{num_factornode:03d}"
        self.factornode_recorder[factornode_name] = factornode
        factornode.format_name(factornode_name)

        connections = factornode.get_connections()
        for varnode_name in connections:
            varnode = self.varnode_recorder[varnode_name]
            varnode.register_connection(factornode_name)

        factornode.parent = self.varnode_recorder[connections[0]]
        for varnode_name in connections[1:]:
            varnode = self.varnode_recorder[varnode_name]
            varnode.parent = factornode

    def pmf(self):
        varnode_names = list(self.varnode_recorder.keys())
        varnodes = list(self.varnode_recorder.values())
        factornodes = list(self.factornode_recorder.values())
        var_dim = [variable.rv_dim for variable in varnodes]
        assert len(var_dim) < 32, "max number of vars for brute_force is 32 \
            (numpy's matrix dim limit)"
        joint_acc = np.ones(var_dim)
        for factor in factornodes:
            which_dims = [varnode_names.index(v)
                          for v in factor.get_connections()]
            factor_acc = np.ones(var_dim)
            for joint_coord in np.ndindex(tuple(var_dim)):
                factor_coord = tuple([joint_coord[i] for i in which_dims])
                factor_acc[joint_coord] *= factor.potential[factor_coord]
            joint_acc *= factor_acc
        joint_prob = joint_acc / np.sum(joint_acc)
        return joint_prob

    def exact_marginal(self):
        varnodes = list(self.varnode_recorder.values())
        prob_tensor = self.pmf()

        marginal_list = self.cal_marginal_from_tensor(prob_tensor, varnodes)
        for node, marginal in zip(varnodes, marginal_list):
            node.bfmarginal = marginal

    def first_belief_propagation(self):
        for node in self.nodes:
            for recipient_name in node.connections:
                val = node.make_init_message(recipient_name)
                message = Message(node, val)
                self.node_recorder[recipient_name].store_message(message)

    @staticmethod
    def cal_marginal_from_tensor(prob_tensor, varnode_list):
        rtn_marginal = []
        for i, _ in enumerate(varnode_list):
            rtn_marginal.append(reduction_ndarray(prob_tensor, i))
        return rtn_marginal

    def init_sinkhorn_node(self):
        varnode_names = list(self.varnode_recorder.keys())
        self.sinkhorn_node_coef = {}
        for node_name in self.constrained_recorder:
            node_instance = self.varnode_recorder[node_name]
            self.sinkhorn_node_coef[node_name] = {
                'index': varnode_names.index(node_name),
                'mu': node_instance.constrainedMarginal,
                'u': np.ones(node_instance.rv_dim)
            }
        for node in self.varnode_recorder.values():
            node.sinkhorn = np.ones(node.rv_dim) / node.rv_dim

    def build_big_u(self):
        varnodes = list(self.varnode_recorder.values())
        var_dim = [variable.rv_dim for variable in varnodes]
        joint_acc = np.ones(var_dim)

        for name, recoder in self.sinkhorn_node_coef.items():
            constrained_acc = expand_ndarray(
                recoder['u'], tuple(var_dim), recoder['index'])
            joint_acc *= constrained_acc

        # log_joint_acc -= np.max(log_joint_acc)
        return joint_acc / np.sum(joint_acc)

    # TODO this is a bug!!!!
    def sinkhorn_update(self, tilde_c):
        for node_name, recorder in self.sinkhorn_node_coef.items():
            big_u = self.build_big_u()
            length_var = len(self.varnode_recorder)
            normalized_denominator = (big_u * tilde_c) / \
                np.sum(big_u * tilde_c)

            copy_denominator = reduction_ndarray(
                normalized_denominator, recorder['index'])
            copy_denominator = ndarray_denominator(copy_denominator)
            recorder['u'] = recorder['u'] * recorder['mu'] / copy_denominator

        varnodes = list(self.varnode_recorder.values())
        marginal_list = self.cal_marginal_from_tensor(
            normalized_denominator, varnodes)
        for node, marginal in zip(varnodes, marginal_list):
            node.sinkhorn = marginal

    def check_sinkhorn(self):
        if len(self.constrained_recorder) == 0:
            raise RuntimeError(
                f"There is no constrained nodes, use brutal force")

    def sinkhorn(self, max_iter=5000000, tolerance=1e-6, error_fun=None):
        self.check_sinkhorn()
        step = 0
        epsilons = [1]

        tilde_c = self.pmf()
        self.init_sinkhorn_node()
        cur_marginal = self.export_sinkhorn()
        while (step < max_iter) and tolerance < epsilons[-1]:
            last_marginal = cur_marginal
            step += 1
            if not self.silent:
                print(
                    f"epsilon: {epsilons[-1]:5.4f} | step: {step:5d} {'-'*10}")
            self.sinkhorn_update(tilde_c)
            cur_marginal = self.export_sinkhorn()
            if error_fun:
                epsilons.append(error_fun(cur_marginal, last_marginal))
            else:
                epsilons.append(
                    self.compare_marginals(
                        cur_marginal, last_marginal))
        if not self.silent:
            print('x' * 50)
            print('final epsilon after ' + str(step) + ' iterations = ' + str(
                epsilons[-1]))
        if tolerance <= epsilons[-1]:
            raise RuntimeWarning(
                f"Attention!!! Reach max iterations but not converge")

        return epsilons[1:], step

    def bake(self):
        self.init_node_recorder()
        root = self.varnode_recorder['VarNode_000']
        self.auto_node_coef(root)

    def auto_node_coef(self, node):
        node.is_traversed = True
        for item in node.connections:
            if not self.node_recorder[item].is_traversed:
                self.auto_node_coef(self.node_recorder[item])

        node.auto_coef(self.node_recorder, self.coef_policy)

    def init_node_recorder(self):
        factors = [node for node in self.factornode_recorder.values()]
        variables = [node for node in self.varnode_recorder.values()]
        # in Norm-Product, run factor message first
        self.nodes = factors + variables
        self.node_recorder = {**(self.varnode_recorder),
                              **(self.factornode_recorder)}
        self.constrained_nodes = [self.varnode_recorder[name]
                                  for name in self.constrained_recorder]

    def tree_message_pass(self):
        self.init_cnp_coef()
        self.first_belief_propagation()
        self.two_pass()

    def two_pass(self):
        for node in self.nodes:
            node.marked = False
        if not hasattr(self,'root_node'):
            for node in self.nodes:
                if len(node.connections) == 1:
                    self.root_node = node

        self.send_from(self.root_node)
        self.send_out(self.root_node)

    def send_from(self,node):
        node.marked = True
        for cur_node in node.connected_nodes.values():
            if not cur_node.marked:
                self.send_from(cur_node)
                cur_node.send_message(node)

    def send_out(self,node):
        node.marked = False
        for cur_node in node.connected_nodes.values():
            if cur_node.marked:
                node.send_message(cur_node)
                self.send_out(cur_node)

    def export_marginals(self):
        return dict([
            (n.name, n.marginal()) for n in self.varnode_recorder.values()
        ])

    def export_convergence_marginals(self):
        return dict([
            (n.name, n.marginal()) for n in self.nodes
        ])

    def export_sinkhorn(self):
        return dict([(node_name, node.sinkhorn)
                     for node_name, node in self.varnode_recorder.items()])

    @staticmethod
    def compare_marginals(m1, m2):
        assert not len(np.setdiff1d(m1.keys(), m2.keys()))
        return sum([np.sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])

    def plot(self, png_name='file.png'):
        if pygraphviz is not None:
            graph = pygraphviz.AGraph(directed=False)
            for varnode_name in self.varnode_recorder.keys():
                if varnode_name in self.constrained_recorder:
                    graph.add_node(varnode_name, color='red', style='filled')
                else:
                    graph.add_node(varnode_name, color='blue', style='bold')

            for name, factornode in self.factornode_recorder.items():
                graph.add_node(name, color='green')
                for varnode_name in factornode.get_connections():
                    graph.add_edge(name, varnode_name)

            graph.layout(prog='neato')
            graph.draw(png_name)
        else:
            raise ValueError("must have pygraphviz installed for visualization")

    def to_json(self, separators=(',', ':'), indent=4):
        return json.dumps({
            'class': 'GraphModel',
            'varnodes': [json.loads(node.to_json()) for node in self.varnode_recorder.values()],
            'factornodes': [json.loads(node.to_json()) for node in self.factornode_recorder.values()],
        }, separators=separators, indent=indent)

    @classmethod
    def from_json(cls, j):
        d_context = json.loads(j)

        if d_context['class'] != 'GraphModel':
            raise IOError(
                f"Need a GraphModel class json to construct GraphModel instead of {d_context['class']}")
        varnodes = [VarNode.from_json(json.dumps(info))
                    for info in d_context['varnodes']]
        factornodes = [
            FactorNode.from_json(
                json.dumps(info)) for info in d_context['factornodes']]

        graph = cls()
        for node in varnodes:
            graph.varnode_recorder[node.name] = node
            if node.isconstrained:
                graph.constrained_recorder.append(node.name)
        for node in factornodes:
            graph.factornode_recorder[node.name] = node

        return graph

    def __eq__(self, value):
        if isinstance(value, type(self)):
            flag = []
            for varnode_name in self.varnode_recorder.keys():
                flag.append(
                    self.varnode_recorder[varnode_name] == value.varnode_recorder[varnode_name])
            flag.append(self.constrained_recorder == value.constrained_recorder)
            for factornode_name in self.factornode_recorder.keys():
                flag.append(
                    self.factornode_recorder[factornode_name] == value.factornode_recorder[factornode_name])
            if np.sum(flag) == len(flag):
                return True

        return False
