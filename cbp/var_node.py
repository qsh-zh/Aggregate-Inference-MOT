import numpy as np
import json
import uuid

from .base_node import BaseNode
from .np_utils import expand_ndarray, ndarray_denominator


class VarNode(BaseNode):
    def __init__(
            self,
            rv_dim,
            potential=None,
            constrainedMarginal=None,
            node_coef=1):
        super().__init__(node_coef, potential)

        self.rv_dim = rv_dim
        if (self.potential is None):
            self.potential = np.ones([rv_dim])
        assert(self.potential.shape[0] == rv_dim)

        if constrainedMarginal is None:
            self.isconstrained = False
        else:
            assert(constrainedMarginal.shape[0] == rv_dim)
            assert(abs(np.sum(constrainedMarginal) - 1) < 1e-6)
            self.isconstrained = True
        self.constrainedMarginal = constrainedMarginal

    def auto_coef(self, node_map, assign_policy=None):
        if assign_policy is None:
            super().auto_coef(node_map)
        else:
            self.node_coef = assign_policy(self, node_map)
            self.register_nodes(node_map)
        

        sum_i_alpha = 0
        unset_edge = None
        for item in self.connected_nodes.values():
            i_alpha = item.get_i_alpha(self.name)
            if i_alpha != None:
                sum_i_alpha += i_alpha
            else:
                unset_edge = item.name
        if unset_edge:
            new_i_alpha = self.node_coef - \
                (1 - len(self.connections)) - sum_i_alpha
            self.connected_nodes[unset_edge].set_i_alpha(self.name, new_i_alpha)

    def cal_cnp_coef(self):
        self.coef_ready = True

        self.hat_c_i = self.node_coef
        for item in self.connections:
            self.hat_c_i += self.connected_nodes[item].node_coef

    def _make_message_first_term(self, recipient_node):
        recipient_index_in_var = self.search_msg_index(self.latest_message,
                                                       recipient_node.name)
        hat_c_ialpha = recipient_node.get_hat_c_ialpha(self.name)
        c_alpha = recipient_node.node_coef
        vals = [message.val for message in self.latest_message]
        if self.isconstrained:
            numerator = np.power(self.constrainedMarginal,self.epsilon)
        else:
            potential_part = np.power(self.potential,1.0/self.hat_c_i)
            message_part = np.power(np.prod(vals, axis=0), 1.0 / self.hat_c_i)
            numerator = potential_part * message_part
        denominator = np.power(vals[recipient_index_in_var], 1.0 / hat_c_ialpha)
        denominator = ndarray_denominator(denominator)

        return np.power(numerator / denominator,c_alpha)

    def make_message_bp(self, recipient_node):
        assert self.coef_ready, f"{self.name} need to cal_cnp_coef by graph firstly"
        # first_term.shape equals (self.rv_dim,)
        first_term = self._make_message_first_term(recipient_node)
        assert first_term.shape[0] == self.rv_dim
        # second_term shape equals shape of recipient_node
        second_term = recipient_node.get_varnode_extra_term(self.name)
        assert second_term.shape == self.connected_nodes[recipient_node.name].potential.shape

        var_index_in_recipient = recipient_node.search_node_index(self.name)

        expanded_first_term = expand_ndarray(
            first_term, second_term.shape, var_index_in_recipient)

        return np.multiply(expanded_first_term, second_term)

    def make_message(self, recipient_node):
        return self.make_message_bp(recipient_node)

    def make_init_message(self, recipient_node_name):
        if self.coef_ready:
            recipient_node = self.connected_nodes[recipient_node_name]
            message_dim = recipient_node.potential.shape
            return np.ones(message_dim)
        else:
            raise RuntimeError(f"Need to call cal_cnp_coef first for {self.name}")

    def marginal(self):
        if self.isconstrained:
            return self.constrainedMarginal
        if len(self.message_inbox):
            vals = [message.val for message in self.latest_message]
            vals_prod = np.prod(vals, axis=0)
            prod = self.potential * vals_prod
            belief = np.power(prod, 1.0 / self.hat_c_i)
            return belief / np.sum(belief)
        else:
            return np.ones(self.rv_dim) / self.rv_dim

    def to_json(self, separators=(',', ':'), indent=4):
        return json.dumps({
            'class': 'VarNode',
            'name': self.name,
            'potential': self.potential.tolist(),
            'node_coef': self.node_coef,
            'constraine_marginal': self.constrainedMarginal.tolist() if self.isconstrained else None,
            'connections': self.connections
        }, separators=separators, indent=indent)

    @classmethod
    def from_json(cls, j):
        d_context = json.loads(j)

        if d_context['class'] != 'VarNode':
            raise IOError(
                f"Need a VarNode class json to construct VarNode instead of {d_context['class']}")

        potential = d_context['potential']
        coef = d_context['node_coef']
        constraine_marginal = d_context['constraine_marginal']
        if isinstance(constraine_marginal, list):
            constraine_marginal = np.asarray(constraine_marginal)
        node = cls(
            len(potential),
            np.asarray(potential),
            constraine_marginal,
            node_coef=coef)
        node.format_name(d_context['name'])
        for factor_node in d_context['connections']:
            node.register_connection(factor_node)
        return node

    def __eq__(self, value):
        if isinstance(value, VarNode):
            flag = []
            flag.append(self.name == value.name)
            flag.append(np.array_equal(self.potential, value.potential))
            flag.append(np.array_equal(self.node_coef, value.node_coef))
            flag.append(self.isconstrained == value.isconstrained)
            flag.append(
                np.array_equal(
                    self.constrainedMarginal,
                    value.constrainedMarginal))
            flag.append(self.node_degree == value.node_degree)
            flag.append(self.connections == value.connections)
            if np.sum(flag) == len(flag):
                return True

        return False
