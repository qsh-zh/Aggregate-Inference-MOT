import json
import uuid

import numpy as np

from .base_node import BaseNode
from .np_utils import expand_ndarray, ndarray_denominator


class FactorNode(BaseNode):
    def __init__(self, connections, potential, coef=1):
        super().__init__(coef, potential)
        self.connections = connections
        self.last_innerparenthese4varnode_msg = {}

        self.i_alpha = {}
        for item in self.connections:
            self.i_alpha[item] = None

    def check_before_run(self, node_map):
        super().check_before_run(node_map)
        self.check_potential(self, node_map)

    def check_potential(self, node_map):
        for i, varnode_name in enumerate(self.connections):
            varnode = node_map[varnode_name]
            assert self.potential.shape[i] == varnode.rv_dim, \
                f"Dimention mismatch! At {i:02d} axis, \
                    rv_dim:{varnode.rv_dim:02d}, potential: {self.potential.shape[i]}\
                        in Factor:{self.name}"

    def auto_coef(self, node_map, assign_policy=None):
        if assign_policy:
            self.node_coef = assign_policy(self, node_map)
            self.register_nodes(node_map)
        else:
            super().auto_coef(node_map)

        sum_i_alpha = 0
        unset_edge = None
        for connected_var in self.connections:
            i_alpha = self.get_i_alpha(connected_var)
            if i_alpha != None:
                sum_i_alpha += i_alpha
            else:
                unset_edge = connected_var
        if unset_edge:
            new_i_alpha = 1 - self.node_coef - sum_i_alpha
            self.set_i_alpha(unset_edge, new_i_alpha)

    def get_i_alpha(self, connection_name):
        return self.i_alpha[connection_name]

    def set_i_alpha(self, connection_name, value):
        self.i_alpha[connection_name] = value

    def cal_cnp_coef(self):
        self.coef_ready = True

        self.hat_c_ialpha = {}
        for item in self.connections:
            hat_c_ialpha = self.node_coef + self.i_alpha[item]
            assert hat_c_ialpha != 0
            self.hat_c_ialpha[item] = hat_c_ialpha

    def get_hat_c_ialpha(self, node_name):
        if self.coef_ready:
            return self.hat_c_ialpha[node_name]

    def get_varnode_extra_term(self, node_name):
        """
        Norm-Product Belief Propagation, n_{i -> alpha} second term
        This term is always 1 in stardard bp
        """
        if node_name not in self.last_innerparenthese4varnode_msg:
            raise RuntimeError(
                f"{node_name} do not have previous msg sent by {self.name}")
        
        if abs(self.i_alpha[node_name]) < 1e-5:
            return np.ones_like(self.last_innerparenthese4varnode_msg[node_name])
        else:
            # TODO when the a^x, a = 0, it has some problem 
            coef_exp = -1.0 * self.i_alpha[node_name] / self.hat_c_ialpha[node_name]
            base = self.last_innerparenthese4varnode_msg[node_name]
            value = np.power(ndarray_denominator(base),coef_exp)
            return value

    def make_message(self, recipient_node):
        assert(recipient_node.name in self.connections)
        if len(self.connections) == 1:
            self.last_innerparenthese4varnode_msg[recipient_node.name] = self.potential
            return self.summation(self.potential, recipient_node)
        else:
            product_out = self.cal_inner_parentheses(recipient_node)

            hat_c_ialpha = self.hat_c_ialpha[recipient_node.name]
            product_out_power = np.power(product_out, 1.0 / hat_c_ialpha)
            message = np.power(self.summation(product_out_power, recipient_node), hat_c_ialpha)
            # FIXME remove me
            if np.isclose(message, 0).all():
                print("debug me")
            return np.power(self.summation(product_out_power, recipient_node),hat_c_ialpha)

    def marginal(self):
        message_val = np.array([message.val for message in self.latest_message])
        prod_messages = np.prod(message_val, axis=0)
        product_out = np.multiply(self.potential, prod_messages)
        return np.power(product_out,1.0/self.node_coef)

    def cal_inner_parentheses(self, recipient_node):
        latest_message = self.latest_message
        filtered_message = [message for message in latest_message
                            if not message.sender.name == recipient_node.name]

        message_val = np.array([message.val for message in filtered_message])
        prod_messages = np.prod(message_val, axis=0)
        # FIXME is this reasonable? 
        prod_messages /= np.mean(prod_messages)
        # FIXME should set a larg enough potential initally
        product_out = np.multiply(self.potential, prod_messages)
        self.last_innerparenthese4varnode_msg[recipient_node.name] = product_out
        if np.isclose(product_out, 0).all():
            print("debug me")
        return product_out

    def store_message(self, message):
        assert message.val.shape == self.potential.shape, \
            f"From {message.sender.name} to {self.name} shape mismatch, \
                expected {self.potential.shape}, recieved {message.val.shape}"
        super().store_message(message)

    def reformat_message(self, message):
        potential_dims = self.potential.shape
        states = message.val
        which_dim = self.connections.index(message.sender.name)

        return expand_ndarray(states, potential_dims, which_dim)

    def summation(self, potential, node):
        potential_dim = potential.shape
        node_index = self.connections.index(node.name)
        out = np.zeros(node.rv_dim)
        assert(potential_dim[node_index] == node.rv_dim)
        for coord in np.ndindex(potential_dim):
            i = coord[node_index]
            out[i] += potential[coord]
        return out

    def to_json(self, separators=(',', ':'), indent=4):
        return json.dumps({
            'class': 'FactorNode',
            'name': self.name,
            'potential': self.potential.tolist(),
            'node_coef': self.node_coef,
            'connections': self.connections
        }, separators=separators, indent=indent)

    def __eq__(self, value):
        if isinstance(value, FactorNode):
            flag = []
            flag.append(self.name == value.name)
            flag.append(np.array_equal(self.potential, value.potential))
            flag.append(np.array_equal(self.node_coef, value.node_coef))
            flag.append(self.connections == value.connections)
            if np.sum(flag) == len(flag):
                return True

        return False

    @classmethod
    def from_json(cls, j):
        d_context = json.loads(j)

        if d_context['class'] != 'FactorNode':
            raise IOError(
                f"Need a FactorNode class json to construct FactorNode \
                instead of {d_context['class']}")

        potential = d_context['potential']
        coef = d_context['node_coef']
        connections = d_context['connections']
        node = cls(connections, np.asarray(potential), coef=coef)

        node.format_name(d_context['name'])

        return node
