import json
import time
from functools import partial

import numpy as np

from .base_graph import BaseGraph
from .coef_policy import *
from .factor_node import FactorNode
from .message import Message
from .np_utils import expand_ndarray, reduction_ndarray
from .var_node import VarNode


class GraphModel(BaseGraph):
    def __init__(self, silent=True, epsilon=1, coef_policy=bp_policy):
        super().__init__(silent=silent, epsilon=epsilon, coef_policy=coef_policy)


    def init_cnp_coef(self):
        for node in self.nodes:
            node.reset()
            node.cal_cnp_coef()

    # decay interface
    def belif_p(self,algo = None):
        if algo == None:
            algo = self.norm_product_bp
        elif algo == self.iterative_scaling:
            assert self.coef_policy == bp_policy, f"its+bp must bp coef"
        self.bake()
        return algo()

    def norm_product_bp(self, max_iter=5000000, tolerance=1e-4, error_fun=None):
        self.init_cnp_coef()
        self.first_belief_propagation()
        return self.engine_loop(
            self.parallel_message,
            max_iter,
            tolerance,
            error_fun)

    def engine_loop(
            self,
            engine_fun,
            max_iter=5000000,
            tolerance=1e-2,
            error_fun=None,
            isoutput=False):
        epsilons = [np.inf]
        start = time.time()
        timer_record = []
        step = 0
        cur_marginals = self.export_convergence_marginals()

        while (step < max_iter) and tolerance * len(self.nodes) < epsilons[-1] or step < 5:
            last_marginals = cur_marginals
            step += 1

            try:
                engine_fun()
                cur_marginals = self.export_convergence_marginals()
                if error_fun:
                    epsilons.append(error_fun(cur_marginals, last_marginals))
                else:
                    epsilons.append(
                        self.compare_marginals(cur_marginals, last_marginals))

                timer_record.append(time.time() - start)
                if not self.silent:
                    print(
                        f"epsilon: {epsilons[-1]:5.4f} | step: {step:5d} {'-'*10}")
                    print(cur_marginals)
                    print(last_marginals)
                    print(epsilons)
                if isoutput:
                    print(f'verobose output {epsilons[-1]}')
            except  RuntimeError:
                print("break")
                break

        return epsilons[1:], step, timer_record

    def iterative_scaling(self):
        self.init_cnp_coef()
        self.first_belief_propagation()

        inner_bind = partial(self.parallel_message, False)
        self.engine_loop(inner_bind)

        return self.engine_loop(self.iterative_scaling_outer_loop)

    def iterative_scaling_outer_counting(self):
        try:
            self.iterative_scaling_outer_cnt += 1
        except AttributeError:
            self.iterative_scaling_outer_cnt = 0

        self.iterative_scaling_outer_cnt %= len(self.constrained_nodes)

    def its_next_looplink(self):
        self.iterative_scaling_outer_counting()
        target_node = self.constrained_nodes[self.iterative_scaling_outer_cnt]
        # target_node.sendout_message()

        next_node = self.constrained_nodes[(
            self.iterative_scaling_outer_cnt + 1) % len(self.constrained_nodes)]

        return target_node,self.find_link(target_node, next_node)

    def iterative_scaling_outer_loop(self):
        cur_node, loop_link = self.its_next_looplink()
        inner_fun = partial(self.iterative_scaling_inner_loop, loop_link)

        self.engine_loop(inner_fun)

    def iterative_scaling_inner_loop(self, loop_link):
        if len(loop_link) == 2:
            return

        for sender, reciever in zip(loop_link[0:-1], loop_link[1:]):
            sender.send_message(reciever)

        loop_link.reverse()
        for sender, reciever in zip(loop_link[0:-1], loop_link[1:]):
            sender.send_message(reciever)

    def find_link(self, node_a, node_b):
        a_2root = self.get_node2root(node_a)
        b_2root = self.get_node2root(node_b)
        while (len(b_2root) > 1 and len(a_2root) > 1):
            if b_2root[-1] == a_2root[-1] and b_2root[-2] == a_2root[-2]:
                b_2root.pop()
                a_2root.pop()
            else:
                break
        b_2root.reverse()
        if len(b_2root) >= 2:
            return a_2root + b_2root[1:]
        else:
            return a_2root[:-1] + b_2root[:]

    def get_node2root(self, node):
        rtn = []
        tmp = node
        while True:
            rtn.append(tmp)
            tmp = tmp.parent
            if not tmp:
                break

        return rtn

    def parallel_message(self, run_constrained=True):
        for target_var in self.varnode_recorder.values():
            connected_factor_names = target_var.connections
            connected_factor = [self.node_recorder[name]
                                for name in connected_factor_names]
            # sendind in messages from factors
            target_var.sendin_message(self.silent)

            if run_constrained or (not target_var.isconstrained):
                target_var.sendout_message(self.silent)
