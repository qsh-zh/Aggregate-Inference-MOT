import gc
import math
import pickle
import sys
import time
from functools import partial
from test.grid_square import GridSquare

import numpy as np

from cbp.coef_policy import *

sys.setrecursionlimit(1500)


gc.disable()


class Compare(object):
    def __init__(self, num_node, d, policy, cls_class, rand_seed=1):
        self.builder = cls_class(num_node, d, policy, rand_seed=rand_seed)
        self.builder.sample(1000)
        self.graph = self.builder()
        self.cmp_book = {"num_node": num_node, "d": d}

    def run_engine(self, func):
        start = time.time()
        _, step, _ = func()
        elapsed = time.time() - start
        self.cmp_book[f'elasped_time'] = elapsed
        self.cmp_book[f'step'] = step
        # self.cmp_book[f'timestamp'] = timestamp
        self.cmp_book[f'gt_marginal'] = self.builder.get_gtmarginal()
        self.cmp_book[f'estimated'] = self.builder.estimated_marginal()
        self.cmp_book[f'real_traj'] = self.builder.traj
        self.cmp_book[f'observation'] = self.builder.observation

    def run_itsbp(self):
        worker = partial(self.graph.belif_p, self.graph.iterative_scaling)
        self.run_engine(worker)

        return self.cmp_book

    def run_cgm(self):
        self.run_engine(self.graph.belif_p)

        return self.cmp_book


def run_cmp(cmp_instance, methods):
    if methods == "cgm":
        return cmp_instance.run_cgm()
    elif methods == "itsbp":
        return cmp_instance.run_itsbp()


def build_cmp(
        method_dict,
        d_min,
        d_max,
        num_node,
        graph_builder,
        cmp_policy,
        rand_seed,
        pkl_name):
    record_board = []

    for d in range(d_min, d_max):
        start = time.time()
        cmp_record = {}
        for name in method_dict:
            cmp_instance = Compare(
                num_node, d, cmp_policy, graph_builder, rand_seed)
            cmp_record[name] = run_cmp(cmp_instance, name)

        elapsed = time.time() - start
        print(
            f"{elapsed:010.4f} {':'*10} run {num_node:05d} \
                nodes with {d:05d} grid")
        record_board.append(cmp_record)
        cmp_instance = None
        gc.collect()

        with open(pkl_name, 'wb') as handle:
            pickle.dump(
                record_board,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
