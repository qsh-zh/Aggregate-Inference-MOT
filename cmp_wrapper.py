
import argparse
from compare import build_cmp
from test.grid_square import GridSquare
from test.grid_cgm import GridCgm
from cbp.coef_policy import *

parser = argparse.ArgumentParser(description='build single compare instance')
parser.add_argument('--d_min', type=int)
parser.add_argument('--d_max', type=int)
parser.add_argument('--num_node', type=int)
parser.add_argument('--graph_builder', type=str)
parser.add_argument('--cmp_policy', type=str)
parser.add_argument('--rand_seed', type=int)
parser.add_argument('--pkl_name', type=str)
parser.add_argument('-method', action='append', default=[])

result = parser.parse_args()

graph_builder = {'cgm': GridCgm,
                 'itsbp': GridSquare}

cmp_policy = {'avg': avg_policy,
              'factor': factor_policy,
              'bp': bp_policy,
              'crazy': crazy_policy}

# g_type = graph_type()

build_cmp(result.method,
        result.d_min,
        result.d_max,
        result.num_node,
        graph_builder[result.graph_builder],
        cmp_policy[result.cmp_policy],
        result.rand_seed,
        f"Data/{result.graph_builder}/{result.pkl_name}.pkl")
