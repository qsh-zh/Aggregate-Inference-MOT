import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter

from cbp.factor_node import FactorNode
from cbp.graph_model import GraphModel
from cbp.np_utils import batch_normal_angle
from cbp.var_node import VarNode

from .build_graph import BuildHMM


class BaseGrid(BuildHMM):
    def __init__(self, length, d_row, d_col, d_obs, policy, rand_seed=1):
        super().__init__(length, d_col * d_row, policy, rand_seed=rand_seed)
        self.angle_wind = np.pi/2
        self.d_col = d_col
        self.d_row = d_row
        self.d_obs = d_obs

        self._produce_transition_potential()

    def _produce_transition_potential(self):
        potential = []
        if not hasattr(self, 'destination'):
            self.destination = (self.d_row, self.d_col)
        col, row = np.meshgrid(np.arange(self.d_col), np.arange(self.d_row))

        for cur_row in range(self.d_row):
            for cur_col in range(self.d_col):
                diff_row = row - cur_row
                diff_col = col - cur_col

                angle_goal = np.arctan2(
                    self.destination[0] - 1 - cur_row,
                    self.destination[1] - 1 - cur_col)

                dist_matrix = np.sqrt(np.power(diff_row, 2) \
                    + np.power(diff_col, 2))
                angle_matrix = np.arctan2(diff_row, diff_col)

                wind_matrix = np.abs(
                    batch_normal_angle(
                        angle_matrix -
                        self.angle_wind))
                goal_matrix = np.abs(
                    batch_normal_angle(
                        angle_matrix - angle_goal))

                sum_up = -1 * dist_matrix - 7 * wind_matrix - 10 * goal_matrix
                sum_up[cur_row, cur_col] += 1
                prob = np.exp(sum_up)

                potential.append(prob.flatten() / np.sum(prob))

        unnormalized_potential = np.array(potential).reshape(self.d, self.d)
        self.transition_potential = unnormalized_potential / \
            np.mean(unnormalized_potential)

    def _produce_observation_potential(self):
        raise NotImplementedError()

    def add_observation_factor(self, name_list):
        if not hasattr(self, 'observation_potential'):
            self._produce_observation_potential()
        factornode = FactorNode(name_list, self.observation_potential)
        self.graph.add_factornode(factornode)
        return factornode

    def add_transition_factor(self, name_list):
        if not hasattr(self, 'transition_potential'):
            self._produce_transition_potential()
        factornode = FactorNode(name_list, self.transition_potential)
        self.graph.add_factornode(factornode)

    def init_graph(self):
        self.cal_fix_marginal()
        assert self.fix_marginal.shape == (self.hmm_length, self.d_obs)

        varnode_namelist = []
        for i in range(0, self.hmm_length):
            varnode_namelist.append(f"VarNode_{i:03d}")
            varnode_namelist.append(f"VarNode_{2 * i + 1:03d}")
            self.add_trivial_node()
            self.add_constrained_node(self.fix_marginal[i, :])

        for i in range(0, self.hmm_length - 1):
            edge_connection = [
                f"VarNode_{2 * i:03d}",
                f"VarNode_{2 * i + 1:03d}"]
            self.add_observation_factor(edge_connection)
            edge_connection = [f"VarNode_{2 * i:03d}", f"VarNode_{2 * i+2:03d}"]
            self.add_transition_factor(edge_connection)

        edge_connection = [
            f"VarNode_{2 * self.hmm_length - 2:03d}",
            f"VarNode_{2*self.hmm_length - 1:03d}"]
        self.add_observation_factor(edge_connection)

    def sample_engine(self, state, dim, potential):
        assert state < self.d
        conditional_prob = potential[state, :]
        next_state = self.rng.choice(
            dim, p=conditional_prob / np.sum(conditional_prob))
        return next_state

    def step(self, state):
        return self.sample_engine(state, self.d, self.transition_potential)

    def observe(self, state):
        return self.sample_engine(state, self.d_obs,self.observation_potential)

    def init_stats_sampler(self):
        return 0

    def sample(self, num_sample):
        self.num_sample = num_sample
        traj_recorder = []
        observation_recorder = []
        for _ in range(num_sample):
            states = []
            observations = []
            single_state = self.init_stats_sampler()
            for _ in range(self.hmm_length):
                single_state = self.step(single_state)
                observations.append(self.observe(single_state))
                states.append(single_state)
            traj_recorder.append(states)
            observation_recorder.append(observations)

        self.traj = np.array(traj_recorder).reshape(num_sample, self.hmm_length)
        self.observation = np.array(observation_recorder).reshape(
            num_sample, self.hmm_length)

    def cal_fix_marginal(self):
        self.fix_marginal = self.empirical_marginal(self.observation, self.d_obs)

    def heatmap(self, data, **kwars):
        for i in range(self.d):
            distribution = data[i, :].reshape(self.d_row, self.d_col)
            ax = sns.heatmap(distribution)
            ax.set_title(f"{self.ind2rowcol(i)}_{kwars['title']}")
            fig = ax.get_figure()
            fig.savefig(f"{kwars['path']}_{i}_step.png")
            plt.close(fig)

    def visualize_sensor(self):
        self.heatmap(
            self.observation_potential,
            title="observation_potential",
            path=os.path.expanduser("~/Desktop/cbp/Data/observation_potential"))

    def visualize_transition(self):
        self.heatmap(
            self.transition_potential,
            title="transition_potential",
            path=os.path.expanduser("~/Desktop/cbp/Data/transition_potential"))

    def visualize(self, data, **kwars):
        num_example, time_length = data.shape
        for i in range(time_length):
            locations = data[:, i]
            bins, _ = np.histogram(locations, np.arange(self.d + 1))
            self.visualize_map_bins(bins, f"{kwars['path']}_{i}_step.png")

    def visualize_map_bins(self, bins, fig_name):
        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        xx = []
        yy = []
        xy_size = []
        for xy, cnts in enumerate(bins):
            if cnts > 0:
                row, col = self.ind2rowcol(xy)
                xx.append(col)
                yy.append(row)
                xy_size.append(int(cnts))
        with sns.axes_style("darkgrid"):
            ax = sns.scatterplot(x=xx,
                                 y=yy,
                                 hue=xy_size,
                                 size=xy_size,
                                 sizes=(5, 200),
                                 palette=cmap)

            ax.set_xlim([0, self.d_col])
            ax.set_ylim([0, self.d_row])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.savefig(fig_name)
            plt.close()

    def visualize_bird(self):
        self.visualize(self.traj,
                       **{"title": "bird traj",
                          "path": os.path.expanduser("~/Desktop/cbp/Data/gt")})

    def visualize_observation(self):
        self.visualize(self.observation,
                       **{"title": "bird traj",
                          "path": os.path.expanduser("~/Desktop/cbp/Data/fake")})

    def visualize_estimated(self):
        marginal = self.estimated_marginal()
        for i in range(self.hmm_length):
            bins = self.num_sample * marginal[i, :]
            png_name = os.path.expanduser(
                f"~/Desktop/cbp/Data/estimated_{i}.png")
            self.visualize_map_bins(bins, png_name)

    def empirical_marginal(self, traj, num_bins):
        marginal = []
        for i in range(traj.shape[1]):
            bins, _ = np.histogram(
                traj[:, i], np.arange(num_bins + 1))
            marginal.append(bins / np.sum(bins))

        return np.array(marginal).reshape(traj.shape[1], num_bins)

    def get_gtmarginal(self):
        gt_marginal = self.empirical_marginal(self.traj,self.d)
        assert gt_marginal.shape == (self.hmm_length, self.d)
        return gt_marginal

    def estimated_marginal(self):
        marginal = []
        for i in range(0, self.hmm_length):
            marginal.append(
                self.graph.varnode_recorder[f'VarNode_{2 * i:03d}'].marginal())

        return np.array(marginal).reshape(self.hmm_length, self.d)

    def ind2rowcol(self, index):
        index = np.array(index).astype(np.int64)
        row = (index / self.d_col).astype(np.int64)
        col = index % self.d_col
        return row, col
