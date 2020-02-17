import numpy as np
from .grid_square import GridSquare
from .cgmodel import CGModel


class GridCgm(GridSquare):
    def __init__(self, length, d, policy, sigma_val=0.3, rand_seed=1):
        super().__init__(length, d, policy, sigma_val=sigma_val, rand_seed=rand_seed)
        self.graph = CGModel(True, coef_policy=policy)

    def init_graph(self):
        gt_node_marginal = self.get_gtmarginal()
        assert gt_node_marginal.shape == (self.hmm_length, self.d)
        assert self.observe_cnt.shape == (self.hmm_length, self.d)
        varnode_namelist = []

        for i in range(0, self.hmm_length):
            varnode_namelist.append(f"VarNode_{i:03d}")
            node_observed_cnt = self.observe_cnt[i, :]
            self.add_trivial_node(node_observed_cnt , gt_node_marginal[i,:])

        for i in range(0, self.hmm_length - 1):
            edge_connection = [
                f"VarNode_{i:03d}",
                f"VarNode_{i+1:03d}"
            ]
            self.add_observation_factor(edge_connection)

        self.graph.num_sample = self.num_sample

    def add_trivial_node(self, observation, gt_distribution):
        varnode = super().add_trivial_node()
        varnode.observation = observation
        varnode.gt_dist = gt_distribution
        return varnode

    def add_observation_factor(self, name_list):
        factornode = super().add_observation_factor(name_list)
        factornode.original_potential = factornode.potential

    def observe_poison(self, traj_summary: np.ndarray) -> np.ndarray:
        assert traj_summary.shape[1] == self.hmm_length
        cnt_bins = []
        for i in range(traj_summary.shape[1]):
            bins, _ = np.histogram(traj_summary[:, i], np.arange(self.d + 1))
            observed = [self.rng.poisson(item, size=1) for item in bins]
            cnt_bins.append(observed)

        return np.array(cnt_bins).reshape(self.hmm_length,-1)

    def sample(self, num_sample):
        self.num_sample = num_sample
        traj_recorder = []
        for _ in range(num_sample):
            states = []
            single_state = 0
            for _ in range(self.hmm_length):
                single_state = self.step(single_state)
                states.append(single_state)
            traj_recorder.append(states)

        self.traj = np.array(traj_recorder).reshape(num_sample, self.hmm_length)
        self.observe_cnt = self.observe_poison(self.traj)
        # TODO
        self.observation = self.observe_cnt

    def estimated_marginal(self):
        marginal = []
        for i in range(0, self.hmm_length):
            marginal.append(
                self.graph.varnode_recorder[f'VarNode_{i:03d}'].marginal())

        return np.array(marginal).reshape(self.hmm_length, self.d)
