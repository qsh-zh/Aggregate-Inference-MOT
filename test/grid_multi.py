from .grid_wifi import GridWifi

class GridMulti(GridWifi):
    def __init__(self, length, d, policy, rand_seed=1):
        super().__init__(length, d, policy, rand_seed=rand_seed)

    def add_observation_factor(self, name_list, sensor_index):
        if not hasattr(self, 'observation_potential'):
            self._produce_observation_potential()
        factornode = FactorNode(name_list, self.observation_potential[sensor_index])
        self.graph.add_factornode(factornode)
        return factornode

    def init_graph(self):
        self.cal_fix_marginal()
        assert self.fix_marginal.shape == (self.hmm_length, self.d_obs, 2)
        
        for i in range(0, self.hmm_length):
            state_node_name = f"VarNode_{(self.d_obs+1)*i:03d}"
            self.add_trivial_node()
            for sensor_ind in range(self.d_obs):
                observation_name = f"VarNode_{(self.d_obs+1)*i + sensor_ind+1:03d}"
                self.add_constrained_node(self.fix_marginal[i, sensor_ind, :])
                self.add_observation_factor([state_node_name, observation_name], sensor_ind)
                
        for i in range(0, self.hmm_length - 1):
            edge_connection = [f"VarNode_{(self.d_obs+1)*i:03d}",
                               f"VarNode_{(self.d_obs+1)*(i+1):03d}"]
            self.add_transition_factor(edge_connection)

    def observe(self, state, sensor_ind):
        return self.sample(state, 1, p=self.observation_potential[sensor_ind][0])
        
    def sample(self, num_sample):
        self.num_sample = num_sample
        traj_recorder = []
        sensor_recorder =[]
        for _ in range(num_sample):
            states = []
            single_state = self.init_stats_sampler()
            agent_sensor_record = []
            for _ in range(hmm_length):
                single_state = self.step(single_state)
                cur_sensor = []
                for sensor_ind in range(self.d_obs):
                    cur_sensor.append(self.observe(single_state, sensor_ind))
                agent_sensor_record.append(cur_sensor)
                states.append(single_state)
            traj_recorder.append(states)
            sensor_recorder.append(agent_sensor_record)
        
        self.traj = np.array(traj_recorder).reshape(num_sample, self.hmm_length)
        self.observation = np.array(observation_recorder).reshape(
            num_sample, self.hmm_length, self.d_obs)
            
    def cal_fix_marginal(self):
        marginal = []
        for time_stamp in range(self.hmm_length):
            cur_marginal = []
            for sensor_ind in range(self.d_obs):
                observed_cnt = np.sum(self.observation[:, i, sensor_ind])
                observed_p = observed_cnt / self.num_sample
                cur_marginal.append([observed_p,1-observed_p])
            marginal.append(cur_marginal)
        
        return np.array(marginal).reshape(self.hmm_length, self.d_obs, 2)
        
    def estimated_marginal(self):
        marginal = []
        for i in range(0, self.hmm_length):
            marginal.append(
                self.graph.varnode_recorder[f'VarNode_{(self.d_obs + 1) * i:03d}'].marginal())

        return np.array(marginal).reshape(self.hmm_length, self.d)
