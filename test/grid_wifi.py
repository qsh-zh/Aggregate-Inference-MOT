import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .base_grid import BaseGrid


class GridWifi(BaseGrid):
    def __init__(self, length, d, policy, rand_seed=1):
        # self.destination = (d/2,d)
        super().__init__(length, d, d, 0, policy, rand_seed=rand_seed)
        self.hotspot = []

    def register_hotspot(self, row, col):
        self.d_obs += 1
        self.hotspot.append((row, col))

    def compile(self):
        self._produce_observation_potential()

    def _produce_observation_potential(self):
        potential = []
        for cur_row in range(self.d_row):
            for cur_col in range(self.d_col):
                cur_potential = np.zeros(self.d_obs)

                for i, sensor in enumerate(self.hotspot):
                    distance = np.linalg.norm(
                        [cur_row - sensor[0], cur_col - sensor[1]])
                    cur_potential[i] = np.min([0.99, 1.5 * np.exp(-distance)])

                potential.append(cur_potential)

        self.observation_potential = np.array(
            potential).reshape(self.d, self.d_obs)

    def init_stats_sampler(self):
        if np.random.normal() > 0:
            return 0
        else:
            return int(self.d_col / 2)

    def fixed_sensors(self, d=1):
        for cur_row in range(0, self.d_row, d):
            for cur_col in range(0, self.d_col, d):
                self.register_hotspot(cur_row, cur_col)

    def random_sensor(self, num):
        for _ in range(num):
            row = self.rng.uniform(0, self.d_row - 1)
            col = self.rng.uniform(0, self.d_col - 1)
            self.register_hotspot(row,col)
            

    def draw_sensor(self):
        sensor_row = []
        sensor_col = []
        for sensor in self.hotspot:
            sensor_row.append(sensor[0])
            sensor_col.append(sensor[1])
        
        col, row = np.meshgrid(np.arange(self.d_col), np.arange(self.d_row))
        
        total_x = np.concatenate((col.flatten(), sensor_col), axis=0)
        total_y = np.concatenate((row.flatten(), sensor_row), axis=0)
        style = ['state'] * col.size + ['sensor'] * len(sensor_row)
        ax = sns.scatterplot(x=total_x,
                            y=total_y,
                            style=style,
                            markers={'state': 's', 'sensor': 'X'})
        plt.savefig(os.path.expanduser("~/Desktop/cbp/Data/sensor.png"))
        plt.close()

