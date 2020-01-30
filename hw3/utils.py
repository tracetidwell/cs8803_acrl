import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2


#import networkx as nx

class Params:

    def __init__(self):
        self.length = 3
        self.width = 2
        self.border = np.array([[self.length/2, self.length/2, -self.length/2, -self.length/2],
                                [self.width/2, -self.width/2, -self.width/2, self.width/2],
                                [1, 1, 1, 1]])
        self.wb = 2
        self.l_radius = 0.25
        self.r_radius = 0.25
        self.d_theta_nom = 0.6
        self.d_theta_max_dev = 0.2
        self.d_theta_reverse = self.d_theta_nom / 3
        self.max_move_count = 100000
        self.observation_radius = 5


class Location:

    def __init__(self, x, y):
        self.x = x
        self.y = y


class State:

    def __init__(self, x, y, theta, move_count, params):
        self.x = x
        self.y = y
        self.theta = theta
        self.H = np.array([[math.cos(self.theta), -math.sin(self.theta), self.x],
                           [math.sin(self.theta), math.cos(self.theta), self.y],
                           [1, 1, 1]])
        self.border = np.matmul(self.H, params.border)
        self.move_count = move_count

    def __str__(self):
        #return 'x: {}, y: {}, theta: {}'.format(self.x, self.y, self.theta)
        return 'x: {:.1f}, y: {:.1f}'.format(self.x, self.y)


class Map:

    def __init__(self, map_name, bridge_locations, bridge_probabilities, seed_map, start, goal, map_samples=None, n_samples=2):
        self.map_name = map_name
        self.bridge_locations = bridge_locations
        self.bridge_probabilities = bridge_probabilities
        self.seed_map = seed_map
        self.start = start
        self.goal = goal
        self.n_samples = 2
        if map_samples:
            self.map_samples = map_samples
        else:
            self.map_samples = []
            for i in range(self.n_samples):
                self.map_samples.append(self.generate_sample())


    def generate_sample(self):
        new_map = np.copy(self.seed_map)
        for i, bridge_location in enumerate(self.bridge_locations):
            probs = [self.bridge_probabilities[i], 1-self.bridge_probabilities[i]]
            new_map[bridge_location] = np.random.choice([1, 0], p=probs)
        return new_map

blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
violet = (180, 0, 180)
yellow = (0, 180, 180)
white = (255, 255, 255)

def display_environment(curr_map, observed_map, state, params, graph=None, path=None):

    plt.ion()
    plt.imshow(observed_map, cmap='gray')
    plt.plot(curr_map.start.x, curr_map.start.y, 'og')
    plt.plot(curr_map.goal.x, curr_map.goal.y, 'or')
    plt.plot(np.concatenate([state.border[0, :], [state.border[0, 0]]]),
            np.concatenate([state.border[1, :], [state.border[1, 0]]]))
    plt.plot(state.x, state.y, 'ob')

    plt.plot([state.x, state.x + params.length/2*math.cos(state.theta)],
             [state.y, state.y + params.length/2*math.sin(state.theta)], 'b')

    if path:
        node_positions = {n: graph.node[n]['pos'] for n in graph.nodes}
        edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
        #nx.draw_networkx_edges(graph, node_positions, edgelist=edges, edge_color='g')


    plt.draw()
    plt.pause(0.0001)
    plt.clf()
