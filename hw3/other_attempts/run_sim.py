from maps import load_map
from utils import Params, Map, State, display_environment
from motion_model import motion_model
import numpy as np
import itertools
import cv2
from search import build_graph, a_star, xy_to_node
import time


curr_map = load_map('map_1.pickle')
#curr_map = load_map('map_2.pickle')
#curr_map = load_map('map_3.pickle')

params = Params()

N, M = curr_map.seed_map.shape
x, y = np.meshgrid(range(N), range(M))

DISPLAY_ON = 1
DISPLAY_TYPE = 1

for actual_map in curr_map.map_samples:

    observed_map = np.copy(curr_map.seed_map)
    state = State(curr_map.start.x, curr_map.start.y, 0, 0, params)
    count = itertools.count()
    flags = 0

    goal = curr_map.goal

    if DISPLAY_ON:
        display_environment(curr_map, observed_map, state, params)


    while state.move_count < params.max_move_count and flags != 2:
        next(count)

        #action = astar()
        graph = build_graph(observed_map)
        start_node = xy_to_node(int(state.x), int(state.y), M)
        goal_node = xy_to_node(curr_map.goal.x, curr_map.goal.y, M)
        #print(start_node, goal_node)
        #print(graph[start_node])
        path, cost = a_star(graph, start_node, goal_node, direction=1)
        #print('here')
        action = 0.1

        time.sleep(.01)

        state, observed_map, flags = motion_model(params, state, action, observed_map, actual_map, goal)

        if flags == 1:
            break

        if DISPLAY_ON:
            display_environment(curr_map, observed_map, state, params, graph, path)
