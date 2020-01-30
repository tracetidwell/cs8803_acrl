from maps import load_map
from utils import Params, Map, State, display_environment
from motion_model import motion_model
import numpy as np
import itertools
import cv2
import time
import math
import copy

from search import build_graph, a_star, xy_to_node
from motion_astar import car_problem, euclid_distance

'''
########################################
SELECT MAP HERE
########################################
'''
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

    #Setup Targets in non-tuple form
    goal_ours = (curr_map.goal.y, curr_map.goal.x)
    state_ours = (curr_map.start.y, curr_map.start.x,  (math.pi/2) - state.theta)

    if DISPLAY_ON:
        display_environment(curr_map, observed_map, state, params)

    #Set Paramaters to ensure proper re-planning
    hist_map = observed_map*0
    pointer = 0
    lim = 0

    while state.move_count < params.max_move_count and flags != 2:
        next(count)
        tf = 3
        '''
        ########################################
        REPLAN IF NEW OBSERVATION OR TIME LIMIT
        ########################################
        '''
        if (pointer >= lim -1) or pointer > 12:
            recalc = True
        else:
            recalc = False

        if (not np.array_equal(hist_map , observed_map)) or recalc:
            hist_map = copy.deepcopy(observed_map)
            aheur = copy.copy(hist_map)

            '''
            ########################################
            BUILD AND SEARCH ROUGH A*
            ########################################
            '''
            start_node = xy_to_node(int(state.x), int(state.y), M)
            goal_node = xy_to_node(curr_map.goal.x, curr_map.goal.y, M)

            graph = build_graph(hist_map, start_node, goal_node, buffer=1)
            path, cost= a_star(graph, start_node, goal_node, direction=1)

            if len(path) == 0:
                graph = build_graph(hist_map, start_node, goal_node, buffer=0)
                path, cost= a_star(graph, start_node, goal_node, direction=1)

            '''
            ########################################
            LIMIT SEARCH REGION FOR FINE CONTROLS A*
            ########################################
            '''
            for p in path[:11]:
                vis = graph.node[p]['pos']
                starti = int(vis[1])
                startj = int(vis[0])
                aheur[starti, startj] = .3

                scope = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4 ,5]
                for i in scope:
                    for j in scope:
                        newi = starti + i
                        newj = startj + j
                        #print(newi, newj)
                        if(newi > 0 and newj > 0 and newi < 50 and newj < 50):
                            aheur[newi, newj] = min(aheur[newi, newj], .3)

            fine_goal = (starti, startj)

            '''
            ########################################
            SNAP TARGET TO GOAL IF NEARBY
            ########################################
            '''
            if euclid_distance(fine_goal, goal_ours) < 2:
                fine_goal = goal_ours
                starti = int(fine_goal[0])
                startj = int(fine_goal[1])

                aheur[starti, startj] = .3
                scope = [-3, -2, -1, 0, 1, 2, 3]
                for i in scope:
                    for j in scope:
                        newi = starti + i
                        newj = startj + j
                        #print(newi, newj)
                        if(newi > 0 and newj > 0 and newi < 50 and newj < 50):
                            aheur[newi, newj] = min(aheur[newi, newj], .3)

            aheur[aheur != .3] = 0
            aheur[aheur == .3] = 1
            aheur[starti, startj] = .3

            '''
            ########################################
            FINE CONTROLS A*
            ########################################
            '''
            second_map = car_problem(aheur, fine_goal, heuristic_only=False, delay=1, alpha=5, uset=2)
            ta, vs, qs, success = second_map.astar(state_ours, spatial_chunk=25, angle_chunk=25, times=tf)

            '''
            ########################################
            REVERSE APPROPRIATELY
            ########################################
            '''
            if success == 0:
                #BETTER PLANNING OF BACKUP ROUTINE - CHOOSE THE ONE THAT WORKS IN SITUATION
                lim = 50
                actions = np.ones(lim+1)*2
                for i in range(0, len(actions)):
                    if (i+1) % 2 == 0:
                        actions[i] = .99
            else:
                lim = len(ta[2])*2
                actions = ta[2]

            pointer = 0
                #print(lim)

        '''
        ########################################
        TAKE ACTION, MOVE POINTER, ITERATE
        ########################################
        '''
        action = -actions[int(pointer/tf)]
        pointer += 1

        time.sleep(.01)

        state, observed_map, flags = motion_model(params, state, action, observed_map, actual_map, goal)
        state_ours = (state.y, state.x,  (math.pi/2) - state.theta)

        if flags == 1:
            break

        if DISPLAY_ON:
            display_environment(curr_map, observed_map, state, params, graph=graph, path=path)


cv2.waitKey(5000000)
cv2.destroyAllWindows()
