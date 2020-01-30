import numpy as np
import scipy

import time
import queue
import copy
import heapq
import cv2
import math
import random

from scipy.spatial import distance

class car_problem():
    def __init__(self, map, goal, heuristic_only=False, delay=20, alpha=5, uset=0):
        self.map = map
        self.goal = goal

        #Different U sets for searching
        if uset == 1:
            self.uset = [-.99, -.4, 0, .4, .99]
            self.uset = [0, .4, -.4, -.99, .99]
        if uset == 2:
            self.uset = [-.99, -.6, -.2, -.1, 0, .2, .6, .99]
        else:
            self.uset = [-.99, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, .99]


        self.car_length = 3
        self.car_width = 2

        self.res = 5
        self.alpha = alpha
        self.heuristic_map = np.zeros(self.map.shape)
        self.prep_heuristic_map()
        self.heuristic_only = heuristic_only

        self.viz = copy.copy(map)
        self.delay = delay

    #Rough (fast) collision check
    def check_collision(self, state):
        for i in np.arange(-self.car_width/2, self.car_width/2, self.car_width/self.res):
            for j in np.arange(-self.car_length/2, self.car_length/2, self.car_length/self.res):
                testx = int(max(0, min(49, round(state[0] + i))))
                testy = int(max(0, min(49, round(state[1] + j))))

                if self.map[testx, testy] == 0:
                    return True

        return False

    def isgoal(self, state):
        return euclid_distance(state, self.goal) < 1

    #Simple Neighbors for BFS
    def simple_neighbors(self, pos):
        i = pos[0]
        j = pos[1]

        pos_list = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

        neighbors = []
        for p in pos_list:
            neighbors.append(p)

        return neighbors

    #Performs BFS To prepare heuristic map for Astar
    def prep_heuristic_map(self):
        self.heuristic_map = np.ones((100, 100)) * -1

        #Create FIFO QUEUE
        q = queue.Queue()
        intgoal = (int(self.goal[0]*2), int(self.goal[1]*2))
        q.put((0, intgoal))

        #Pop Shallowest available path (breadth)
        while not q.empty():
            #Pop Next path from list, obtain last element
            next = q.get()
            #Iterate Through Children
            for kid in self.simple_neighbors(next[1]):

                #Add Children to visited set
                collkid = (kid[0]/2.0, kid[1]/2.0)
                if not self.check_collision(collkid):
                    if self.heuristic_map[kid[0], kid[1]] == -1:
                        #Create new paths and add to queue - Alphabetically POPPED
                        new_kid = (next[0] + 1, kid)
                        q.put(new_kid)
                        self.heuristic_map[kid[0], kid[1]] = next[0] + 1


        mx = np.max(self.heuristic_map)

        if mx == -1:
            self.heuristic_map*0

    #Account for discretization of heuristic distance
    def heuristic_distance(self, state):
        indexi = int(state[0]*2)
        indexj = int(state[1]*2)
        return math.sqrt(max(1, self.heuristic_map[indexi, indexj]))

    #Dynamically Generate Graph Neighbors based on simulated dynamics
    def dynamics_neighbors(self, state, spatial_chunk, angle_chunk, times):
        transitions = []

        for u in self.uset:
            new_state = discretize_state(forward_dynamics(state, u, times), dec=0, spatial_chunk=spatial_chunk, angle_chunk=angle_chunk)
            traveled = euclid_distance(state, new_state)

            if not self.check_collision(new_state):
                distance = self.heuristic_distance(new_state)*self.alpha
                transitions.append((distance, new_state, traveled, u))

        return transitions

    #Perform Heuristic Astar search
    def astar(self, position, spatial_chunk=5, angle_chunk=8, times=2):
        dist = (self.heuristic_distance(position)*self.alpha , 0)

        q = []
        heapq.heapify(q)
        heapq.heappush(q, (dist, [position], []))

        best_dist = self.heuristic_distance(position)*self.alpha*4
        best_hist = [None]

        visited = set()
        visited.add(position)
        q_pops = 0
        while q:
            next_node = heapq.heappop(q)
            q_pops += 1

            #Timeout if searching too long
            if q_pops > 20000:
                return next_node, visited, q_pops, 0

            if self.viz[int(next_node[1][-1][0]), int(next_node[1][-1][1])] != .3:
                self.viz[int(next_node[1][-1][0]), int(next_node[1][-1][1])] = .3
                '''self.region_viz() TURN ON IF WANT TO VISUALIZE SEARCH '''

            if self.isgoal(next_node[1][-1]):

                return  next_node, visited, q_pops, 1
            else:
                for kid_distance in self.dynamics_neighbors(next_node[1][-1], spatial_chunk=spatial_chunk, angle_chunk=angle_chunk, times=times):
                    kid = kid_distance[1]
                    traveled = kid_distance[2]
                    actu = kid_distance[3]


                    if kid not in visited:
                        dist = kid_distance[0]
                        #Heuristic only
                        if self.heuristic_only == True:
                            priority = (dist, 0)
                        else:
                            priority = (dist + traveled + next_node[0][1], next_node[0][1] + traveled)

                        kid_list = next_node[1][:]
                        kid_list.append(kid)

                        action_list = next_node[2][:]
                        action_list.append(actu)

                        heapq.heappush(q, (priority, kid_list, action_list))
                        visited.add(kid)

                        if dist < best_dist:
                            best_dist = dist
                            best_hist = (priority, kid_list, action_list)

        return next_node, visited, q_pops, 0

    def region_viz(self):
        map2 = cv2.resize(self.viz,(int(500),int(500)), cv2.INTER_NEAREST)
        cv2.imshow('image',map2)
        cv2.waitKey(self.delay)
        pass

#Estimated Forward Dynamics (not the same as once used in run_sim)
def forward_dynamics(state, u, times=2):
    x = state[0]
    y = state[1]
    theta = state[2]

    #Constants
    nom_delta = .6
    max_delta = .2
    theta_reverse = .2
    l_radius = .25
    r_radius = .25
    wheelbase = 2

    if u == -2:
        delta_left = -theta_reverse
        delta_right = -theta_reverse

        R = r_radius * delta_right
        L = l_radius * delta_left

        xp = x + times * (((R + L) / 2)*math.cos(theta))
        yp = y + times * (((R + L) / 2)*math.sin(theta))

        wb_constant = (R - L) / wheelbase
        thetap = theta + times* wb_constant

    else:
        #Wheel Turns
        delta_right = nom_delta + (u*max_delta)
        delta_left = nom_delta - (u*max_delta)

        R = r_radius * delta_right
        L = l_radius * delta_left

        wb_constant = (R - L) / wheelbase

        if (R == L):
            xp = x + times * ((R + L) / 2)*math.cos(theta)
            yp = y + times * ((R + L) / 2)*math.sin(theta)
        else:
            #New State
            diff_constant = (wheelbase / 2) * ((R + L) / (R - L))
            xp = x + times * (diff_constant * (math.sin(wb_constant + theta) - math.sin(theta)))
            yp = y - times * (diff_constant * (math.cos(wb_constant + theta) - math.cos(theta)))

        thetap = theta + times * wb_constant
    return (xp, yp, thetap)

#Discretize state into number of bins
def discretize_state(state, dec=0, spatial_chunk=5, angle_chunk=8):
    #print(dec, spatial_chunk, angle_chunk)
    s0 = round(state[0]*spatial_chunk, dec) / spatial_chunk
    s1 = round(state[1]*spatial_chunk, dec) / spatial_chunk
    s2 = (round(angle_chunk*(state[2] % (2*math.pi)), dec) / angle_chunk)

    return (s0, s1, s2)

#Euclidean distance wrapper
def euclid_distance(s, sp):
    return distance.euclidean((s[0], s[1]),(sp[0], sp[1]))
