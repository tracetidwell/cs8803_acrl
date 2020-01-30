import numpy as np
import scipy

import time
import queue
import copy
import heapq
import cv2
import math

from scipy.spatial import distance



class car_problem():
    def __init__(self, map, goal, heuristic_only=False, delay=20, alpha=5):
        self.map = map
        self.goal = goal

        self.uset = [-.99, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, .99, -2]
        #self.uset = [-.99, -.5, -.2, 0, .2, .5, .99, -2]

        self.car_length = 3
        self.car_width = 2

        self.res = 5
        self.alpha = alpha
        self.heuristic_map = np.zeros(self.map.shape)
        self.prep_heuristic_map()
        self.heuristic_only = heuristic_only

        self.viz = copy.copy(map)
        self.delay = delay

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

    def simple_neighbors(self, pos):
        i = pos[0]
        j = pos[1]

        pos_list = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

        neighbors = []
        for p in pos_list:
            if map[p[0],p[1]] != 0:
                neighbors.append(p)

        return neighbors

    def prep_heuristic_map(self):
        self.heuristic_map = np.ones(self.map.shape) * -1

        #Create FIFO QUEUE
        q = queue.Queue()
        q.put((0, self.goal))

        #Pop Shallowest available path (breadth)
        while not q.empty():
            #Pop Next path from list, obtain last element
            next = q.get()
            #print(next)
            #Iterate Through Children
            for kid in self.simple_neighbors(next[1]):
                #print(kid)
                #Add Children to visited set
                if not self.check_collision(kid):
                    if self.heuristic_map[kid[0], kid[1]] == -1:
                        #Create new paths and add to queue - Alphabetically POPPED
                        new_kid = (next[0] + 1, kid)
                        q.put(new_kid)
                        self.heuristic_map[kid[0], kid[1]] = next[0] + 1


    def heuristic_distance(self, state):
        return math.sqrt(max(1, self.heuristic_map[round(state[0]), round(state[1])]))

    def dynamics_neighbors(self, state):
        transitions = []
        for u in self.uset:
            new_state = discretize_state(forward_dynamics(state, u))
            traveled = euclid_distance(state, new_state) + math.sqrt(abs(u))*.5
            #traveled = abs(u)

            if not self.check_collision(new_state):
                distance = self.heuristic_distance(new_state)*self.alpha
                transitions.append((distance, new_state, traveled, u))

        return transitions

    def astar(self, position):
        dist = (self.heuristic_distance(start)*self.alpha , 0)

        q = []
        heapq.heapify(q)
        heapq.heappush(q, (dist, [position], []))

        visited = set()
        visited.add(start)
        q_pops = 0
        while q:
            next_node = heapq.heappop(q)
            q_pops += 1

            if self.viz[int(next_node[1][-1][0]), int(next_node[1][-1][1])] != .3:
                self.viz[int(next_node[1][-1][0]), int(next_node[1][-1][1])] = .3
                self.region_viz()

            if self.isgoal(next_node[1][-1]):
                return  next_node, visited, q_pops
            else:
                for kid_distance in self.dynamics_neighbors(next_node[1][-1]):
                    dist = kid_distance[0]
                    kid = kid_distance[1]
                    traveled = kid_distance[2]
                    actu = kid_distance[3]

                    if kid not in visited:
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

        print("FAILED TO FIND SOLUTION")
        return next_node, visited, q_pops

    def region_viz(self):
        map2 = cv2.resize(self.viz,(int(500),int(500)), cv2.INTER_NEAREST)
        cv2.imshow('image',map2)
        cv2.waitKey(self.delay)
        pass


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
        times = 2
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

def discretize_state(state, dec=0, spatial_chunk=5, angle_chunk=8):
    s0 = round(state[0]*spatial_chunk, dec) / spatial_chunk
    s1 = round(state[1]*spatial_chunk, dec) / spatial_chunk
    s2 = (round(angle_chunk*(state[2] % (2*math.pi)), dec) / angle_chunk)

    #print((s0, s1, s2))
    #time.sleep(.1)
    return (s0, s1, s2)

def euclid_distance(s, sp):
    return distance.euclidean((s[0], s[1]),(sp[0], sp[1]))


map = np.ones((50,50))

#Standard MAP 1
map[0] = 0
map[49] = 0
map[:, 0] = 0
map[:, 49] = 0
map[11:38, 11:38] = 0

#Obstacles
map[23:26, 0:4] = 0
map[23:26, 9:12] = 0

map[15:20, 0:8] = 0
map[15:20, 11:12] = 0

map[29:32, 0:1] = 0
map[29:32, 5:12] = 0

map[38:46, 20] = 0
map[0:8, 20] = 0
map[25, 25:42] = 0
map[25, 46:49] = 0

#map[11:38,11:50] = 0

goal = (5,5)
start = (42, 35, -2)

map[goal[0], goal[1]] = -1
map[start[0], start[1]] = -1

first_map = car_problem(map, goal, heuristic_only=False, delay=1, alpha=50)
a, vs, qs = first_map.astar(start)
print("Searched {} Nodes, Final Path {} Steps".format(qs, len(a[1])))

pathviz = copy.copy(map)
state = start
print(len(a[2]))
print(len(a[1]))

'''
for i, vis in enumerate(a[1][:300]):
    starti = int(vis[0])
    startj = int(vis[1])
    pathviz[starti, startj] = .3

    print(a[2][i])

    map2 = cv2.resize(pathviz,(int(500),int(500)), cv2.INTER_NEAREST)
    cv2.imshow('image',map2)
    cv2.waitKey(10)
'''


for vis in a[2][:418]:
    #print(vis)
    state = forward_dynamics(state, vis, times=2)
    #print(state)

    statei = int(state[0])
    statej = int(state[1])
    pathviz[statei, statej] = .3

    map2 = cv2.resize(pathviz,(int(500),int(500)), cv2.INTER_NEAREST)
    cv2.imshow('image',map2)
    cv2.waitKey(3)


exit(0)
for i in range (0,10):
    first_map = car_problem(map, goal, heuristic_only=False, delay=1, alpha=30)

    #for i in range(0, 10):
    a, vs, qs = first_map.astar(start)
    print("Searched {} Nodes, Final Path {} Steps".format(qs, len(a[1])))

    aheur = copy.copy(map)

    for vis in a[1][:100]:
        starti = int(vis[0])
        startj = int(vis[1])
        aheur[starti, startj] = .3

        scope = [-2, -1, 0, 1, 2]
        for i in scope:
            for j in scope:
                newi = starti + i
                newj = startj + j
                #print(newi, newj)
                if(newi > 0 and newj > 0 and newi < 50 and newj < 50):
                    aheur[newi, newj] = min(aheur[newi, newj], .3)

        map2 = cv2.resize(aheur,(int(500),int(500)), cv2.INTER_NEAREST)
        cv2.imshow('image',map2)
        cv2.waitKey(4)

    aheur[aheur != .3] = 0
    aheur[aheur == .3] = 1
    aheur[starti, startj] = .3

    map2 = cv2.resize(aheur,(int(500),int(500)), cv2.INTER_NEAREST)
    cv2.imshow('image',map2)
    cv2.waitKey(400)

    fine_goal = (starti, startj)
    if euclid_distance(fine_goal, goal) < 2:
        fine_goal = goal

    second_map = car_problem(aheur, fine_goal, heuristic_only=False, delay=1, alpha=50)
    a, vs, qs = second_map.astar(start)
    print("Searched {} Nodes, Final Path {} Steps".format(qs, len(a[1])))

    for vis in a[1][:100]:
        starti = int(vis[0])
        startj = int(vis[1])
        aheur[starti, startj] = .3

        map2 = cv2.resize(aheur,(int(500),int(500)), cv2.INTER_NEAREST)
        cv2.imshow('image',map2)
        cv2.waitKey(4)


    if euclid_distance(a[1][-1], goal) < 2:
        print("MADE IT TO THE GOAL")
        break

    start = a[1][-3]


cv2.waitKey(5000000)
cv2.destroyAllWindows()
