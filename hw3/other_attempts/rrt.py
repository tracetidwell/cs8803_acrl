import numpy as np
import scipy

import time
import queue
import copy
import heapq
import cv2
import math

from scipy.spatial import distance
from bresenham import bresenham

class node():
    def __init__(self, x, y, theta, parent):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None

class car_problem():
    def __init__(self, map, goal, heuristic_only=False, delay=20, alpha=5):
        self.map = map
        self.goal = goal

        self.uset = [-.99, -.8, -.6, -.4, -.3, -.2, 0, .2, .3, .4, .6, .8, .99, -2]
        #self.uset = [-.99, -.5, -.2, 0, .2, .5, .99, -2]

        self.car_length = 3
        self.car_width = 2
        self.res = 5

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

    def dynamics_neighbor_togoal(self, state, target):

        ref_distance = euclid_distance(state, target)

        ref_distance = 100

        transitions = []
        found = False
        for u in self.uset:
            new_state = discretize_state(forward_dynamics(state, u))

            if not self.check_collision(new_state):
                distance = euclid_distance(new_state, target)
                transitions.append((distance, new_state, u))
                found = True

        transitions.sort()

        if found:
            if transitions[0][0] < ref_distance:
                return transitions[0]
            else:
                return None
        else:
            return None

    def check_straight_line_possible(self, nearest, ss):

        pp = bresenham(int(nearest[0]), int(nearest[1]),int(ss[0]), int(ss[1]))
        for x in pp:
            if self.map[x[0], x[1]] == 0:
                return False

        return True

    def rrt(self, position):
        def sample(size=50):
            sample = np.random.uniform(0,size,3)
            X_sample = (round(sample[0],2), round(sample[1],2), ((2*math.pi*sample[2])/size))
            return X_sample

        def nn(graph_list, new_point, k=5):
            min_distance = 100000
            closest = None

            min_list = []
            for compare in graph_list:
                dist = euclid_distance(compare, new_point)
                if dist < min_distance and dist > 2:
                    min_distance = dist
                    closest = compare
                    min_list.append((dist, compare))

            return sorted(min_list)

        vertices = set([])
        vertices.add(position)

        c = 0
        for x in range(0,100000):
            c += 1
            if ((np.random.sample() > .05) or (c < 10)):
                ss = sample()
                while self.check_collision(ss):
                    ss = sample()
            else:
                ss = self.goal

            nearestk = nn(vertices, ss)
            #print(nearestk)
            nearest = nearestk[0][1]

            '''
            dx = ss[0] - nearest[0]
            dy = ss[1] - nearest[1]
            dist = nearestk[0][0]
            print(ss)
            print(nearest)
            print(dx, dy, dist)
            ss = (nearest[0] + dx/np.sqrt(dist), nearest[1] + dy/np.sqrt(dist), 0)

            k = min(50, len(nearestk))
            best_ends = []

            for i in range(0,k):

                nearest_k = nearestk[i]
                nearest = nearest_k[1]

                if self.check_straight_line_possible(nearest, ss):
                    best = self.dynamics_neighbor_togoal(nearest, ss)

                    loops = 0
                    loop_limit = 100
                    prev_best = None

                    while ((best != None) and loops < loop_limit):
                        loops += 1

                        prev_best = best

                        if loops % 1 == 0 and(euclid_distance(nearest, best[1]) > 1):
                            vertices.add(best[1])

                            if self.viz[int(best[1][0]), int(best[1][1])] != .3:
                                self.viz[int(best[1][0]), int(best[1][1])] = .3
                                self.region_viz(self.delay)

                        if self.isgoal(best[1]):
                            print("Steps taken {}".format(c))
                            self.region_viz(0)
                            exit(0)

                        if best[0] < 1:

                            vertices.add(best[1])
                            if self.viz[int(best[1][0]), int(best[1][1])] != .3:
                                self.viz[int(best[1][0]), int(best[1][1])] = .3
                                self.region_viz(self.delay)

                            best = None
                        else:
                            best = self.dynamics_neighbor_togoal(best[1], ss)

                    best_ends.append(prev_best)

            if(len(best_ends) < 1):
                pass
            else:

                if self.viz[int(ss[0]), int(ss[1])] != .6:
                    self.viz[int(ss[0]), int(ss[1])] = .6
                    self.region_viz(self.delay)


                print(sorted(best_ends))
                best = sorted(best_ends)[0]
                vertices.add(best[1])
                if self.viz[int(best[1][0]), int(best[1][1])] != .3:
                    self.viz[int(best[1][0]), int(best[1][1])] = .3
                    self.region_viz(self.delay)


            '''
            #Working Tree
            if self.check_straight_line_possible(nearest, ss):
                #self.viz[int(ss[0]), int(ss[1])] = .6
                #self.region_viz(self.delay)
                best = self.dynamics_neighbor_togoal(nearest, ss)

                loops = 0
                loop_limit = 10
                while ((best != None) and loops < loop_limit):
                    loops += 1

                    if loops % 1 == 0 and(euclid_distance(nearest, best[1]) > 1):
                        vertices.add(best[1])

                        if self.viz[int(best[1][0]), int(best[1][1])] != .3:
                            self.viz[int(best[1][0]), int(best[1][1])] = .3
                            self.region_viz(self.delay)

                    if self.isgoal(best[1]):
                        print("Steps taken {}".format(c))
                        self.region_viz(0)
                        exit(0)

                    if best[0] < 1:
                        vertices.add(best[1])
                        if self.viz[int(best[1][0]), int(best[1][1])] != .3:
                            self.viz[int(best[1][0]), int(best[1][1])] = .3
                            self.region_viz(self.delay)
                        best = None
                    else:
                        best = self.dynamics_neighbor_togoal(best[1], ss)


        self.region_viz(0)

    def region_viz(self, timer):
        map2 = cv2.resize(self.viz,(int(500),int(500)), cv2.INTER_NEAREST)
        cv2.imshow('image',map2)
        cv2.waitKey(timer)
        pass


def forward_dynamics(state, u, times=1):
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

def discretize_state(state, dec=3, spatial_chunk=1, angle_chunk=1):
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
map[0:9, 20] = 0
map[25, 25:42] = 0
map[25, 46:49] = 0

#map[11:38,11:50] = 0

#Gate close:
#map[23:26, 0:12] = 0

goal = (5,5,0)
start = (42, 28, -2)

map[goal[0], goal[1]] = -1
map[start[0], start[1]] = -1

first_map = car_problem(map, goal, heuristic_only=False, delay=1, alpha=50)

first_map.rrt(start)


exit(0)
#for i in range(0, 10):
a, vs, qs = first_map.astar(start)
print("Searched {} Nodes, Final Path {} Steps".format(qs, len(a[1])))

for vis in a[1]:
    map[int(vis[0]), int(vis[1])] = .3
    map2 = cv2.resize(map,(int(500),int(500)), cv2.INTER_NEAREST)
    cv2.imshow('image',map2)
    cv2.waitKey(4)

cv2.waitKey(5000000)
cv2.destroyAllWindows()
