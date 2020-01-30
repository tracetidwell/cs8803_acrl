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
from scipy.spatial import Voronoi

import matplotlib.pyplot as plt

map = np.ones((50,50))

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

n,m = map.shape

points_list = []
for x in range(n):
    for y in range(m):
        if map[x, y] == 0:
            points_list.append([x, y])

points = np.array(points_list)

from scipy.spatial import Voronoi, voronoi_plot_2d

vor = Voronoi(points)
fig = voronoi_plot_2d(vor)
#plt.scatter(x=points[:, 0], y=points[:, 1])
#plt.scatter(x=vor.vertices[:, 0], y=vor.vertices[:, 1])
plt.show()
