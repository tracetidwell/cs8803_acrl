import heapq
import itertools
from heapq import heappush, heappop
import numpy as np
import pickle
import itertools
import os
from collections import OrderedDict, Counter
import networkx as nx
import copy
#from utils import *


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.


    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self, queue=None):
        """Initialize a new Priority Queue."""
        if queue:
            self.queue = queue
            heapq.heapify(self.queue)
        else:
            self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        return heapq.heappop(self.queue)

    def remove_node(self, node):
        self.queue.remove([item for item in self.queue if item[1] == node][0])
        heapq.heapify(self.queue)

    def remove(self, item):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node (int): Index of node in queue.
        """
        self.queue.remove(item)
        #heapq.heapify(self.queue)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def add(self, node):
        self.queue.append(node)

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        heapq.heappush(self.queue, node)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [item[1] for item in self.queue]


    def contains_node(self, node):
        """
        Containment Check operator for 'in'

        Args:
            node: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        # return key in [n for _, n in self.queue]
        return node in self.queue

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top_val(self):
        return self.queue[0][0]

    def is_empty(self):
        return self.size() == 0

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """
        if len(self.queue) == 0:
            return float('inf'), None, None
        else:
            return self.queue[0]


def euclidean_dist_heuristic(graph, u, v):
    """
    Returns Euclidean distance between two nodes

    Parameters
    ----------
    graph : ExplorableGraph
            Undirected graph to search
    u : int
        Node for which to find distance
    v : int
        Node for which to find distance

    Returns
    -------
    distance : float
        Euclidean distance between two nodes

    """
    x1, y1 = graph.node[u]['pos']
    x2, y2 = graph.node[v]['pos']

    return ((y2-y1)**2 + (x2-x1)**2)**0.5

def euclidean_dist_danger_zone(graph, u, v):
    x1, y1 = graph.node[u]['pos']
    x2, y2 = graph.node[v]['pos']
    #print("danger node")
    d = (((y2-y1)**2 + (x2-x1)**2)**0.5) + graph.node[u]['dz'] + graph.node[v]['dz']
    print(x1, y1, d)
    return d


def a_star(graph, start, goal, heuristic_fn=euclidean_dist_heuristic, direction=2):
    """
    Returns optimal path between start and goal nodes

    Parameters
    ----------
    graph : ExplorableGraph
            Undirected graph to search
    start : int
            Node from which to start search
    goal : int
            Node at which to end search
    heuristic_fn: func, default euclidean_distance()
            Function used to calculate heruistic distance
    direction: {1, 2}, default 2
            - 1 is for unidirectional search, going only from start to goal
            - 2 is for bidirectional search, searching from both start and goal

    Returns
    -------
    path : [int]
            Optimal path as list of ints representing order in which nodes are visited
    cost : float
            Cost of optimal path as sum of edge weights of optimal path
    explored_nodes : dict
            Contains nodes as keys and number of times each node was explored as values
    """

    # Initialize variables
    mu = float('inf')
    goals = [start, goal]
    preds = {goal: {} for goal in goals}
    costs = {goal: {goal: 0} for goal in goals}
    visiteds = {goal: set() for goal in goals}
    queues = {goal: PriorityQueue() for goal in goals}
    min_costs = {pair: float('inf') for pair in itertools.permutations(goals, 2)}

    # If start and goal are same, return empty list with path 0
    if start == goal:
        return [], mu, explored, snapshots

    # Initialize frontiers based on uni- or bi-directional search
    for i in range(direction):
        queues[goals[i]].append((0, goals[i]))

    # While the smallest item in the frontier is less than the cost of the best path found, search
    while min([queue.top()[0] for queue in queues.values()]) < mu:
        _, start = min([(queue.top(), start) for start, queue in queues.items() if queue.size() > 0])
        val, node = queues[start].pop()
        visiteds[start].add(node)
        goal = goals[goals.index(start) - 1]
        path = []

        # If the node is the goal, a path has been found
        if node == goal:
            path.append(goal)

            # Iterate through predecessors until start node reached
            while path[-1] != start:
                path.append(preds[start][path[-1]])
            path = path[::-1]

            opt_path = list(path)
            mu = costs[start][node]

            return opt_path, mu

        # If the node has been visited by the other search
        elif node in visiteds[goal]:
            # Crossover points are all nodes in current search's frontier that are also in opposite
            # search's frontier or visited set
            goal_frontier = [item[1] for item in queues[goal].queue]
            crossover_points = visiteds[start].intersection(visiteds[goal].union(goal_frontier))
            # For each crossover node, a path is formed
            for crossover_point in crossover_points:
                # If cost to crossover node from both searches is less than current minimum, update
                # minimum to new cost and initialize a path
                if costs[start][crossover_point] + costs[goal][crossover_point] < mu:
                    mu = costs[start][crossover_point] + costs[goal][crossover_point]
                    path = [crossover_point]  # [crossover_point]

            # Iterate through predecessors until start node reached, then reverse partial path
            if path:
                while path[-1] != start:
                    path.append(preds[start][path[-1]])
                path = path[::-1]

                # Iterate through predecessors until goal node reached
                while path[-1] != goal:
                    path.append(preds[goal][path[-1]])
                opt_path = list(path)

        else:
            # For every neighbor of the current node being visited
            #print(node)
            for neighbor, edge in graph[node].items():
                # If the neighbor is in the frontier and a lower cost for it has been found,
                # remove it from the frontier
                if neighbor in queues[start] and costs[start][node] + edge['weight'] < costs[start][neighbor]:
                    queues[start].remove_node(neighbor)
                # If the neighbor has not been visited and is not in the frontier, add it, add the current node
                # as its predecessor, and update the cost to visit the neighbor
                if neighbor not in visiteds[start] and neighbor not in queues[start]:
                    preds[start][neighbor] = node
                    costs[start][neighbor] = costs[start][node] + edge['weight']
                    queues[start].append((costs[start][neighbor] + heuristic_fn(graph, neighbor, goal), neighbor))

    return [], mu


def xy_to_node(x, y, M):
    return x*M + y

#Builda a buffer around the map - Didnt work
def buffer_map(map):
    buffer = copy.copy(map)

    nodes = []

    steps = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, 1), (-1, 1), (1, -1)]

    steps2 = [(-2, 0), (2, 0), (0, -2), (0, 2),
            (-2, -2), (2, 2), (-2, 2), (2, -2)]

    N, M = buffer.shape
    for i in range(N):
        for j in range(M):
            waller = False
            for s in steps:
                newi = i + s[0]
                newj = j + s[1]

                if(newi > 0 and newj > 0 and newi < N and newj < M):
                    if buffer[newi, newj] == 0:
                        waller = True

            for s in steps2:
                newi = i + s[0]
                newj = j + s[1]

                if(newi > 0 and newj > 0 and newi < N and newj < M):
                    if buffer[newi, newj] == 0:
                        waller = True

            if waller == True:
                nodes.append((i*M + j, {'pos': (i, j), 'dz':100000000000}))
            else:
                nodes.append((i*M + j, {'pos': (i, j), 'dz':0}))

    return nodes

#Builds computational graph to search from our observed map
def build_graph(observed_map, start_node=0, goal_node=0, buffer=1):
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, 1), (-1, 1), (1, -1)]

    N, M = observed_map.shape


    nodes = [(i*M + j, {'pos': (i, j)}) for i in range(N) for j in range(M)]
    #nodes = buffer_map(observed_map)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for node in graph.nodes:
        x, y = graph.node[node]['pos']
        edge_pos = [(np.clip(x + del_x, 0, N-1), np.clip(y + del_y, 0, M-1)) for del_x, del_y in steps]
        edges = [(node, i*M + j, 1) for i, j in edge_pos if i*M + j != node]
        graph.add_weighted_edges_from(edges)

    x_wall, y_wall = np.where(observed_map==0)
    remove_nodes = [xy_to_node(y, x, M) for x, y in zip(x_wall, y_wall)]
    graph.remove_nodes_from(remove_nodes)

    #Remove nodes adjacent to walls
    if buffer == 1:
        danger_start = False
        for s in steps:
            remove_nodes = []
            for x, y in zip(x_wall, y_wall):
                xy = xy_to_node(y + s[1], x + s[0], M)
                if not(xy_to_node(y + s[1], x + s[0], M) == start_node):
                    remove_nodes.append(xy)
                else:
                    danger_start = True

            graph.remove_nodes_from(remove_nodes)

    return graph
