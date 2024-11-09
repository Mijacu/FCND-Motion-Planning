from queue import PriorityQueue
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString
import numpy as np
import networkx as nx
import time

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min), int(north_max), int(east_max)

class Poly:

    def __init__(self, coords, height):
        self._polygon = Polygon(coords)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coords(self):
        return list(self._polygon.exterior.coords)[:-1]
    
    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return (self._polygon.centroid.x, self._polygon.centroid.y)

    def contains(self, point):
        point = Point(point)
        return self._polygon.contains(point)

    def crosses(self, other):
        return self._polygon.crosses(other)


def extract_polygons(data, safety_distance):

    polygons = []
    # t1 = time.time()
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        
        obstacle = [north - d_north - safety_distance, north + d_north + safety_distance, east - d_east - safety_distance, east + d_east + safety_distance]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
        
        height = alt + d_alt + safety_distance

        p = Poly(corners, height)
        polygons.append(p)
    # print(f'elapsed time: {time.time() - t1}')
    return polygons

class Planner:

    def __init__(self, data, num_samples, target_altitude, safety_distance):
        print("---Init Sampler")
        self._polygons = extract_polygons(data, safety_distance)
        self._xmin = np.min(data[:, 0] - data[:, 3])
        self._xmax = np.max(data[:, 0] + data[:, 3])

        self._ymin = np.min(data[:, 1] - data[:, 4])
        self._ymax = np.max(data[:, 1] + data[:, 4])

        # self._zmin = 0
        # limit z-axis
        # self._zmax = 20
        
        # Record maximum polygon dimension in the xy plane
        # multiply by 2 since given sizes are half widths
        # This is still rather clunky but will allow us to 
        # cut down the number of polygons we compare with by a lot.
        self._max_poly_xy = 2 * np.max((data[:, 3], data[:, 4]))
        centers = np.array([p.center for p in self._polygons])
        self._polygons_tree = KDTree(centers, metric='euclidean')
        self._num_samples = num_samples
        self.target_altitude = target_altitude
        self.safety_distance = safety_distance
        self.sampled = False
        # for each node connect try to connect to k nearest nodes
        self._k = 5
        self._nodes = list()

    @property
    def polygons(self):
        return self._polygons
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def graph(self):
        return self._graph

    def sample(self, start_position, goal_position):
        """Implemented with a k-d tree for efficiency."""
        print("---Sample")
        if self.sampled:
            return
        pts = [start_position, goal_position]
        while len(pts) < self._num_samples:
            # print("Sample: ", s)
            s = (np.random.uniform(self._xmin, self._xmax), np.random.uniform(self._ymin, self._ymax))
            in_collision = False
            idxs = list(self._polygons_tree.query_radius(np.array([s[0], s[1]]).reshape(1, -1), r=self._max_poly_xy + self.safety_distance)[0])
            if len(idxs) == 0:
                # print("Not indices")
                pts.append(s)
                continue
            for ind in idxs: 
                p = self._polygons[int(ind)]
                if p.contains(s) and p.height >= self.target_altitude:
                    # print("In collision!")
                    in_collision = True
                    break
            if not in_collision:
                # print("Not in collision!")
                pts.append(s)
        self._nodes = pts

    def create_graph(self):
        print("---Create Graph")
        graph = nx.Graph()
        # print('Nodes tree')
        tree = KDTree(self._nodes)
        for n1 in self._nodes:
            # print('node: ', n1)
            # for each node connect try to connect to k nearest nodes
            idxs = tree.query([n1], self._k, return_distance=False)[0]
            # print('idxs: ', idxs)
            for idx in idxs:
                n2 = self._nodes[idx]
                if n2 == n1:
                    # print('Same node!')
                    continue   
                if self.__can_connect(n1, n2):
                    # print('Can connect!')
                    # print("node 1: ", n1)
                    # print("node 2: ", n2)
                    weight = np.linalg.norm(np.array(n2) - np.array(n1))
                    # print("weight: ", weight)
                    graph.add_edge(n1, n2, weight=weight)
        self._graph = graph

    def a_star(self, start, goal):
        """Modified A* to work with NetworkX graphs."""
        path = []
        queue = PriorityQueue()
        queue.put((0, start))
        visited = set(start)

        branch = {}
        found = False
        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            if current_node == start:
                current_cost = 0.0
            else:              
                current_cost = branch[current_node][0]
            if current_node == goal:        
                print('Found a path.')
                found = True
                break
            for next_node in self.graph[current_node]:
                if next_node in visited: 
                    continue  
                visited.add(next_node)
                cost = self.graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost             
                branch[next_node] = (branch_cost, current_node)
                queue_cost = branch_cost + self.heuristic(next_node, goal)             
                queue.put((queue_cost, next_node))
        if not found:
            print("Path was not found!")
            return list(), 0
        elif len(branch) == 0:
            print("We are already in the goal position. A path is not necessary.")
            return list(), 0
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while n != start:
            path.append(n)
            n = branch[n][1]
        path.append(n)

        return path[::-1], path_cost

    def heuristic(self, position, goal_position):
        return np.linalg.norm(np.array(position) - np.array(goal_position))

    def __can_connect(self, n1, n2):
        # print('---Can connect')
        l = LineString([n1, n2])
        # print('Polygons size: ', len(self._polygons))
        # print(f"n1: {n1}, n2: {n2}")
        nodes_distance = np.linalg.norm(np.array([n1[0], n1[1]]) - np.array([n2[0], n2[1]]))
        # print(f"nodes distance: {nodes_distance}")
        query_radius = np.ceil(nodes_distance + self._max_poly_xy + self.safety_distance)
        # print(f"query radius: {query_radius}")
        center = np.array([(n1[0] + n2[0] // 2), (n1[1] + n2[1] // 2)]).reshape(1, -1)
        # print(f"center: {center}")
        idxs = list(self._polygons_tree.query_radius(center, r=query_radius)[0])
        # print(f"idxs: {idxs}")
        if len(idxs) == 0:
            # print("idxs is empty")
            return True
        for ind in idxs: 
            p = self._polygons[int(ind)]
            if p.crosses(l) and p.height >= self.target_altitude:
                return False
        return True