import argparse
import time
import msgpack
from enum import Enum, auto
import random
import numpy as np
import matplotlib.pyplot as plt

from planning_utils import Planner, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.data = np.loadtxt('colliders.csv', delimiter=',', dtype='float64', skiprows=2)
        self.num_samples = 800
        self.target_altitude = 40
        self.safety_distance = 8
        self.planner = Planner(self.data, self.num_samples, self.target_altitude, self.safety_distance)

        # Define a grid for a particular altitude and safety margin around obstacles
        self.grid, self.n_min, self.e_min, self.n_max, self.e_max = create_grid(self.data, self.target_altitude, self.safety_distance)
        print("North min = {0}, East min = {1}, North max = {2}, East max = {3}".format(self.n_min, self.e_min, self.n_max, self.e_max))
        
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.error = 1.0

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    path, _ = self.plan_path()
                    self.draw_fig(path)
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if not self.armed and not self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def at_slow_velocity(self):
        return np.linalg.norm(self.local_velocity) < self.error
    
    def plan_path(self):
        self.flight_state = States.PLANNING

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = np.loadtxt('colliders.csv', delimiter= ',', dtype='str', max_rows=1) 
        lat0 = float(lat0.split()[1])
        lon0 = float(lon0.split()[1])

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)
        # TODO: retrieve current global position
        global_position = self.global_position
        # TODO: convert to current local position using global_to_local()
        local_position = global_to_local(global_position, self.global_home)
        print('global home {0}, global position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Define starting point on the grid (this is just grid center)
        # TODO: convert start position to current position rather than map center
        self.start_position = (int(local_position[0]), int(local_position[1]))
        
        # Set goal as some arbitrary position on the grid
        # TODO: adapt to set goal as latitude / longitude position and convert
        goal_positions = list()
        # global_goal - np.array([-122.395606, 37.793719, 5])
        goal_positions.append(np.array([-122.402571, 37.794815, self.target_altitude])) # Justin Herman Plaza
        goal_positions.append(np.array([-122.396064, 37.793991, self.target_altitude])) # Corner of Drumm and California
        goal_positions.append(np.array([-122.397222, 37.795100, self.target_altitude])) # A few blocks over
        goal_positions.append(np.array([-122.397614, 37.796688, self.target_altitude])) 
        goal_position = random.choice(goal_positions)
        print(f"Goal position: {goal_position}")
        local_goal = global_to_local(goal_position, self.global_home)
        self.goal_position = (int(local_goal[0]), int(local_goal[1]))
        print('Local Start and Goal: ', self.start_position, self.goal_position)

        print("Searching for a path ...")
        # Sample random nodes in the configuration space.
        self.planner.sample(self.start_position, self.goal_position)
        print(f"Sampled {len(self.planner.nodes)} nodes")
        # Create a graph by connecting the random nodes.
        print('Creating Probabilistic Roadmap')
        self.planner.create_graph()
        print(f"Found graph with {len(self.planner.graph.nodes)} nodes ")
        print("Number of edges: ", len(self.planner.graph.edges))

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        path, path_cost = self.planner.a_star(self.start_position, self.goal_position)
        print(f"Path cost: {path_cost}")
        if len(path) == 0:
            self.disarming_transition()
            return
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[int(p[0]), int(p[1]), self.target_altitude, 0] for p in path]
        print("waypoints: ", waypoints)
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()
        return path, path_cost

    def draw_fig(self, path):
        # Plot the Drone planning path
        fig = plt.figure()
        plt.imshow(self.grid, cmap='Greys', origin='lower')
        # draw edges
        for (n1, n2) in self.planner.graph.edges:
            plt.plot([n1[1] - self.e_min, n2[1] - self.e_min], [n1[0] - self.n_min, n2[0] - self.n_min], 'green' , alpha=0.5)
        # draw all nodes
        # for n1 in self.planner.nodes:
        #     plt.scatter(n1[1] - self.e_min, n1[0] - self.n_min, c='blue')
        # draw connected nodes
        for n1 in self.planner.graph.nodes:
            plt.scatter(n1[1] - self.e_min, n1[0] - self.n_min, c='red')
        # draw grid start and goal
        plt.scatter(self.start_position[1] - self.e_min, self.start_position[0] - self.n_min, c='olive')
        plt.scatter(self.goal_position[1] - self.e_min, self.goal_position[0] - self.n_min, c='orange')
        # draw path
        path_pairs = zip(path[:-1], path[1:])
        for (n1, n2) in path_pairs:
            plt.plot([n1[1] - self.e_min, n2[1] - self.e_min], [n1[0] - self.n_min, n2[0] - self.n_min], 'pink')
        plt.xlabel('EAST')
        plt.ylabel('NORTH')

        plt.savefig('Drone_path.png')
        plt.close(fig)

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
