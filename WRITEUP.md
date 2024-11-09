# Project Motion Planning Write-up

## Explain the Starter Code

1. Test that motion_planning.py is a modified version of backyard_flyer_solution.py for simple path planning. Verify that both scripts work. Then, compare them side by side and describe in words how each of the modifications implemented in motion_planning.py is functioning.

- The motion planning script introduces a step to plan a route that the drone follows to get to a specified goal position. Once that it finishes planning the route, then it traverses the plan by moving to different waypoints until it reaches the goal.

## Implementing Your Path Planning Algorithm

1. In the starter code, we assume that the home position is where the drone first initializes, but in reality you need to be able to start planning from anywhere. Modify your code to read the global home location from the first line of the colliders.csv file and set that position as global home (self.set_home_position())

- I parsed the first line of the colliders file and set the global position to be the drone's home position.

2. In the starter code, we assume the drone takes off from map center, but you'll need to be able to takeoff from anywhere. Retrieve your current position in geodetic coordinates from self._latitude, self._longitude and self._altitude. Then use the utility function global_to_local() to convert to local position (using self.global_home as well, which you just set)

- I got the global position of the drone and converted it to a local position.

3. In the starter code, the start point for planning is hardcoded as map center. Change this to be your current local position.

- Used the drone's local position as the start position for the planning algorithm.

4. In the starter code, the goal position is hardcoded as some location 10 m north and 10 m east of map center. Modify this to be set as some arbitrary position on the grid given any geodetic coordinates (latitude, longitude)

- I added a list of arbitraries goal positions using global coordinates and randomly choose one of the list to use it as goal position in the Drone simulation.

5. Write your search algorithm. Minimum requirement here is to add diagonal motions to the A* implementation provided, and assign them a cost of sqrt(2). However, you're encouraged to get creative and try other methods from the lessons and beyond!

- I used the probabilistic roadmap technique to sample multiple node that doesn't collide with object in the map, and created a connected graph. Then I used the graph version of the A* algortithm to find a path from the start position to the goal position.

6. Cull waypoints from the path you determine using search.

- Used the path found with the A* algorithm to determine the waypoint in the drone's trajectory.

## Executing the flight

1. This is simply a check on whether it all worked. Send the waypoints and the autopilot should fly you from start to goal!

- Checked this point using several hardcoded goal coordinates.