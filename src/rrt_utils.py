#!/usr/bin/env python3

import random
import numpy as np

# treeNode class
class treeNode:
    def __init__(self, locationX, locationY):
        self.locationX = locationX          # X Location
        self.locationY = locationY          # Y Location
        self.children = []                  # children List
        self.parent = None                  # parent node reference

# RRT Algorithm class
class RRTAlgorithm():
    def __init__(self, start, goal, grid, stepSize):
        # The RRT (root position)
        self.randomTree     = treeNode(start[0], start[1])
        self.goal           = treeNode(goal[0], goal[1])            # goal position
        self.nearestNode    = None                                  # nearest node
        self.grid           = grid                                  # the map
        self.rho            = stepSize                              # length of each branch
        self.path_distance  = 0                                     # total path distance
        self.nearestDist    = 10000                                 # distance to neareast node
        self.numWaypoints   = 0                                     # number of waypoints
        self.Waypoints      = []                                    # the waypoints

    # add the point to the nearest node and add goal when reached
    def addChild(self, locationX, locationY):
        if (locationX == self.goal.locationX):
            # add the goal node to the children of the nearest node
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            tempNode = treeNode(locationX, locationY)
            # add tempNode to children of nearest node
            self.nearestNode.children.append(tempNode)
            tempNode.parent = self.nearestNode

    # sample a random point within grid Limits
    def sampleAPoint(self):
        x = random.randint(1, self.grid.shape[1])
        y = random.randint(1, self.grid.shape[0])
        point = np.array([x, y])
        return point

    # steer a distance stepsize from start to end Location
    def steerToPoint(self, locationStart, locationEnd):
        offset = self.rho*self.unitVector(locationStart, locationEnd)
        point = np.array([locationStart.locationX + offset[0],
                         locationStart.locationY + offset[1]])
        if point[0] >= self.grid.shape[1]:
            point[0] = self.grid.shape[1] - 1
        if point[1] >= self.grid.shape[0]:
            point[1] = self.grid.shape[0] - 1
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0
        return point

    # check if obstacle lies between the start and end point of the edge
    def isInObstacle(self, locationStart, locationEnd):
        u_hat = self.unitVector(locationStart, locationEnd)

        testPoint = np.array([0.0, 0.0])

        for i in range(self.rho):
            testPoint[0] = locationStart.locationX + i*u_hat[0]
            testPoint[1] = locationStart.locationY + i*u_hat[1]

            y = np.round(testPoint[1]).astype(np.int64)
            x = np.round(testPoint[0]).astype(np.int64)
            if (y < 0 or x < 0 or y >= self.grid.shape[0] or x >= self.grid.shape[1] or 
                self.grid[x, y] > 30 or self.grid[x-1, y] > 30 or self.grid[x, y-1] > 30 or self.grid[x-1, y-1] > 30
                or self.grid[x+1, y] > 30 or self.grid[x, y+1] > 30 or self.grid[x+1, y+1] > 30):
                return True
        return False

    # check if obstacle lies between the start and end point of the edge
    def isInObstacle_waypoint(self, locationStart, locationEnd):
        u_hat = self.unitVector(locationStart, locationEnd)
        testPoint = np.array([0.0, 0.0])
        
        for i in range(int(np.linalg.norm(np.array([locationStart.locationX, locationStart.locationY]) - locationEnd))):
            testPoint[0] = locationStart.locationX + i*u_hat[0]
            testPoint[1] = locationStart.locationY + i*u_hat[1]

            y = np.round(testPoint[1]).astype(np.int64)
            x = np.round(testPoint[0]).astype(np.int64)
            if (y < 0 or x < 0 or y >= self.grid.shape[0] or x >= self.grid.shape[1] or 
                self.grid[x, y] > 30 or self.grid[x-1, y] > 30 or self.grid[x, y-1] > 30 or self.grid[x-1, y-1] > 30
                or self.grid[x+1, y] > 30 or self.grid[x, y+1] > 30 or self.grid[x+1, y+1] > 30):
                return True
        return False
    
    # find unit vector between 2 points which form a vector
    def unitVector(self, locationStart, locationEnd):
        v = np.array([locationEnd[0] - locationStart.locationX,
                     locationEnd[1] - locationStart.locationY])
        u_hat = v/np.linalg.norm(v)
        return u_hat

    # find the nearest node from a given unconnected point (Euclidean distance)
    def findNearest(self, root, point):
        if not root:
            return
        dist = self.distance(root, point)
        if dist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dist
        # recursively call by iterating through the childern
        for child in root.children:
            self.findNearest(child, point)

    # find euclidean distance between a node and an XY point
    def distance(self, node1, point):
        dist = np.sqrt(
            (node1.locationX - point[0])**2 + (node1.locationY - point[1])**2)
        return dist

    # check if the goal has been reached within step size
    def goalFound(self, point):
        if self.distance(self.goal, point) <= self.rho:
            return True

    # reset nearestNode and nearest Distance
    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000

    # trace the path from goal to start
    def retraceRRTPath(self, goal):
        if goal.locationX == self.randomTree.locationX:
            return
        self.numWaypoints += 1
        currentPoint = np.array([goal.locationX, goal.locationY])
        self.Waypoints.insert(0, currentPoint)
        self.path_distance += self.rho
        self.retraceRRTPath(goal.parent)


def simplify_path(rrt, waypoints):
    """
    Simplifies the path by connecting waypoints directly if no obstacles lie between them.
    """
    simplified_waypoints = [waypoints[0]]  
    current = waypoints[0]

    # Iterate backward through the waypoints
    for i in range(len(waypoints) - 1):
        next_point = waypoints[i+1]

        # Check if there's an obstacle between the current point and the next point
        if not rrt.isInObstacle_waypoint(treeNode(current[0], current[1]), next_point):
            # If no obstacle, skip intermediate points and connect directly
            if next_point[0] == waypoints[-1][0] and next_point[1] == waypoints[-1][1]:
                simplified_waypoints.append(next_point)
                break
            continue 
        else:
            # If there's an obstacle, add the previous point to the path
            simplified_waypoints.append(waypoints[i])
            current = waypoints[i]

    # Ensure the goal is in the simplified waypoints
    if simplified_waypoints[-1][0] != waypoints[-1][0] or simplified_waypoints[-1][1] != waypoints[-1][1]:
        simplified_waypoints.append(waypoints[-1])

    return simplified_waypoints

def get_waypoints(rrt, i, start, goal):
    totalNodes = 0
    totalIterations = 0

    while True:
        totalIterations += 1
        # Reset nearest values
        rrt.resetNearestValues()
        # algo begins
        point = rrt.sampleAPoint()
        rrt.findNearest(rrt.randomTree, point)
        new = rrt.steerToPoint(rrt.nearestNode, point)
        bool = rrt.isInObstacle(rrt.nearestNode, new)
        i = i - 1
        if (bool == False):
            totalNodes += 1
            rrt.addChild(new[0], new[1])
            # if goal found, append the path
            if (rrt.goalFound(new)):
                rrt.addChild(goal[0], goal[1])
                print("\nGoal Found!")
                break
        else:
            i = i + 1

    # trace back path returned, and add start to waypoints
    rrt.retraceRRTPath(rrt.goal)
    rrt.Waypoints.insert(0, start)

    # Connecting free waypoints directly
    waypoints = simplify_path(rrt, rrt.Waypoints)

    return waypoints

# Input: Start (2, ), Goal (2, ), Grid (H, W). (All numpy arrays)
# Output: Waypoints. 
def get_global_path(start, goal, grid, stepSize=10, i=5000):
    rrt = RRTAlgorithm(start, goal, grid, stepSize)
    waypoints = get_waypoints(rrt, i, start, goal)
    return waypoints
