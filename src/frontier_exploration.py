#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool, ColorRGBA
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point 
from visualization_msgs.msg import Marker, MarkerArray


import random
import transforms3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN



class Frontier(Node): 
    def __init__(self, free_threshold=20, unknown_min=20, unknown_max=50):
        super().__init__('frontier')

        # Initialize variables 
        self.grid_data          = None
        self.robot_pose_w       = None   # Robot's current pose 
        self.robot_pose_g       = None   # Robot's current pose
        self.goal_pose_w        = None   # 2D navigation goal pose wprld
        self.goal_pose_g        = None   # 2D navigation goal pose grid
        self.find_frontier      = False
        self.frontiers          = []
        self.best_frontier      = None
        self.explored           = False


        self.free_threshold     = free_threshold
        self.unknown_min        = unknown_min
        self.unknown_max        = unknown_max

        self.fr_dist_weight         = 1
        self.fr_orient_weight       = 2
        self.fr_cluster_weight      = 2

        self.reset_values()  


        # Subscribers
        self.odometry_subscription  = self.create_subscription( Odometry,       '/odom',            self.odometry_callback, 10 )
        self.map_subscription       = self.create_subscription( OccupancyGrid,  '/map',             self.occupancy_grid_callback, 10 )
        self.frontier_subscription  = self.create_subscription(Bool,            '/find_frontier',   self.frontier_callback, 10)

        # Publishinf the goal 
        self.goal_pub               = self.create_publisher(PoseStamped, '/goal_pose_frontier', 10)  # Topic name is '/goal_pose'
        self.cluster_pub            = self.create_publisher(MarkerArray, '/frontiers_visual', 10)


        # Add a timer to delay calling `get_frontier`
        self.timer = self.create_timer(2.0, self.delayed_get_frontier)  # Delay by 2 seconds
        self.timer_called = False                                       # Flag to ensure `get_frontier` is only called once


    def delayed_get_frontier(self):
        """
        Delays the call to `get_frontier` by a set time and ensures it is called only once.
        """
        if not self.timer_called:
            print("Finding frontier once")
            self.get_frontier()
            self.timer_called = True
            self.timer.cancel()  # Stop the timer after calling the function once



    def reset_values(self):
        
        self.frontiers      = []
        self.clusters       = []
        self.centroids      = []
        self.best_frontier  = None


    def yaw_from_quat(self, quaternion):

        x, y, z, w = quaternion
        # Get the rotation matrix from the quaternion
        rotation_matrix = transforms3d.quaternions.quat2mat([w, x, y, z])

        # Extract yaw from the rotation matrix
        _, _, yaw = transforms3d.euler.mat2euler(rotation_matrix, axes='sxyz')
        return yaw

    def grid_to_world(self, grid_x, grid_y):
        world_x = (53 - grid_x) / 20
        world_y = (53 - grid_y) / 20
        return world_x, world_y

    def world_to_grid(self, world_x, world_y):
        grid_x = -20 * world_x + 53
        grid_y = -20 * world_y + 53
        return int(grid_x), int(grid_y)
    
    def odometry_callback(self, msg: Odometry):
        """Callback to update the robot's pose."""
        # Extract position (x, y)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract orientation (quaternion)
        quaternion = (
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w
                    )
        
        yaw = self.yaw_from_quat(quaternion)

        # Update robot pose
        self.robot_pose_w   = (x, y, yaw)
        r_xg, r_yg          = self.world_to_grid(x, y)    
        self.robot_pose_g   = np.array([r_xg, r_yg]).reshape(2, )

    def occupancy_grid_callback(self, msg: OccupancyGrid):

        # Extract grid data and metadata
        self.grid_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_data = np.fliplr(np.rot90(self.grid_data, k=1))




    def find_frontiers(self):
        """
        Finds all frontier points in the grid.

        Returns:
            list of tuple: A list of coordinates (row, col) of all frontier points.
        """
        self.frontiers = []

        rows, cols = self.grid_data.shape

        # Define offsets for 8-neighbor connectivity
        neighbors = [
            [-1, 0], [1, 0], [0, -1], [0, 1],  # Up, Down, Left, Right
            [-1, -1], [-1, 1], [1, -1], [1, 1]  # Diagonals
        ]

        for r in range(rows):
            for c in range(cols):
                # Check if the cell is free
                if 0 <= self.grid_data[r, c] <= self.free_threshold:
                    # Check neighbors for unknown cells
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc

                        # Ensure neighbor is within bounds
                        if 0 <= nr < rows and 0 <= nc < cols:
                            # If an adjacent cell is unknown, this is a frontier point 
                            if self.unknown_min < self.grid_data[nr, nc] <= self.unknown_max:
                                self.frontiers.append([r, c])
                                break  # No need to check further neighbors 
        
        if not self.frontiers:
            self.explored = True
            self.get_logger().info("No frontiers found.")
        


    def cluster_frontiers(self, eps=1.5, min_samples=4): # 1.5, 4
        """ 
        Clusters the frontier points using DBSCAN and discards isolated points.

        Parameters:
            frontiers (list of list): List of frontier points as [row, col].
            eps (float): Maximum distance between points to consider them part of the same cluster.
            min_samples (int): Minimum number of points in a cluster.

        Returns:
            list of np.ndarray: List of clusters, each as a numpy array of [row, col] points.
        """
        if not self.frontiers:
            return []

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(self.frontiers)

        # Group points by cluster labels
        self.clusters = []
        labels = clustering.labels_
        for cluster_id in set(labels):
            if cluster_id != -1:  # Discard noise points (label -1)
                self.clusters.append(np.array([self.frontiers[i] for i in range(len(self.frontiers)) if labels[i] == cluster_id]))

        if not self.clusters:
            self.explored = True
            self.get_logger().info("No clusters found.")

    def find_cluster_centroids(self):
        """
        Computes the centroid of each cluster.

        Parameters:
            clusters (list of np.ndarray): List of clusters, each as a numpy array of [row, col] points.

        Returns:
            list of tuple: List of centroids as (row, col) for each cluster.
        """
        self.centroids = []
        for cluster in self.clusters:
            centroid_row = np.mean(cluster[:, 0])
            centroid_col = np.mean(cluster[:, 1])
            self.centroids.append([centroid_row, centroid_col])




    def get_perpen_dir(self, cluster):
        # Perform PCA to find the principal axis
        pca = PCA(n_components=1)
        pca.fit(cluster)

        # The principal component direction
        principal_direction = pca.components_[0]

        # Perpendicular direction (rotate by 90 degrees)
        perpendicular_direction = np.array([-principal_direction[1], principal_direction[0]])

        # Normalize the perpendicular direction vector
        perpendicular_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)

        return perpendicular_direction

    # Function to check if a cell is free (for simplicity, assuming grid_map is loaded and has a threshold)
    def is_cell_free(self, r, c):
        # Ensure the indices are within bounds
        if 0 <= r < self.grid_data.shape[0] and 0 <= c < self.grid_data.shape[1]:
            return self.grid_data[r, c] != -1 and self.grid_data[r, c] <= self.free_threshold  # Free if cell value is not -1 and less than threshold
        return False

    def get_offset_centroid(self, centroid, cluster, offset = 7):
        
        # Get the perpendicular direction of the cluster
        perpendicular_direction = self.get_perpen_dir(cluster)
        
        # Centroid coordinates
        r, c = centroid
        offset_centroid = None
        valid_offset_centroid = False

        # Check both positive and negative perpendicular directions (2 cells away)
        positive_adjacent = (r + 2 * perpendicular_direction[0], c + 2 * perpendicular_direction[1])
        negative_adjacent = (r - 2 * perpendicular_direction[0], c - 2 * perpendicular_direction[1])

        # Check which direction is free
        positive_is_free = self.is_cell_free(int(np.round(positive_adjacent[0])), int(np.round(positive_adjacent[1])))
        negative_is_free = self.is_cell_free(int(np.round(negative_adjacent[0])), int(np.round(negative_adjacent[1])))

        if positive_is_free or negative_is_free:
            if positive_is_free:
                # Calculate the new point 5 cells in the positive perpendicular direction
                offset_centroid = (r + perpendicular_direction[0] * offset, c + perpendicular_direction[1] * offset)
                
            elif negative_is_free:
                # Calculate the new point 5 cells in the negative perpendicular direction
                offset_centroid = (r - perpendicular_direction[0] * offset, c - perpendicular_direction[1] * offset)

            offset_centroid = [int(np.round(offset_centroid[0])), int(np.round(offset_centroid[1]))]
            valid_offset_centroid = self.is_cell_free(offset_centroid[0], offset_centroid[1])

            return offset_centroid, valid_offset_centroid
        
        else:
            print("Neither direction is free. No valid offset available.")
            return offset_centroid, valid_offset_centroid

    def get_valid_offset_centroid(self):

        self.valid_centroid = []
        self.valid_clusters = []

        for i in range(len(self.centroids)):
            offset_centroid, valid_offset_centroid = self.get_offset_centroid(self.centroids[i], self.clusters[i])
            if valid_offset_centroid:
                self.valid_centroid.append(offset_centroid)
                self.valid_clusters.append(self.clusters[i])

        # replce the centroids and clusters with valid ones
        self.centroids  = self.valid_centroid
        self.clusters   = self.valid_clusters

        if not self.centroids:
            self.explored = True
            self.get_logger().info("No valid offset centroids found.")


    # Function to calculate Euclidean distance
    def calculate_distance(self, centroid):
        r_robot, c_robot = self.robot_pose_g
        r_centroid, c_centroid = centroid
        return np.sqrt((r_robot - r_centroid)**2 + (c_robot - c_centroid)**2)
    
    # Function to calculate the orientation difference between robot orientation and the vector to centroid
    def calculate_orientation_difference(self, centroid):

        centroid_w = self.grid_to_world(centroid[0], centroid[1])

        # Vector from robot to centroid
        r_wx, r_wy = self.robot_pose_w[:2]
        centroid_wx, centroid_wy = centroid_w

        vector_to_centroid = np.array([centroid_wx - r_wx, centroid_wy - r_wy])

        # Calculate the angle of the vector with respect to x-axis
        vector_angle = np.arctan2(vector_to_centroid[1], vector_to_centroid[0])
        
        # Calculate the angle difference (in radians)
        angle_diff = np.abs(self.robot_pose_w[2] - vector_angle)
        
        # Normalize angle to be between 0 and pi (180 degrees)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        return angle_diff

    # normalize goal_dist_cost considering minimum as zero and maximum as (max distance from the goal_dist_cost_list)
    def normalize_goal_dist_cost(self, dist_np):
        max_goal_dist = np.max(dist_np) 
        normalized_distances = dist_np / max_goal_dist
        return normalized_distances.reshape(-1, ) 

    def normalize_orientation_cost(self, orientation_np):
        max_orientation = np.pi
        normalized_orientation = orientation_np / max_orientation
        return normalized_orientation.reshape(-1, )

    def normalize_cluster_cost(self, cluster_np): 
        max_cluster = np.max(cluster_np)
        normalized_cluster = 1 - (cluster_np / max_cluster)
        return normalized_cluster.reshape(-1, )

    def compute_cost(self, distance_list, orientation_diff_list, cluster_length_list):
        # Normalize the cost factors
        distance_np             = np.array(distance_list)
        orientation_diff_np     = np.array(orientation_diff_list)
        cluster_length_np       = np.array(cluster_length_list)

        normalized_distances    = self.normalize_goal_dist_cost(distance_np)
        normalized_orientation  = self.normalize_orientation_cost(orientation_diff_np)
        normalized_cluster      = self.normalize_cluster_cost(cluster_length_np)

        # compute weighted cost 
        cost = (
                self.fr_dist_weight     * normalized_distances   +
                self.fr_orient_weight   * normalized_orientation +
                self.fr_cluster_weight  * normalized_cluster
                )
        return cost

    # Function to compute the best centroid based on the cost function 
    def compute_best_centroid(self):

        distance_list = []
        orientation_diff_list = []
        cluster_length_list = []
        
        # Iterate over all centroids and clusters
        for centroid, cluster in zip(self.centroids, self.clusters):

            # 1. Calculate distance
            distance = self.calculate_distance(centroid)
            distance_list.append(distance)
            
            # 2. Calculate orientation difference (between robot and vector to centroid)
            orientation_diff = self.calculate_orientation_difference(centroid)
            orientation_diff_list.append(orientation_diff)
            
            # 3. Calculate cluster length
            cluster_length = len(cluster)
            cluster_length_list.append(cluster_length)

        cost = self.compute_cost(distance_list, orientation_diff_list, cluster_length_list)


        # 4. Find the centroid with the minimum cost
        best_index          = np.argmin(cost)
        self.best_frontier  = self.centroids[best_index]

        best_frontier_w = self.grid_to_world(self.best_frontier[0], self.best_frontier[1])

        return best_frontier_w


    def goal_publisher(self, best_frontier_w):
        """
        Publishes the goal point [x, y] as a PoseStamped message.

        Parameters:
            x (float): The x-coordinate of the goal.
            y (float): The y-coordinate of the goal.
        """
        goal_msg = PoseStamped()

        # Set the header information
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'  # Assuming the frame is 'map'

        # Set the position (x, y) and orientation (no rotation, facing forward)
        goal_msg.pose.position.x = best_frontier_w[0]
        goal_msg.pose.position.y = best_frontier_w[1]

        # Publish the goal message 
        self.goal_pub.publish(goal_msg) 
        self.get_logger().info(f'Published goal: x={best_frontier_w[0]}, y={best_frontier_w[1]}')


    def numpy_to_point(self, point):
        """
        Convert a numpy array point to a geometry_msgs/Point message.
        """

        # convert point from grid to world
        point_w = self.grid_to_world(point[0], point[1])

        return Point(x=float(point_w[0]), y=float(point_w[1]), z=0.0)


    def random_color(self):
        """
        Generate a random RGBA color.
        """
        return ColorRGBA(
            r=random.uniform(0.0, 1.0),
            g=random.uniform(0.0, 1.0),
            b=random.uniform(0.0, 1.0),
            a=1.0  # Fully opaque
        )


    def visualize_frontiers(self):
        """
        Visualize clusters in Rviz. Each cluster will have a unique color.

        Parameters:
            clusters (list of np.array): List of numpy arrays, where each array represents a cluster of points.
        """
        markers = MarkerArray()  # Create a MarkerArray to publish all clusters at once

        for i, cluster in enumerate(self.clusters):
            # Create a marker for the cluster
            marker = Marker()
            marker.header.frame_id = "map"  
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "clusters"  
            marker.id = i 
            marker.type = Marker.POINTS  
            marker.action = Marker.ADD  

            # Set scale of the points
            marker.scale.x = 0.05  # Point width
            marker.scale.y = 0.05  # Point height

            # Assign a random color to each cluster
            marker.color = self.random_color()

            # Add points to the marker
            for point in cluster:
                marker.points.append(self.numpy_to_point(point))

            # Add the marker to the MarkerArray
            markers.markers.append(marker)

        # Publish all markers in one message
        self.cluster_pub.publish(markers)

        self.get_logger().info(f"Published {len(self.clusters)} clusters.")


    def get_frontier(self):
        
        # Reset the values
        self.reset_values()

        # Find frontiers 
        self.find_frontiers()

        # Cluster the frontiers 
        self.cluster_frontiers()

        if not self.explored:
        
            # Compute the centroid of each cluster
            self.find_cluster_centroids()

            # get valid offset centroid
            self.get_valid_offset_centroid()

            if not self.explored:

                # find best centroid as a goal
                best_frontier_w = self.compute_best_centroid() # [x, y]

                self.goal_publisher(best_frontier_w)

                # Visualize the frontiers - self.clusters = [np.array([N1, 2]), np.array([N1, 2]), ...]
                self.visualize_frontiers()

                self.find_frontier = False


    def frontier_callback(self, msg: Bool):
        self.find_frontier = msg.data
        if self.find_frontier:
            print("Finding frontier")
            self.get_frontier()










def main(args=None):
    rclpy.init(args=args)
    frontier = Frontier()

    try:
        rclpy.spin(frontier)
    except KeyboardInterrupt:
        frontier.get_logger().info('Node stopped by user.')
    finally:
        frontier.destroy_node()
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
