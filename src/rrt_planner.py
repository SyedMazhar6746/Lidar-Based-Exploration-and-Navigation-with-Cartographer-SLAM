#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

import transforms3d
from scipy.ndimage import binary_dilation
from rrt_utils import get_global_path


class RRT_Planner(Node): 
    def __init__(self):
        super().__init__('rrt_planner')

        # Initialize variables 
        self.grid_data          = None
        self.robot_pose_w       = None      # Robot pose in world coordinates
        self.robot_pose_g       = None      # Robot pose in grid coordinates
        self.goal_pose_w        = None      # Goal pose in world coordinates
        self.goal_pose_g        = None      # Goal pose in grid coordinates
        self.waypoints_world    = None

        # Subscribers 
        self.map_subscription       = self.create_subscription( OccupancyGrid,  '/map',                 self.occupancy_grid_callback, 10 )
        self.odometry_subscription  = self.create_subscription( Odometry,       '/odom',                self.odometry_callback, 10 )
        self.goal_subscription      = self.create_subscription( PoseStamped,    '/goal_pose_frontier',  self.goal_callback, 10 )

        # Publishers
        self.marker_publisher = self.create_publisher(Marker, 'rrt_global_path', 10)


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

    def update_grid_config(self, robot_radius_cells = 3):

        # Modify the values in the MAP
        self.grid_data[self.grid_data == -1]    = 100
        self.grid_data[self.grid_data < 30]     = 0  # Set all values less than 30 to 0
        self.grid_data[self.grid_data >= 30]    = 100  # Set all other values to 100

        # Create a structuring element for dilation (8-connectivity)
        structuring_element = np.ones((2 * robot_radius_cells + 1, 2 * robot_radius_cells + 1), dtype=np.uint8)

        # Apply dilation to inflate obstacles  
        self.grid_data = binary_dilation(self.grid_data == 100, structure=structuring_element).astype(np.uint8) * 100

        
    def occupancy_grid_callback(self, msg: OccupancyGrid):

        # Extract grid data and metadata
        self.grid_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_data = np.fliplr(np.rot90(self.grid_data, k=1))

        # Enter configuration space 
        self.update_grid_config()

    def publish_path(self, waypoints_world):
        """Publish the path for visualization in RViz."""
        marker_msg = Marker()
        
        # Set header information for the marker
        marker_msg.header.stamp = self.get_clock().now().to_msg()  # Current time
        marker_msg.header.frame_id = 'map'  # Set your desired frame_id
        
        # Set the marker type to LINE_STRIP for continuous path
        marker_msg.type = Marker.LINE_STRIP
        
        # Set the action (add/update the marker)
        marker_msg.action = Marker.ADD
        
        # Set the line width (thickness of the path)
        marker_msg.scale.x = 0.03  # Adjust this value to increase/decrease line thickness
        
        # Set the color of the line
        marker_msg.color.a = 1.0  # Alpha (opacity) 1.0 means fully opaque
        marker_msg.color.r = 0.0  # Red
        marker_msg.color.g = 1.0  # Green
        marker_msg.color.b = 0.0  # Blue (Red path)

        # Add each waypoint to the marker as a Point
        for point in waypoints_world:
            point_msg = Point()
            point_msg.x = point[0]
            point_msg.y = point[1]
            point_msg.z = 0.05  # Assuming 2D path, so z=0
            
            # Append the point to the marker's points list
            marker_msg.points.append(point_msg)
        
        # Publish the marker
        self.marker_publisher.publish(marker_msg)



    # Dynamically generate offsets for the specified number of layers
    def generate_offsets(self, num_layers):
        offsets = []
        for layer in range(1, num_layers + 1):
            for dx in range(-layer, layer + 1):
                for dy in range(-layer, layer + 1):
                    if abs(dx) == layer or abs(dy) == layer:  # Only outer edge of the layer
                        offsets.append([dx, dy])
        return offsets
    

    def goal_callback(self, msg: PoseStamped):
        """Callback to update the navigation goal."""

        # Extract position (x, y) and orientation (quaternion)
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        quad        = msg.pose.orientation
        goal_quad   = (quad.x, quad.y, quad.z, quad.w)
        goal_yaw    = self.yaw_from_quat(goal_quad)

        self.goal_pose_w    = (goal_x, goal_y, goal_yaw)
        g_xg, g_yg          = self.world_to_grid(goal_x, goal_y)
        self.goal_pose_g    = np.array([g_xg, g_yg]).reshape(2, )

        # Check if the goal is in an obstacle
        if self.grid_data[g_xg, g_yg] == 100:
            self.get_logger().info('\033[91mGoal is in obstacle, searching for nearby free cells...\033[0m')
            goal_found = False

            num_layers = 3  # Change this to specify how many layers to search
            offsets = self.generate_offsets(num_layers)

            for dx, dy in offsets:
                nx, ny = g_xg + dx, g_yg + dy

                # Ensure indices are within bounds
                if 0 <= nx < self.grid_data.shape[0] and 0 <= ny < self.grid_data.shape[1]:
                    # Check if the cell is free
                    if self.grid_data[nx, ny] < 30 and self.grid_data[nx, ny] != -1:
                        self.get_logger().info(f'\033[92mFound a free cell at ({nx}, {ny}). Setting it as the goal.\033[0m')
                        g_xg, g_yg = nx, ny
                        self.goal_pose_g = np.array([g_xg, g_yg]).reshape(2, )
                        goal_found = True
                        break

            if not goal_found:
                self.get_logger().info('\033[91mNo free cells found around the goal. Cannot find a path.\033[0m')
                return

        # Convert the grid goal to world coordinates
        goal_x, goal_y      = self.grid_to_world(g_xg, g_yg)
        self.goal_pose_w    = (goal_x, goal_y, goal_yaw)

        # Calculate the global path
        start   = self.robot_pose_g
        goal    = self.goal_pose_g
        waypoints_grid = get_global_path(start, goal, self.grid_data)

        self.waypoints_world = [self.grid_to_world(float(point[0]), float(point[1])) for point in waypoints_grid]

        print("Waypoints in world coordinates: ", self.waypoints_world)

        # Publish the path for visualization in RViz
        self.publish_path(self.waypoints_world)







def main(args=None):
    rclpy.init(args=args)
    rrt_planner = RRT_Planner()

    try:
        rclpy.spin(rrt_planner)
    except KeyboardInterrupt:
        rrt_planner.get_logger().info('Node stopped by user.')
    finally:
        rrt_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
