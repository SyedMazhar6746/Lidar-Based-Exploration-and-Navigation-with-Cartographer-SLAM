#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool  
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker

import transforms3d
from scipy.ndimage import binary_dilation 


class DWA_Planner(Node):   
    def __init__(self):
        super().__init__('dwa_planner')

        # Initialize variables  
        self.grid_data          = None
        self.resolution         = None
        self.origin             = None
        self.robot_pose_w       = None   # Robot pose in world coordinates
        self.robot_pose_g       = None   # Robot pose in grid coordinates
        self.goal_pose_w        = None   # Goal pose in world coordinates
        self.goal_pose_g        = None   # Goal pose in grid coordinates
        self.waypoints_w        = None
        self.obstacle_cells     = None

        # Subscribers
        self.map_subscription       = self.create_subscription( OccupancyGrid,  '/map',             self.occupancy_grid_callback, 10 )
        self.odometry_subscription  = self.create_subscription( Odometry,       '/odom',            self.odometry_callback, 10 )
        self.goal_subscription      = self.create_subscription( Marker,         '/rrt_global_path', self.rrt_path_w, 10 )

        # Publisher for the path visualization
        self.path_pub           = self.create_publisher(Path,  '/trajectory', 10)
        self.vel_publisher      = self.create_publisher(Twist, '/cmd_vel', 10)
        self.frontier_publisher = self.create_publisher(Bool,  '/find_frontier', 10)


        # Timer for DWA loop (periodic execution)
        self.timer = self.create_timer(0.1, self.dwa_step)  # Timer callback every 0.1 seconds

        # Turtlebot3 specifications for waffle 
        self.max_lin_vel = 0.16  # 0.16 m/s
        self.max_ang_vel = 0.8  # 1.0 rad/s (104.27 degrees/s)

        self.max_lin_acc = 0.8   # 1.5 m/s^2
        self.max_ang_acc = 2.0   # 4.0 rad/s^2

        # Weights for the cost function
        self.g_weight    = 2.0  # Goal distance weight # 5
        self.obs_weight  = 1.0  # Obstacle distance weight
        self.dt          = 0.1  # Time step

        # Get the current velocities
        self.cur_lin_vel = 0.0 # Initial Linear velocity
        self.cur_ang_vel = 0.0 # Initial Angular velocity

        # Dynamic window
        self.v_min         = 0.0
        self.v_max         = 0.0
        self.w_min         = 0.0
        self.w_max         = 0.0

        self.obs_radius    = 3      # 14 cells = 0.7 m
        self.duration      = 3.0    # Duration for which to simulate the trajectory

        self.reached_goal       = False
        self.current_waypoint   = 0
        self.best_v             = 0.0
        self.best_w             = 0.0
        self.best_trajectory    = []

    def goal_distance(self):
        """
        Compute the Euclidean distance from the robot's current position to the goal.
        """
        return np.sqrt((self.robot_pose_w[0] - self.goal_pose_w[0])**2 + (self.robot_pose_w[1] - self.goal_pose_w[1])**2)
    
    def get_reachable_velocities(self):
        """
        Calculate the set of reachable velocities based on current velocities,
        maximum acceleration, and time step.
        """
        self.v_min = 0.0  # Robot cannot go backward at a greater speed than deceleration
        self.v_max = min(self.max_lin_vel, self.cur_lin_vel + self.max_lin_acc * self.dt)  # Max linear velocity

        self.w_min = max(-self.max_ang_vel, self.cur_ang_vel - self.max_ang_acc * self.dt)
        self.w_max = min(self.max_ang_vel , self.cur_ang_vel + self.max_ang_acc * self.dt)

    def goal_dist_cost(self, each_trajectory):
        """
        Compute the cost of the trajectory based on the distance from the goal.
        """
        final_x, final_y, _ = each_trajectory[-1]
        goal_dist_cost = np.sqrt((final_x - self.goal_pose_w[0])**2 + (final_y - self.goal_pose_w[1])**2)

        return goal_dist_cost
    
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

        # update the current velocities
        self.cur_lin_vel = msg.twist.twist.linear.x
        self.cur_ang_vel = msg.twist.twist.angular.z

    def update_grid_config(self, robot_radius_cells = 4):

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
        self.obstacle_cells = np.argwhere(self.grid_data == 100)

    def np_world_to_grid(self, world_x, world_y):
        grid_x = -20 * world_x + 53
        grid_y = -20 * world_y + 53
        return np.int32(grid_x), np.int32(grid_y)

    def traj_world_to_grid(self, traj_w):
        """
        Convert all trajectory points from world coordinates to grid coordinates.
        :param traj: List of trajectory points, where each point is [x, y, theta]
        :return: Numpy array of grid coordinates for all trajectory points
        """
        # Extract x and y points from the trajectory
        traj_array = np.array(traj_w)  # Convert to NumPy array for efficient processing
        world_x = traj_array[:, 0]  # Extract x-coordinates
        world_y = traj_array[:, 1]  # Extract y-coordinates

        # Convert world coordinates to grid coordinates
        traj_gx, traj_gy = self.np_world_to_grid(world_x, world_y)

        # Combine grid_x and grid_y into a single array
        traj_g = np.vstack((traj_gx, traj_gy)).T  # Shape: (N, 2)

        return traj_g # Shape: (N, 2) (trajectory points in grid coordinates)

    def check_trajectory(self, traj_grid_coords):
        """
        Check if any value in the grid along the trajectory is 100.
        :param grid: The occupancy grid map
        :param traj_grid_coords: The trajectory in grid coordinates
        :return: True if all values are not 100, False otherwise
        """
        if np.any(self.grid_data[traj_grid_coords[:, 0], traj_grid_coords[:, 1]] == 100):
            return False
        return True
    

    # Step 2: Compute minimum distances 
    def compute_min_distances(self, end_point):
        """
        Compute the minimum distance from each endpoint to the closest obstacle cell.
        
        :param end_points: Array of trajectory endpoints (N x 2)
        :param obstacle_cells: Array of obstacle cell coordinates (M x 2)
        :return: Array of minimum distances for each endpoint
        """
        # Compute pairwise distances using broadcasting
        # Shape: (N, M), where N = number of endpoints, M = number of obstacle cells
        distances = np.sqrt(
            (end_point[:, np.newaxis, 0] - self.obstacle_cells[:, 0])**2 +
            (end_point[:, np.newaxis, 1] - self.obstacle_cells[:, 1])**2
        )
        
        # Find the minimum distance for each endpoint with in a circle of radius 14, else 0
        min_distance = np.where(np.min(distances, axis=1) < self.obs_radius, np.min(distances, axis=1), 0)
        
        # normalize the values between 0 and 1 consider 0 as miinimum and radius as maximum
        normalized_distance = np.where( min_distance > 0, 1 - (min_distance / self.obs_radius), 0 )
        
        return normalized_distance
    
    def generate_trajectory(self, v, w):
        """
        Generate a trajectory based on the current velocities and time step.
        """
        # Initialize the trajectory with the current robot state
        x, y, theta = self.robot_pose_w
        trajectory = [[float(x), float(y), float(theta)]]

        # Simulate the trajectory
        num_steps = int(self.duration / self.dt)
        for _ in range(num_steps):
            # Update position and orientation based on motion model
            theta += w * self.dt
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt

            # Append the current state to the trajectory
            trajectory.append([float(x), float(y), float(theta)])

        traj_grid_coords = self.traj_world_to_grid(trajectory)

        # Check if the trajectory is valid
        valid = self.check_trajectory(traj_grid_coords)

        obs_cost = self.compute_min_distances(traj_grid_coords[-1:])

        return trajectory, valid, obs_cost

    # normalize goal_dist_cost considering minimum as zero and maximum as (max distance from the goal_dist_cost_list)
    def normalize_goal_dist_cost(self, goal_dist_np):
        max_goal_dist = np.max(goal_dist_np) 
        normalized_distances = goal_dist_np / max_goal_dist
        return normalized_distances.reshape(-1, ) 


    def send_velocities(self, ):
        """
        Publish the linear and angular velocities to the robot.
        """
        # Create a Twist message
        twist = Twist()
        if self.reached_goal:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x  = self.best_v
            twist.angular.z = self.best_w

        # Publish the Twist message
        self.vel_publisher.publish(twist)


    def visualize_trajectory(self):
        # Create a Path message
        path = Path()
        path.header.frame_id = 'map'  # Use 'odom' or another frame if applicable
        path.header.stamp = self.get_clock().now().to_msg()

        # Add each trajectory point to the Path
        for x, y, _ in self.best_trajectory[0]:
            pose = PoseStamped()
            pose.header.frame_id = path.header.frame_id
            pose.header.stamp = self.get_clock().now().to_msg()

            # Set position
            pose.pose.position.x = x
            pose.pose.position.y = y

            # Add the pose to the path
            path.poses.append(pose)

        # Publish the path
        self.path_pub.publish(path) 
        # self.get_logger().info('Published trajectory')


    def rrt_path_w(self, msg: Marker):
        # Extract the waypoints from the marker message
        waypoints = msg.points 

        # Convert the waypoints to a list of lists
        self.waypoints_w = [[point.x, point.y] for point in waypoints]
        
        print("path in world: ", self.waypoints_w) 
        self.current_waypoint = 0
        self.reached_goal = False
        if self.waypoints_w:
            self.goal_pose_w = self.waypoints_w[self.current_waypoint]
            self.goal_pose_g = self.world_to_grid(self.goal_pose_w[0], self.goal_pose_w[1])




    def dwa_step(self):

        """Single step of the DWA loop executed periodically."""
        if not self.waypoints_w or self.reached_goal:
            self.send_velocities()
            return
        
        # Check if the robot has reached the current goal
        if self.goal_distance() < 0.2:
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.waypoints_w):
                self.reached_goal = True
                self.get_logger().info("All waypoints reached!")

                self.get_logger().info("Asking a frontier point")
                # Publishing a Bool message to '/find_frontier'
                msg = Bool()
                msg.data = True  # or False, depending on your requirement
                self.frontier_publisher.publish(msg)

                return

            print("Reached waypoint ", self.current_waypoint)

            # Update the goal to the next waypoint
            self.goal_pose_w = self.waypoints_w[self.current_waypoint]
            self.goal_pose_g = self.world_to_grid(self.goal_pose_w[0], self.goal_pose_w[1])

            
        self.get_reachable_velocities()

        trajectory          = []
        all_trajectory      = []
        all_v_w             = []
        goal_dist_cost_list = []
        obs_cost_list       = []
        costs               = np.array([])

        # Evaluate the trajectories (simplified for now to just goal distance)
        for v in np.linspace(self.v_min+0.01, self.v_max, 5):  # Sampling 5 velocities in the range
            for w in np.linspace(self.w_min, self.w_max, 5):  # Sampling 5 angular velocities 
                
                # obs_cost is normalized cost
                trajectory, valid, obs_cost = self.generate_trajectory(v, w)

                # if valid is True, append the trajectory to all_trajectory, else discard
                if valid:   
                    goal_dist_cost = self.goal_dist_cost(trajectory)

                    all_trajectory.append(trajectory) 
                    goal_dist_cost_list.append(goal_dist_cost)
                    all_v_w.append([v, w])
                    obs_cost_list.append(obs_cost)
                
        if goal_dist_cost_list:
            # # Normalize the goal_dist_cost_list
            goal_dist_cost_np   = np.array(goal_dist_cost_list).reshape(-1, )
            obs_cost_np         = np.array(obs_cost_list).reshape(-1, )

            normalized_goal_dist_cost = self.normalize_goal_dist_cost(goal_dist_cost_np) 

            costs = (
                        self.g_weight * normalized_goal_dist_cost +
                        self.obs_weight * obs_cost_np
                    )

            best_index                  = np.argmin(costs) 
            self.best_trajectory        = [all_trajectory[best_index]]
            self.best_v, self.best_w    = all_v_w[best_index]
        
            # send best velocity and angular velocity to the robot
            self.send_velocities()
            self.visualize_trajectory()







def main(args=None):
    rclpy.init(args=args)
    dwa_planner = DWA_Planner()

    try:
        rclpy.spin(dwa_planner)
    except KeyboardInterrupt:
        dwa_planner.get_logger().info('Node stopped by user.')
    finally:
        dwa_planner.destroy_node()
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
