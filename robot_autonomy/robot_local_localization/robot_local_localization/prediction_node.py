#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math
from tf_transformations import quaternion_from_euler


class PredictionNode(Node):

    def __init__(self):
        super().__init__('prediction_node')
        self.declare_parameter('wheel_separation', 0.45)
        self.declare_parameter('wheel_radius', 0.1)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('process_noise_v', 0.01)
        self.declare_parameter('process_noise_w', 0.01)

        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.rate = self.get_parameter('rate').value
        self.process_noise_v = self.get_parameter('process_noise_v').value
        self.process_noise_w = self.get_parameter('process_noise_w').value

        self.dt = 1.0 / self.rate
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.P = np.diag([0.2, 0.2, 0.2])

        self.last_v = 0.0
        self.last_w = 0.0

        self.subscription = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10
        )

        self.prediction_pub = self.create_publisher(Odometry, '/ekf/motion_model', 10)

        self.path_pub = self.create_publisher(Path, '/ekf/prediction_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.timer = self.create_timer(self.dt, self.prediction_update)

    def cmd_vel_callback(self, msg):
        self.last_v = msg.linear.x
        self.last_w = msg.angular.z

    def prediction_update(self):
        v = self.last_v
        w = self.last_w
        dt = self.dt

        theta_prev = self.theta

        if abs(w) < 1e-6:
            self.x += v * dt * math.cos(theta_prev)
            self.y += v * dt * math.sin(theta_prev)
        else:
            R = v / w
            self.x += R * (-math.sin(theta_prev) + math.sin(theta_prev + w * dt))
            self.y += R * (math.cos(theta_prev) - math.cos(theta_prev + w * dt))

        self.theta = (theta_prev + w * dt + math.pi) % (2 * math.pi) - math.pi

        if abs(w) < 1e-6:
            F = np.array([
                [1.0, 0.0, -v * dt * math.sin(theta_prev)],
                [0.0, 1.0,  v * dt * math.cos(theta_prev)],
                [0.0, 0.0, 1.0]
            ])
        else:
            R = v / w
            F = np.array([
                [1.0, 0.0, R * (-math.cos(theta_prev) + math.cos(theta_prev + w * dt))],
                [0.0, 1.0, R * (-math.sin(theta_prev) + math.sin(theta_prev + w * dt))],
                [0.0, 0.0, 1.0]
            ])

        Q = np.diag([
            (self.process_noise_v * dt)**2,
            (self.process_noise_v * dt)**2,
            (self.process_noise_w * dt)**2
        ])

        self.P = F @ self.P @ F.T + Q

        odom_msg = self.create_odometry_message()
        self.prediction_pub.publish(odom_msg)

        self.publish_path()

    def create_odometry_message(self):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        q = quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        pose_cov = [0.0] * 36
        pose_cov[0] = self.P[0, 0]   # x-x
        pose_cov[1] = self.P[0, 1]   # x-y
        pose_cov[6] = self.P[1, 0]   # y-x
        pose_cov[7] = self.P[1, 1]   # y-y
        pose_cov[35] = self.P[2, 2]  # yaw-yaw
        odom.pose.covariance = pose_cov

        # Twist
        odom.twist.twist.linear.x = self.last_v
        odom.twist.twist.angular.z = self.last_w

        twist_cov = [0.0] * 36
        twist_cov[0] = self.process_noise_v ** 2
        twist_cov[35] = self.process_noise_w ** 2
        odom.twist.covariance = twist_cov

        return odom

    def publish_path(self):
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y

        q = quaternion_from_euler(0, 0, self.theta)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
