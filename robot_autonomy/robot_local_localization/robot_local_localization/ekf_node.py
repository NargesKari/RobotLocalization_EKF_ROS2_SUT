#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import tf_transformations
import math
from typing import Optional


class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        self._init_parameters()
        self._init_state()
        self._init_interfaces()

        self.ekf_timer = self.create_timer(self.dt, self.ekf_loop)

    # -------------------------
    # Parameters
    # -------------------------
    def _init_parameters(self):
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ekf_rate', 100.0)
        self.declare_parameter('max_path_length', 1000)  # optional limit

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ekf_rate = float(self.get_parameter('ekf_rate').value)
        self.max_path_length = int(self.get_parameter('max_path_length').value)

        self.dt = 1.0 / self.ekf_rate

    # -------------------------
    # State
    # -------------------------
    def _init_state(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.state_vector = np.array([self.x, self.y, self.theta], dtype=float)

        # initial (large) covariance if you're uncertain
        self.P = np.diag([10.0, 10.0, 10.0])

        self.last_pred_msg: Optional[Odometry] = None
        self.last_meas_msg: Optional[Odometry] = None

        self.last_update_time = self.get_clock().now()

        self.current_v_x = 0.0
        self.current_omega_z = 0.0

    # -------------------------
    # Interfaces
    # -------------------------
    def _init_interfaces(self):
        self.ekf_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.path_pub = self.create_publisher(Path, '/ekf/path', 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame
        self.path_msg.poses = []

        self.create_subscription(Odometry, '/ekf/motion_model', self.prediction_callback, 10)
        self.create_subscription(Odometry, '/measurement_model/odom', self.measurement_callback, 10)

    # -------------------------
    # Callbacks
    # -------------------------
    def prediction_callback(self, msg: Odometry):
        self.last_pred_msg = msg
        self.current_v_x = msg.twist.twist.linear.x
        self.current_omega_z = msg.twist.twist.angular.z

    def measurement_callback(self, msg: Odometry):
        self.last_meas_msg = msg

    # -------------------------
    # EKF loop
    # -------------------------
    def ekf_loop(self):
        if self.last_pred_msg is None:
            return

        # use the motion_model odometry as the prediction (you already publish P there)
        self._prediction_step_from_msg(self.last_pred_msg)

        update_occurred = False
        if self.last_meas_msg is not None:
            if self._is_stamp_newer(self.last_meas_msg.header.stamp, self.last_update_time.to_msg()):
                self._update_step_from_msg(self.last_meas_msg)
                update_occurred = True

        if update_occurred:
            publish_stamp = self.last_meas_msg.header.stamp
            self.publish_ekf_output(publish_stamp)

    # -------------------------
    # Prediction: adopt motion_model pose & covariance
    # -------------------------
    def _prediction_step_from_msg(self, msg: Odometry):
        q = msg.pose.pose.orientation
        theta_pred = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        x_pred = msg.pose.pose.position.x
        y_pred = msg.pose.pose.position.y

        P_pred_flat = msg.pose.covariance
        if len(P_pred_flat) == 36:
            P_pred_full = np.array(P_pred_flat).reshape((6, 6))
            # Extract 3x3 for [x, y, yaw] (indices 0,1,5 in 6x6)
            P_pred = np.array([
                [P_pred_full[0, 0], P_pred_full[0, 1], P_pred_full[0, 5]],
                [P_pred_full[1, 0], P_pred_full[1, 1], P_pred_full[1, 5]],
                [P_pred_full[5, 0], P_pred_full[5, 1], P_pred_full[5, 5]]
            ], dtype=float)
        else:
            # fallback if covariance missing/incorrect
            P_pred = np.diag([1.0, 1.0, 0.5])

        self.state_vector = np.array([x_pred, y_pred, theta_pred], dtype=float)
        self.P = P_pred

    # -------------------------
    # Update: fuse measurement
    # -------------------------
    def _update_step_from_msg(self, msg: Odometry):
        q = msg.pose.pose.orientation
        theta_meas = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        x_meas = msg.pose.pose.position.x
        y_meas = msg.pose.pose.position.y
        Z_k = np.array([x_meas, y_meas, theta_meas], dtype=float)

        # measurement covariance R_k (3x3)
        R_k_flat = msg.pose.covariance
        if len(R_k_flat) == 36:
            R_k_full = np.array(R_k_flat).reshape((6, 6))
            R_k = np.array([
                [R_k_full[0, 0], R_k_full[0, 1], R_k_full[0, 5]],
                [R_k_full[1, 0], R_k_full[1, 1], R_k_full[1, 5]],
                [R_k_full[5, 0], R_k_full[5, 1], R_k_full[5, 5]]
            ], dtype=float)
        else:
            R_k = np.diag([0.05**2, 0.05**2, 0.02**2])

        H_k = np.identity(3, dtype=float)

        S_k = H_k @ self.P @ H_k.T + R_k

        # compute Kalman gain safely
        try:
            K = self.P @ H_k.T @ np.linalg.inv(S_k)
        except np.linalg.LinAlgError:
            self.get_logger().warn("S_k is singular; skipping update.")
            return

        Z_pred = self.state_vector.copy()
        innovation = Z_k - Z_pred

        # normalize angle innovation
        innovation[2] = self.wrap_to_pi(innovation[2])

        self.state_vector = self.state_vector + K @ innovation
        self.state_vector[2] = self.wrap_to_pi(self.state_vector[2])

        I = np.identity(3, dtype=float)
        self.P = (I - K @ H_k) @ self.P

        # update timestamp of last measurement consumed
        self.last_update_time = self.get_clock().now()

    # -------------------------
    # Publish results
    # -------------------------
    def publish_ekf_output(self, stamp):
        odom_msg = self.create_odom_message(stamp)
        self.ekf_pub.publish(odom_msg)
        self.publish_path(stamp)

    def create_odom_message(self, stamp) -> Odometry:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = float(self.state_vector[0])
        odom.pose.pose.position.y = float(self.state_vector[1])

        q = self.yaw_to_quaternion(float(self.state_vector[2]))
        odom.pose.pose.orientation = q

        # build 6x6 covariance and flatten
        cov6 = np.zeros((6, 6), dtype=float)
        cov6[0, 0] = float(self.P[0, 0])
        cov6[0, 1] = float(self.P[0, 1])
        cov6[0, 5] = float(self.P[0, 2])

        cov6[1, 0] = float(self.P[1, 0])
        cov6[1, 1] = float(self.P[1, 1])
        cov6[1, 5] = float(self.P[1, 2])

        cov6[5, 0] = float(self.P[2, 0])
        cov6[5, 1] = float(self.P[2, 1])
        cov6[5, 5] = float(self.P[2, 2])

        odom.pose.covariance = cov6.flatten().tolist()

        odom.twist.twist.linear.x = float(self.current_v_x)
        odom.twist.twist.angular.z = float(self.current_omega_z)

        twist_cov = np.diag([0.05**2, 100.0, 100.0, 100.0, 100.0, 0.05**2])
        odom.twist.covariance = twist_cov.flatten().tolist()

        return odom

    def publish_path(self, stamp):
        self.path_msg.header.stamp = stamp

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.odom_frame
        pose_stamped.pose.position.x = float(self.state_vector[0])
        pose_stamped.pose.position.y = float(self.state_vector[1])

        q = self.yaw_to_quaternion(float(self.state_vector[2]))
        pose_stamped.pose.orientation = q

        self.path_msg.poses.append(pose_stamped)

        # limit path length to avoid unbounded growth
        if len(self.path_msg.poses) > self.max_path_length:
            self.path_msg.poses = self.path_msg.poses[-self.max_path_length :]

        self.path_pub.publish(self.path_msg)

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        q = Quaternion()
        q.x = quat[0]
        q.y = quat[1]
        q.z = quat[2]
        q.w = quat[3]
        return q

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _is_stamp_newer(self, stamp_msg, last_time_msg) -> bool:
        # stamp_msg is a builtin_interfaces/Time (with sec/nanosec)
        # last_time_msg is same type (from self.last_update_time.to_msg())
        if stamp_msg.sec > last_time_msg.sec:
            return True
        if stamp_msg.sec == last_time_msg.sec and stamp_msg.nanosec > last_time_msg.nanosec:
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()
    try:
        rclpy.spin(ekf_node)
    except KeyboardInterrupt:
        pass
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
