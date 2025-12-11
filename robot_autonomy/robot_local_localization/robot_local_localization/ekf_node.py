#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, TransformStamped, PoseStamped
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

    def _init_parameters(self):
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        
        self.ekf_rate = 100.0
        self.dt = 1.0 / self.ekf_rate

    def _init_state(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.state_vector = np.array([self.x, self.y, self.theta])
        
        self.P = np.diag([10.0, 10.0, 10.0])
        
        self.last_pred_msg: Optional[Odometry] = None
        self.last_meas_msg: Optional[Odometry] = None
        
        self.last_update_time = self.get_clock().now()
        
        self.current_v_x = 0.0
        self.current_omega_z = 0.0

    def _init_interfaces(self):
        self.ekf_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.path_pub = self.create_publisher(Path, '/ekf/path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        
        self.create_subscription(
            Odometry,
            '/ekf/motion_model',
            self.prediction_callback,
            10
        )
        
        self.create_subscription(
            Odometry,
            '/measurement_model/odom',
            self.measurement_callback,
            10
        )

    def prediction_callback(self, msg: Odometry):
        self.last_pred_msg = msg
        self.current_v_x = msg.twist.twist.linear.x
        self.current_omega_z = msg.twist.twist.angular.z

    def measurement_callback(self, msg: Odometry):
        self.last_meas_msg = msg

    def ekf_loop(self):
        
        if self.last_pred_msg is None:
            return

        self._prediction_step_from_msg(self.last_pred_msg)
        
        update_occurred = False
        if self.last_meas_msg is not None:
            
            if self.last_meas_msg.header.stamp.sec > self.last_update_time.to_msg().sec or \
               (self.last_meas_msg.header.stamp.sec == self.last_update_time.to_msg().sec and 
                self.last_meas_msg.header.stamp.nanosec > self.last_update_time.to_msg().nanosec):
                
                self._update_step_from_msg(self.last_meas_msg)
                update_occurred = True
        
        # اگر آپدیت رخ نداده باشد، هیچ خروجی منتشر نمی‌شود.
        if update_occurred:
            publish_stamp = self.last_meas_msg.header.stamp
            
            self.publish_ekf_output(publish_stamp)


    def _prediction_step_from_msg(self, msg: Odometry):
        
        q = msg.pose.pose.orientation
        theta_pred = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        x_pred = msg.pose.pose.position.x
        y_pred = msg.pose.pose.position.y

        P_pred_flat = msg.pose.covariance
        P_pred_full = np.array(P_pred_flat).reshape(6, 6)
        
        P_pred = np.array([
            [P_pred_full[0, 0], P_pred_full[0, 1], P_pred_full[0, 5]],
            [P_pred_full[1, 0], P_pred_full[1, 1], P_pred_full[1, 5]],
            [P_pred_full[5, 0], P_pred_full[5, 1], P_pred_full[5, 5]]
        ])

        self.state_vector = np.array([x_pred, y_pred, theta_pred])
        self.P = P_pred


    def _update_step_from_msg(self, msg: Odometry):
        
        q = msg.pose.pose.orientation
        theta_meas = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        x_meas = msg.pose.pose.position.x
        y_meas = msg.pose.pose.position.y
        Z_k = np.array([x_meas, y_meas, theta_meas])
        
        R_k_flat = msg.pose.covariance
        R_k_full = np.array(R_k_flat).reshape(6, 6)
        
        R_k = np.array([
            [R_k_full[0, 0], R_k_full[0, 1], R_k_full[0, 5]],
            [R_k_full[1, 0], R_k_full[1, 1], R_k_full[1, 5]],
            [R_k_full[5, 0], R_k_full[5, 1], R_k_full[5, 5]]
        ])
        
        H_k = np.identity(3)
        
        S_k = H_k @ self.P @ H_k.T + R_k
        
        try:
            K = self.P @ H_k.T @ np.linalg.inv(S_k)
        except np.linalg.LinAlgError:
            return
        
        Z_pred = self.state_vector
        innovation = Z_k - Z_pred

        innovation[2] = self.wrap_to_pi(innovation[2])
        
        self.state_vector = self.state_vector + K @ innovation
        self.state_vector[2] = self.wrap_to_pi(self.state_vector[2])
        
        I = np.identity(3)
        self.P = (I - K @ H_k) @ self.P
        
        self.last_update_time = self.get_clock().now()

    def publish_ekf_output(self, stamp):
        
        odom_msg = self.create_odom_message(stamp)
        self.ekf_pub.publish(odom_msg)
        
        self.publish_path(stamp)


    def create_odom_message(self, stamp) -> Odometry:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.state_vector[0]
        odom.pose.pose.position.y = self.state_vector[1]
        
        q = self.yaw_to_quaternion(self.state_vector[2])
        odom.pose.pose.orientation = q

        cov = np.diag([100.0] * 6)
        
        cov[0, 0] = self.P[0, 0]; cov[0, 1] = self.P[0, 1]; cov[0, 5] = self.P[0, 2]
        cov[1, 0] = self.P[1, 0]; cov[1, 1] = self.P[1, 1]; cov[1, 5] = self.P[1, 2]
        cov[5, 0] = self.P[2, 0]; cov[5, 1] = self.P[2, 1]; cov[5, 5] = self.P[2, 2]
        
        odom.pose.covariance = cov.flatten().tolist()
        
        odom.twist.twist.linear.x = self.current_v_x
        odom.twist.twist.angular.z = self.current_omega_z

        twist_cov = np.diag([0.05**2, 100.0, 100.0, 100.0, 100.0, 0.05**2])
        odom.twist.covariance = twist_cov.flatten().tolist()
        
        return odom

    def publish_path(self, stamp):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = 'odom'
        
        pose_stamped.pose.position.x = self.state_vector[0]
        pose_stamped.pose.position.y = self.state_vector[1]
        
        q = self.yaw_to_quaternion(self.state_vector[2])
        pose_stamped.pose.orientation = q
        
        self.path_msg.poses.append(pose_stamped)
        self.path_pub.publish(self.path_msg)


    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
        q = Quaternion()
        q.x = quaternion[0]
        q.y = quaternion[1]
        q.z = quaternion[2]
        q.w = quaternion[3]
        return q

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi


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