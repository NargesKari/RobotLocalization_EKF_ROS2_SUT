#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from typing import Optional


class MeasurementNode(Node):

    def __init__(self):
        super().__init__("measurement_node")

        self.declare_parameter("imu_topic", "/zed/zed_node/imu/data_raw")
        self.declare_parameter("vo_topic", "/vo/odom")
        self.declare_parameter("measurement_topic", "/measurement_model/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("imu_yaw_stddev", 0.02)

        self.imu_topic = self.get_parameter("imu_topic").value
        self.vo_topic = self.get_parameter("vo_topic").value
        self.measurement_topic = self.get_parameter("measurement_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.imu_yaw_stddev = float(self.get_parameter("imu_yaw_stddev").value)
        self.imu_yaw_var = self.imu_yaw_stddev ** 2

        self.last_imu_yaw: Optional[float] = None

        self.vo_z_offset: Optional[float] = None
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)
        self.create_subscription(Odometry, self.vo_topic, self.vo_callback, 10)
        self.meas_pub = self.create_publisher(Odometry, self.measurement_topic, 10)

        self.get_logger().info("✅ Measurement node started.")

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.last_imu_yaw = yaw
        if len(msg.orientation_covariance) == 9 and msg.orientation_covariance[8] > 0.0:
            self.imu_yaw_var = float(msg.orientation_covariance[8])

    def vo_callback(self, msg: Odometry):

        if self.last_imu_yaw is None:
            return 

        meas = Odometry()
        meas.header = msg.header
        meas.header.frame_id = self.odom_frame
        meas.child_frame_id = self.base_frame

        # Position (normalize Z)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        meas = Odometry()
        meas.header = msg.header
        meas.header.frame_id = self.odom_frame
        meas.child_frame_id = self.base_frame

        meas.pose.pose.position.x = x
        meas.pose.pose.position.y = y
        meas.pose.pose.position.z = 0.0

        meas.pose.pose.orientation = self.yaw_to_quaternion(self.last_imu_yaw)

        meas.pose.covariance = [0.0] * 36

        if len(msg.pose.covariance) == 36:
            meas.pose.covariance[0] = msg.pose.covariance[0]  # cov(x)
            meas.pose.covariance[7] = msg.pose.covariance[7]  # cov(y)
        else:
            meas.pose.covariance[0] = 0.05 ** 2
            meas.pose.covariance[7] = 0.05 ** 2

        meas.pose.covariance[35] = self.imu_yaw_var

        meas.twist = msg.twist

        self.meas_pub.publish(meas)

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        half = yaw * 0.5
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(half)
        q.w = math.cos(half)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
