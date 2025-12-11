import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import tf_transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped

class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction_node')

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.declare_parameter('wheel_separation', 0.45)
        self.declare_parameter('wheel_radius', 0.1)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('rate', 100.0) # افزایش نرخ برای دقت بیشتر
        self.declare_parameter('process_noise_v', 0.05)
        self.declare_parameter('process_noise_w', 0.05)

        self.wheel_separation = self.get_parameter('wheel_separation').get_parameter_value().double_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        self.process_noise_v = self.get_parameter('process_noise_v').get_parameter_value().double_value
        self.process_noise_w = self.get_parameter('process_noise_w').get_parameter_value().double_value
        
        self.dt = 1.0 / self.rate
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.P = np.diag([0.1, 0.1, 0.1])
        
        self.last_v = 0.0
        self.last_w = 0.0

        self.subscription = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10 # QoS
        )
        
        # ناشر Odometry
        self.prediction_pub = self.create_publisher(Odometry, '/ekf/motion_model', 10) 
        
        # ناشر Path (مسیر) - جدید
        self.path_pub = self.create_publisher(Path, '/ekf/path', 10) 
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        
        self.timer = self.create_timer(self.dt, self.prediction_update)
        self.get_logger().info('Prediction Node started.')


    def cmd_vel_callback(self, msg):
        """دریافت آخرین فرمان سرعت (v, w) از cmd_vel"""
        self.last_v = msg.linear.x
        self.last_w = msg.angular.z
        
        self.get_logger().info(f'Received Cmd_Vel: v={self.last_v:.2f}, w={self.last_w:.2f}')

    def prediction_update(self):
        """اجرای گام پیش‌بینی EKF (مدل حرکت)"""
        
        v = self.last_v
        w = self.last_w
        dt = self.dt
        
        # محاسبه تغییرات موقعیت (مدل دیفرانسیل)
        if abs(w) < 1e-6:
            delta_x = v * dt * np.cos(self.theta)
            delta_y = v * dt * np.sin(self.theta)
        else:
            R = v / w
            delta_x = R * (-np.sin(self.theta) + np.sin(self.theta + w * dt))
            delta_y = R * (np.cos(self.theta) - np.cos(self.theta + w * dt))
            
        self.x += delta_x
        self.y += delta_y
        self.theta = (self.theta + w * dt + np.pi) % (2 * np.pi) - np.pi # نرمال‌سازی زاویه
                
        # به‌روزرسانی کوواریانس (P)
        if abs(w) < 1e-6:
            F = np.array([
                [1.0, 0.0, -v * dt * np.sin(self.theta)],
                [0.0, 1.0, v * dt * np.cos(self.theta)],
                [0.0, 0.0, 1.0]
            ])
        else:
            R = v / w
            F = np.array([
                [1.0, 0.0, R * (-np.cos(self.theta) + np.cos(self.theta + w * dt))],
                [0.0, 1.0, R * (-np.sin(self.theta) + np.sin(self.theta + w * dt))],
                [0.0, 0.0, 1.0]
            ])
            
        Q = np.diag([
            (self.process_noise_v * dt)**2 * np.cos(self.theta)**2,
            (self.process_noise_v * dt)**2 * np.sin(self.theta)**2,
            (self.process_noise_w * dt)**2
        ])
        
        self.P = F @ self.P @ F.T + Q

        # انتشار TF (تبدیل odom به base_link)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom' 
        t.child_frame_id = 'base_link' # تغییر داده شد

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)

        # انتشار Odometry
        odom_msg = self.create_odometry_message()
        self.prediction_pub.publish(odom_msg)
        
        # انتشار Path (جدید)
        self.publish_path()

    def create_odometry_message(self):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom' 
        odom.child_frame_id = 'base_link' # تغییر داده شد

        # موقعیت و جهت
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]
        
        # کوواریانس
        cov = np.zeros((6, 6))
        cov[0, 0] = self.P[0, 0]; cov[0, 1] = self.P[0, 1]
        cov[1, 0] = self.P[1, 0]; cov[1, 1] = self.P[1, 1]
        cov[5, 5] = self.P[2, 2] # yaw-yaw
        
        # اصلاح کوواریانس برای Z, Roll, Pitch (به 100.0)
        cov[2, 2] = 100.0
        cov[3, 3] = 100.0
        cov[4, 4] = 100.0
        
        odom.pose.covariance = cov.flatten().tolist()
        
        return odom

    def publish_path(self):
        """اضافه کردن پوز فعلی به مسیر و انتشار آن."""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'odom'
        
        pose_stamped.pose.position.x = self.x
        pose_stamped.pose.position.y = self.y
        
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        pose_stamped.pose.orientation.x = quaternion[0]
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]
        
        self.path_msg.poses.append(pose_stamped)
        
        # if len(self.path_msg.poses) > 100:
        #     self.path_msg.poses.pop(0)
            
        self.path_pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    prediction_node = PredictionNode()
    rclpy.spin(prediction_node)
    prediction_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()