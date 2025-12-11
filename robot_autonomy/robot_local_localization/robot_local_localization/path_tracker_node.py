#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from typing import Dict
import math

class PathTrackerNode(Node):
    
    def __init__(self):
        super().__init__('path_tracker_node')
        self.get_logger().info('✅ Path Tracker Node started. Sending commands and plotting paths...')

        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.3  # rad/s
        self.segment_time = 5.236 
        self.get_logger().info(f'Trajectory segment time is {self.segment_time:.3f} seconds for 90-degree turns.')

        
        self.paths: Dict[str, Path] = {}
        
        self.odom_topics = {
            'ekf': '/ekf/odom',
            'motion_model': '/ekf/motion_model',
            'measurement': '/measurement_model/odom',
            'real': '/ground_truth_odom'  
        }
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.path_publishers = {}
        for key in self.odom_topics.keys():
            path_topic = f'/path/{key}'
            self.path_publishers[key] = self.create_publisher(Path, path_topic, 10)
            self.paths[key] = Path()
            self.paths[key].header.frame_id = 'odom'


        for key, topic in self.odom_topics.items():
            self.create_subscription(
                Odometry,
                topic,
                lambda msg, k=key: self.odom_callback(msg, k),
                10
            )

        self.step = 0 
        self.timer = self.create_timer(0.1, self.publish_commands) 
        self.get_logger().info('Trajectory segment time is 5.0 seconds.')
        self.command_timer = self.create_timer(self.segment_time, self.next_command_step)
        
    def odom_callback(self, msg: Odometry, key: str):
        """جمع‌آوری داده‌های موقعیت از هر منبع و افزودن به مسیر."""
        
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.header.frame_id = 'odom'
        pose_stamped.pose = msg.pose.pose
        
        self.paths[key].poses.append(pose_stamped)
        
        self.path_publishers[key].publish(self.paths[key])

    def publish_commands(self):
        """انتشار فرمان سرعت بر اساس مرحله فعلی مسیر."""
        twist_msg = Twist()
        
        if self.step % 2 == 0:
            # مرحله زوج: حرکت مستقیم (ضلع مربع)
            twist_msg.linear.x = self.linear_speed
            twist_msg.angular.z = 0.0
            self.get_logger().debug(f'Step {self.step}: Moving straight.')
        else:
            # مرحله فرد: چرخش 90 درجه (گوشه مربع)
            twist_msg.linear.x = 0.0
            
            # 🟢 اصلاح: چرخش ثابت در یک جهت (مثلاً همیشه به چپ)
            if self.step in [1, 3, 5, 7]:
                twist_msg.angular.z = self.angular_speed  # چرخش به چپ
            else:
                twist_msg.angular.z = 0.0 # توقف در صورت اتمام چرخه

            self.get_logger().debug(f'Step {self.step}: Turning 90 degrees.')
        
        self.cmd_vel_pub.publish(twist_msg)

    def next_command_step(self):
        """تغییر مرحله مسیر پس از سپری شدن زمان هر سگمنت."""
        self.step += 1
        
        if self.step >= 8:
            self.step = 0
            self.get_logger().info('Rectangular trajectory cycle completed. Restarting.')

def main(args=None):
    rclpy.init(args=args)
    path_tracker_node = PathTrackerNode()
    
    try:
        rclpy.spin(path_tracker_node)
    except KeyboardInterrupt:
        pass
    
    path_tracker_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()