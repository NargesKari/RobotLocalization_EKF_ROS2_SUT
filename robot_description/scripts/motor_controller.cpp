#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <string>
#include <cmath>
#include <memory>
#include <vector>

class MotorControllerNode : public rclcpp::Node {
public:
    MotorControllerNode()
        : rclcpp::Node("motor_controller_node") {
        
        wheel_radius_ = this->declare_parameter<double>("wheel_radius", 0.1);    // R (متر)
        wheel_separation_ = this->declare_parameter<double>("wheel_separation", 0.45); // L (متر)
        
        motor_commands_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/motor_commands", 10);

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel",
            10,
            std::bind(&MotorControllerNode::cmdVelCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "MotorControllerNode ready. Publishing to /motor_commands.");
    }

private:
    double wheel_radius_;      
    double wheel_separation_;  

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr motor_commands_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        const double v_x = msg->linear.x;
        const double w_z = msg->angular.z;
        
        const double v_r = v_x + (w_z * wheel_separation_ / 2.0);
        const double v_l = v_x - (w_z * wheel_separation_ / 2.0);

        const double omega_r = v_r / wheel_radius_;
        const double omega_l = v_l / wheel_radius_;
        
        auto motor_commands_msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
        
         motor_commands_msg->data.push_back(omega_l);
        motor_commands_msg->data.push_back(omega_r);

        motor_commands_pub_->publish(std::move(motor_commands_msg));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorControllerNode>());
    rclcpp::shutdown();
    return 0;
}