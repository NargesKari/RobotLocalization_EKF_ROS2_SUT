#include <cmath>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Dense>

class WheelOdomNode : public rclcpp::Node {
public:
    WheelOdomNode()
            : rclcpp::Node("wheel_odom_node") {
        initParameters();
        initState();
        initCommunication();
    }

private:
    using JointState = sensor_msgs::msg::JointState;
    using Odometry = nav_msgs::msg::Odometry;

    std::string left_joint_name_;
    std::string right_joint_name_;
    std::string odom_frame_;
    std::string base_frame_;
    std::string odom_topic_;
    double wheel_radius_{0.0};
    double wheel_separation_{0.0};

    Eigen::Vector3d x_;
    bool first_{true};
    double last_left_pos_{0.0};
    double last_right_pos_{0.0};
    rclcpp::Time last_time_{0, 0, RCL_ROS_TIME};

    rclcpp::Subscription<JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<Odometry>::SharedPtr odom_pub_;

    void initParameters() {
        left_joint_name_ = declare_parameter<std::string>("left_wheel_joint", "drivewhl_l_joint");
        right_joint_name_ = declare_parameter<std::string>("right_wheel_joint", "drivewhl_r_joint");
        wheel_radius_ = declare_parameter<double>("wheel_radius", 0.1);
        wheel_separation_ = declare_parameter<double>("wheel_separation", 0.45);
        odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
        base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
        odom_topic_ = declare_parameter<std::string>("odom_topic", "/wheel_encoder/odom");
    }

    void initState() {
        x_.setZero();
    }

    void initCommunication() {
        odom_pub_ = create_publisher<Odometry>(odom_topic_, 10);

        joint_sub_ = create_subscription<JointState>(
                "/joint_states",
                10,
                std::bind(&WheelOdomNode::jointStateCallback, this, std::placeholders::_1));
    }

    static double wrapToPi(double a) {
        while (a <= -M_PI) {
            a += 2.0 * M_PI;
        }
        while (a > M_PI) {
            a -= 2.0 * M_PI;
        }
        return a;
    }

    void jointStateCallback(const JointState::SharedPtr msg) {
        if (msg->name.empty() || msg->position.empty()) {
            return;
        }

        size_t idxL = 0;
        size_t idxR = 0;
        if (!getWheelIndices(*msg, idxL, idxR)) {
            return;
        }

        if (idxL >= msg->position.size() || idxR >= msg->position.size()) {
            RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 2000,
                    "JointState positions not available for both wheel joints.");
            return;
        }

        const double left_pos = msg->position[idxL];
        const double right_pos = msg->position[idxR];
        const rclcpp::Time t = msg->header.stamp;

        if (first_) {
            first_ = false;
            last_left_pos_ = left_pos;
            last_right_pos_ = right_pos;
            last_time_ = t;
            return;
        }

        const double dt = (t - last_time_).seconds();
        if (dt <= 0.0) {
            return;
        }

        updateState(left_pos, right_pos);
        publishOdom(t, left_pos, right_pos, dt);

        last_left_pos_ = left_pos;
        last_right_pos_ = right_pos;
        last_time_ = t;
    }

    bool getWheelIndices(const JointState &msg, size_t &idxL, size_t &idxR) {
        std::unordered_map <std::string, size_t> name_to_index;
        name_to_index.reserve(msg.name.size());
        for (size_t i = 0; i < msg.name.size(); ++i) {
            name_to_index[msg.name[i]] = i;
        }

        const auto itL = name_to_index.find(left_joint_name_);

const auto itR = name_to_index.find(right_joint_name_);
        if (itL == name_to_index.end() || itR == name_to_index.end()) {
            RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 2000,
                    "Wheel joints '%s' or '%s' not found in JointState.",
                    left_joint_name_.c_str(), right_joint_name_.c_str());
            return false;
        }

        idxL = itL->second;
        idxR = itR->second;
        return true;
    }

    void updateState(double left_pos, double right_pos) {
        const double d_left = wheel_radius_ * (left_pos - last_left_pos_);
        const double d_right = wheel_radius_ * (right_pos - last_right_pos_);

        const double d_center = 0.5 * (d_right + d_left);
        const double d_theta = (d_right - d_left) / wheel_separation_;

        const double theta = x_(2);
        const double theta_mid = theta + 0.5 * d_theta;

        x_(0) += d_center * std::cos(theta_mid);
        x_(1) += d_center * std::sin(theta_mid);
        x_(2) = wrapToPi(theta + d_theta);
    }

    void publishOdom(const rclcpp::Time &stamp,
                     double left_pos,
                     double right_pos,
                     double dt) {
        const double d_left = wheel_radius_ * (left_pos - last_left_pos_);
        const double d_right = wheel_radius_ * (right_pos - last_right_pos_);
        const double d_center = 0.5 * (d_right + d_left);
        const double d_theta = (d_right - d_left) / wheel_separation_;

        const double v = d_center / dt;
        const double omega = d_theta / dt;

        Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id = base_frame_;

        odom.pose.pose.position.x = x_(0);
        odom.pose.pose.position.y = x_(1);
        odom.pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, x_(2));
        q.normalize();

        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        odom.twist.twist.linear.x = v;
        odom.twist.twist.angular.z = omega;

        odom_pub_->publish(odom);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WheelOdomNode>());
    rclcpp::shutdown();
    return 0;
}