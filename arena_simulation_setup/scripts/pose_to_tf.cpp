#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

class PoseToTF : public rclcpp::Node
{
public:
  PoseToTF()
  : Node("pose_to_tf")
  {
    this->declare_parameter<std::string>("odom_frame", "odom");
    odom_frame_ = this->get_parameter("odom_frame").as_string();

    this->declare_parameter<std::string>("pose_topic", "/jackal/pose");
    std::string pose_topic = this->get_parameter("pose_topic").as_string();

    // Publish an identity transform at origin immediately so the odom frame
    // always exists in the TF tree from node startup — before the first /pose
    // message arrives. Nav2 will never see "odom does not exist".
    last_transform_.header.frame_id = "map";
    last_transform_.child_frame_id = odom_frame_;
    last_transform_.transform.rotation.w = 1.0;

    subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      pose_topic, 10, std::bind(&PoseToTF::topic_callback, this, std::placeholders::_1));
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // Republish last known transform at 20 Hz so the odom frame stays alive
    // in the TF buffer even when Isaac Sim pauses during reset.
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&PoseToTF::timer_callback, this));
  }

private:
  void topic_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    last_transform_.header.frame_id = "map";
    last_transform_.child_frame_id = odom_frame_;
    last_transform_.transform.translation.x = msg->pose.position.x;
    last_transform_.transform.translation.y = msg->pose.position.y;
    last_transform_.transform.translation.z = msg->pose.position.z;
    last_transform_.transform.rotation = msg->pose.orientation;
  }

  void timer_callback()
  {
    last_transform_.header.stamp = this->now();
    tf_broadcaster_->sendTransform(last_transform_);
  }

  geometry_msgs::msg::TransformStamped last_transform_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string odom_frame_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseToTF>());
  rclcpp::shutdown();
  return 0;
}
