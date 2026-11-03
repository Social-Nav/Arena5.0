#ifndef ARENA_SEMANTIC_COSTMAP_PLUGIN_SEMANTIC_LAYER_HPP_
#define ARENA_SEMANTIC_COSTMAP_PLUGIN_SEMANTIC_LAYER_HPP_

#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "std_msgs/msg/string.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace arena_semantic_costmap_plugin
{

class SemanticLayer : public nav2_costmap_2d::Layer
{
public:
  SemanticLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return false;}
  bool isDiscretized() {return false;}

private:
  void semanticCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void getParameters();
  void restoreWrittenCells(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j);
  void publishStatus(uint64_t nonzero_cells, int max_cost, int sample_cost);

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;

  std::mutex mutex_;
  nav_msgs::msg::OccupancyGrid grid_;
  bool grid_valid_{false};
  bool enabled_{true};
  bool semantic_enabled_{true};
  double ttl_sec_{0.0};
  std::string semantic_topic_;
  std::string status_topic_;

  // semanticCallback message count; updateCosts is counted separately.
  uint64_t semantic_msg_count_{0};
  uint64_t update_costs_count_{0};
  uint64_t writes_changed_{0};

  // Previous semantic writes: master index -> (old_cost, written_cost).
  std::map<unsigned int, std::pair<unsigned char, unsigned char>> written_cells_;
  int sample_mx_{-1};
  int sample_my_{-1};
  int sample_old_cost_{0};
  int sample_new_cost_{0};

  // Last applied grid extent, so updateCosts is called over the old region
  // after TTL expiry/disable and can actually clear the semantic contribution.
  bool last_extent_valid_{false};
  double last_min_x_{0.0};
  double last_min_y_{0.0};
  double last_max_x_{0.0};
  double last_max_y_{0.0};
};

}  // namespace arena_semantic_costmap_plugin

#endif  // ARENA_SEMANTIC_COSTMAP_PLUGIN_SEMANTIC_LAYER_HPP_
