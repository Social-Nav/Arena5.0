#include "arena_semantic_costmap_plugin/semantic_layer.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "nav2_costmap_2d/cost_values.hpp"

namespace arena_semantic_costmap_plugin
{

void SemanticLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("SemanticLayer requires a node");
  }
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("semantic_enabled", rclcpp::ParameterValue(true));
  declareParameter("semantic_topic", rclcpp::ParameterValue(""));
  declareParameter("semantic_ttl_sec", rclcpp::ParameterValue(0.0));
  declareParameter("status_topic", rclcpp::ParameterValue("/lisn_semantic_layer_status"));
  getParameters();

  status_pub_ = node->create_publisher<std_msgs::msg::String>(status_topic_, 10);
  if (!semantic_topic_.empty()) {
    semantic_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
      semantic_topic_, rclcpp::QoS(1).reliable(),
      std::bind(&SemanticLayer::semanticCallback, this, std::placeholders::_1));
    RCLCPP_INFO(logger_, "SemanticLayer: subscribed to %s", semantic_topic_.c_str());
  } else {
    RCLCPP_INFO(logger_, "SemanticLayer: no semantic_topic; layer is a no-op");
  }
}

void SemanticLayer::getParameters()
{
  auto node = node_.lock();
  if (!node) {
    return;
  }
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".semantic_enabled", semantic_enabled_);
  node->get_parameter(name_ + ".semantic_ttl_sec", ttl_sec_);
  node->get_parameter(name_ + ".semantic_topic", semantic_topic_);
  node->get_parameter(name_ + ".status_topic", status_topic_);
}

void SemanticLayer::semanticCallback(
  const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  grid_ = *msg;
  grid_valid_ = true;
  ++semantic_msg_count_;
}

void SemanticLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  getParameters();
  nav_msgs::msg::OccupancyGrid grid;
  bool valid = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    valid = grid_valid_;
    if (valid) {
      grid = grid_;
    }
  }
  if (enabled_ && semantic_enabled_ && valid && grid.info.resolution > 0.0) {
    const double origin_x = grid.info.origin.position.x;
    const double origin_y = grid.info.origin.position.y;
    const double end_x = origin_x + grid.info.width * grid.info.resolution;
    const double end_y = origin_y + grid.info.height * grid.info.resolution;
    *min_x = std::min(*min_x, origin_x);
    *min_y = std::min(*min_y, origin_y);
    *max_x = std::max(*max_x, end_x);
    *max_y = std::max(*max_y, end_y);
  }
  if (last_extent_valid_) {
    // Keep the previous applied region in the update window so expiry/disable
    // can actually clear the cells the layer wrote last cycle.
    *min_x = std::min(*min_x, last_min_x_);
    *min_y = std::min(*min_y, last_min_y_);
    *max_x = std::max(*max_x, last_max_x_);
    *max_y = std::max(*max_y, last_max_y_);
  }
}

void SemanticLayer::restoreWrittenCells(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  const unsigned int size_x = master_grid.getSizeInCellsX();
  for (auto & entry : written_cells_) {
    const unsigned int index = entry.first;
    const unsigned char old_cost = entry.second.first;
    const unsigned char written_cost = entry.second.second;
    const unsigned int mx = index % size_x;
    const unsigned int my = index / size_x;
    if (static_cast<int>(mx) < min_i || static_cast<int>(mx) >= max_i ||
      static_cast<int>(my) < min_j || static_cast<int>(my) >= max_j)
    {
      continue;
    }
    // Only remove our contribution.  If another layer raised the cell above
    // our written cost (static/obstacle lethal), leave it untouched.
    if (master_grid.getCost(mx, my) == written_cost) {
      master_grid.setCost(mx, my, old_cost);
    }
  }
  written_cells_.clear();
}

void SemanticLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  getParameters();
  ++update_costs_count_;
  nav_msgs::msg::OccupancyGrid grid;
  bool apply = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    restoreWrittenCells(master_grid, min_i, min_j, max_i, max_j);
    writes_changed_ = 0;
    sample_mx_ = -1;
    sample_my_ = -1;
    sample_old_cost_ = 0;
    sample_new_cost_ = 0;
    if (enabled_ && semantic_enabled_ && grid_valid_) {
      grid = grid_;
      apply = true;
    } else {
      last_extent_valid_ = false;
    }
  }
  if (!apply) {
    publishStatus(0, 0, 0);
    return;
  }
  if (ttl_sec_ > 0.0) {
    const double age = (clock_->now() - rclcpp::Time(grid.header.stamp)).seconds();
    if (age > ttl_sec_) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        grid_valid_ = false;
        last_extent_valid_ = false;
      }
      publishStatus(0, 0, 0);
      return;
    }
  }
  const std::string global_frame = layered_costmap_->getGlobalFrameID();
  if (grid.header.frame_id != global_frame) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, 5000,
      "SemanticLayer: grid frame %s != costmap frame %s; skipping",
      grid.header.frame_id.c_str(), global_frame.c_str());
    publishStatus(0, 0, 0);
    return;
  }
  const double resolution = grid.info.resolution;
  if (resolution <= 0.0 || grid.info.width == 0 || grid.info.height == 0) {
    publishStatus(0, 0, 0);
    return;
  }
  uint64_t nonzero = 0;
  int max_cost = 0;
  int sample_cost = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t iy = 0; iy < grid.info.height; ++iy) {
      const double wy = grid.info.origin.position.y + (iy + 0.5) * resolution;
      for (uint32_t ix = 0; ix < grid.info.width; ++ix) {
        const int8_t value = grid.data[iy * grid.info.width + ix];
        if (value <= 0) {
          continue;
        }
        const double wx = grid.info.origin.position.x + (ix + 0.5) * resolution;
        unsigned int mx = 0;
        unsigned int my = 0;
        if (!master_grid.worldToMap(wx, wy, mx, my)) {
          continue;
        }
        if (static_cast<int>(mx) < min_i || static_cast<int>(mx) >= max_i ||
          static_cast<int>(my) < min_j || static_cast<int>(my) >= max_j)
        {
          continue;
        }
        const int cost = std::min(254, static_cast<int>(value) * 254 / 100);
        const unsigned char old_cost = master_grid.getCost(mx, my);
        if (old_cost == nav2_costmap_2d::NO_INFORMATION) {
          continue;
        }
        const unsigned char new_cost = std::max(old_cost, static_cast<unsigned char>(cost));
        if (new_cost != old_cost) {
          master_grid.setCost(mx, my, new_cost);
          written_cells_[master_grid.getIndex(mx, my)] =
            std::make_pair(old_cost, new_cost);
          if (sample_mx_ < 0) {
            sample_mx_ = static_cast<int>(mx);
            sample_my_ = static_cast<int>(my);
            sample_old_cost_ = static_cast<int>(old_cost);
            sample_new_cost_ = static_cast<int>(new_cost);
          }
          ++writes_changed_;
        }
        ++nonzero;
        max_cost = std::max(max_cost, cost);
      }
    }
    sample_cost = sample_new_cost_;
    last_min_x_ = grid.info.origin.position.x;
    last_min_y_ = grid.info.origin.position.y;
    last_max_x_ = grid.info.origin.position.x + grid.info.width * resolution;
    last_max_y_ = grid.info.origin.position.y + grid.info.height * resolution;
    last_extent_valid_ = true;
  }
  publishStatus(nonzero, max_cost, sample_cost);
}

void SemanticLayer::publishStatus(uint64_t nonzero_cells, int max_cost, int sample_cost)
{
  if (!status_pub_) {
    return;
  }
  bool enabled = true;
  bool semantic_enabled = true;
  double ttl_sec = 0.0;
  bool valid = false;
  std::string grid_frame;
  uint64_t grid_cells = 0;
  uint64_t semantic_msg_count = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled = enabled_;
    semantic_enabled = semantic_enabled_;
    ttl_sec = ttl_sec_;
    valid = grid_valid_;
    semantic_msg_count = semantic_msg_count_;
    if (valid) {
      grid_frame = grid_.header.frame_id;
      grid_cells = static_cast<uint64_t>(grid_.info.width) * grid_.info.height;
    }
  }
  std_msgs::msg::String message;
  message.data = std::string("{\"schema\":\"arena.lisn.semantic_layer.v1\",")
    + "\"layer\":\"arena_semantic_costmap_plugin::SemanticLayer\","
    + "\"enabled\":" + (enabled ? "true" : "false") + ","
    + "\"semantic_enabled\":" + (semantic_enabled ? "true" : "false") + ","
    + "\"semantic_msg_count\":" + std::to_string(semantic_msg_count) + ","
    + "\"update_costs_count\":" + std::to_string(update_costs_count_) + ","
    + "\"writes_changed\":" + std::to_string(writes_changed_) + ","
    + "\"nonzero_cells\":" + std::to_string(nonzero_cells) + ","
    + "\"max_cost\":" + std::to_string(max_cost) + ","
    + "\"sample_cost\":" + std::to_string(sample_cost) + ","
    + "\"sample_mx\":" + std::to_string(sample_mx_) + ","
    + "\"sample_my\":" + std::to_string(sample_my_) + ","
    + "\"sample_old_cost\":" + std::to_string(sample_old_cost_) + ","
    + "\"sample_new_cost\":" + std::to_string(sample_new_cost_) + ","
    + "\"ttl_sec\":" + std::to_string(ttl_sec) + ","
    + "\"grid_valid\":" + (valid ? "true" : "false") + ","
    + "\"grid_frame\":\"" + grid_frame + "\","
    + "\"grid_cells\":" + std::to_string(grid_cells) + "}";
  status_pub_->publish(message);
}

void SemanticLayer::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  grid_valid_ = false;
  written_cells_.clear();
  last_extent_valid_ = false;
}

}  // namespace arena_semantic_costmap_plugin

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  arena_semantic_costmap_plugin::SemanticLayer,
  nav2_costmap_2d::Layer)
