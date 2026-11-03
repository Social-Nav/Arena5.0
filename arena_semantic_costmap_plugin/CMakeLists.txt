cmake_minimum_required(VERSION 3.8)
project(arena_semantic_costmap_plugin)

set(lib_name ${PROJECT_NAME}_core)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(${lib_name} SHARED src/semantic_layer.cpp)
target_include_directories(${lib_name} PUBLIC include)
ament_target_dependencies(${lib_name}
  rclcpp
  nav2_costmap_2d
  pluginlib
  nav_msgs
  std_msgs)

install(TARGETS ${lib_name} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

pluginlib_export_plugin_description_file(nav2_costmap_2d semantic_layer.xml)

ament_package()
