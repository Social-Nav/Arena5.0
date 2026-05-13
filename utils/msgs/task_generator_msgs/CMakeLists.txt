cmake_minimum_required(VERSION 3.5)
project(task_generator_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetEnvironments.srv"
  "srv/GetObstacles.srv"
  "srv/GetParametrizeds.srv"
  "srv/GetRobotScenarios.srv"
  "srv/GetRobots.srv"
  "srv/GetScenarios.srv"
  "srv/GetWorlds.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
