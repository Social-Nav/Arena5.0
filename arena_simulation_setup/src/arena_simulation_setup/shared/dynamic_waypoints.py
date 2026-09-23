from __future__ import annotations


def normalize_dynamic_waypoints(waypoints) -> tuple[list[list], list[float | None]]:
    """Normalize legacy and speed-annotated pedestrian waypoints."""
    normalized_waypoints = []
    waypoint_velocities = []
    for index, waypoint in enumerate(waypoints or []):
        desired_velocity = None
        if isinstance(waypoint, dict):
            desired_velocity = waypoint.get('desired_velocity')
            waypoint_pose = waypoint.get('pose', waypoint.get('position'))
            if waypoint_pose is None and 'x' in waypoint and 'y' in waypoint:
                waypoint_pose = [
                    waypoint['x'],
                    waypoint['y'],
                    waypoint.get('yaw', waypoint.get('heading', 0.0)),
                ]
        else:
            waypoint_pose = waypoint
            if isinstance(waypoint, (list, tuple)) and len(waypoint) == 4:
                waypoint_pose = waypoint[:3]
                desired_velocity = waypoint[3]

        if not isinstance(waypoint_pose, (list, tuple)) or len(waypoint_pose) not in (2, 3):
            raise ValueError(
                f"waypoints[{index}] must be [x,y], [x,y,heading], "
                "[x,y,heading,desired_velocity], or a mapping with pose"
            )
        normalized_waypoints.append(list(waypoint_pose))
        if desired_velocity is None:
            waypoint_velocities.append(None)
        else:
            desired_velocity = float(desired_velocity)
            if desired_velocity <= 0.0:
                raise ValueError(
                    f"waypoints[{index}].desired_velocity must be positive, "
                    f"got {desired_velocity}"
                )
            waypoint_velocities.append(desired_velocity)

    return normalized_waypoints, waypoint_velocities
