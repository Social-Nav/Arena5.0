"""Associate action statuses with one reset epoch, without ROS dependencies."""


class GoalEpoch:
    RUNNING = frozenset({1, 2})

    def __init__(self):
        self.seen = set()
        self.blocked = set()
        self.active = None
        self.generation = 0
        self.armed = False

    @staticmethod
    def key(status):
        return bytes(status.goal_info.goal_id.uuid)

    def begin(self):
        self.blocked = set(self.seen)
        self.active = None
        self.generation += 1
        self.armed = True

    def observe(self, statuses):
        statuses = list(statuses)
        self.seen.update(self.key(status) for status in statuses)
        if not self.armed:
            return None
        if self.active is None:
            candidates = [status for status in statuses
                          if self.key(status) not in self.blocked
                          and status.status in self.RUNNING]
            if not candidates:
                return None
            latest = max(candidates, key=lambda status: (
                status.goal_info.stamp.sec, status.goal_info.stamp.nanosec))
            self.active = self.key(latest)
        return next((status for status in statuses if self.key(status) == self.active), None)
