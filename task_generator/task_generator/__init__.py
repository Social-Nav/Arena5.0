import typing

import rclpy
import rclpy.callback_groups
import rclpy.impl.rcutils_logger
import rclpy.node

try:
    from typing import Self
except ImportError:
    Self = typing.TypeVar('Self')


class SafeCallbackNode(rclpy.node.Node):
    """
    Automatically make clients part of a new MutuallyExclusiveCallbackGroup to avoid deadlocks.
    """

    @property
    def default_callback_group(self) -> rclpy.callback_groups.CallbackGroup:
        return rclpy.callback_groups.ReentrantCallbackGroup()


if typing.TYPE_CHECKING:
    from task_generator.node import TaskGenerator
else:
    TaskGenerator = object()


class NodeInterface:

    def __init__(self, *args, node: TaskGenerator, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__node = node

    @property
    def node(self) -> TaskGenerator:
        return self.__node

    @property
    def _logger(self) -> rclpy.impl.rcutils_logger.RcutilsLogger:
        return self.node.get_logger().get_child(type(self).__name__)
