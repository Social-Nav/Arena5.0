#! /usr/bin/env python3
import asyncio
import threading
import traceback
import rclpy
import rclpy.action
import rclpy.executors
from .node import TaskGenerator


def patch_action_client_lock_race():
    """Pre-create ActionClient._lock before it is added to the executor.

    ROS 2 Jazzy's rclpy ActionClient initializes ``_lock`` at the end of
    ``__init__``, after the waitable has already been registered on the node.
    The task generator spins a MultiThreadedExecutor while async setup is still
    creating the Nav2 action client, so the executor can briefly observe a
    partially constructed ActionClient and call get_num_entities() before
    ``_lock`` exists.  Pre-creating the lock keeps that construction window safe
    while preserving the upstream implementation for all normal behavior.
    """
    action_client_cls = rclpy.action.ActionClient
    if getattr(action_client_cls, '_arena_precreates_lock', False):
        return

    original_init = action_client_cls.__init__

    def _arena_safe_init(self, *args, **kwargs):
        if not hasattr(self, '_lock'):
            self._lock = threading.Lock()
        original_init(self, *args, **kwargs)

    action_client_cls.__init__ = _arena_safe_init
    action_client_cls._arena_precreates_lock = True


def spin_blocking(executor):
    try:
        executor.spin()
    except rclpy.executors.ExternalShutdownException:
        pass


async def app_logic(node):
    node.get_logger().info('Beginning client, shut down with CTRL-C')
    await node.setup()
    stop_event = asyncio.Event()
    await stop_event.wait()


async def main_async(args=None):
    del args
    rclpy.init()
    patch_action_client_lock_race()
    loop = asyncio.get_running_loop()

    executor = rclpy.executors.MultiThreadedExecutor()

    node = TaskGenerator()
    node.event_loop = loop

    executor.add_node(node)

    spin_future = loop.run_in_executor(None, spin_blocking, executor)
    app_task = asyncio.create_task(app_logic(node))

    try:
        try:
            import aiomonitor
        except ImportError:
            aiomonitor = None

        if aiomonitor is None:
            node.get_logger().warn('`aiomonitor` not installed; continuing without async monitor')
            done, _ = await asyncio.wait(
                [spin_future, app_task],
                return_when=asyncio.FIRST_COMPLETED
            )
        else:
            with aiomonitor.start_monitor(loop=loop, locals=locals()):
                done, _ = await asyncio.wait(
                    [spin_future, app_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

        if spin_future in done:
            spin_future.result()

        if app_task in done:
            app_task.result()

    except asyncio.CancelledError:
        node.get_logger().info('Shutting down.')
    except Exception:
        node.get_logger().error(traceback.format_exc())
        raise
    finally:
        if not app_task.done():
            app_task.cancel()

        executor.shutdown()

        try:
            await spin_future
        except Exception:
            pass

        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


def main(args=None):
    try:
        asyncio.run(main_async(args=args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import time
    time.sleep(5)
    main()
