from __future__ import annotations

import asyncio
import functools
import inspect
import typing

import rclpy.client
import rclpy.node
import rclpy.qos

import launch

from arena_rclpy_mixins.Time import TimeNode

T = typing.TypeVar('T')


class AsyncLaunchManager:
    def __init__(self):
        self.active_tasks = set()

    async def launch_description(self, description: launch.LaunchDescription):
        """ Launch a launch description asynchronously

        Args:
            description (launch.LaunchDescription): _launch description to launch
        """
        ls = launch.LaunchService()
        ls.include_launch_description(description)
        task = asyncio.create_task(ls.run_async())
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)
        return task

    async def kill_all(self):
        """ Kill all active launch description tasks asynchronously
        """
        if not self.active_tasks:
            return
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.active_tasks, return_exceptions=True)


class AsyncNode(TimeNode, rclpy.node.Node):
    """Async utils for rclpy nodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._launch_manager = AsyncLaunchManager()

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """active event loop of node
        """
        return self._loop

    @event_loop.setter
    def event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def wait_for_service_async(
        self,
        client: rclpy.client.Client,
        timeout: float | None = None,
        interval: float = 0.1,
    ) -> bool:
        """
        Asynchronously wait for service to be available
        """
        start_time = self.wall_time
        while True:
            if client.wait_for_service(timeout_sec=interval):
                return True
            if timeout is not None:
                if (self.wall_time - start_time).to_seconds() >= timeout:
                    return False
            await asyncio.sleep(interval)

    async def do_launch(self, launch_description: launch.LaunchDescription) -> None:
        """
        Asynchronously launch a launch description
        """
        async def _launcher():
            await self._launch_manager.launch_description(launch_description)
        asyncio.run_coroutine_threadsafe(_launcher(), self._loop)

    def wait_for(self, future: typing.Awaitable[T]) -> T:
        """
        Wait for an awaitable to complete in a blocking manner
        """
        async def coro() -> T:
            return await future
        return asyncio.run_coroutine_threadsafe(coro(), self.event_loop).result()

    def sync_wrap(self, fn: typing.Callable[..., typing.Awaitable[T]]) -> typing.Callable[..., T]:
        """
        Wrap an asynchronous function to be called synchronously
        """
        @functools.wraps(fn)
        def sync_fn(*args, **kwargs) -> T:
            return self.wait_for(fn(*args, **kwargs))
        return sync_fn

    def syncify(self, fn: typing.Callable[..., T] | typing.Callable[..., typing.Awaitable[T]]) -> typing.Callable[..., T]:
        """
        Wrap any function to be called synchronously
        """
        if inspect.iscoroutinefunction(fn):
            return self.sync_wrap(fn)

        @functools.wraps(fn)
        def sync_fn(*args, **kwargs) -> T:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                return self.wait_for(result)
            return result
        return sync_fn

    async def await_ros(self, ros_future: asyncio.Future[T]) -> T:
        """
        Wraps a ROS Future into an Asyncio Future so it can be awaited.
        """
        aio_future = self._loop.create_future()

        def on_ros_done(fut):
            if aio_future.cancelled():
                return

            try:
                result = fut.result()
                self._loop.call_soon_threadsafe(aio_future.set_result, result)
            except Exception as e:
                self._loop.call_soon_threadsafe(aio_future.set_exception, e)

        ros_future.add_done_callback(on_ros_done)

        return await aio_future

    # rclpy.node.Node overrides
    def create_subscription(self, msg_type, topic: str, callback: typing.Callable[[rclpy.node.MsgType], None] | typing.Callable[[rclpy.node.MsgType], typing.Awaitable[None]], qos_profile: rclpy.client.QoSProfile | int, *, callback_group: rclpy.client.CallbackGroup | None = None, event_callbacks: rclpy.node.SubscriptionEventCallbacks | None = None, qos_overriding_options: rclpy.node.QoSOverridingOptions | None = None, raw: bool = False) -> rclpy.node.Subscription:
        callback = self.syncify(callback)
        return super().create_subscription(msg_type, topic, callback, qos_profile, callback_group=callback_group, event_callbacks=event_callbacks, qos_overriding_options=qos_overriding_options, raw=raw)

    def create_service(self, srv_type, srv_name: str, callback: typing.Callable[[rclpy.node.SrvTypeRequest, rclpy.node.SrvTypeResponse], rclpy.node.SrvTypeResponse] | typing.Callable[[rclpy.node.SrvTypeRequest, rclpy.node.SrvTypeResponse], typing.Awaitable[rclpy.node.SrvTypeResponse]], *, qos_profile: rclpy.client.QoSProfile = rclpy.qos.qos_profile_services_default, callback_group: rclpy.client.CallbackGroup | None = None) -> rclpy.node.Service:
        callback = self.syncify(callback)
        return super().create_service(srv_type, srv_name, callback, qos_profile=qos_profile, callback_group=callback_group)

    def create_client_wrapper(self, srv_type, srv_name: str, timeout: float = 60.0, *, qos_profile: rclpy.client.QoSProfile = rclpy.qos.qos_profile_services_default, callback_group: rclpy.client.CallbackGroup | None = None) -> ClientWrapper:
        return ClientWrapper(
            self,
            self.create_client(srv_type, srv_name, qos_profile=qos_profile, callback_group=callback_group),
            timeout=timeout,
        )


class AsyncUtil:
    # timeout wrappers
    @classmethod
    async def timeout(cls, coro: typing.Awaitable[T], timeout_sec: float) -> T | None:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None


ServiceT = typing.TypeVar('ServiceT')


class ClientWrapper(typing.Generic[ServiceT]):
    """A wrapper around rclpy.client.Client to provide async utilities.
    """

    def __init__(self, node: AsyncNode, client: rclpy.client.Client, timeout: float = 60.0):
        """
        Args:
            node (AsyncNode | None): node that will own the client.
            client (rclpy.client.Client): The rclpy client to wrap.
            timeout (float, optional): Timeout in seconds. Defaults to 60.0.
        """
        self._node: AsyncNode = node
        self._client: rclpy.client.Client = client
        self._timeout: float = timeout

    @property
    def client(self) -> rclpy.client.Client:
        """The underlying rclpy client.
        """
        return self._client

    async def call_timeout(
        self,
        request: ServiceT.Request,
        timeout_sec: float | None = None,
    ) -> ServiceT.Response | None:
        """Call the service with a timeout.

        Args:
            request (rclpy.client.SrvTypeRequest): The service request object.
            timeout_sec (float | None, optional): Timeout in seconds. Defaults to constructor timeout.

        Returns:
            rclpy.client.SrvTypeResponse | None: The service response object or None if timed out.
        """
        if timeout_sec is None:
            timeout_sec = self._timeout
        res = await AsyncUtil.timeout(
            self._node.await_ros(self._client.call_async(request)),
            timeout_sec=timeout_sec
        )
        if res is None:
            self._node.get_logger().warning(f"Service call to {self._client.srv_name} timed out after {timeout_sec} seconds")
        return res

    def call_timeout_sync(
        self,
        request: ServiceT.Request,
        timeout_sec: float | None = None,
    ) -> ServiceT.Response | None:
        """Call the service with a timeout in a blocking manner.

        Args:
            request (ServiceT.Request): The service request object.
            timeout_sec (float | None, optional): Timeout in seconds. Defaults to constructor timeout.

        Returns:
            ServiceT.Response | None: The service response object or None if timed out.
        """
        if timeout_sec is None:
            timeout_sec = self._timeout

        return self._node.wait_for(
            self.call_timeout(
                request,
                timeout_sec=timeout_sec,
            )
        )

    async def ensure(self, timeout_sec: float | None = None) -> bool:
        """Ensure the service is available within the given timeout.

        Args:
            timeout_sec (float | None, optional): Timeout in seconds. If None, wait forever. Defaults to None.
        """
        return await self._node.wait_for_service_async(self._client, timeout=timeout_sec)
