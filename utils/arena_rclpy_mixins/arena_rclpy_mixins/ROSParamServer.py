import abc
import traceback
import typing

import rcl_interfaces.msg
import rclpy.exceptions
import rclpy.node

from .shared import DefaultParameter

T = typing.TypeVar('T')
U = typing.TypeVar('U')


class ROSParamT(abc.ABC, typing.Generic[T]):

    @abc.abstractmethod
    def __init__(
        self,
        /,
        name: str,
        value: typing.Any,
        *,
        type_: typing.Optional[rclpy.Parameter.Type] = None,
        parse: typing.Optional[typing.Callable[[
            typing.Any], T]] = None,
        **kwargs,
    ) -> None:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get name.
        """

    @property
    @abc.abstractmethod
    def value(self) -> T:
        """
        Get cached value.
        """

    @value.setter
    @abc.abstractmethod
    def value(self, value):
        """
        Set value and publish.
        """

    @abc.abstractmethod
    def callback(self, value: typing.Any) -> bool:
        """
        Callback function for setting value.
        """


class _ROSParam(ROSParamT[T], typing.Generic[T]):
    """
    Wrapper that handles callbacks.
    """

    _node: typing.ClassVar["ROSParamServer"]
    _name: str

    _type: typing.TypeVar
    _from_param: typing.Callable[[typing.Any], T]

    _value: T
    _parameter_value: typing.Any

    @staticmethod
    def identity(x, *args):
        """
        lambda x: x
        """
        del args
        return x

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: typing.Any):
        self._node.set_parameters([
            rclpy.Parameter(
                name=self._name,
                value=value
            )
        ])

    @property
    def param(self) -> typing.Any:
        return self._parameter_value

    @param.setter
    def param(self, value: typing.Any):
        self._parameter_value = value
        self._value = self._from_param(value)

    def callback(self, value: typing.Any) -> bool:
        self.param = value
        return True

    def __init__(
        self,
        /,
        name: str,
        value: typing.Optional[typing.Any] = None,
        *,
        type_: typing.Optional[rclpy.Parameter.Type] = None,
        parse: typing.Optional[typing.Callable[[typing.Any], T]] = None,
        **kwargs,
    ) -> None:
        self._name = name

        if parse is None:
            parse = self.identity
        self._from_param = parse

        if type_ is not None:
            self._node.rosparam.declare_safe(
                self.name,
                type_,
            )

        self._node.register_param(self, value, **kwargs)


counter = 0


class _rosparam(typing.Generic[T]):
    """
    Light-weight stateless interface for singular typed rosparam actions (short-lived).
    Runtime checks are not performed.
    Use ROSParam instead for most use cases.
    """

    _node: typing.ClassVar["ROSParamServer"]

    class _UNSET(object):
        ...

    @classmethod
    def declare_safe(
        cls, param_name: str,
        value: typing.Any = None,
        **kwargs
    ) -> None:
        if cls._node.has_parameter(param_name):
            return

        try:
            cls._node.declare_parameter(param_name, value, **kwargs)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

    @classmethod
    def get_unsafe(
        cls,
        param_name: str,
        default: T | typing.Type[_UNSET] = _UNSET
    ) -> T:
        """
        Get value of parameter.
        """

        _default = default

        default_value = None
        if default is not cls._UNSET:
            default_value = DefaultParameter(default)

        result = cls._node.get_parameter_or(
            param_name,
            default_value,
        )
        if result.type_ is rclpy.Parameter.Type.NOT_SET:
            if _default is not cls._UNSET:
                return _default  # type: ignore
            raise ValueError(
                f'parameter {param_name} is unset and no default passed'
            )

        return typing.cast(T, result.value)

    @classmethod
    def get(cls, param_name: str, default: T) -> T:
        """
        Get value of parameter. Declare if undeclared.
        """
        if default is not None:
            cls.declare_safe(param_name, default)
        return cls.get_unsafe(param_name, default)

    @classmethod
    def set_unsafe(cls, param_name: str, value: T) -> bool:
        """
        Set value of parameter.
        """

        return cls._node.set_parameters([
            rclpy.Parameter(param_name, value=value)
        ])[0].successful

    @classmethod
    def set(cls, param_name: str, value: T) -> bool:
        """
        Set value of parameter. Declare if undeclared.
        """

        try:
            return cls.set_unsafe(param_name, value)
        except rclpy.exceptions.ParameterNotDeclaredException:
            cls._node.declare_parameter(param_name, value)
            return True

    @classmethod
    def callback(
        cls,
        param_name: str,
        callback: typing.Callable[[typing.Any], bool]
    ):
        try:
            value = cls.get_unsafe(param_name)
        except ValueError:
            value = None

        cls._node.add_param_callback(param_name, callback)

        if value is None:
            return

        if cls._node.executor is not None:
            # An exception raised inside an ``executor.create_task`` coroutine is
            # NOT reported when it happens.  ``rclpy`` re-raises a failed Task
            # only on a *later* return from ``wait_for_ready_callbacks``
            # (rclpy/executors.py), and a node that has no timers and no inbound
            # traffic at startup never gets one -- ``create_task`` trips the
            # guard condition exactly once, which is enough to run the task and
            # no more.  So an initial parameter callback that raised used to be
            # lost completely, and the first sign of trouble was whatever
            # downstream deadline expired minutes later, naming the wrong
            # subsystem.  Report it here instead, at the moment it happens.
            cls._node.executor.create_task(
                lambda: cls._node.report_param_callback_failure(
                    param_name, value, callback,
                )
            )
        else:
            # Deliberately NOT wrapped: without an executor the call runs on the
            # caller's stack, so an exception propagates to a caller that can see
            # it.  Wrapping this branch too would make the diagnostics *worse* by
            # swallowing an exception that is currently loud.  The asymmetry in
            # handling is what makes the two branches symmetric in loudness.
            callback(value)


class ParamCallbackFailure(typing.NamedTuple):
    """One parameter callback that raised, kept so a later waiter can see it.

    A failure that is only logged is invisible to code, and a failure that is
    only recorded is invisible to a human reading the log.  Both are produced,
    from one place, so the two can never disagree.
    """

    param_name: str
    value: typing.Any
    exception: BaseException
    traceback_text: str

    def summary(self) -> str:
        """One-line cause, suitable for another subsystem's error message."""
        return (
            f"parameter {self.param_name!r} callback raised "
            f"{type(self.exception).__name__}: {self.exception}"
        )


class ROSParamServer(rclpy.node.Node):
    """
    Interface for interacting with this node's ros2 parameters.
    """

    # this confuses my type checker
    # ROSParam: type[_ROSParam[typing.Any]]
    # rosparam: type[_rosparam[typing.Any]]

    _callbacks: dict[
        str,
        typing.Set[typing.Callable[[typing.Any], bool]]
    ]

    _param_callback_failures: list[ParamCallbackFailure]

    @property
    def param_callback_failures(self) -> tuple[ParamCallbackFailure, ...]:
        """Parameter callbacks that raised, oldest first.

        Populated by :meth:`report_param_callback_failure`, i.e. by the *initial*
        dispatch in ``_ROSParam.callback``, which is the one path whose exception
        cannot reach any caller.  Read this instead of waiting for a downstream
        deadline: a subsystem blocked on something an initial callback was
        supposed to produce can consult it and fail immediately, naming the real
        cause.
        """

        return tuple(getattr(self, '_param_callback_failures', ()))

    def report_param_callback_failure(
        self,
        param_name: str,
        value: typing.Any,
        callback: typing.Callable[[typing.Any], bool],
    ) -> bool:
        """Run ``callback(value)``, reporting rather than losing a raise.

        Used for the initial dispatch only.  Subsequent parameter sets go through
        :meth:`_callback`, which already wraps the same callback and returns the
        formatted traceback to the setter in ``SetParametersResult.reason``; this
        method is that path's sibling for the case where there is no setter to
        return anything to.

        Returns:
            bool: the callback's own result, or ``False`` if it raised.
        """

        try:
            return bool(callback(value))
        except BaseException as e:  # noqa: BLE001 -- nothing above can report it
            traceback_text = ''.join(
                traceback.TracebackException.from_exception(e).format())
            failure = ParamCallbackFailure(
                param_name=param_name,
                value=value,
                exception=e,
                traceback_text=traceback_text,
            )
            if not hasattr(self, '_param_callback_failures'):
                self._param_callback_failures = []
            self._param_callback_failures.append(failure)
            self.get_logger().error(
                f'initial configuration of parameter {param_name} with value '
                f'{value!r} FAILED: {type(e).__name__}: {e}\n{traceback_text}'
            )
            return False

    def add_param_callback(
        self,
        param_name: str,
        callback: typing.Callable[[typing.Any], bool]
    ):
        """
        Add callback for parameter changes.
        """

        self._callbacks.setdefault(param_name, set()).add(callback)

    def register_param(self, param: ROSParamT[T], value: typing.Any, **kwargs):
        del kwargs  # unused

        current_value = self.rosparam[T].get(
            param.name,
            value
        )

        self.add_param_callback(
            param.name,
            param.callback,
        )

        result = self._callback([
            rclpy.Parameter(
                name=param.name,
                value=current_value
            )
        ])

        if not result.successful:
            raise RuntimeError(
                f'initial configuration of parameter {param.name} failed with {result.reason}')

    def _callback(self, params: list[rclpy.Parameter]):
        # Sibling of report_param_callback_failure(): this is the path taken by
        # every parameter set *after* registration.  It reports a raise by
        # returning the formatted traceback to the setter in
        # SetParametersResult.reason.  The initial dispatch has no setter to
        # return to, which is why it has its own reporter rather than sharing
        # this one.
        successful = True
        reason: list[str] = []
        for param in params:
            for callback in self._callbacks.get(param.name, set()):
                self.get_logger().debug(
                    f"setting param {param.name} with value {param.value} (callback {callback})")
                try:
                    successful &= callback(param.value)
                except BaseException as e:
                    self.get_logger().warn(
                        f'setting parameter {param.name} with value {param.value} failed: {e}')
                    reason.append(
                        ''.join(
                            traceback.TracebackException.from_exception(e).format()))
                    successful = False

        if not successful:
            # this function can cause inconsistent states when some callbacks
            # succeed, some fail. revert back to old value here.
            ...

        return rcl_interfaces.msg.SetParametersResult(
            successful=successful,
            reason="\n".join(reason)
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = {}
        self._param_callback_failures = []
        self.add_on_set_parameters_callback(self._callback)
        self._setup_rosparam()

    def _setup_rosparam(self):
        class ROSParam_impl(_ROSParam[T], typing.Generic[T]):
            _node = self
        self.ROSParam = ROSParam_impl

        class rosparam_impl(_rosparam[T], typing.Generic[T]):
            _node = self
        self.rosparam = rosparam_impl
