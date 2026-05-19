from __future__ import annotations

import abc
import traceback

from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree import IdentifierProtocol

from task_generator import NodeInterface
from task_generator.constants import Constants
from task_generator.utils.registry import Registry

from ._interface import ObstacleITF, PedestrianITF, RobotITF, WorldITF


class BaseSim(NodeInterface, ObstacleITF, PedestrianITF, RobotITF, WorldITF, abc.ABC):

    _namespace: Namespace

    def __init__(self, *args, namespace: Namespace, **kwargs):
        super().__init__(*args, **kwargs)
        self._namespace = namespace

    @abc.abstractmethod
    async def before_reset_task(self) -> bool:
        """
        Is executed each time before the task is reset. This is useful in
        order to pause the simulation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def after_reset_task(self) -> bool:
        """
        Is executed after the task is reset. This is useful to unpause the
        simulation.
        """
        raise NotImplementedError()

    # Utils
    async def safe_resolve(self, identifier: IdentifierProtocol):
        try:
            return await identifier.resolve()
        except Exception as e:
            self._logger.error(f"Failed to resolve {identifier}:\n{e}\n{traceback.format_exc()}")
            return None


SimulatorRegistry = Registry[Constants.SimSimulator, BaseSim]()


@SimulatorRegistry.register(Constants.SimSimulator.DUMMY)
async def lazy_dummy(**kwargs):
    from .dummy_simulator import DummySimulator
    return DummySimulator(**kwargs)


# @SimulatorRegistry.register(Constants.SimSimulator.FLATLAND)
# async def lazy_flatland(**kwargs):
#     from .flatland_simulator import FlatlandSimulator
#     return FlatlandSimulator(**kwargs)


@SimulatorRegistry.register(Constants.SimSimulator.GAZEBO)
async def lazy_gazebo(**kwargs):
    from .gazebo_simulator import GazeboSimulator
    return await GazeboSimulator.create(**kwargs)


# @SimulatorRegistry.register(Constants.SimSimulator.UNITY)
# async def lazy_unity(**kwargs):
#     from .unity_simulator import UnitySimulator
#     return UnitySimulator(**kwargs)


@SimulatorRegistry.register(Constants.SimSimulator.ISAAC)
async def lazy_isaac(**kwargs):
    from .isaac_simulator import IsaacSimulator
    return await IsaacSimulator.create(**kwargs)


@SimulatorRegistry.register(Constants.SimSimulator.ISAAC_EVAL)
async def lazy_isaac_eval(**kwargs):
    from .isaac_eval_simulator import IsaacEvalSimulator
    return await IsaacEvalSimulator.create(**kwargs)
