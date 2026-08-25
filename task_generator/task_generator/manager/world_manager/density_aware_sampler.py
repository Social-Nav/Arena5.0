"""Density-aware pedestrian route sampling on a WorldMap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from task_generator.shared import Position

from .utils import WorldMap


CandidateProvider = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class DensityAwareSamplingConfig:
    """Explicit geometry constraints for pedestrian starts and routes.

    Clearance fields are margins beyond ``pedestrian_radius_m``.  The defaults
    retain TM_Random's previous one-metre total static clearance while avoiding
    its accidental global packing constraint on route goals.
    """

    pedestrian_radius_m: float = 0.35
    static_clearance_m: float = 0.65
    spawn_min_center_dist_m: float = 1.0
    route_point_clearance_m: float = 0.65
    route_min_length_m: float = 1.0
    goals_per_agent: int = 2
    require_same_component: bool = True
    allow_point_reuse: bool = False
    max_start_restarts: int = 32

    def __post_init__(self):
        numeric_nonnegative = (
            'pedestrian_radius_m',
            'static_clearance_m',
            'spawn_min_center_dist_m',
            'route_point_clearance_m',
            'route_min_length_m',
        )
        for field in numeric_nonnegative:
            if float(getattr(self, field)) < 0.0:
                raise ValueError(f'{field} must be non-negative')
        if self.goals_per_agent < 1:
            raise ValueError('goals_per_agent must be at least one')
        if self.max_start_restarts < 1:
            raise ValueError('max_start_restarts must be at least one')


@dataclass(frozen=True)
class PedestrianRoute:
    start: Position
    goals: tuple[Position, ...]


@dataclass(frozen=True)
class DensityAwareSample:
    routes: tuple[PedestrianRoute, ...]
    diagnostics: dict


class DensityAwareSamplingError(RuntimeError):
    """Raised when no complete validated pedestrian route set is found."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


class DensityAwarePositionSampler:
    """Sample globally separated starts and locally constrained route goals."""

    def __init__(
        self,
        *,
        world_map: WorldMap,
        rng: np.random.Generator,
        candidate_provider: CandidateProvider,
        config: DensityAwareSamplingConfig | None = None,
    ):
        self._map = world_map
        self._rng = rng
        self._candidate_provider = candidate_provider
        self._config = config or DensityAwareSamplingConfig()

    def _candidates(self, clearance_m: float) -> np.ndarray:
        clearance_cells = clearance_m / self._map.resolution
        candidates = np.asarray(
            self._candidate_provider(self._map.occupancy.grid, clearance_cells),
            dtype=np.int64,
        )
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError(f'candidate provider returned invalid shape {candidates.shape}')
        return candidates

    @staticmethod
    def _key(candidate: np.ndarray) -> tuple[int, int]:
        return int(candidate[0]), int(candidate[1])

    def _to_position(self, candidate: np.ndarray) -> Position:
        return self._map.tf_grid2pos(self._key(candidate))

    def sample(self, agent_count: int) -> DensityAwareSample:
        if agent_count < 0:
            raise ValueError('agent_count must be non-negative')
        if agent_count == 0:
            return DensityAwareSample(routes=(), diagnostics={
                'agent_count': 0,
                'config': asdict(self._config),
                'start_candidate_count': 0,
                'route_candidate_count': 0,
                'restart_count': 0,
            })

        cfg = self._config
        start_clearance = cfg.pedestrian_radius_m + cfg.static_clearance_m
        route_clearance = cfg.pedestrian_radius_m + cfg.route_point_clearance_m
        start_candidates = self._candidates(start_clearance)
        if np.isclose(start_clearance, route_clearance):
            route_candidates = start_candidates
        else:
            route_candidates = self._candidates(route_clearance)

        # Connectivity is evaluated on the less restrictive of the two already
        # footprint-safe candidate sets. This asks whether the route endpoints
        # share traversable free space without inventing a new map convention.
        connectivity_candidates = (
            start_candidates
            if start_clearance <= route_clearance
            else route_candidates
        )
        component_mask = np.zeros(self._map.shape, dtype=np.uint8)
        if len(connectivity_candidates):
            component_mask[connectivity_candidates[:, 0], connectivity_candidates[:, 1]] = 1
        labels, component_count = ndimage.label(
            component_mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )

        route_by_component: dict[int, np.ndarray] = {}
        if cfg.require_same_component:
            route_labels = labels[route_candidates[:, 0], route_candidates[:, 1]] if len(route_candidates) else []
            for component in np.unique(route_labels):
                component = int(component)
                if component:
                    route_by_component[component] = route_candidates[np.asarray(route_labels) == component]

        diagnostics = {
            'agent_count': agent_count,
            'config': asdict(cfg),
            'map_shape': list(self._map.shape),
            'resolution': self._map.resolution,
            'start_total_static_clearance_m': start_clearance,
            'route_total_static_clearance_m': route_clearance,
            'start_candidate_count': int(len(start_candidates)),
            'route_candidate_count': int(len(route_candidates)),
            'component_count': int(component_count),
            'component_sizes': {
                str(component): int(len(candidates))
                for component, candidates in route_by_component.items()
            },
            'restart_count': 0,
            'start_candidates_examined': 0,
            'route_candidates_examined': 0,
        }

        if len(start_candidates) < agent_count:
            raise DensityAwareSamplingError(
                f'not enough footprint-safe start candidates: requested={agent_count}, available={len(start_candidates)}',
                diagnostics,
            )

        spawn_min_cells = cfg.spawn_min_center_dist_m / self._map.resolution
        route_min_cells = cfg.route_min_length_m / self._map.resolution

        for restart in range(cfg.max_start_restarts):
            diagnostics['restart_count'] = restart + 1
            selected_starts: list[np.ndarray] = []
            for index in self._rng.permutation(len(start_candidates)):
                candidate = start_candidates[index]
                diagnostics['start_candidates_examined'] += 1
                if selected_starts and np.any(
                    np.linalg.norm(np.asarray(selected_starts) - candidate, axis=1) < spawn_min_cells
                ):
                    continue
                if cfg.require_same_component:
                    component = int(labels[candidate[0], candidate[1]])
                    if component == 0 or len(route_by_component.get(component, ())) < cfg.goals_per_agent:
                        continue
                selected_starts.append(candidate)
                if len(selected_starts) == agent_count:
                    break

            if len(selected_starts) != agent_count:
                continue

            used = {self._key(start) for start in selected_starts} if not cfg.allow_point_reuse else set()
            selected_routes: list[tuple[np.ndarray, list[np.ndarray]]] = []
            failed = False
            for start in selected_starts:
                if cfg.require_same_component:
                    pool = route_by_component[int(labels[start[0], start[1]])]
                else:
                    pool = route_candidates
                previous = start
                goals: list[np.ndarray] = []
                for _ in range(cfg.goals_per_agent):
                    goal = None
                    for index in self._rng.permutation(len(pool)):
                        candidate = pool[index]
                        diagnostics['route_candidates_examined'] += 1
                        if not cfg.allow_point_reuse and self._key(candidate) in used:
                            continue
                        if np.linalg.norm(candidate - previous) < route_min_cells:
                            continue
                        goal = candidate
                        break
                    if goal is None:
                        failed = True
                        break
                    goals.append(goal)
                    if not cfg.allow_point_reuse:
                        used.add(self._key(goal))
                    previous = goal
                if failed:
                    break
                selected_routes.append((start, goals))

            if failed:
                continue

            start_array = np.asarray(selected_starts)
            pairwise = np.linalg.norm(start_array[:, None, :] - start_array[None, :, :], axis=2)
            pairwise[pairwise == 0.0] = np.inf
            diagnostics['min_start_center_dist_m'] = float(pairwise.min() * self._map.resolution)
            diagnostics['route_lengths_m'] = [
                [
                    float(np.linalg.norm(current - previous) * self._map.resolution)
                    for previous, current in zip([start, *goals[:-1]], goals)
                ]
                for start, goals in selected_routes
            ]
            diagnostics['selected_component_ids'] = [
                int(labels[start[0], start[1]]) for start in selected_starts
            ]
            routes = tuple(
                PedestrianRoute(
                    start=self._to_position(start),
                    goals=tuple(self._to_position(goal) for goal in goals),
                )
                for start, goals in selected_routes
            )
            return DensityAwareSample(routes=routes, diagnostics=diagnostics)

        raise DensityAwareSamplingError(
            f'failed to sample {agent_count} complete pedestrian routes after '
            f'{cfg.max_start_restarts} start restarts',
            diagnostics,
        )
