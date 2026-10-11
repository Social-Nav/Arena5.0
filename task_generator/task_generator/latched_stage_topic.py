"""Keep episode readiness as the final latched sample on eval_ready.

External FollowPath clients consume stage='episode' with KEEP_LAST(1).
Other stages go to eval_ready_status, so neither a late subscriber nor a
busy executor loses the release/reset signal to an unrelated stage.
"""

from __future__ import annotations

import typing

import attrs

#: The stage the external ``eval_ready`` consumers filter for.  Anything else
#: they discard, so anything else must not be the latched sample.
EVAL_READY_CONSUMER_STAGE = 'episode'

#: Relative topic carrying the contract: only stages every consumer accepts.
EVAL_READY_TOPIC = 'eval_ready'

#: Relative topic carrying the full lifecycle, including stages the external
#: consumers discard (``startup``, ``world_geometry``, ``episode_start``).  Also
#: latched, so a human or a late-joining diagnostic tool still sees the last
#: state, and nothing that used to be published is lost.
EVAL_READY_STATUS_TOPIC = 'eval_ready_status'


@attrs.define(frozen=True)
class LatchedStageTopicContract:
    """Which stages may be published on a depth-1 latched, stage-filtered topic.

    Args:
        consumer_filters: One entry per external consumer, each the set of
            ``stage`` values that consumer accepts.  Held as a tuple of
            ``frozenset`` so the contract is hashable and cannot be mutated by a
            caller after the fact.
    """

    consumer_filters: typing.Tuple[typing.FrozenSet[str], ...] = attrs.field(
        converter=lambda filters: tuple(frozenset(entry) for entry in filters)
    )

    @classmethod
    def eval_ready(cls) -> "LatchedStageTopicContract":
        """The generic external client's episode readiness contract."""
        return cls(consumer_filters=(frozenset({EVAL_READY_CONSUMER_STAGE}),))

    @property
    def contract_stages(self) -> typing.FrozenSet[str]:
        """Stages every registered consumer accepts.  Empty if none is registered."""
        if not self.consumer_filters:
            return frozenset()
        return frozenset.intersection(*self.consumer_filters)

    def carries(self, stage: str) -> bool:
        """Whether ``stage`` may be published on the contract topic."""
        return str(stage) in self.contract_stages


def obtainable_sample(
    publications: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> typing.Optional[typing.Mapping[str, typing.Any]]:
    """The only sample a depth-1 subscriber is *guaranteed* to be able to obtain.

    Calibrated against the measurement described in the module docstring: with
    ``KEEP_LAST(1)`` on the subscriber, neither a back-to-back burst nor a
    late-join transient-local replay reliably surfaces anything but the final
    sample.  A subscriber that happens to keep up may see more; a correctness
    argument may not rely on it.

    Args:
        publications: The samples published on one topic, in order.

    Returns:
        The last sample, or ``None`` if nothing was published.
    """
    return publications[-1] if publications else None


def starved_consumer_filters(
    publications: typing.Sequence[typing.Mapping[str, typing.Any]],
    contract: LatchedStageTopicContract,
    *,
    stage_key: str = 'stage',
) -> typing.List[typing.FrozenSet[str]]:
    """Consumer filters that cannot obtain any sample from ``publications``.

    Args:
        publications: The samples published on the contract topic, in order.
        contract: The contract whose consumers are being checked.
        stage_key: Field naming the stage in each sample.

    Returns:
        The filters for which the obtainable sample is discarded.  Empty means no
        consumer can be starved by this publication sequence.
    """
    latched = obtainable_sample(publications)
    if latched is None:
        # Nothing published yet is a normal startup state, not starvation: a
        # transient-local subscriber will receive whatever is published next.
        return []
    stage = str(latched.get(stage_key))
    return [entry for entry in contract.consumer_filters if stage not in entry]
