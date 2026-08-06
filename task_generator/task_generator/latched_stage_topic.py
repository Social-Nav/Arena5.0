"""The publication contract of a depth-1 latched, stage-filtered status topic.

Why this module exists
----------------------
``<task_generator>/eval_ready`` is a ``depth=1, RELIABLE, TRANSIENT_LOCAL``
``String`` topic carrying JSON with a ``stage`` field.  Two *external* consumers
subscribe to it, and both discard every message whose stage is not the one they
are waiting for:

* the official InternNav client
  (``deps/InternNav/scripts/realworld/http_internvla_client.py:882``:
  ``if payload.get('stage') != 'episode': return``), which refuses to plan while
  its gate is unsatisfied and therefore never commands the robot; and
* the InternNav timing manager
  (``arena_bringup/arena_bringup/internnav_timing_manager.py:280``, identical
  filter), which publishes 20 Hz zero velocity while its gate is unsatisfied.

Both subscribe at ``depth=1`` as well, and neither is ours to change: the client
is the official upstream component, so what it *accepts* is the contract, not a
variable.

The measured delivery semantics
-------------------------------
The episode-start barrier added a second publication (``stage='episode_start'``)
on this topic immediately after the ``stage='episode', ready=True`` message the
consumers wait for.  Both consumers then stopped seeing the message they needed,
in 4/4 evaluation runs, and the robot was never commanded at all.

Rather than reason about DDS history semantics, the behaviour was measured in the
production middleware (``rmw_fastrtps_cpp``, ROS 2 Jazzy, this workspace's own
container) with the production QoS on both ends.  20 trials per case,
``tmp/lane_w2_eval_ready_fix/qos_probe_replicate.py``:

===========================================  ==============  ============
case                                          publisher depth  gate satisfied
===========================================  ==============  ============
one publication only (pre-barrier)                        1     20/20
a second, non-matching publication after it                1      0/20
   ... with publisher depth raised to 10                  10      0/20
   ... with publisher depth raised to 50                  50      0/20
second publication on a *separate* topic                   1     20/20
===========================================  ==============  ============

Two conclusions, both measurements rather than inferences:

1. **Only the final publication on such a topic is reliably obtainable.**  A
   depth-1 subscriber's own ``KEEP_LAST(1)`` cache is overwritten by the next
   sample before its executor takes the earlier one, and a late joiner receives
   only the latest retained sample.  So a stage-filtering subscriber can be
   starved by *any* later publication its filter discards.
2. **Raising the publisher's history depth does not help**, because the binding
   constraint is the *subscriber's* depth, which we do not own.  That design
   alternative is rejected on measurement, not on argument.

The rule this module enforces
-----------------------------
A stage may ride the contract topic only if **every** registered consumer filter
accepts it.  Everything else goes to a sidecar diagnostics topic, so no
information is lost and the contract topic cannot be starved *by construction* --
not by ordering, and not by a future caller who has never read this file.

Deliberately free of ROS imports so the rule can be tested without a simulator.
"""

from __future__ import annotations

import typing

import attrs

#: The stage the two external ``eval_ready`` consumers filter for.  Anything else
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
        """The live ``eval_ready`` contract: the client and the timing manager.

        Both filter for the same single stage, so the contract set is that one
        stage.  Listing them separately is deliberate: if a third consumer with a
        different filter is ever added, the intersection shrinks and
        :meth:`carries` starts routing the previously-contracted stage to the
        diagnostics topic, which is a loud test failure rather than a silent
        starvation in production.
        """
        return cls(
            consumer_filters=(
                frozenset({EVAL_READY_CONSUMER_STAGE}),  # http_internvla_client.py:882
                frozenset({EVAL_READY_CONSUMER_STAGE}),  # internnav_timing_manager.py:280
            )
        )

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
