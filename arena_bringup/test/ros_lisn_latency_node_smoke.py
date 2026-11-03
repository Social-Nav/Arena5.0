#!/usr/bin/env python3
"""Real LisnSlowLoopNode + fake producer + ROS-clock latency smoke.

No GPU/Isaac/model.  The node's real latency timer polls the real
LatencyCommitGate; a fake producer enqueues two responses whose arrival order
differs from release order.  The commit callback is a deterministic test double
so this smoke proves scheduling/exact-once/source identity, not controller
apply.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import rclpy

from arena_bringup.lisn_slow_loop_ros import LisnSlowLoopNode


def main() -> int:
    rclpy.init(args=sys.argv)
    evidence = Path('/tmp/lisn_latency_node_smoke/evidence.jsonl')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text('')
    node = LisnSlowLoopNode(
        policy_path=Path('arena_bringup/configs/lisn/slow_loop_grounding_scan_prox_v1.json'),
        evidence_path=evidence,
        controller_node='/task_generator_node/Ai2_Bot2/controller_server',
        proposal_topic='/task_generator_node/Ai2_Bot2/lisn_slow_loop/proposal',
        target_frame='map',
        episode_id='latency-smoke',
        engineering=True,
        wiring_live=True,
        producer_endpoint='http://127.0.0.1:8000/v1/chat/completions',
        latency_mode='delay',
        latency_ms=500.0,
    )
    commits = []

    def fake_commit(payload):
        commits.append(payload['proposal']['id'])
        return {'decision': 'accepted', 'tick_index': len(commits)}

    node._finalize_commit = fake_commit
    gate = node._latency_gate
    if gate is None:
        print(json.dumps({'ok': False, 'reason': 'latency_gate_missing'}))
        node.node.destroy_node()
        rclpy.shutdown()
        return 1
    identity_a = {'image_sha256': hashlib.sha256(b'a').hexdigest()}
    identity_b = {'image_sha256': hashlib.sha256(b'b').hexdigest()}
    identity_drop = {'image_sha256': hashlib.sha256(b'd').hexdigest()}
    gate.latency_ms = 500.0
    gate.enqueue(
        source_stamp_ns=100, source_frame_id='map', source_identity=identity_a,
        payload={'proposal': {'id': 'A'}})
    gate.latency_ms = 0.0
    gate.enqueue(
        source_stamp_ns=200, source_frame_id='map', source_identity=identity_b,
        payload={'proposal': {'id': 'B'}})
    gate.mode = 'drop'
    gate.enqueue(
        source_stamp_ns=300, source_frame_id='map', source_identity=identity_drop,
        payload={'proposal': {'id': 'DROP'}})
    gate.mode = 'delay'
    gate.latency_ms = 500.0
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(commits) < 2:
        rclpy.spin_once(node.node, timeout_sec=0.05)
    release_receipts = [r for r in gate.receipts if r.get('phase') == 'release']
    drop_receipts = [r for r in gate.receipts if r.get('dropped')]
    receipt = {
        'schema': 'arena.lisn.latency_node_smoke.v1',
        'commit_order': commits,
        'release_receipts': release_receipts,
        'drop_receipts': drop_receipts,
        'source_identities': [r.get('source_identity') for r in release_receipts],
        'ok': (
            commits == ['B', 'A']
            and len(release_receipts) == 2
            and len(drop_receipts) == 1
            and all(r.get('exact_once') is True for r in release_receipts)
            and all(r.get('source_unchanged') is True for r in release_receipts)
            and all(r.get('refreshed') is False for r in release_receipts)
        ),
    }
    if getattr(node, '_executor', None) is not None:
        node._executor.shutdown(wait=False)
    if getattr(node, '_commit_executor', None) is not None:
        node._commit_executor.shutdown(wait=False)
    node.node.destroy_node()
    rclpy.shutdown()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
