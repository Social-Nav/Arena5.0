# Runtime source-binding Gate0

Host-side Git checks do not prove which source a long-lived Docker container
will execute.  Qualified simulator/evaluator sessions must overlay the reviewed
superproject worktree into both runtime containers:

```bash
export ARENA_BENCHMARK_SOURCE_ROOT=/absolute/reviewed/worktree
docker compose \
  -f src/Arena/_meta/docker/docker-compose.yaml \
  -f src/Arena/_meta/docker/features/benchmark/docker-compose.source-binding.yaml \
  config
```

The override mounts the worktree at `/opt/arena_ws/src/Arena` read-only for the
`arena` and `isaac` services.  It does not copy files into the primary checkout
and does not make a host-side hash equivalent to a container check.

After creating the idle containers but before any model, Isaac process, ROS
launch or episode, a versioned Gate0 package must invoke
`container_source_gate` with a frozen JSON contract.  The contract includes the
expected absolute host source root, superproject commit, `arena_isaac` gitlink,
and SHA256 for every runtime-critical file.  The gate collects from each live
container:

- the exact Docker mount at `/opt/arena_ws/src/Arena` and `RW=false`;
- host `git rev-parse HEAD` and `git ls-tree` from the exact mount source
  reported by Docker inspect;
- the populated submodule HEAD from that same host source, which must equal the
  superproject gitlink;
- each required file SHA256 from the container path.

This split identity is intentional. A linked worktree's `.git` file commonly
points to an absolute host gitdir outside the source root, so mounting only the
worktree cannot make that metadata visible inside a container. The gate does
not claim otherwise: Docker binds host identity to the source, and container
hashes prove the bytes actually visible at the import path.

Missing paths, a parent-only legacy mount, writable source, commit/gitlink
drift, missing camera modules, payload drift, or disagreement between the arena
and Isaac containers fails before model/ROS/Isaac launch. Any collection error
is itself a failure. The output path must be absolute and new; evidence is
created atomically and never overwritten.

The real positive fixture uses two `arena:dev` containers with `--network none`,
no GPU device request, and the populated integration root mounted read-only. It
runs the production collector, then removes only those exact fixture
containers.
