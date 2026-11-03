# Camera image backends and dynamic TF

Arena freezes exactly one camera **image** backend before an episode. The project default is `native`, which uses the Isaac Sim Replicator ROS writers. `manual` is a **DEPRECATED** isolated diagnostic/legacy path; it is never an automatic runtime fallback.

```bash
export ARENA_CAMERA_PUBLISH_BACKEND=native   # project default
```

The compatibility alias `ARENA_LISN_CAMERA_PUBLISH_BACKEND=spawnusdrobot` resolves to `native`, but new launch files should use the canonical variable. `dual`, `auto`, and unknown values fail closed. Native and manual publishers must never publish the same camera topics in one process or episode.

## Episode lock and manifest

Qualified runs must set both:

```bash
export ARENA_REQUIRE_CAMERA_BACKEND_MANIFEST=1
export ARENA_CAMERA_BACKEND_MANIFEST=/absolute/new/output/camera_backend_manifest.json
```

The manifest records the frozen backend, image producer, depth semantics, intrinsics source, dynamic TF producer, episode identifiers, selection source, and `auto_switch_allowed=false`. The backend is process-locked before robot camera publishers are created. An environment change or switch request after freeze is an error.

## Native backend

`native` uses:

- Replicator ROS image and CameraInfo writers;
- `DistanceToImagePlane` depth;
- intrinsics from `isaacsim.ros2.bridge.read_camera_info`;
- a separate `DynamicCameraTfPublisher` using live USD base/camera world matrices, the ROS optical basis conversion, and the same actual simulation capture stamp.

Dynamic TF is not part of the image backend. It remains active for native and explicitly selected manual runs.

## DEPRECATED manual image backend

`ManualReplicatorCameraPublisher` is retained only for explicit, isolated diagnostics and legacy compatibility. Its class, constructor, writer initialization, publish entry, and call site carry `DEPRECATED` tags. Explicit initialization emits one warning; the per-frame loop does not repeat it.

Manual currently uses:

- Replicator `rgb`, `camera_params`, and radial `distance_to_camera` annotators;
- replacement of non-finite depth with `0.0`;
- CameraInfo derived from `camera_params`, with a legacy `fx=fy=500` fallback that is invalid for qualification.

These inputs are not scientifically equivalent to native `DistanceToImagePlane` and native CameraInfo. Manual is deprecated and restricted to isolated diagnostic/legacy use pending removal; it is not a model-evaluation fallback candidate. Validators must report the semantic difference and must not relax it or imply that matching semantics would restore manual as a routine evaluation backend.

There is no scheduled removal version or deadline. Removal can be considered only after actual robot/GRScenes native camera qualification and complete sim+eval+model three-Docker validation pass, followed by compatibility observation across later releases.

## Qualification boundary

The fixed small-scene A/B/C diagnostic proved that native writers can emit real RGB/depth/CameraInfo in headless mode and that `disable_viewport_updates=true` did not suppress those payloads in that case. This does **not** qualify every GRScenes asset, robot camera, TF tree, model input, or full benchmark.

Before an actual robot/GRScenes run, Gate0 must bind:

1. source/submodule/image/interpreter hashes and the backend manifest path;
2. exactly one image backend and one render product per camera;
3. depth semantics and intrinsics source;
4. live dynamic TF producer, actual camera/base prim paths, ROS optical conversion, and same-stamp capture contract;
5. ROS domain/RMW/QoS and one non-empty DDS endpoint GID per topic;
6. fresh RGB/depth/CameraInfo frames, lossless raw artifacts, cleanup, GPU, disk, lock, and evidence-finalization budgets.

Missing TF is an independent qualification failure; it must not be rewritten as a writer no-image result.

Camera runtime records are owned by the exact robot prim path and namespace. A
same-robot respawn unregisters its prior writers, render products, image specs,
and TF specs before replacement. `DeletePrims` performs the same exact cleanup
before destroying the robot prim; a cleanup error blocks deletion and retains
the failed handles for bounded retry. Other live robots are never broad-cleared.
An episode reset that only moves the same live robot deliberately keeps its
owned backend and TF records; replacement/deletion must use the lifecycle paths
above.
