# Clean eval integration mapping
Baseline: `c275eb4` (origin/main = origin/feat/internnav-eval-progress).
Source: `feat/internnav-eval-progress` HEAD `14f6bea` (140 commits ahead).
Clean branch: `feat/eval-clean-20261103`.
Production feature tree equivalent; `git diff --stat 14f6bea HEAD` is limited to
the listed 7 additive contract/test/mapping files (+423 lines). It is not an
empty diff and is not compared to a temporary hash.

## New feature commits
| New commit | Theme | Original exploration commits |
|---|---|---|
| `c1dd660` | E0 qualification, density, artifact provenance | 1-5 |
| `6ed7d8c` | Unified model executors + native FollowPath benchmark runtime | 6-25 |
| `6ed7d8c` | Task34/runner/model-param runtime wiring | 56-66 |
| `6cdf42e` | LISN slow-loop, grounding, semantic, Prox/Qwen A-track | 26-34 |
| `6cdf42e` | LISN producer/provider/timing hardening | 37-55 |
| `6cdf42e` | LISN grounding/goal/SocialMPC/semantic/Prox implementation | 60-60 |
| `6cdf42e` | LISN/Prox implementation continued | 72-80 |
| `6cdf42e` | LISN/Prox/semantic/diagnostic implementation | 84-104 |
| `6cdf42e` | Prox A-track scan grounding and diagnostics | 109-114 |
| `6cdf42e` | Prox diagnostic renderer fixes | 116-116 |
| `d8b7676` | Strict-B fast guard + paper pipeline contracts | 118-129 |
| `dde7c5e` | Agentic lifecycle/provenance/method routing | 130-137 |
| `dde7c5e` | Agentic current-source drift fail-close | 139-140 |
| `fd09b85` | Frozen eval500 world/profile dataset swap | 138-138 |
| `041bdeb` | Docs, runbooks, coverage, stable run index, openspec | 67-71 |
| `041bdeb` | Docs: HuNav sf60 batch and Prox runbook | 82-82 |
| `041bdeb` | Docs: Prox runbook/coverage/hash refresh | 115-115 |
| `041bdeb` | Docs: v2 hash refresh | 117-117 |
| `041bdeb` | Docs: v2 hash refresh after renderer fixes | 120-120 |
| `5983d71` | HuNav people bridge + submodule integration pointers | 81-83 |
| `5983d71` | HuNav people bridge/costmap frame | 105-108 |
| `88412b1` | Integration fix: remove stale pre-eval500 world duplicates | integration fix (no source commit; removes stale pre-eval500 duplicates) |

## Original commit mapping

### `c1dd660` — E0 qualification, density, artifact provenance
- `a853b4b` Add E0 benchmark qualification evidence
- `f32b08a` Enforce density obstacle consumer contract
- `5f430e9` Bind E0 runtime to isolated overlay provenance
- `3853699` Separate artifact validity from task outcome
- `e4266bf` Record versioned postprocess parser provenance

### `6ed7d8c` — Unified model executors + native FollowPath benchmark runtime
- `42e51ef` Add parallel native FollowPath benchmark runtime
- `8b2eb17` Integrate FollowPath lifecycle and isolate result paths
- `29f4f50` Enable colcon pytest discovery and document integration
- `659bb0e` Mount benchmark model registry from NFS
- `1b7025a` Integrate native camera backend contract
- `6d255de` Integrate runtime source and odom ownership gates
- `3097f4d` feat(vln): unify model adapters behind one FollowPath executor
- `ccf7e6a` fix(vln): isolate stale inference and serialize cancel confirmation
- `1001c7a` fix(vln): gate session transition and bind authoritative robot frames
- `5dafec8` fix(vln): declare executor parameters and unshadow pathlib Path
- `b6500c8` fix(vln): pair RGB and depth by exact stamp before inference
- `1e58825` fix(vln): recognize unified executor evidence in artifact validation
- `dd928f1` feat(vln): record overlay provenance beside the base runtime
- `ba0331b` fix(benchmark): propagate unified executor counters into summary
- `48a9ba3` fix(task_generator): require explicit external model status before episode start
- `d09bcf2` fix(task_generator): hold episode start for harness pre-episode gate release
- `d3ac419` fix(task_generator): match LightNav client VOLATILE QoS for model status
- `47b8fca` chore(arena_isaac): pin exact-stamp camera TF image echo
- `14d4946` chore(arena_isaac): pin lazy camera TF image echo subscription
- `760fae8` fix(diagnostics): bind external FollowPath client evidence to one session

### `6ed7d8c` — Task34/runner/model-param runtime wiring
- `3b9fa48` fix(vln): bound overlay provenance env payload to avoid E2BIG
- `9dcfffb` fix(nav2): add none planner/interplanner configs for unified FollowPath
- `3eac607` fix(vln): publish pre-episode gate from unified executor
- `dfaa6af` fix(vln): complete pre-episode gate publisher wiring
- `589f34d` fix(lisn): match documented minimal Seed-2.1 ChatCompletions payload
- `08548f9` config(task34): update Ai2_Bot2 RPP/smoother speed ceiling to 0.8
- `0d71d18` feat(task34): record raw/delivered/odom speed trace with human and turn conditions
- `b959d33` feat(task34): add 20fps Task34 base profile and runner base-profile selection
- `03daaee` feat(task34): sample eval video on sim time and dedup map view
- `bea01e4` feat(task34): forward bounded model params to the evaluator
- `8bc9ccd` fix(task34): register model-params-json CLI argument

### `6cdf42e` — LISN slow-loop, grounding, semantic, Prox/Qwen A-track
- `183e561` feat(lisn): add slow-loop contract and ROS parameter bridge
- `da8bce6` feat(lisn): add real producer path and exact-binding engineering profile
- `a8fdec4` fix(lisn): wire the real producer into the live node and runner
- `6ca398b` fix(lisn): close D1-D3 launch, single-produce and social_mpc identity gaps
- `c60b70c` feat(lisn): add Ark/OpenAI-compatible producer config and safe image encoding
- `25c664f` feat(lisn): add ModelHub and Ark provider resolution to the VLM producer
- `5caa786` feat(lisn): add scoped DMXAPI provider and model-list probe
- `1ee3f18` feat(lisn): add Gemini 2.5 Pro ModelHub provider and env auto-detection
- `e1d711a` feat(lisn): add 30 s engineering profile, live apply receipt and ready status

### `6cdf42e` — LISN producer/provider/timing hardening
- `4cae3be` fix(lisn): publish a session-bound model/status heartbeat
- `4b6f17c` fix(lisn): match eval_ready TRANSIENT_LOCAL QoS in the sidecar
- `daf3c2e` fix(lisn): bind provider sections exactly and latch 401/403
- `c154a22` fix(lisn): structured redacted errors, explicit provider rules and node latch
- `a44eb36` fix(lisn): allowlisted error codes, static binding errors and key fail-close
- `aaa54a5` feat(lisn): expose redacted response metadata for real probes
- `f9e546d` fix(lisn): real-probe prompt, SDK alias parsing and 130 s engineering profile
- `d83a296` fix(lisn): explicit producer timeout passthrough and bounded timeout latch
- `4f5e803` feat(lisn): safe error categories and a global producer deadline
- `e612ef9` fix(lisn): truly abort the owned HTTP request at the global deadline
- `4cdc742` fix(lisn): absolute deadline bounds, bounded IPC and shared passthrough
- `02daf15` fix(lisn): include start and decode in the absolute deadline; JSON IPC
- `ee83ca3` test(lisn): bound JSON decode inside the absolute deadline
- `a3c34f6` fix(lisn): owner process group covers bootstrap and JSON decode
- `8f9409a` test(lisn): prove the ROS tick stays non-blocking with a slow producer
- `7b7cfb7` feat(lisn): offline DMX native timing helpers and failure receipts
- `827d1e1` feat(lisn): wire the actual DMX timing entry to the repo helper
- `61b0b20` feat(lisn): add the Seed-Evolving Responses adapter and timing entry
- `2d8724f` feat(lisn): Seed-2.1 Pro ChatCompletions adapter and single-request timing entry

### `6cdf42e` — LISN grounding/goal/SocialMPC/semantic/Prox implementation
- `589f34d` fix(lisn): match documented minimal Seed-2.1 ChatCompletions payload

### `6cdf42e` — LISN/Prox implementation continued
- `99a75b7` fix(lisn): host-owned producer identity/provenance and goal fail-closed
- `918b3aa` feat(lisn): grounding-live profile without hard age gate + projection resolver
- `2cd8c78` feat(lisn): allow LISN-only SocialMPC cruise target override
- `501135c` fix(lisn): preserve commit proposal metadata on exception path
- `12be4c0` feat(lisn): consume grounded map goal via common FollowPath sender
- `28bca83` fix(lisn): grounded map goal identity and LISN tick lifecycle
- `f4864a1` feat(lisn): same-stamp depth/CameraInfo/TF grounding and runner launch args
- `e627e26` fix(lisn): freeze source bundle, publish after apply, isolate grounding prompt, robust depth decode
- `17338a1` feat(lisn): env bridge for grounding goal consumer and cruise target

### `6cdf42e` — LISN/Prox/semantic/diagnostic implementation
- `d3f28ed` fix(lisn): tick release gate, honest timing path, LISN run identity
- `cb4a308` fix(lisn): supply host source frame/size/stamp to grounding prompt
- `fd4326d` fix(lisn): read-only observation for frozen controller parameters
- `a5299eb` fix(lisn): sim-time snapshot clock and bounded people_weight observation
- `c2f6a4b` fix(lisn): prepend map start pose for single-point grounded goals
- `07a2d48` fix(lisn): current map pose as SocialMPC path start; 0.8 normal cruise prompt
- `62ff2a4` fix(lisn): path-direction heading for grounded goal pose
- `5569625` fix(lisn): straight-segment heading on both grounded path poses
- `8bd7b0e` fix(lisn): raise rotation-shim disengage threshold for slow-loop subgoals
- `1ed7b7f` fix(lisn): load SocialMPC directly instead of RotationShim wrapper
- `691d36b` fix(lisn): verify direct SocialMPC plugin identity
- `027a619` fix(diagnostics): recognize LISN grounded-goal control loop
- `41546eb` feat(lisn): real semantic entity projection and graded costmap layer
- `335f348` fix(semantic): expand costmap bounds to grid extent
- `8a7fb05` chore(lisn): info-level launch log for semantic costmap diagnosis
- `0d9e4d5` fix(lisn): 30s waypoint lifecycle transition timeout
- `ed40028` fix(lisn): 60s waypoint transition timeout for semantic costmap activation
- `85fe963` fix(semantic): start layer disabled and enable after episode barrier
- `41f7db6` feat(semantic): persist raw/projected entities and grid data in receipt
- `81654e5` feat(lisn): compact semantic response and 30s semantic profile
- `78d5648` fix(semantic): coerce numeric VLM instance_id to contract string

### `6cdf42e` — Prox A-track scan grounding and diagnostics
- `9b0d8b7` feat(prox): wire ProxMPC A-track with real Isaac RTX LiDAR
- `4661c7b` feat(lisn): add ProxMPC scan-grounding A-track and collision monitor
- `c059f22` test(lisn): cover scan-range grounding resolver
- `5a2cb41` feat(lisn): add Prox diagnostic recorder, VLM trace, and debug renderers
- `50f55c2` test(lisn): cover out-of-bounds scan grounding rejection
- `640a90b` fix(lisn): exact source lookup, panel layout, and v2 timebase audit

### `6cdf42e` — Prox diagnostic renderer fixes
- `36e8704` fix(lisn): draw off-window goal indicator and expose model/schema in diag panel

### `d8b7676` — Strict-B fast guard + paper pipeline contracts
- `4907f22` feat(paper): add B/TTC, scenario, latency, teacher, and metrics fail-close contracts
- `df99150` test(paper): cover B error action and document executed CLI smoke
- `6f56867` docs(benchmark): refresh v2 hashes after offline renderer fixes
- `86acc37` fix(paper): wire B flow-risk veto, scenario env, latency release, and owned cleanup
- `7e52fb8` feat(bfast): add off/on launch profiles, real ROS veto smoke, and release queue
- `b6e609a` docs(benchmark): record B launch parse and ROS smoke receipt
- `ab65ece` test(bfast): add actual executor ROS smoke and B profile identity validation
- `a69e1a7` feat(paper): add B runner profile plan and latency commit gate
- `8925918` feat(paper): wire B eval parameter chain and non-blocking latency commits
- `26e2d5e` fix(paper): enforce Qwen B identity, smoke-only settings, and immutable metrics roots
- `13dabbc` fix(paper): mark B plan non-executable and add scenario/C provenance audits
- `7611c56` fix(paper): pin Hunav config SHA and add source-run provenance audit

### `dde7c5e` — Agentic lifecycle/provenance/method routing
- `4a4b1a5` feat(agentic): add operator contract, request schema, and evidence index
- `693dacc` docs(agentic): pin latest source-run audit receipt
- `4eb3f97` fix(agentic): bind operator plan to profile and protect source runs
- `a9f2227` feat(agentic): add lifecycle response and deterministic evidence verdict
- `4cce6ef` feat(agentic): bind read/audit/finalize and add recording contracts
- `d01b9c1` fix(agentic): bind audit to run identity and require provenance
- `78c59aa` fix(agentic): authenticate provenance and add qwen_prox_a audit role
- `780ff3e` feat(agentic): add versioned method routing plan contract

### `dde7c5e` — Agentic current-source drift fail-close
- `f590c2f` fix(agentic): restore lifecycle CLI and pin canonical method bundles
- `14f6bea` fix(agentic): fail close on current profile and common-case drift

### `fd09b85` — Frozen eval500 world/profile dataset swap
- `8bb1896` feat(eval): swap legacy grscenes worlds for the frozen eval500 dataset

### `041bdeb` — Docs, runbooks, coverage, stable run index, openspec
- `abd95aa` docs(benchmark): add stable lane profile mapping and run index
- `fe85494` docs(benchmark): record LISN static-review metadata and LightNav run row
- `8ee1481` docs(lisn): add fresh-revalidation design candidate
- `1061c17` docs(lisn): correct 20s wait-cap vs 5s monotonic age semantics
- `8f68637` docs(lisn): add Qwen3-VL-8B self-host feasibility and budget study

### `041bdeb` — Docs: HuNav sf60 batch and Prox runbook
- `ad92e4c` docs(benchmark): record HuNav sf60 three-VLN rerun batch

### `041bdeb` — Docs: Prox runbook/coverage/hash refresh
- `0943fc4` docs(benchmark): add ProxMPC A-track runbook and paper pipeline coverage

### `041bdeb` — Docs: v2 hash refresh
- `4ea8782` docs(benchmark): pin final v2 run timing hash

### `041bdeb` — Docs: v2 hash refresh after renderer fixes
- `6f56867` docs(benchmark): refresh v2 hashes after offline renderer fixes

### `5983d71` — HuNav people bridge + submodule integration pointers
- `cbc135c` config(hunav): add auditable social_force_factor 60 profile
- `ad92e4c` docs(benchmark): record HuNav sf60 three-VLN rerun batch
- `ddde060` feat(hunav): select default config via ARENA_HUNAV_CONFIG

### `5983d71` — HuNav people bridge/costmap frame
- `a87dc55` fix(lisn): semantic layer deadlock/clearing, real people bridge, map writer
- `50c36dd` fix(people): transform HuNav tracks into costmap frame; LISN fov_angle
- `9ed9aa6` fix(people): use lookup_transform scalars for mixed Python bindings
- `2a7f4e0` fix(validation): explicit controller_people_zero incomplete marker

## Notes
- The original branch, its 24 worktrees, dirty files, outputs, and failure evidence were not reset, rebased, amended, or deleted.
- `arena_isaac` and `arena_robots` gitlinks were updated to the target tree via `git update-index`; the original submodule working trees were not modified.
- The referenced submodule commits were pushed normally to `feat/eval-clean-submodule-20261103`: `arena_isaac=98a6f45…` to `https://github.com/Social-Nav/arena-isaac`, and `arena_robots=e116700…` to both the configured local origin and `https://github.com/Social-Nav/robots.git`. A fresh recursive clone verified `arena_isaac`, `arena_robots`, and nested `jackal/robot_usds/turtlebot4` at their pinned SHAs.
- The clean branch uses 12 reviewable commits; it does not squash the 140 commits into one commit and does not drop TIMEOUT/collision functionality.
- `eval500` is preserved as its own feature commit `fd09b85`; stale pre-eval500 world duplicates were removed in `88412b1`. The production feature tree is equivalent to `14f6bea`; the explicit delta is the 7 additive contract/test/mapping files listed above.

## Integration fixes

The clean branch adds one small integration-fix commit after the feature split:

- `arena_vln_models/test/test_lisn_tick_lifecycle.py`: initialize the test harness
  `_fast_guard_veto_topic` attribute so the harness matches the production
  executor's fast-guard veto contract. This is a test-harness repair, not a
  production behavior change.
- `arena_vln_models/test/test_model_adapter_registry.py`: add
  `model_backend`, `runner_entry`, and `actual_model_backend` to the
  model-private plan whitelist; these fields legitimately differ by model.

No production code was changed by this integration-fix commit. The original
branch and its committed behavior remain untouched.

## Stage 2 arena_isaac integration

- Independent submodule branch: `feat/eval-clean-arena-isaac-20261103`.
- Commit `53234f7` is a targeted cherry-pick of
  `839c87390e1f081824a75f89b89dd34f49cf5c66` on top of the current
  `98a6f45` integration base. It does not merge the whole 272-commit
  `feature/social_yielding` branch.
- Two conflicts were resolved explicitly:
  - `SpawnPedestrians.py` keeps the existing rendered-state backend and adds
    the categorized `CharacterPoolSelector`; `Person` receives both
    `resolved_model` and `backend`.
  - `person.py` keeps the existing camera/render state code and adds the
    character graph/appearance profile.
- Offline verification: 18 character-asset/identity/selector tests pass;
  deterministic pool hits are 22 pedestrian, 2 doctor, 4 construction_worker,
  4 police; the catalog has 32 identities / 252 asset variants; unknown
  models fail closed. The full 1125-asset inventory requires the Isaac People
  asset root and was not verifiable in this offline lane.
- The parent clean branch now pins `arena_isaac=53234f7` and
  `arena_robots=e116700`. A fresh recursive clone checked out `arena_isaac`
  `53234f7`, `arena_robots` `e116700`, and nested
  `jackal/robot_usds/turtlebot4` at their pinned SHAs.
- Real Isaac spawn/asset-selection/video smoke was not executed because the
  current authorization boundary forbids starting an Isaac episode; it remains
  the next runtime-only validation step.

