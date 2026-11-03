# HuNav social_force_factor 60 — three-VLN rerun batch (2026-10-31)

Status: **completed engineering batch**. This is not a scientific comparison with
the old `social_force_factor=10` runs and must not be aggregated with them.

## Configuration

- Case: `grscenes_20_v1` / `default_2` / `Ai2_Bot2`, 2 humans, pedestrian seed 0,
  300 sim seconds, shared RPP control path.
- Instruction SHA256: `b6649036379d7b7eb951aff084e2f8987beefed409dbc3184cf1479eaa1743b3`.
- HuNav config: `arena_bringup/configs/hunav/default_sf60.yaml`,
  SHA256 `a834b6bb42e6c8f14ea4734aa5ab1fd03791a1897f9445d24c2fb7ccade9eeb1`.
- Loader selection: `ARENA_HUNAV_CONFIG=default_sf60.yaml`; effective
  `social_force_factor=60.0`. The user's dirty
  `src/Arena/.../hunav/default.yaml` (SHA `e427195b...`) was not overwritten.
- The installed share used by the lane is the target-install copy of the same
  `default_sf60.yaml`; the scenario `default_2` has no per-agent
  `social_force_factor` override.

## Results

| Lane | Run prefix | eval_rc | End reason | Artifact pass | Path length | Min human distance | Videos |
|---|---|---|---:|---|---|---:|---:|---|
| InternNav | `outputs/arena_hunav_sf60_20261031/internnav/20261030_161945_grscenes_20_v1_Ai2_Bot2_internnav` | 0 | sim_timeout (TIMEOUT) | true | 26.215 m | 0.672 m | 3/3 complete |
| NaVILA | `outputs/arena_hunav_sf60_20261031/navila_fixed/20261030_163919_grscenes_20_v1_Ai2_Bot2_navila` | 0 | goal_reached | true | 5.194 m | 0.721 m | 3/3 complete |
| LightNav | `outputs/arena_hunav_sf60_20261031/lightnav/20261030_165058_grscenes_20_v1_Ai2_Bot2_lightnav` | 0 | goal_reached | true | 5.223 m | 0.551 m | 3/3 complete |

All three runs keep the original 20 fps / timebase / xN recording contract.
`benchmark_success` remains false where the social-distance or qualification
checks fail; `goal_reached` is reported separately and is not rewritten.

## Human reaction (observed, not causal proof)

| Lane | Human | Closest approach | Post lateral deviation | Robot speed near encounter |
|---|---|---:|---:|---:|
| InternNav | 1 | 0.672 m | 2.54 m | 0.001 m/s |
| InternNav | 2 | 0.831 m | 0.622 m | 0.029 m/s |
| NaVILA | 1 | 0.614 m | 0.603 m | 0.002 m/s |
| LightNav | 1 | 0.552 m | 0.812 m | 0.053 m/s |

The robot moved and slowed near the pedestrians, and the humans showed lateral
displacement during close encounters. Causal attribution of the displacement to
`social_force_factor=60` is **UNKNOWN** without an `sf10` counterfactual; no human
path was edited.

## Index

Machine-readable index:
`outputs/arena_hunav_sf60_20261031/BATCH_INDEX.json`
(SHA256 `09239b4e8275f0f970efaa5d648fa0280f0962b5215fad141eef3807efb57d95`).

## Scientific limits

- Do not aggregate with the old `sf10` results.
- The three VLN lanes share RPP; LISN SocialMPC is a separate control path and
  is not a fair cross-controller comparison.
- Human reaction is observed correlation, not a controlled causal result.
