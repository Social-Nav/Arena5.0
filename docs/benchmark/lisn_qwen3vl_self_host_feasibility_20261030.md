# LISN self-hosted Qwen3-VL-8B feasibility (read-only, no download/build/serve)

Status: **feasibility study only**. No weights were downloaded, no image was
pulled or built, no server was started, no episode and no model request was
issued. Existing cloud secrets and receipts are untouched. The monotonic 5s
source-age gate, the P1A frozen results and every existing threshold remain
unchanged.

Verified against worktree HEAD `1061c174d7aab470adf68bd8f3c8f604599c8b57`.

## 0. Scope

- Pause further cloud-API POSTs. Do not delete existing secret files, receipts
  or the frozen 1/120 observation evidence.
- Evaluate self-hosting `Qwen3-VL-8B-Instruct` (FP8 or AWQ 4-bit candidates) with
  vLLM inside an OVE GPU model container, reachable from
  `arena-arena_jazzy_ws-arena-1`.
- This document only records read-only findings and budget estimates. It does
  not implement, install, download or serve anything.

Read-only probes used for this document:

- `curl` against the Hugging Face model API (`api/models/<repo>` and
  `?blobs=true`) for revision SHAs and exact file bytes.
- `curl` against PyPI JSON for wheel sizes and release metadata.
- `curl` against vLLM release-tagged documentation (`v0.11.0`, `v0.19.1`,
  `v0.32.0`) for supported-model and quantization matrices.
- `docker run --rm --network none` inside the existing local images **only** to
  read Python/package versions. No model was loaded and no server was started.
- `docker image inspect`, `df`, `mount`, `nvidia-smi`, `docker system df`
  for local resource facts.

## 1. Official model identity and exact revisions

| Artifact | Repository | Revision (commit) | License | Bytes | GiB |
|---|---|---|---|---|---|
| BF16 base | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | 17,545,915,883 | 16.341 |
| FP8 quantized (official) | `Qwen/Qwen3-VL-8B-Instruct-FP8` | `9cdc6310a8cb770ce18efaf4e9935334512aee45` | Apache-2.0 | 10,605,572,552 | 9.877 |
| 4-bit quantized (third party) | `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` | `914bc902ca94c0135be5aa70f25a616bfc853140` | Apache-2.0 declared; calibration data CC-BY-4.0 | 7,565,893,679 | 7.046 |
| Upstream code | `QwenLM/Qwen3-VL` | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | — | — | — |

Model card facts:

- `architectures = ["Qwen3VLForConditionalGeneration"]`, text model type
  `qwen3_vl_text`, vision model type `qwen3_vl`.
- Text backbone: 36 layers, hidden 4096, 32 attention heads, 8 KV heads,
  head_dim 128, vocab 151,936.
- Vision tower: depth 27, hidden 1152, 16 heads, patch 16.
- Native context 256K (config `max_position_embeddings = 262144`).
- The FP8 model card states Transformers cannot load the FP8 weights directly
  and recommends vLLM or SGLang. It also asks for `qwen_vl_utils>=0.0.14` for
  offline inference; the OpenAI-compatible server path does its own
  preprocessing through the Transformers processor.

Official Qwen repositories for this size are BF16, FP8 and GGUF only. There is
**no official Qwen AWQ or GPTQ checkpoint** for `Qwen3-VL-8B-Instruct`. The
`cyankiwi` 4-bit repository is third-party; its `quantization_config` is
`format: pack-quantized`, `group_size: 32`, `num_bits: 4`, i.e. an
llm-compressor compressed-tensors W4A16 checkpoint, not AutoAWQ. All vision
blocks are in the ignore list, so only the text backbone is quantized.

## 2. Quantization candidates

### 2.1 FP8 (official)

- `quantization_config`: `quant_method: fp8`, `fmt: e4m3`,
  `activation_scheme: dynamic`, block size 128 (per model card).
- `ignored_layers` include `lm_head`, the whole visual merger/patch/position
  path and every visual block, so the vision tower stays BF16.
- Intended deployment is vLLM (per the official model card).

### 2.2 4-bit (third party, compressed-tensors W4A16)

- `format: pack-quantized`, `num_bits: 4`, `group_size: 32`, symmetric int,
  `strategy: group`, MSE observer.
- All `model.visual.*` blocks are ignored; only the text backbone is quantized.
- Provenance is third-party. The declared license is Apache-2.0, inherited from
  the Apache-2.0 base model, but the calibration dataset
  `nvidia/Llama-Nemotron-Post-Training-Dataset` is CC-BY-4.0. This is a
  permissive combination, not a license blocker, but it is **not an official
  Qwen artifact** and needs an explicit Main decision before use.

### 2.3 BF16 baseline

- No quantization risk, 16.341 GiB weights. Kept only as a reference row; the
  co-residency estimate makes it impractical on this single L40.

## 3. Released vLLM support matrix

Only released versions are listed. No roadmap item is treated as support.

| vLLM release | Qwen3-VL architecture | FP8 W8A8 on Ada | INT4 W4A16 on Ada | AutoAWQ on Ada |
|---|---|---|---|---|
| `0.11.0` (PyPI 2025-10-04) | released: "Qwen3-VL series" in release notes | released (generic fp8 method) | released (CC > 8.0) | released |
| `0.19.1` (PyPI 2026-04-18) | listed in supported-models docs | released | released | released |
| `0.32.0` (current latest) | listed in supported-models docs | released | released | released |

Evidence:

- vLLM `v0.11.0` release notes list "Qwen3-VL series (#24727)" under new
  architectures; tag commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`.
- `v0.11.0` and `v0.19.1` `docs/models/supported_models.md` both list
  `Qwen3VLForConditionalGeneration` as supported.
- `v0.19.1` `docs/features/quantization/README.md` hardware table: FP8 (W8A8)
  supported on Ada and Hopper only; AWQ supported on Ampere/Ada/Hopper.
- `v0.19.1` `docs/features/quantization/int4.md`: INT4 W4A16 requires compute
  capability > 8.0 (Ampere, Ada, Hopper, Blackwell).
- vLLM `0.19.1` tag commit `b1388b1fbf5aaef47937fabe98931211684666a6`;
  `requires_dist` pins `torch==2.10.0`, `compressed-tensors==0.15.0.1`,
  `flashinfer-python==0.6.6`, `transformers>=4.56.0` with 5.0–5.5.0 excluded.

What is **not** claimed:

- No released vLLM note claims an end-to-end tested Qwen3-VL + FP8 pairing;
  support is architecture-level plus the generic `fp8` quant method. It must
  still be verified with one real checkpoint load before adoption.
- There is no official Qwen3-VL AutoAWQ checkpoint, so "AWQ" here means the
  third-party compressed-tensors W4A16 repository above.
- GGUF multimodal serving was not evaluated.
- A known FP8 throughput regression is reported for Qwen3-VL-32B-FP8 on
  vLLM 0.25.1 (issue #49259); it does not name 0.19.1, and no claim is made
  either way.

## 4. Reusable pinned base image and incremental layer

The host already has an image with a compatible, internally consistent vLLM
stack. Read-only version probe results:

| Image | Size (bytes) | Python | vLLM | torch | transformers | compressed-tensors | flashinfer |
|---|---|---|---|---|---|---|---|
| `lightnav-single:cea0cc6-lock3e0c5546-nounsetfix1` (ID `sha256:b2a321aced4f9e12b7bc5b3b35ea4e0b89951d3d5f36d13c82444443c8a1cf0b`) | 15,338,587,184 (14.29 GiB) | 3.11.14 (`/opt/lightnav-model-venv`) | 0.19.1 | 2.10.0+cu128 | 5.8.0 | 0.15.0.1 | 0.6.6 |
| `arena-navila:76b98f2-20261028-fixed` | 7.54 GB | 3.10.12 | absent | 2.3.0+cu121 | 4.37.2 | — | — |
| `arena:dev` | 4.81 GB | 3.12.3 | absent | absent | absent | — | — |

`vllm-flash-attn` and `flash-attn` are absent in the LightNav venv, but vLLM
0.19.1 pins `flashinfer-python`/`flashinfer-cubin` 0.6.6, which are present.
`triton` 3.6.0, `torchvision` 0.25.0 and `torchaudio` 2.10.0 match the vLLM
0.19.1 pins.

Option A (recommended for a future authorized round): reuse the LightNav image
as the immutable base and add only a thin server wrapper.

- New unique overlay: server script (order of 10 KiB) plus optionally
  `qwen-vl-utils==0.0.14` (8,120 B wheel). Total well below 1 MiB.
- The new image tag reports the shared 14.29 GiB base size, but Docker layer
  sharing means the **unique** delta is only the wrapper layer.
- No rebuild of the base, no new base image pull, no host/Mac dependency.

Option B: build a slim image from `arena:dev` and pip-install the vLLM stack.
Wheel download sum for the exact pinned set is about **4.51 GiB compressed**
(torch 2.10.0 x86_64 915.7 MB, vllm 0.19.1 433.1 MB, cudnn 706.8 MB,
cublas 594.3 MB, cusparse 288.2 MB, cusparselt 287.2 MB, nccl 322.4 MB,
cusolver 267.5 MB, flashinfer-cubin 267.7 MB, triton 188.4 MB, plus smaller
CUDA wheels), and installed size is larger. This requires network downloads and
is **BLOCKED_NO_DOWNLOAD_AUTHORIZATION** for this round.

Option C: pull a prebuilt `vllm/vllm-openai` image. Not present locally; pull
size is not recorded and would be a new download. **BLOCKED** under the same
rule.

Existing build cache is 37.52 GB (26.2 GB reported reclaimable). It is **not**
deleted or pruned by this study.

## 5. NAS read-only mount plan and disk budget

NAS facts:

- `/mnt/bn/robot-mllm-data-hl` is NFSv4.1, total 10 P, used 95 T, available 10 P.
- Existing layout: `arena_vln_baselines/{checkpoints,cache,installs,logs,outputs,repos,venvs}`.
- `arena_vln_baselines/cache/huggingface` already exists.

Proposed layout for a future authorized round (not created now):

- Weights: `/mnt/bn/robot-mllm-data-hl/arena_vln_baselines/checkpoints/Qwen3-VL-8B-Instruct-FP8` or `...-AWQ-4bit`, mounted **read-only** (`:ro`) into the model container. Weights never enter an image layer.
- Cache: `HF_HOME` and `VLLM_CACHE_ROOT`/`TORCHINDUCTOR_CACHE_DIR` pointed at a
  writable NAS cache directory (for example
  `arena_vln_baselines/cache/qwen3vl_vllm`) so the local root partition does not
  absorb compile/HF caches.

Local disk facts (read-only):

- `/dev/vda2`: 492 G total, 445 G used, **28 G available, 95% used**.

Budget accounting depends on whether the 10 GiB cap counts NAS weights:

| Candidate | Weight bytes / GiB | Local new (Option A) | Strict total if NAS weights count | Verdict |
|---|---|---|---|---|
| FP8 official | 10,605,572,552 / 9.877 | wrapper < 1 MiB + local cache 0–1 GiB | 9.878 GiB (cache on NAS) to 10.877 GiB (1 GiB local cache) | **PASS only with cache on NAS**; with a 1 GiB local cache it exceeds 10 GiB by ~0.88 GiB → **BLOCKED** under strict total |
| AWQ 4-bit third party | 7,565,893,679 / 7.046 | wrapper < 1 MiB + local cache 0–1 GiB | 8.046–9.046 GiB | **PASS** under both accountings |
| BF16 official | 17,545,915,883 / 16.341 | wrapper < 1 MiB | ≥16.341 GiB | **BLOCKED** under both accountings |

The exact FP8 result is therefore a budget-accounting decision, not a technical
one: FP8 fits if NAS weights are outside the 10 GiB cap and the compile cache
lives on NAS; AWQ fits under either reading.

## 6. GPU VRAM estimate and dependency compatibility

Host GPU: one `NVIDIA L40`, 46,068 MiB, driver `570.211.01` (CUDA 12.8 branch),
Ada SM 8.9. Idle usage at study time was 4,696 MiB (desktop/Isaac container
idle, no eval process).

KV-cache math for the 8B model:

- 36 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes (fp16)
  = 147,456 B per token ≈ 144 KiB/token.
- 8,192-token context ≈ 1.13 GiB; 16,384-token context ≈ 2.25 GiB.
- One 1024×768 image adds on the order of 10^3 visual tokens after patch
  merging, so 8,192 is a reasonable starting `--max-model-len`.

Rough resident estimates (weights + KV 8,192 + ~1.5 GiB runtime/CUDA graphs):

| Candidate | Weights | KV @8192 | Runtime overhead | Estimated peak |
|---|---|---|---|---|
| FP8 | 9.88 GiB | 1.13 GiB | ~1.5 GiB | **~12.5 GiB** |
| AWQ 4-bit | 7.05 GiB | 1.13 GiB | ~1.5 GiB | **~9.7 GiB** |
| BF16 | 16.34 GiB | 1.13 GiB | ~1.5 GiB | **~19.0 GiB** |

Co-residency with Isaac Sim is the main unresolved risk. The existing
`render2_probe/RENDER2_RECEIPT.json` records a peak of **40,491 MiB** for a
render=2 internnav-lane run; the receipt does not separate Isaac from any
co-resident model server, so Isaac-only peak is unknown. With 46,068 MiB total,
an FP8 server capped at
`--gpu-memory-utilization 0.30` (~13.8 GiB) leaves ~32.2 GiB for Isaac; an AWQ
server at 0.25 (~11.5 GiB) leaves ~34.5 GiB. Whether Isaac fits under those
ceilings is **UNKNOWN** and must be measured in a future authorized run. If
Isaac alone needs more than ~34 GiB, neither candidate can co-reside on this
single GPU, and the design is **BLOCKED_CO_RESIDENCY**.

Dependency compatibility:

- Driver 570.211.01 supports the CUDA 12.8 runtime used by torch 2.10.0+cu128.
- vLLM 0.19.1 pins `torch==2.10.0`, `torchaudio==2.10.0`,
  `torchvision==0.25.0`, `compressed-tensors==0.15.0.1`,
  `flashinfer-python==0.6.6`, `transformers>=4.56.0` excluding 5.0–5.5.0.
  The LightNav venv already satisfies every pin.
- `qwen_vl_utils` is not required for the OpenAI-compatible server; it is only
  used by the offline `LLM(...)` example in the model card.
- FP8 requires Ada/Hopper per the released vLLM table; L40 is Ada, so the
  hardware class is supported. End-to-end loading of this exact checkpoint on
  this host remains unverified.

## 7. Interface boundary (design only, no code change)

Self-hosting replaces only the producer inference source. Nothing else in the
LISN chain changes.

- Producer resolution already supports a generic OpenAI-compatible endpoint:
  `ARENA_LISN_VLM_ENDPOINT` (`lisn_slow_loop_ros.py:627-632`), with
  `ARENA_LISN_VLM_API_KEY_ENV` (`:629`) and `ARENA_LISN_VLM_MODEL`.
- For a local server, the endpoint would be
  `http://<model-container-bridge-ip>:8000/v1/chat/completions`. `arena-1` is
  host-network-only, so the model container must be reachable by its bridge IP
  (same pattern used for the NaVILA TCP server), with no host port published.
- Credential handling: a generic provider fails closed when no key is present
  (`lisn_slow_loop_ros.py:787-793`). A local vLLM server can be given a
  non-secret local placeholder token through `ARENA_LISN_VLM_API_KEY_ENV`; the
  value must never enter the repo, argv, logs or receipts. No cloud secret is
  reused.
- Request body already matches the vLLM OpenAI server: `image_url` with a
  `data:<mime>;base64,<...>` JPEG (`lisn_slow_loop_ros.py:920-921`), text part,
  `max_tokens` (`:925`). vLLM's own docs show the same
  `data:image/jpeg;base64,...` chat-message form.
- Contract gates stay byte-identical: monotonic received-age 5s
  (`lisn_slow_loop_contract.py:537-541`), source identity match (`:559`),
  `valid_until_ns` at snapshot commit time (`lisn_slow_loop_ros.py:1226-1244`),
  atomic apply/readback and the P1A frozen results.
- The producer wait cap stays an explicit run parameter
  (`runner.py:392/394`, defaults 120.0). A local 8B model will often answer
  faster than the cloud API, but no guarantee is made that it answers within
  5s; the contract is not changed by this study. Any TTL/threshold change
  requires a separate request.

## 8. Blockers and decisions for Main

1. **Budget accounting**: does the 10 GiB cap include the NAS-resident weights?
   - If yes: FP8 is blocked with any local cache ≥ ~0.12 GiB; AWQ is the only
     candidate that fits.
   - If no: FP8 fits, provided the compile cache is on NAS.
2. **Co-residency**: Isaac-only peak VRAM is not recorded. The design is
   **BLOCKED_CO_RESIDENCY** until one bounded measurement shows Isaac + a
   capped vLLM instance fit in 46,068 MiB.
3. **Third-party artifact approval**: the 4-bit candidate is
   `cyankiwi/...AWQ-4bit` (compressed-tensors W4A16), not an official Qwen
   checkpoint. Apache-2.0 plus CC-BY-4.0 calibration data is permissive, but
   using it needs an explicit decision and provenance recording.
4. **Download/build authorization**: Option B/C and any weight fetch require a
   separate authorization. None was performed here.

## 9. Non-actions (explicit)

- No model weights downloaded, no image pulled, no image built.
- No vLLM server started, no model loaded, no GPU allocation.
- No LISN episode, no cloud POST, no API request.
- No secret file read, printed, hashed or deleted.
- No source default, contract threshold, P1A result or test assertion changed.
- InternNav `073420`, NaVILA `095026` and LightNav `102601` artifacts were not
  re-run; the 1894 docs were not redone.
- No host or Mac dependency installed; no host conda environment changed.
