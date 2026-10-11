# Installation

This guide installs the current Arena benchmark on a clean Ubuntu workstation.
Arena manages two Docker Compose services: ROS 2 Jazzy in `arena-1` and
Isaac Sim 5.1 in `isaac-1`. The model repository owns its external FollowPath client. Do not install or run ROS 2 on the host, and do not use Gazebo as
an installation test or fallback.

Commands below assume an x86-64 Ubuntu host and use `~/arena_jazzy_ws` as the
workspace. If you choose another basename, Docker container names will also
change because the project name is `arena-$(basename "$ARENA_WS_DIR")`.

## 1. Prepare the NVIDIA host

Use an NVIDIA GPU and driver supported by
[Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
On Ubuntu, install the recommended driver with the distribution package manager:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

After the reboot, the host must report the GPU without an error:

```bash
nvidia-smi
```

### Install Docker Engine

These commands follow Docker's
[Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git git-lfs unzip python3 python3-venv
git lfs install
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the user to the `docker` group, then verify both
Docker and Compose:

```bash
docker run --rm hello-world
docker compose version
```

### Install NVIDIA Container Toolkit

These commands follow NVIDIA's
[Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

The GPU must also work inside a container:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

!!! note
    The first Arena build pulls large ROS, Isaac Sim, and model-runtime images.
    Ensure that Docker's data root and the dataset filesystem have ample free
    space (at least 100 GB free is a practical starting point). The compressed
    commercial scene archive alone is about 23.8 GB. If
    `nvcr.io/nvidia/isaac-sim:5.1.0` requires authentication, sign in to NGC
    with `docker login nvcr.io` and accept the applicable NVIDIA license terms.

## 2. Clone Arena

Unified FollowPath evaluation is published on the
[Arena `skz/vln-integration` branch](https://github.com/Social-Nav/Arena5.0/tree/skz/vln-integration).
Use this branch for the model evaluation guide; remote `main` does not yet include
this adaptation. General Arena installations may select `main` instead. The
corresponding Isaac code is published on the
[arena-isaac `skz/vln-integration` branch](https://github.com/Social-Nav/arena-isaac/tree/skz/vln-integration)
and pinned by Arena's submodule commit. If you already have the adaptation
checkout, set `ARENA_WS_DIR` to its workspace and skip cloning. Record the commit
of the checkout you actually use.

```bash
export ARENA_WS_DIR="$HOME/arena_jazzy_ws"
export ARENA_BRANCH=skz/vln-integration
mkdir -p "$ARENA_WS_DIR/src"
git clone --branch "$ARENA_BRANCH" \
  https://github.com/Social-Nav/Arena5.0.git "$ARENA_WS_DIR/src/Arena"
git -C "$ARENA_WS_DIR/src/Arena" rev-parse HEAD
```

Arena initializes its own submodules during Docker bootstrap. Install model weights and clients according to the chosen model repository; Arena does not install or stage them.

## 3. Prepare the asset downloader

This host environment only downloads the external scene files used below:

```bash
export DOWNLOAD_VENV="$HOME/.venvs/arena-downloads"
python3 -m venv "$DOWNLOAD_VENV"
"$DOWNLOAD_VENV/bin/pip" install --upgrade pip huggingface_hub gdown
```

## 4. Install GRScenes commercial assets

Arena already tracks the 30 `grscenes_*_v1` world definitions, maps, and episode
scenario files in Git. The earlier
[world and episode Drive folder](https://drive.google.com/drive/folders/1I3IopJpOwnh0iX5GOGbY82U54W0tXkzW)
is retained as source/archive material; it is **not** an additional required
download for the current checkout.

The large USD dependency remains external to Git. Download only the commercial
archive from the public
[InternRobotics/GRScenes dataset](https://huggingface.co/datasets/InternRobotics/GRScenes):

```bash
export GRSCENES_DIR="$HOME/datasets/GRScenes-100"
export GRSCENES_DOWNLOAD_DIR="$HOME/Downloads/GRScenes"
mkdir -p "$GRSCENES_DIR" "$GRSCENES_DOWNLOAD_DIR"

"$DOWNLOAD_VENV/bin/hf" download \
  InternRobotics/GRScenes \
  scenes/GRScenes-100/commercial_scenes.zip \
  --repo-type dataset \
  --local-dir "$GRSCENES_DOWNLOAD_DIR"

unzip \
  "$GRSCENES_DOWNLOAD_DIR/scenes/GRScenes-100/commercial_scenes.zip" \
  -d "$GRSCENES_DIR"
```

The extraction is correct only if it produces this layout without an extra
`target_30_new` or `GRScenes-100` directory level:

```text
$GRSCENES_DIR/
└── commercial_scenes/
    ├── Materials/
    ├── models/
    └── scenes/
        └── <scene-id>_usd/
            └── start_result_navigation.usd
```

Keep `Materials`, `models`, `metadata.json`, textures, and other referenced files
in place. A USD file copied without its sibling assets can open with missing
geometry or materials. GRScenes is distributed under its own
CC BY-NC-SA 4.0 terms; review the dataset card before use.

### Apply the cleaned navigation USD files

Download the Arena-maintained
[cleaned USD folder](https://drive.google.com/drive/folders/1eqnvl05QMjUZkK-iPT9sduJiPIk7LS6T)
with a browser, or use `gdown`:

```bash
export CLEANED_USD_DIR="$HOME/Downloads/arena-grscenes-cleaned"
mkdir -p "$CLEANED_USD_DIR"
"$DOWNLOAD_VENV/bin/gdown" --folder \
  https://drive.google.com/drive/folders/1eqnvl05QMjUZkK-iPT9sduJiPIk7LS6T \
  --output "$CLEANED_USD_DIR/"
```

If Google Drive rate-limits the CLI, download the folder in a browser and extract
it into `CLEANED_USD_DIR`. The expected files are organized as
`grscenes_<n>/start_result_navigation.usd`. The folder may intentionally contain
only the scenes that have been cleaned; all other worlds continue using the
original GRScenes navigation USD.

Run the following from Bash. It derives the dataset scene ID from each tracked
`world.yaml`, saves the original once as `start_result_navigation.usd.upstream`,
and then installs every cleaned file supplied by the Drive folder:

```bash
cd "$ARENA_WS_DIR/src/Arena"
applied=0
while IFS= read -r -d '' replacement; do
  world_name="$(basename "$(dirname "$replacement")")"
  case "$world_name" in
    grscenes_[0-9]*) ;;
    *) echo "Unexpected cleaned USD path: $replacement" >&2; exit 1 ;;
  esac

  world_yaml="arena_simulation_setup/worlds/${world_name}_v1/world.yaml"
  test -f "$world_yaml" || { echo "Missing $world_yaml" >&2; exit 1; }
  target_rel="$(sed -n \
    's|^[[:space:]]*path: "/data/scenes/\(.*\)"|\1|p' \
    "$world_yaml")"
  test -n "$target_rel" || { echo "No dataset path in $world_yaml" >&2; exit 1; }

  target="$GRSCENES_DIR/$target_rel"
  test -f "$target" || { echo "Missing upstream USD: $target" >&2; exit 1; }
  if test ! -e "$target.upstream"; then
    cp --preserve=timestamps "$target" "$target.upstream"
  fi
  cp --preserve=timestamps "$replacement" "$target"
  echo "Applied $world_name -> $target_rel"
  applied=$((applied + 1))
done < <(find "$CLEANED_USD_DIR" -type f \
  -name start_result_navigation.usd -print0)
test "$applied" -gt 0 || { echo "No cleaned USD files found" >&2; exit 1; }
echo "Applied $applied cleaned USD file(s)."
```

Do not commit the downloaded archive, extracted asset tree, cleaned binary USDs,
or model checkpoints to the Arena repository.

### Validate all configured scene paths

This check reads the 30 current `_v1` world configs and verifies every configured
USD against the host dataset root. It is valid whether the cleaned folder covers
a subset or all scenes.

```bash
export ARENA_WS_DIR GRSCENES_DIR
python3 - <<'PY'
import os
import re
from pathlib import Path

arena = Path(os.environ["ARENA_WS_DIR"]) / "src/Arena"
root = Path(os.environ["GRSCENES_DIR"])
worlds = sorted((arena / "arena_simulation_setup/worlds").glob("grscenes_*_v1/world.yaml"))
if len(worlds) != 30:
    raise SystemExit(f"Expected 30 grscenes_*_v1 configs, found {len(worlds)}")

missing = []
for config in worlds:
    match = re.search(r'path:\s*"/data/scenes/(.+start_result_navigation\.usd)"',
                      config.read_text())
    if not match:
        missing.append(f"{config}: no standard /data/scenes path")
        continue
    target = root / match.group(1)
    if not target.is_file():
        missing.append(str(target))

if missing:
    raise SystemExit("Missing configured GRScenes assets:\n" + "\n".join(missing))
print(f"OK: all {len(worlds)} configured navigation USDs exist")
PY
```

## 5. Configure the dataset mount

Both `arena-1` and `isaac-1` receive the same host directory at `/data/scenes`.
Create or update the workspace environment file before the Docker bootstrap:

```bash
touch "$ARENA_WS_DIR/.env"
sed -i '/^HOST_SCENES_DIR=/d' "$ARENA_WS_DIR/.env"
printf 'HOST_SCENES_DIR=%s\n' "$GRSCENES_DIR" >> "$ARENA_WS_DIR/.env"
export HOST_SCENES_DIR="$GRSCENES_DIR"
```

Use an absolute host path. Keep secrets and private proxy credentials out of the
repository; if they are required locally, place only their environment variable
assignments in the untracked workspace `.env`.

Verify the intended bind mount without starting ROS or Isaac Sim:

```bash
docker run --rm \
  -v "$GRSCENES_DIR:/data/scenes:ro" \
  ubuntu test -f \
  /data/scenes/commercial_scenes/scenes/MWF4WLIKTIFZIAABAAAAACA8_usd/start_result_navigation.usd
```

That sample is the asset used by `grscenes_20_v1`.

## 6. Build the simulation environment

Install the Jazzy evaluator and Isaac Sim features:

```bash
cd "$ARENA_WS_DIR/src/Arena"
./install.sh --install-isaac
```

Arena no longer installs model images, weights or clients. Follow
[Model Integration and Evaluation](Model_Evaluation.md) to download and run the
chosen model. New clients implement the shared [FollowPath interface](Model_Evaluation.md#interface).

## 7. Acceptance checks

Validate the common installation:

```bash
cd "$ARENA_WS_DIR"
source arena -c 'arena validate bootstrap'
```

Only for the unified FollowPath adaptation described in section 2, also run:

```bash
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
```

`doctor` checks Arena/Nav2 package availability. Actual external-client readiness is checked at the episode barrier. Continue with [Model Integration and Evaluation](Model_Evaluation.md#benchmark-commands).
