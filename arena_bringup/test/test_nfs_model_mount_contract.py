from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_COMPOSE = REPO_ROOT / "_meta/docker/docker-compose.yaml"
INTERNNAV_COMPOSE = REPO_ROOT / "_meta/docker/features/internnav/docker-compose.yaml"
MOUNT = '${HOST_ARENA_MODELS_DIR:-${HOST_ARENA_WS_DIR}/deps/models}:/opt/arena_ws/deps/models:ro'


def test_all_runtime_services_use_read_only_model_root_override():
    base = BASE_COMPOSE.read_text(encoding="utf-8")
    internnav = INTERNNAV_COMPOSE.read_text(encoding="utf-8")
    assert base.count(MOUNT) == 2
    assert internnav.count(MOUNT) == 1


def test_model_root_override_preserves_local_checkout_fallback():
    combined = BASE_COMPOSE.read_text(encoding="utf-8") + INTERNNAV_COMPOSE.read_text(encoding="utf-8")
    assert "/tmp/arena_models" not in combined
    assert "${HOST_ARENA_WS_DIR}/deps/models" in combined


def test_model_mount_does_not_change_container_model_paths():
    internnav = INTERNNAV_COMPOSE.read_text(encoding="utf-8")
    assert "ARENA_INTERNNAV_MODEL_PATH=/opt/arena_ws/deps/models/InternVLA-N1-DualVLN" in internnav
    assert "INTERNNAV_DEPTH_ANYTHING_CKPT=/opt/arena_ws/deps/models/depth-anything-v2-metric-hypersim-small/" in internnav
