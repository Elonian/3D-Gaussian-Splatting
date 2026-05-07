#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

config_path="${1:-configs/default.yml}"
if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 2
fi

if ! python -c "import yaml" >/dev/null 2>&1; then
    echo "PyYAML is missing; installing it so the YAML config can be read."
    python -m pip install PyYAML
fi

env_file="$(mktemp)"
python - "$config_path" "$env_file" <<'PY'
from pathlib import Path
import shlex
import sys
import yaml

config_path = Path(sys.argv[1])
env_path = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text()) or {}

def nested(*keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

def as_bool(value):
    return "1" if bool(value) else "0"

scenes = nested("render", "scenes", default=["chair", "lego", "materials", "drums"])
if not isinstance(scenes, list) or not all(isinstance(scene, str) for scene in scenes):
    raise SystemExit("render.scenes must be a list of scene names")

num_views = nested("render", "num_views", default=None)
if num_views is not None and (not isinstance(num_views, int) or num_views <= 0):
    raise SystemExit("render.num_views must be null or a positive integer")

values = {
    "CFG_INSTALL_REQUIREMENTS": as_bool(nested("pipeline", "install_requirements", default=True)),
    "CFG_INSTALL_SIMPLE_KNN": as_bool(nested("pipeline", "install_simple_knn", default=False)),
    "CFG_RENDER": as_bool(nested("pipeline", "render", default=True)),
    "CFG_EVALUATE": as_bool(nested("pipeline", "evaluate", default=True)),
    "CFG_SCENES": "\n".join(scenes),
    "CFG_DEVICE_TYPE": str(nested("render", "device_type", default="cuda")),
    "CFG_CUDA_VISIBLE_DEVICES": str(nested("render", "cuda_visible_devices", default="0")),
    "CFG_OUTPUT_ROOT": str(nested("render", "output_root", default="outputs")),
    "CFG_NUM_VIEWS": "" if num_views is None else str(num_views),
    "CFG_REF_ROOT": str(nested("evaluation", "ref_root", default="data/nerf_synthetic")),
    "CFG_EVAL_OUTPUT_FILE": str(nested("evaluation", "output_file", default="evaluation.txt")),
    "CFG_LOG_DIR": str(nested("logging", "directory", default="logs")),
    "CFG_LOG_PREFIX": str(nested("logging", "file_prefix", default="run")),
    "CFG_LOG_FILE": str(nested("logging", "file_name", default="")),
    "CFG_CHECKPOINT_DIR": str(nested("checkpoints", "directory", default="outputs/checkpoints")),
    "CFG_CHECKPOINT_RUN_NAME": str(nested("checkpoints", "run_name", default=nested("project", "name", default="run"))),
}

env_path.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
PY

# shellcheck disable=SC1090
source "$env_file"
rm -f "$env_file"

resolve_path() {
    local value="$1"
    if [[ "$value" = /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s/%s\n' "$project_root" "$value"
    fi
}

log_dir="$(resolve_path "$CFG_LOG_DIR")"
mkdir -p "$log_dir"
if [[ -n "$CFG_LOG_FILE" ]]; then
    if [[ "$CFG_LOG_FILE" = /* ]]; then
        log_file="$CFG_LOG_FILE"
    else
        log_file="${log_dir}/${CFG_LOG_FILE}"
    fi
else
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${log_dir}/${CFG_LOG_PREFIX}_${timestamp}.log"
fi
mkdir -p "$(dirname "$log_file")"

checkpoint_root="$(resolve_path "$CFG_CHECKPOINT_DIR")"
checkpoint_dir="${checkpoint_root}/${CFG_CHECKPOINT_RUN_NAME}"
mkdir -p "$checkpoint_dir"

exec > >(tee "$log_file") 2>&1

on_error() {
    local exit_code=$?
    echo "[ERROR] Pipeline failed at line $1 with exit code $exit_code"
    echo "[ERROR] Log file: $log_file"
    echo "failed $(date -Is) line=$1 exit_code=$exit_code" > "${checkpoint_dir}/FAILED.txt"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

echo "=== 3DGS pipeline started: $(date -Is) ==="
echo "Project root: $project_root"
echo "Config: $config_path"
echo "Log file: $log_file"
echo "Checkpoint dir: $checkpoint_dir"
echo "Python: $(python --version)"
echo "Scenes:"
while IFS= read -r scene; do
    [[ -n "$scene" ]] && echo "  - $scene"
done <<< "$CFG_SCENES"
echo "started $(date -Is)" > "${checkpoint_dir}/STARTED.txt"

if [[ "$CFG_INSTALL_REQUIREMENTS" == "1" || "$CFG_INSTALL_SIMPLE_KNN" == "1" ]]; then
    echo "=== Setup ==="
    if [[ "$CFG_INSTALL_REQUIREMENTS" == "1" ]]; then
        setup_skip_requirements=0
    else
        setup_skip_requirements=1
    fi
    if [[ "$CFG_INSTALL_SIMPLE_KNN" == "1" ]]; then
        setup_skip_simple_knn=0
    else
        setup_skip_simple_knn=1
    fi
    SKIP_REQUIREMENTS="$setup_skip_requirements" \
    SKIP_SIMPLE_KNN="$setup_skip_simple_knn" \
    bash execution_scripts/setup_env.sh
else
    echo "=== Setup skipped by config ==="
fi

out_root="$(resolve_path "$CFG_OUTPUT_ROOT")"
ref_root="$(resolve_path "$CFG_REF_ROOT")"
eval_output_file="$(resolve_path "$CFG_EVAL_OUTPUT_FILE")"

if [[ "$CFG_RENDER" == "1" ]]; then
    echo "=== Rendering ==="
    render_args=(--device-type "$CFG_DEVICE_TYPE" --out-root "$out_root")
    if [[ -n "$CFG_NUM_VIEWS" ]]; then
        render_args+=(--num-views "$CFG_NUM_VIEWS")
    fi
    while IFS= read -r scene; do
        [[ -z "$scene" ]] && continue
        echo "--- Rendering scene: $scene ---"
        echo "started $(date -Is)" > "${checkpoint_dir}/render_${scene}.started"
        CUDA_VISIBLE_DEVICES="$CFG_CUDA_VISIBLE_DEVICES" \
        python utils/render.py --scene-type "$scene" "${render_args[@]}"
        echo "done $(date -Is)" > "${checkpoint_dir}/render_${scene}.done"
    done <<< "$CFG_SCENES"
else
    echo "=== Rendering skipped by config ==="
fi

if [[ "$CFG_EVALUATE" == "1" ]]; then
    echo "=== Evaluation ==="
    mapfile -t scene_array <<< "$CFG_SCENES"
    python evaluation/evaluate.py \
        --ref-root "$ref_root" \
        --out-root "$out_root" \
        --output-file "$eval_output_file" \
        --scenes "${scene_array[@]}"
    echo "done $(date -Is)" > "${checkpoint_dir}/evaluation.done"
else
    echo "=== Evaluation skipped by config ==="
fi

echo "completed $(date -Is)" > "${checkpoint_dir}/COMPLETED.txt"
echo "=== 3DGS pipeline completed: $(date -Is) ==="
echo "Log file: $log_file"
echo "Checkpoint dir: $checkpoint_dir"
