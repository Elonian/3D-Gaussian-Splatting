#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

config_path="${1:-configs/default.yml}"
bash execution_scripts/run_with_config.sh "$config_path"
