#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  printf 'Usage: bash run.sh REPOSITORY {building3d|paper-range-center} INPUT.xyz OUTPUT.obj\n' >&2
  exit 2
fi
recipe="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(realpath -- "$1")"
[[ "$(git -C "$repo" rev-parse --show-toplevel)" == "$repo" ]] || exit 2
[[ "$(git -C "$repo" rev-parse HEAD)" == 168c6fe6de0886ecb28ee8e917da41bc82577bf1 ]] || { echo 'Unsupported reproduction revision' >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked reproduction files have local changes' >&2; exit 2; }
case "$2" in
  building3d) config=building3d_tuned.json ;;
  paper-range-center) config=paper_range_center.json ;;
  *) echo 'Unknown profile' >&2; exit 2 ;;
esac
input="$(realpath -- "$3")"
output="$(realpath -m -- "$4")"
[[ ! -e "$output" ]] || { echo 'Output already exists; refusing overwrite' >&2; exit 2; }
cd "$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" -m xin_reproduce extract "$input" "$output" --params "$recipe/$config" --fpo-workers 1 --save-features
