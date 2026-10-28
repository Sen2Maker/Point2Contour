# Release candidate review

This branch is a review candidate, not a published release. It does not change main.

## Included

- Source-only comparison recipes with pinned external repositories. No third-party
  comparison implementations or pretrained weights are vendored. Xin's independent
  implementation remains in its separate repository.
- Complete frozen CPU graph reconstruction, with geometry, learned editing and
  multi-round ordinary/mode endpoint evidence. Public entry points use process
  names, not historical development version labels.
- A read-only network evidence exporter enabled by `--export-evidence`.
- A separate hash-bound frozen-editor asset bundle; no model assets or experiments
  are stored inside this code checkout.
- Explicit completed-epoch checkpoint metadata and corrected restart boundaries.

## Verification

Tests are run in an independent experiment directory. The review includes
saved-evidence parity on Tallinn and Tokyo, actual ordinary/mode connection
completion cases, empty/degenerate evidence, and a real original-checkpoint
evidence-export/OBJ smoke test. The latter reproduces all 15 saved arrays exactly
and the final reference OBJ byte-for-byte for the tested building.

This is sample-level parity, not a full-dataset rerun or a new platform score.
Use the network environment described in `postprocessing/README.md` for both
network inference and CPU reconstruction. The frozen editor state is exported as
numeric trees without refitting; historical sklearn pickle loading is unnecessary.
A clean-machine dependency installation remains untested.

Unified-environment checks: all five numeric editors exactly matched 22,196
reference feature rows, including every split threshold, adjacent floating-point
values and missing values. All 24 real Tallinn/Tokyo cases and two empty-input
boundaries passed. The maximum deviation from historical OBJ geometry was
1.025e-8 m (below the 1e-6 m tolerance); the exported-evidence smoke OBJ was
byte-identical. All 21 unit tests passed. No training or fresh GPU inference was
needed for this migration. The previous dual-environment implementation remains
available in commit `4355e3b`, with its original asset preserved separately.

Checkpoint tests cover old before-epoch snapshots and new after-epoch snapshots.
Legacy `model_latest` / `model_epoch` files resume at their recorded, not-yet-run
epoch. New files resume at `next_epoch` after completed training, validation and
scheduler stepping. Filenames keep zero-based epoch indices: `0089` is the last
epoch of a 90-epoch run. No new training was started for this repair.

## Intentionally unchanged

The existing no-edge fallback, evaluation definitions, and batch inference exit
policy were not changed in this review. They must not be assumed to implement
different experiment protocols merely because reconstruction now runs.

The 3DLineDetection and Xin wrappers are not complete standalone experiment
packages. External code, environments, data adapters and weights remain the user's
responsibility. No third-party code has been added to fix this limitation.

The repository still has no top-level redistribution license; model asset rights
and public hosting must be approved separately. No push or publication is implied.
