# Comparison recipes

These recipes keep external implementations separate from Point2Contour.
They do not download data or weights, train models, or select parameters on a new test set.

| Method | Entry | What changes |
| --- | --- | --- |
| [3DLineDetection](3DLineDetection/README.md) | `bash apply_params.sh UPSTREAM` | Nine existing geometric parameter constants |
| [NerVE](NerVE/README.md) | `bash apply_params.sh UPSTREAM` | Cube/face inference thresholds and curve connection parameters |
| [Xin2024](Xin2024/README.md) | `bash run.sh REPOSITORY PROFILE INPUT OUTPUT` | Explicit configuration selection for an independent reproduction |

Patch recipes support `--dry-run` and `--restore`, pin upstream commits, and reject unexpected source edits. Rebuild compiled methods after applying or restoring parameters. Use each upstream project's own environment and license.

The configurations were selected during previous experiments; they are not evidence of blind-test tuning independence. Keep training splits, checkpoint provenance, input normalization and evaluation protocols with any reported results.
