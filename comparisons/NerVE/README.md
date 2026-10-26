# NerVE

Upstream: https://github.com/uhzoaix/NerVE

```bash
git clone https://github.com/uhzoaix/NerVE.git
git -C NerVE checkout 20730c0415ebfb7f87c8124560bf87e29ac91806
bash apply_params.sh --dry-run /path/to/NerVE
bash apply_params.sh /path/to/NerVE
# Restore upstream values:
bash apply_params.sh --restore /path/to/NerVE
```

The patch changes inference cube/face probability thresholds from 0.5 to 0.2, the curve query radius from 4 to 6, the deletion-length limit from 5 to 0, and smoothing iterations from 3 to 0. Training and validation thresholds remain unchanged. The probability thresholds apply to the upstream BCE prediction branch, not its alternative classification branch.

Cube and face thresholds affect **Pre as well as Post**. These changes must not be described as postprocessing-only changes. Other parameters remain upstream defaults; use final checkpoints for all three networks.

This is a parameter recipe, not a Building3D dataset adapter or a checkpoint release. Our evaluated Building3D runs used EdgeDataset normalization associated with GT geometry. Preserve that disclosure; these results do not demonstrate GT-independent preprocessing. The upstream demo alone does not reproduce the Building3D input preparation.
