# 3DLineDetection

This directory provides a script that applies our Building3D-tuned configuration
to the [original 3DLineDetection implementation](https://github.com/xiaohulugo/3DLineDetection).
It adjusts nine existing parameters while preserving the geometric algorithm.
The original implementation is downloaded separately.

Requirements: Bash, Git and Python 3.8+.

From this directory, download the supported version and apply the parameters:

```bash
git clone https://github.com/xiaohulugo/3DLineDetection.git upstream
cd upstream
git checkout --detach f7a0976a86007b8cc82943c7d0ab63c0a68af495
bash ../apply_params.sh
```

Apply the script to the clean checkout before making other source edits. Then
configure the input/output paths in the original example and rebuild the project
following its instructions. If you use your own entry point, pass `k = 48` to
`detector.run`.

To reproduce our experiment conditions, also read the first three XYZ columns
of each input line, export OBJ coordinates with 17 significant digits, and run
one building per process with `OMP_NUM_THREADS=1`. These input/output settings
are separate from the parameter patch and are required for reproducing the
reported results.

To restore the original parameters, run `bash ../apply_params.sh --restore`.
