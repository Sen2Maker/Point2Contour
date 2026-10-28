# Structured wireframe reconstruction

This directory provides the frozen CPU graph decoder, geometric repair, learned
editing and multi-round connection completion used by the evaluated pipeline.
It is separate from the original geometric utility `tools/post.py`.
The trusted frozen editor asset is distributed separately from the code checkout.

## Running

First export evidence in the **network environment**. This records the existing
forward pass without changing its outputs; it does not add another refinement pass.

```bash
python pre_fromxyz.py --xyz-root /path/to/roof.xyz \
  --model-dir /path/to/original-split-model --output-dir /path/to/results \
  --export-evidence --stop-on-error
```

Then use the **CPU reconstruction environment** (Python 3.8.3, NumPy 1.23.5,
SciPy 1.5.0, scikit-learn 0.23.1). An environment specification is provided in
`environment.yml`; installation on a clean machine has not been independently
tested. Do not load the historical sklearn asset in the newer GPU environment.

```bash
python -m postprocessing.reconstruct \
  --input /path/to/results/roof/input.npz \
  --distributions /path/to/results/roof/distributions.npz \
  --assets /path/to/Point2Contour-postprocessing-assets \
  --output /path/to/results/roof/structured.obj
```

Run from the repository root or add it to `PYTHONPATH`. The decoder reconstructs
the primary graph from the supplied evidence, not cached per-building graphs.
It writes metric-coordinate OBJ plus a JSON hash receipt and refuses existing
output paths. The asset SHA-256 is verified before deserialization. Pickle is not
a safe format for arbitrary untrusted downloads; use only the trusted asset.

Input arrays are `pc[N,3]`, `ep[E,3]`, `raw[L,2,3]` in metres. Each of the three
rounds has aligned `raw_k`, `mode_k`, `candidate_i_k` and `bin_k` arrays; `raw`
must match `raw_2`. NMS-only OBJ files cannot substitute for these records.

## Process

1. **Endpoint clustering and graph assembly.** Round metric input coordinates to
   eight decimals, normalize by point-cloud mean and maximum radius, cluster line
   endpoints, and assemble shared vertices with attachment, direction and support
   constraints.
2. **Geometric graph refinement.** Apply geometric simplification, repair,
   evidence-based augmentation, and repair again. This stage can move vertices,
   merge edges and bridge supported gaps.
3. **Learned graph editing.** Apply frozen actor, pruning, local repair, residual
   repair and constrained existing-vertex connection modules. This yields the
   primary graph. These are learned components, not purely geometric rules.
4. **Multi-round evidence selection.** For each query/direction identity, retain
   its latest valid response. Collect both the ordinary endpoint estimate and
   the mode endpoint estimate from the same selected response.
5. **Auxiliary graph reconstruction.** Run the same complete decoder on the two
   selected line clouds. Reconstruct every graph from the current arrays; never
   reuse an unrelated building's cached graph.
6. **Constrained connection completion.** Snap auxiliary edges to existing primary
   vertices within 0.35 m. Require distinct endpoints, length at least 0.2 m,
   absence from the primary graph, and support from ordinary selected responses.
   Support uses 21 samples, finite-segment distance at most 0.25 m, direction
   difference at most 12 degrees, supporting line length greater than 0.15 m,
   and coverage at least 0.65.
7. **Learned selection and export.** Encode each candidate with the frozen
   174-feature extractor and rank with the frozen pair classifier. Select at most
   one candidate with score strictly greater than 0.55, using stable ordering.
   Ordinary-response proposals precede previously unseen mode proposals. Preserve
   all primary vertices and edges and export metric-coordinate OBJ.

Only the final completion stage is append-only. The complete decoder is not.
The three decoder calls are CPU graph processing, not three additional neural
network forward passes. Graphs need not be closed or manifold.

## Inputs and provenance

Required evidence includes the point cloud, predicted edge points, final ordinary
line cloud, and three rounds of valid query/direction IDs with ordinary and mode
segments. NMS-only OBJ files do not contain enough evidence to reproduce this
pipeline. The main-network checkpoint, training split and frozen postprocessor
assets must be recorded separately.

Use the existing original-split checkpoints; this recipe does not retrain the
network. Avoid interpreting a directory named `full` as proof that all labeled
buildings were used for training.

Inference without directly reading target GT does not establish independence of
training or parameter selection from evaluation feedback. Preserve the provenance
of learned editors and the disclosed GT-associated input normalization. Do not
describe this pipeline as parameter-free, learning-free or an untouched blind test.

## Verification and scope

The source-only runtime closure preserves the evaluated numerical operations;
only fixed dispatch, dependency binding and process-oriented naming were changed.
`_primitives.py` contains the geometry and editing functions; `_runtime.py` binds
their separate namespaces and frozen constants without importing experiment code.
There are no training, scoring or historical experiment entry points in this path.

Saved-evidence parity and an actual old-checkpoint evidence-export smoke test are
performed outside the code checkout. These checks are not a new platform score,
not a full-dataset rerun, and not evidence that every installation is supported.
Keep the provenance of the six original frozen editors with the separate compact
runtime asset. Licensing and public hosting are independent release decisions.
