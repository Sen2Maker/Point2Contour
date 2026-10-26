# Structured wireframe reconstruction

**Review draft:** this directory documents the current method and provides its
response-selection and geometric support utilities. The complete frozen graph
decoder and learned feature runtime are not yet packaged. It is not an executable
replacement for the evaluated pipeline. Do not use `tools/post.py` as a substitute
and label its results as this method.

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

## Release requirements

Before this directory can be advertised as a reproducible inference package:

- Extract the complete graph decoder and feature runtime without evaluation,
  training, private paths or historical experiment dependencies.
- Package the six required frozen model assets with verified hashes, compatible
  runtime versions, provenance and redistribution permission.
- Verify complete-output equivalence against existing saved array fixtures in
  an experiment workspace, including empty and degenerate inputs.
- Connect a network evidence exporter and CLI, with explicit input/output schemas.

Current utilities have synthetic unit tests only; those tests do not certify the
complete pipeline or the learned components.
