# Release candidate review

This branch is a **partial review draft**, not a complete public reproduction.
Existing model/training/evaluation code is unchanged.

## Included

- Independent comparison directories with pinned upstream revisions and explicit
  parameter configurations; no external method source tree is vendored.
- The existing 3DLineDetection parameter recipe, preserving its distinction
  between parameter changes and separate input/output adaptations.
- A reversible NerVE parameter patch and a configuration-selecting Xin wrapper.
- A process-named description of current wireframe reconstruction, synthetic-tested
  evidence utilities, and hashes of required (not included) frozen assets.

## Verified in this draft

- NerVE patch dry run, application, idempotence, restoration and clean upstream
  readback. No model execution or metrics were run.
- Shell syntax and six synthetic evidence tests.
- Added public material contains no local experiment paths or numbered historical
  postprocessing names.

The copied 3DLineDetection recipe has not been re-executed against a fresh upstream
checkout in this review. The Xin wrapper has not been used for extraction in this
review. Their existing experiment validation is not a test of this new packaging.

## Not ready for release

1. Complete postprocessor runtime extraction, evidence-export CLI and frozen model
   packaging remain unfinished. There is no complete runnable wireframe CLI here.
2. Complete-output equivalence with saved evidence has not been tested for the new
   packaging. Unit tests cannot replace this check.
3. The repository has no top-level redistribution license. The owner must choose
   one before this can be called a licensed open-source release. External projects
   retain their own licenses; model asset rights/provenance must also be checked.

Do not publish the partial utilities as a replacement for the evaluated pipeline.
