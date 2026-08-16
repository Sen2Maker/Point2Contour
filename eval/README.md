# Line-cloud Evaluation

`eval_lineseg.py` evaluates the Point2Contour line-cloud output using the seven metrics reported in the paper. It evaluates every scene independently and then computes the unweighted mean of the scene-level results.

## Input Layout

The target directory must contain one subdirectory per scene:

```text
inference_results/
├── scene_0001/
│   ├── gt_wire.obj
│   ├── pre_seg.obj
│   └── pre_seg_nms.obj
└── scene_0002/
    ├── gt_wire.obj
    ├── pre_seg.obj
    └── pre_seg_nms.obj
```

OBJ files must use `v x y z` records for vertices and `l i j` records for line segments. Coordinates are evaluated directly in metric space. Do not normalize predictions or ground truth to `[-1, 1]` before evaluation.

By default, the script uses:

- `gt_wire.obj` as ground truth;
- `pre_seg.obj` as the non-NMS prediction;
- `pre_seg_nms.obj` as the NMS prediction;
- `0.2 m` as the matching threshold.

## Running Evaluation

Run the primary line-cloud evaluation with:

```bash
python eval/eval_lineseg.py --target-dir /path/to/inference_results
```

Each predicted and ground-truth segment is sampled at a default interval of `0.1 m`, equal to half of the default threshold. Distances are measured from sampled points to the nearest segment in the opposite line set. Predicted corners are defined as the unique terminals of the predicted contour lines.

### Metrics

| Metric | Name | Description |
|---|---|---|
| EGR | Edge Geometry Recall | Percentage of sampled GT points within the threshold of a predicted contour line. |
| EGP | Edge Geometry Precision | Percentage of sampled prediction points within the threshold of a GT edge. |
| GF1 | Edge Geometry F1 | Harmonic mean of EGR and EGP. |
| EGD | Edge Geometry Distance | Mean distance from sampled prediction points to the nearest GT edge. |
| CR | Corner Recall | Percentage of GT corners whose Hungarian-matched predicted terminal lies within the threshold. |
| CD | Corner Distance | Mean Euclidean distance between Hungarian-matched GT corners and predicted terminals. |
| LN | Line Number | Number of predicted contour lines per scene. |

EGR, EGP, GF1, and CR are written as percentages. EGD and CD are measured in meters. LN is averaged across scenes in the dataset report.

### Options

The paper uses a threshold of `0.2 m` and a sampling interval of `0.1 m`. When changing the threshold, set `--density` to half of that threshold to retain the same sampling rule:

```bash
python eval/eval_lineseg.py \
  --target-dir /path/to/inference_results \
  --thresholds 0.4 \
  --density 0.2 \
  --output-dir /path/to/linecloud_metrics
```

Use repeated `--prediction NAME=FILENAME` arguments to evaluate different files:

```bash
python eval/eval_lineseg.py \
  --target-dir /path/to/inference_results \
  --prediction refined=pre_seg.obj \
  --prediction refined_nms=pre_seg_nms.obj
```

`--gt-filename` changes the ground-truth filename.

### Outputs

If `--output-dir` is omitted, outputs are written to the target directory:

```text
eval_report_<timestamp>.txt
eval_scene_metrics_<threshold>m_<timestamp>.csv
```

The text report contains the dataset means. Each CSV preserves the metrics for every scene and prediction target.

## Command Reference

```bash
python eval/eval_lineseg.py --help
```
