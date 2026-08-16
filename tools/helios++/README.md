# Woerden Data Preparation with HELIOS++

These scripts reproduce the synthetic Woerden dataset used in Point2Contour. They download the fixed 3DBAG tiles, extract roof geometry, simulate airborne laser scanning with HELIOS++, and create paired point clouds and wireframes.

Run all commands from the Point2Contour project root.

## 1. Install HELIOS++

```bash
conda env create -f "tools/helios++/environment.yml"
conda activate point2contour-helios
helios --version
```

The environment uses HELIOS++ 2.2.2.

## 2. Generate the Dataset

Set one working directory and run the four scripts in order:

```bash
WORK_DIR=/path/to/woerden_work

# Download the fixed 3DBAG tiles.
# For another area, pass the same one-ID-per-line tile list to steps 01 and 02; also add --skip-checksum to step 01.
python "tools/helios++/01_download_3dbag.py" \
  --output-dir "$WORK_DIR/3dbag_obj"

# Extract one roof mesh and GT wireframe per building.
python "tools/helios++/02_prepare_roofs.py" \
  --zip-dir "$WORK_DIR/3dbag_obj" \
  --output-dir "$WORK_DIR/woerden/train"

# Simulate one ALS point cloud per building.
python "tools/helios++/03_run_helios.py" \
  --prepared-dir "$WORK_DIR/woerden/train"

# Validate the generated pairs and create all.txt.
python "tools/helios++/04_verify_dataset.py" \
  --dataset-dir "$WORK_DIR/woerden/train" \
  --expected-scenes 35450
```

The simulation can be resumed. Existing non-empty XYZ files are skipped automatically.

The final dataset is stored as:

```text
woerden/
└── train/
    ├── xyz/<id>.xyz
    ├── wireframe/<id>.obj
    ├── roof_mesh/<id>.obj
    ├── all.txt
    └── metadata/
```

## 3. Use the Dataset

To prepare it for training, return to the Point2Contour environment and run:

```bash
conda activate point2contour

python data_pare.py \
  --data-root "$WORK_DIR/woerden" \
  --output-dir "$WORK_DIR/woerden_processed"
```

To run the pretrained Tallinn model directly on the generated XYZ files:

```bash
conda activate point2contour

python pre_fromxyz.py \
  --xyz-root "$WORK_DIR/woerden/train/xyz" \
  --gt-root "$WORK_DIR/woerden/train/wireframe" \
  --model-dir results/tallinn_full
```

## Data Source

The committed tile list fixes 30 Woerden-area tiles from 3DBAG release `v20250903`. The download script retrieves them from the official 3DBAG service and verifies their checksums.

3DBAG data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Required attribution: **© 3DBAG by tudelft3d and 3DGI**. HELIOS++ is installed separately from its [official repository](https://github.com/3dgeo-heidelberg/helios); no HELIOS++ source or binary is redistributed here.
