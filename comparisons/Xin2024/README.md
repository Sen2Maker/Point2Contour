# Xin2024 independent reproduction

This recipe uses our independent implementation of the paper, **not the authors' executable or source code**.

Repository: https://github.com/Sen2Maker/Accurate_And_Complete_Line_Segment_Extraction_Reproduction

```bash
git clone https://github.com/Sen2Maker/Accurate_And_Complete_Line_Segment_Extraction_Reproduction.git XinReproduction
git -C XinReproduction checkout 168c6fe6de0886ecb28ee8e917da41bc82577bf1
# Install the reproduction in its own environment first.
bash run.sh /path/to/XinReproduction building3d input.xyz output.obj
bash run.sh /path/to/XinReproduction paper-range-center input.xyz output.obj
```

The wrapper explicitly selects a bundled configuration; it never edits the upstream source. `paper-range-center` is a reference configuration, not a claim of the authors' exact settings. The Building3D configuration changes feature neighbors (32 → 40), retained feature ratio (0.2 → 0.1), minimum cluster size (2 → 10), and epsilon factor (4 → 1). Both profiles cap FPO at 30 iterations; this does not guarantee strict convergence.

Validated experimental environment: Linux, Python 3.8, NumPy 1.23.5, SciPy 1.5.0. Use one FPO worker per building. Inputs and outputs retain metric XYZ coordinates.
