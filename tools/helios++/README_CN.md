# 使用 HELIOS++ 从 3DBAG 构建 Point2Contour 数据

本目录用于复现 Point2Contour 使用的 Woerden 合成迁移数据。目录中只包含脚本、固定的 3DBAG 瓦片清单和环境信息，不包含 3DBAG 原始模型、HELIOS++ 本体、仿真点云或生成的线框。

完整流程如下：

```text
3DBAG OBJ 瓦片（LoD2.2）
        |
        v
逐建筑屋顶网格 + GT 屋顶线框
        |
        v
HELIOS++ 机载激光扫描仿真
        |
        v
dataset/
└── train/
    ├── xyz/<id>.xyz
    ├── wireframe/<id>.obj
    ├── roof_mesh/<id>.obj
    ├── all.txt
    └── metadata/
```

父目录 `dataset/` 符合项目根目录 `data_pare.py` 接受的训练数据组织方式；生成的 `train/xyz` 也可以直接交给 `pre_fromxyz.py` 预测。

## 数据与软件许可

3DBAG 是采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 的开放数据。官方条款允许复制、分发和改编，但需要署名、附上许可链接并说明做过的修改。因此，在开源仓库中提供 3DBAG 官网入口、固定瓦片 ID 和自动下载脚本是允许的，不必也不建议把数据文件重新上传到本仓库。

请使用 3DBAG 要求的署名：

> © 3DBAG by tudelft3d and 3DGI

本流程对原始数据的修改包括：选择 LoD2.2 屋顶面、派生屋顶线框，以及使用 HELIOS++ 仿真 LiDAR 回波。具体条款见 [3DBAG 版权说明](https://docs.3dbag.nl/en/copyright/) 和 [官方 OBJ 文档](https://docs.3dbag.nl/en/delivery/obj/)。

HELIOS++ 通过官方渠道单独安装，采用 GPL-3.0-or-later 与 LGPL-3.0-or-later 双许可，详见 [HELIOS++ 官方仓库](https://github.com/3dgeo-heidelberg/helios)。Point2Contour 不复制 HELIOS++ 的源码或二进制文件。

## 固定的复现对象

[`woerden_tiles_v20250903.txt`](woerden_tiles_v20250903.txt) 固定了以下内容：

- 3DBAG 版本：`v20250903`
- 区域：Woerden 及其紧邻区域
- 格式：逐瓦片 Wavefront OBJ 压缩包
- 瓦片数量：30
- 片区锚点：`8-416-592`、`9-416-584`、`9-436-596` 和 `8-432-584`；提交的清单列出了实验实际使用的相交瓦片
- 几何层级：LoD2.2（压缩包内写作 `LoD22`）
- 坐标：Amersfoort / RD New 平面坐标与 NAP 高程，单位为米

瓦片顺序参与数字场景 ID 的映射。若需要复现既有 ID，请不要调整清单顺序。换用其他 3DBAG 版本可能改变建筑对象、材质编号或几何内容，应视为新的数据版本。

固定版本共解析出 35,453 个屋顶对象。其中历史 ID `20436`、`23657` 和 `32031` 对应的 3DBAG 源 OBJ 含有异常坐标；默认的 1,000 m 包围盒检查会拒绝这三个对象，最终得到 35,450 个有效配对场景。所有拒绝记录都会写入 `metadata/buildings.csv`，不会静默删除。

## 1. 安装 HELIOS++

HELIOS++ 官方推荐使用 Conda。在 Point2Contour 根目录执行：

```bash
conda env create -f "tools/helios++/environment.yml"
conda activate point2contour-helios
helios --version
helios --test
```

复现环境固定 HELIOS++ 2.2.2，这是整理本流程时使用的版本。仿真脚本还会把实际可执行文件、版本、资产目录、Python 版本和参数保存到 `metadata/simulation_config.json`。2.2.2 的 Conda 构建会随平台变化；如果受支持平台无法解析固定的 Python 版本，只删除 `environment.yml` 中的 `python=3.10` 一行，并继续保留 `helios=2.2.2`。

也可以从 [HELIOS++ Releases](https://github.com/3dgeo-heidelberg/helios/releases) 使用官方独立安装包。如果环境中无法导入 `pyhelios`，运行 `03_run_helios.py` 时需传入 `--assets-root`；该目录下必须包含 `data/platforms.xml` 和 `data/scanners_als.xml`。

本流程不要求从源码编译 HELIOS++。如确有开发需求，请遵循上游仓库的最新说明。

## 2. 下载 3DBAG 瓦片

[3DBAG 官方下载页](https://3dbag.nl/en/download)提供当前版本的交互式瓦片选择器和瓦片索引。精确复现时应使用本目录固定的版本和瓦片清单：

```bash
python "tools/helios++/01_download_3dbag.py" \
  --output-dir /path/to/woerden_work/3dbag_obj
```

脚本仅从 `https://data.3dbag.nl` 下载文件，首先按 [`woerden_archives_v20250903.sha256`](woerden_archives_v20250903.sha256) 校验每个文件，再检查压缩包是否包含预期的 LoD2.2 OBJ、执行 ZIP 完整性测试，最后写出 `download_manifest.csv`。再次执行时会校验已有文件，传入 `--overwrite` 才会重新下载。`--skip-checksum` 只用于有意更换版本或自定义瓦片集的情况。

只查看全部下载地址而不下载：

```bash
python "tools/helios++/01_download_3dbag.py" \
  --output-dir /path/to/unused \
  --dry-run
```

## 3. 生成屋顶网格和 GT 线框

```bash
python "tools/helios++/02_prepare_roofs.py" \
  --zip-dir /path/to/woerden_work/3dbag_obj \
  --output-dir /path/to/woerden_work/woerden/train
```

该阶段对每个建筑对象执行：

1. 从瓦片压缩包直接读取 `LoD22-3D.obj`；
2. 只保留材质 `1` 对应的屋顶面；
3. 只导出被屋顶面引用的顶点；
4. 保留屋顶外边界和非流形边；
5. 两个屋顶面共享边时，若两侧法向量的无向夹角大于 4 度，则保留为屋脊或折线；
6. 包围盒对角线超过 1,000 m 时，判定源几何异常并拒绝。

这些默认参数对应历史处理设置，但不再先把所有瓦片合并成一个超大 OBJ。可通过 `--roof-material`、`--crease-angle` 和 `--max-bbox-diagonal` 显式调整。

该阶段会对每个源压缩包计算 SHA-256，输出结构为：

```text
woerden/
└── train/
    ├── roof_mesh/                 # HELIOS++ 场景网格
    ├── wireframe/                 # GT OBJ，使用 v 和 l
    └── metadata/
        ├── buildings.csv          # 来源对象、瓦片、包围盒、数量和状态
        ├── preparation.json       # 参数与瓦片清单
        └── prepared_ids.txt
```

为防止新 GT 与旧点云混用，当同一目录已经存在仿真 XYZ 时，`--overwrite` 会拒绝替换屋顶几何。修改源版本或提取参数后应使用新的输出目录。

## 4. 使用 HELIOS++ 仿真 ALS 点云

先运行三个场景进行冒烟测试：

```bash
conda activate point2contour-helios
python "tools/helios++/03_run_helios.py" \
  --prepared-dir /path/to/woerden_work/woerden/train \
  --max-scenes 3
```

检查生成的 XYZ 和 `_helios/logs/` 后，去掉 `--max-scenes` 运行全部数据：

```bash
python "tools/helios++/03_run_helios.py" \
  --prepared-dir /path/to/woerden_work/woerden/train
```

已有的非空 XYZ 默认跳过，因此中断后可以继续运行。每次调用都会在 `metadata/simulation_runs/` 保存一份不可覆盖的配置，每个已仿真场景则通过 `metadata/simulation.csv` 指向对应运行。每栋建筑的 HELIOS++ 日志保存在 `_helios/logs/`；LAS 中间结果在成功转成 XYZ 后默认删除，使用 `--keep-las` 可以保留。

历史命令使用 8 个 HELIOS++ 工作线程。固定随机种子能够保持实验协议和点数行为，但并行调度可能改变扫描噪声随机数的消费顺序，因此重复执行 `--jobs 8` 不保证输出文件逐字节一致。如果更重视严格确定性而不是历史并行设置，请使用单线程；本机连续两次单线程冒烟测试得到的 XYZ 完全一致：

```bash
python "tools/helios++/03_run_helios.py" \
  --prepared-dir /path/to/woerden_work/woerden/train \
  --jobs 1
```

历史 ALS 参数已经设为默认值：

| 参数 | 数值 |
|---|---:|
| 平台 | `sr22` |
| 扫描仪 | `leica_als50-ii` |
| 随机种子 | 42 |
| 飞行 Z 坐标 | 150 m |
| X/Y 航线外扩 | 50 m / 20 m |
| 平台速度 | 50 m/s |
| 脉冲频率 | 70,000 Hz |
| 扫描角 | 60 度 |
| 扫描频率 | 50 Hz |
| 轨迹间隔 | 0.05 s |
| HELIOS++ 线程数 | 8 |

仿真阶段始终保留真实米制坐标，不平移、不缩放。Point2Contour 会在数据预处理或原始 XYZ 前向时执行自己的逐场景归一化。

## 5. 验证最终数据

全量仿真后检查所有配对文件，并生成 `all.txt`：

```bash
python "tools/helios++/04_verify_dataset.py" \
  --dataset-dir /path/to/woerden_work/woerden/train \
  --expected-scenes 35450
```

脚本检查 XYZ 有限坐标、OBJ 索引、退化边、缺失配对和预期场景数量，并写出 `metadata/validation.csv` 与 `metadata/validation.json`。发现问题时返回非零退出码。

## 6A. 准备 Point2Contour 训练数据

如果要用合成数据进行监督训练，从父级数据集目录运行集成预处理：

```bash
python data_pare.py \
  --data-root /path/to/woerden_work/woerden \
  --output-dir /path/to/woerden_work/woerden_processed
```

脚本会处理 `train/` 下全部配对场景，并以固定方式生成 80%/20% 的训练和验证列表。如果只用于迁移测试而不重新划分，请直接采用下面的原始 XYZ 前向预测。

## 6B. 直接从 XYZ 前向预测

直接预测不需要预生成 PKL：

```bash
python pre_fromxyz.py \
  --xyz-root /path/to/woerden_work/woerden/train/xyz \
  --gt-root /path/to/woerden_work/woerden/train/wireframe \
  --ids /path/to/woerden_work/woerden/train/all.txt \
  --model-dir /path/to/training_result \
  --checkpoint final \
  --device cuda
```

只需要预测时可以省略 `--gt-root`。原始 XYZ 前向使用的分块、KNN、特征传播和球面射线参数应与训练预处理保持一致。

## 文件说明

| 文件 | 作用 |
|---|---|
| `01_download_3dbag.py` | 从官方地址下载并校验固定的 OBJ 瓦片 |
| `02_prepare_roofs.py` | 生成逐建筑屋顶网格和 GT 线框 |
| `03_run_helios.py` | 生成 XML、运行 HELIOS++ 并将 LAS 转成 XYZ |
| `04_verify_dataset.py` | 检查配对数据并生成场景 ID 清单 |
| `woerden_tiles_v20250903.txt` | 有顺序、固定版本的区域定义 |
| `woerden_archives_v20250903.sha256` | 精确的源压缩包校验值 |
| `environment.yml` | 固定的 HELIOS++ 仿真环境 |

所有脚本均可通过 `--help` 查看完整参数。实验归档时应同时保留生成的 CSV 和 JSON，它们是数字 ID、3DBAG 瓦片与 BAG 建筑对象之间的审计链。
