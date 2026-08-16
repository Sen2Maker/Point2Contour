#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate per-building ALS point clouds with HELIOS++.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Building3D-style dataset root; defaults to prepared-dir",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--helios-bin", default="helios")
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="Directory containing data/platforms.xml and data/scanners_als.xml",
    )
    parser.add_argument("--ids", default=None, help="Comma-separated IDs or a text file")
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flight-z", type=float, default=150.0)
    parser.add_argument("--margin-x", type=float, default=50.0)
    parser.add_argument("--margin-y", type=float, default=20.0)
    parser.add_argument("--move-speed", type=float, default=50.0)
    parser.add_argument("--pulse-frequency", type=int, default=70000)
    parser.add_argument("--scan-angle", type=float, default=60.0)
    parser.add_argument("--scan-frequency", type=float, default=50.0)
    parser.add_argument("--trajectory-interval", type=float, default=0.05)
    parser.add_argument("--platform", default="data/platforms.xml#sr22")
    parser.add_argument("--scanner", default="data/scanners_als.xml#leica_als50-ii")
    parser.add_argument("--keep-las", action="store_true")
    parser.add_argument("--show-helios-output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def resolve_helios_binary(value):
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError("HELIOS++ executable not found: {}".format(candidate))
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(
            "HELIOS++ executable '{}' is not on PATH. Activate the HELIOS++ environment or pass --helios-bin.".format(
                value
            )
        )
    return resolved


def resolve_assets_root(value):
    if value is not None:
        root = value.expanduser().resolve()
    else:
        try:
            import pyhelios
        except ImportError as exc:
            raise RuntimeError(
                "Cannot locate HELIOS++ assets automatically. Run this script in the HELIOS++ Conda environment or pass --assets-root."
            ) from exc
        root = Path(pyhelios.__file__).resolve().parent
    required = [root / "data" / "platforms.xml", root / "data" / "scanners_als.xml"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing HELIOS++ assets: {}".format(", ".join(missing)))
    return root


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "scene_id",
        "roof_mesh",
        "status",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError("Missing columns in {}: {}".format(path, sorted(missing)))
    return rows


def parse_id_source(value):
    if value is None:
        return None
    possible_path = Path(value).expanduser()
    text = possible_path.read_text(encoding="utf-8") if possible_path.is_file() else value
    tokens = text.replace(",", " ").replace(";", " ").split()
    return {str(int(token)) for token in tokens}


def select_rows(rows, wanted, start_id, end_id, max_scenes):
    selected = []
    for row in rows:
        if row["status"] != "ok":
            continue
        scene_id = int(row["scene_id"])
        if wanted is not None and str(scene_id) not in wanted:
            continue
        if start_id is not None and scene_id < start_id:
            continue
        if end_id is not None and scene_id > end_id:
            continue
        selected.append(row)
    selected.sort(key=lambda row: int(row["scene_id"]))
    if max_scenes is not None:
        selected = selected[:max_scenes]
    return selected


def write_scene(path, mesh_path):
    mesh_value = quoteattr(str(mesh_path.resolve()))
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="roof_scene" name="roof_scene">
        <part id="0">
            <filter type="objloader">
                <param type="string" key="filepath" value={mesh} />
                <param type="string" key="up" value="z" />
            </filter>
        </part>
    </scene>
</document>
""".format(mesh=mesh_value),
        encoding="utf-8",
    )


def write_survey(path, scene_relative, scene_id, bbox, args):
    xmin, xmax, ymin, ymax = bbox
    x0 = xmin - args.margin_x
    x1 = xmax + args.margin_x
    y0 = ymin - args.margin_y
    y1 = (ymin + ymax) / 2.0
    values = {
        "scene": quoteattr(scene_relative.as_posix() + "#roof_scene"),
        "name": quoteattr("roof_als_{}".format(scene_id)),
        "platform": quoteattr(args.platform),
        "scanner": quoteattr(args.scanner),
        "pulse": args.pulse_frequency,
        "angle": args.scan_angle,
        "frequency": args.scan_frequency,
        "flight_z": args.flight_z,
        "speed": args.move_speed,
        "interval": args.trajectory_interval,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scannerSettings id="set" active="true" pulseFreq_hz="{pulse}" scanAngle_deg="{angle}" scanFreq_hz="{frequency}" />
    <survey name={name} scene={scene} platform={platform} scanner={scanner}>
        <FWFSettings beamSampleQuality="3" binSize_ns="0.25" winSize_ns="1" />
        <detectorSettings rangeMin_m="1" rangeMax_m="1700" />
        <leg>
            <platformSettings x="{x0:.3f}" y="{y0:.3f}" z="{flight_z:.3f}" onGround="false" movePerSec_m="{speed}" />
            <scannerSettings template="set" trajectoryTimeInterval_s="{interval}" />
        </leg>
        <leg>
            <platformSettings x="{x1:.3f}" y="{y0:.3f}" z="{flight_z:.3f}" onGround="false" movePerSec_m="{speed}" />
            <scannerSettings template="set" trajectoryTimeInterval_s="{interval}" />
        </leg>
        <leg>
            <platformSettings x="{x0:.3f}" y="{y1:.3f}" z="{flight_z:.3f}" onGround="false" movePerSec_m="{speed}" />
            <scannerSettings template="set" trajectoryTimeInterval_s="{interval}" />
        </leg>
        <leg>
            <platformSettings x="{x1:.3f}" y="{y1:.3f}" z="{flight_z:.3f}" onGround="false" movePerSec_m="{speed}" />
            <scannerSettings active="false" />
        </leg>
    </survey>
</document>
""".format(**values),
        encoding="utf-8",
    )


def merge_las_to_xyz(las_root, xyz_path):
    try:
        import laspy
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("03_run_helios.py requires laspy and numpy") from exc

    las_files = sorted(las_root.rglob("*.las"))
    temporary = xyz_path.with_suffix(".xyz.tmp")
    total_points = 0
    used_files = 0
    with temporary.open("w", encoding="utf-8") as output:
        for las_path in las_files:
            las = laspy.read(las_path)
            if len(las.points) == 0:
                continue
            coordinates = np.column_stack((las.x, las.y, las.z))
            np.savetxt(output, coordinates, fmt="%.6f")
            total_points += len(coordinates)
            used_files += 1
    os.replace(str(temporary), str(xyz_path))
    return len(las_files), used_files, total_points


def load_existing_results(path):
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        return {row["scene_id"]: row for row in csv.DictReader(stream)}


def write_results(path, rows):
    fields = [
        "scene_id",
        "status",
        "points",
        "las_files",
        "las_files_used",
        "xyz_file",
        "helios_returncode",
        "log_file",
        "run_id",
        "run_config",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene_id in sorted(rows, key=lambda value: int(value)):
            writer.writerow(rows[scene_id])
    os.replace(str(temporary), str(path))


def remove_scene_temp(path, parent):
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path.parent != resolved_parent:
        raise RuntimeError("Refusing to remove a directory outside {}: {}".format(parent, path))
    shutil.rmtree(str(path))


def command_version(helios_binary):
    result = subprocess.run(
        [helios_binary, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def count_nonempty_lines(path):
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        return sum(1 for line in stream if line.strip())


def copy_wireframes(selected, prepared_dir, output_dir):
    if prepared_dir == output_dir:
        return
    destination_dir = output_dir / "wireframe"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        relative = row.get("wireframe", "")
        if not relative:
            raise ValueError("Scene {} has no wireframe path".format(row["scene_id"]))
        source = prepared_dir / relative
        if not source.is_file():
            raise FileNotFoundError("Wireframe not found: {}".format(source))
        shutil.copy2(str(source), str(destination_dir / "{}.obj".format(row["scene_id"])))


def main():
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be positive")

    prepared_dir = args.prepared_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else prepared_dir
    )
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_dir / "_helios"
    )
    manifest_path = prepared_dir / "metadata" / "buildings.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Preparation manifest not found: {}".format(manifest_path))

    helios_binary = resolve_helios_binary(args.helios_bin)
    assets_root = resolve_assets_root(args.assets_root)
    rows = read_manifest(manifest_path)
    selected = select_rows(
        rows,
        parse_id_source(args.ids),
        args.start_id,
        args.end_id,
        args.max_scenes,
    )
    if not selected:
        raise RuntimeError("No prepared scenes match the requested selection")
    copy_wireframes(selected, prepared_dir, output_dir)

    xyz_dir = output_dir / "xyz"
    metadata_dir = output_dir / "metadata"
    scenes_dir = work_dir / "scenes"
    surveys_dir = work_dir / "surveys"
    las_dir = work_dir / "las"
    logs_dir = work_dir / "logs"
    for directory in (xyz_dir, metadata_dir, scenes_dir, surveys_dir, las_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results_path = metadata_dir / "simulation.csv"
    results = load_existing_results(results_path)
    helios_version = command_version(helios_binary)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
    run_config_relative = "simulation_runs/{}.json".format(run_id)
    run_config_path = metadata_dir / run_config_relative
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "helios_binary": helios_binary,
        "helios_version": helios_version,
        "python": sys.version,
        "assets_root": str(assets_root),
        "parameters": {
            "jobs": args.jobs,
            "seed": args.seed,
            "flight_z_m": args.flight_z,
            "margin_x_m": args.margin_x,
            "margin_y_m": args.margin_y,
            "move_speed_m_per_s": args.move_speed,
            "pulse_frequency_hz": args.pulse_frequency,
            "scan_angle_degrees": args.scan_angle,
            "scan_frequency_hz": args.scan_frequency,
            "trajectory_interval_s": args.trajectory_interval,
            "platform": args.platform,
            "scanner": args.scanner,
        },
    }
    serialized_provenance = json.dumps(provenance, indent=2, ensure_ascii=True) + "\n"
    run_config_path.write_text(serialized_provenance, encoding="utf-8")
    (metadata_dir / "simulation_config.json").write_text(
        serialized_provenance, encoding="utf-8"
    )
    print("HELIOS++: {}".format(helios_version or "version unavailable"))
    print("Scenes selected: {}".format(len(selected)))

    success_count = 0
    failure_count = 0
    for position, row in enumerate(selected, start=1):
        scene_id = row["scene_id"]
        xyz_path = xyz_dir / "{}.xyz".format(scene_id)
        log_path = logs_dir / "{}.log".format(scene_id)
        if xyz_path.is_file() and xyz_path.stat().st_size > 0 and not args.overwrite:
            print("[{}/{}] {}: existing".format(position, len(selected), scene_id))
            if scene_id not in results:
                results[scene_id] = {
                    "scene_id": scene_id,
                    "status": "existing",
                    "points": count_nonempty_lines(xyz_path),
                    "las_files": "",
                    "las_files_used": "",
                    "xyz_file": xyz_path.relative_to(output_dir).as_posix(),
                    "helios_returncode": "",
                    "log_file": "",
                    "run_id": "",
                    "run_config": "",
                }
                write_results(results_path, results)
            success_count += 1
            continue
        if xyz_path.exists() and args.overwrite:
            xyz_path.unlink()

        mesh_path = prepared_dir / row["roof_mesh"]
        if not mesh_path.is_file():
            raise FileNotFoundError("Roof mesh not found: {}".format(mesh_path))
        scene_path = scenes_dir / "{}_scene.xml".format(scene_id)
        survey_path = surveys_dir / "{}_survey.xml".format(scene_id)
        bbox = tuple(float(row[key]) for key in ("xmin", "xmax", "ymin", "ymax"))
        write_scene(scene_path, mesh_path)
        write_survey(
            survey_path,
            scene_path.relative_to(work_dir),
            scene_id,
            bbox,
            args,
        )

        scene_las_dir = las_dir / scene_id
        if scene_las_dir.exists():
            remove_scene_temp(scene_las_dir, las_dir)
        scene_las_dir.mkdir(parents=True)
        command = [
            helios_binary,
            survey_path.relative_to(work_dir).as_posix(),
            "--assets",
            str(work_dir),
            "--assets",
            str(assets_root),
            "--output",
            str(scene_las_dir),
            "--lasOutput",
            "--seed",
            str(args.seed),
            "--rebuildScene",
            "-j",
            str(args.jobs),
        ]
        print("[{}/{}] {}".format(position, len(selected), scene_id))
        with log_path.open("w", encoding="utf-8") as log_stream:
            log_stream.write("COMMAND\n{}\n\nOUTPUT\n".format(" ".join(command)))
            log_stream.flush()
            if args.show_helios_output:
                result = subprocess.run(command, cwd=str(work_dir), check=False)
            else:
                result = subprocess.run(
                    command,
                    cwd=str(work_dir),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

        las_total = 0
        las_used = 0
        point_count = 0
        status = "helios_failed"
        if result.returncode == 0:
            las_total, las_used, point_count = merge_las_to_xyz(scene_las_dir, xyz_path)
            status = "ok" if point_count > 0 else "empty_xyz"
        results[scene_id] = {
            "scene_id": scene_id,
            "status": status,
            "points": point_count,
            "las_files": las_total,
            "las_files_used": las_used,
            "xyz_file": xyz_path.relative_to(output_dir).as_posix(),
            "helios_returncode": result.returncode,
            "log_file": log_path.relative_to(output_dir).as_posix()
            if output_dir in log_path.parents
            else str(log_path),
            "run_id": run_id,
            "run_config": run_config_relative,
        }
        write_results(results_path, results)

        if status == "ok":
            success_count += 1
            if not args.keep_las:
                remove_scene_temp(scene_las_dir, las_dir)
        else:
            failure_count += 1
            print("Scene {} failed with status {}; see {}".format(scene_id, status, log_path))
            if args.stop_on_error:
                break

    print("Completed: {} successful, {} failed.".format(success_count, failure_count))
    print("XYZ directory: {}".format(xyz_dir))
    print("Simulation manifest: {}".format(results_path))
    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
