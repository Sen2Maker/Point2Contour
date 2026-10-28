"""Reconstruct metric wireframes from saved network evidence, without GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np

ASSET_SHA256 = '0cdea30038fe474332f41f4401bce1a50d444b6d827ab594bc66a79b7bfc1b6c'


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate(arrays, distributions):
    for key, shape in (('pc', (3,)), ('ep', (3,)), ('raw', (2, 3))):
        value = np.asarray(arrays[key])
        if value.ndim != len(shape) + 1 or value.shape[1:] != shape or not np.isfinite(value).all():
            raise ValueError('Invalid evidence array: ' + key)
    if not len(arrays['pc']):
        raise ValueError('The input point cloud must not be empty')
    for stage in range(3):
        raw = distributions['raw_' + str(stage)]
        mode = distributions['mode_' + str(stage)]
        query = distributions['candidate_i_' + str(stage)]
        direction = distributions['bin_' + str(stage)]
        if raw.shape != (len(query), 2, 3) or mode.shape != raw.shape or direction.shape != query.shape or query.ndim != 1:
            raise ValueError('Misaligned round evidence')
        if not np.isfinite(raw).all() or not np.isfinite(mode).all():
            raise ValueError('Nonfinite round evidence')
        for ids in (query, direction):
            if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
                raise ValueError('Query and direction IDs must be nonnegative integers')
        if len(set(zip(query.tolist(), direction.tolist()))) != len(query):
            raise ValueError('Duplicate query/direction identity in one round')
    if not np.array_equal(arrays['raw'], distributions['raw_2']):
        raise ValueError('Final raw evidence does not match round three')


class Reconstructor:
    def __init__(self, assets):
        # The historical sklearn trees are not cross-version portable. Fail
        # before unpickling instead of silently changing predictions.
        import sklearn
        import scipy
        if sys.version_info[:2] != (3, 8) or sklearn.__version__ != '0.23.1' or np.__version__ != '1.23.5' or scipy.__version__ != '1.5.0':
            raise RuntimeError('Use the frozen CPU environment: Python 3.8, NumPy 1.23.5, SciPy 1.5.0, scikit-learn 0.23.1')
        path = Path(assets) / 'frozen_editors.pkl'
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != ASSET_SHA256:
            raise ValueError('Frozen editor asset hash mismatch')
        # Only the trusted, hash-bound release asset may be deserialized.
        state = pickle.loads(payload)
        from ._runtime import build_runtime
        self.runtime = build_runtime(state)

    def decode(self, arrays, distributions):
        validate(arrays, distributions)
        decoded = self.runtime['decode_base'](arrays)
        origin = arrays['pc'].mean(0)
        scale = float(np.linalg.norm(arrays['pc'] - origin, axis=1).max())
        output, metadata, _ = self.runtime['complete_connections'](
            arrays, distributions, decoded['candidate'], origin, scale,
            decoded['parser'] - origin, decoded['balanced'] - origin)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite output geometry')
        return output, metadata, decoded['candidate']

    def write(self, path, lines):
        self.runtime['write_lines'](Path(path), lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='input.npz: pc, ep, raw in metres')
    parser.add_argument('--distributions', required=True, type=Path, help='distributions.npz: three aligned rounds')
    parser.add_argument('--assets', required=True, type=Path, help='Directory containing the trusted frozen editor asset')
    parser.add_argument('--output', required=True, type=Path, help='New OBJ path; existing outputs are refused')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output already exists; refusing overwrite')
    engine = Reconstructor(args.assets)
    arrays, distributions = read_arrays(args.input), read_arrays(args.distributions)
    lines, operation, base = engine.decode(arrays, distributions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.obj.partial')
    if temporary.exists():
        parser.error('Partial output exists; preserve and inspect it first')
    engine.write(temporary, lines)
    temporary.replace(args.output)
    receipt = dict(edges=len(lines), base_edges=len(base), operation=operation,
                   input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   distributions_sha256=hashlib.sha256(args.distributions.read_bytes()).hexdigest(),
                   asset_sha256=ASSET_SHA256,
                   obj_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   base_reconstructed_from_supplied_arrays=True, target_GT_used=False)
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
