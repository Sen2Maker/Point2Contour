"""Response selection and finite-segment support for connection completion.

Pure geometry utilities; no checkpoint loading, inference or metric evaluation.
"""
import numpy as np


def select_latest_responses(rounds):
    """Select aligned ordinary/mode segments from chronological valid rounds.

    Each round contains query_ids, direction_ids, segments and mode_segments.
    Invalid responses must already have been filtered by the network exporter.
    Output order is newest round first, retaining within-round order.
    """
    seen, ordinary, modal, sources = set(), [], [], []
    for index in range(len(rounds) - 1, -1, -1):
        item = rounds[index]
        ids = np.asarray(item['query_ids'])
        directions = np.asarray(item['direction_ids'])
        lines = np.asarray(item['segments'], dtype=np.float64)
        modes = np.asarray(item['mode_segments'], dtype=np.float64)
        n = len(ids)
        if ids.shape != (n,) or directions.shape != (n,):
            raise ValueError('IDs must be one-dimensional and aligned')
        if ids.dtype.kind not in 'iu' or directions.dtype.kind not in 'iu':
            raise ValueError('IDs must be integer arrays')
        if lines.shape != (n, 2, 3) or modes.shape != (n, 2, 3):
            raise ValueError('Segments must have aligned shape (N, 2, 3)')
        if not np.isfinite(lines).all() or not np.isfinite(modes).all():
            raise ValueError('Non-finite response coordinates')
        keys = list(zip(ids.tolist(), directions.tolist()))
        if len(set(keys)) != n:
            raise ValueError('Duplicate query/direction identity within a round')
        for row, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)
            ordinary.append(lines[row])
            modal.append(modes[row])
            sources.append((index, row))
    return (np.asarray(ordinary, dtype=np.float64).reshape(-1, 2, 3),
            np.asarray(modal, dtype=np.float64).reshape(-1, 2, 3), sources)


def support_fraction(line, responses):
    """Return coverage of 21 candidate samples by aligned finite response lines."""
    line = np.asarray(line, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    if line.shape != (2, 3) or responses.ndim != 3 or responses.shape[1:] != (2, 3):
        raise ValueError('Expected line (2,3) and responses (N,2,3)')
    if not np.isfinite(line).all() or not np.isfinite(responses).all():
        raise ValueError('Non-finite coordinates')
    direction = line[1] - line[0]
    unit = direction / max(np.linalg.norm(direction), 1e-20)
    delta = responses[:, 1] - responses[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    units = delta / np.maximum(lengths[:, None], 1e-20)
    accepted = (np.abs(units @ unit) >= np.cos(np.deg2rad(12))) & (lengths > 0.15)
    segments = responses[accepted]
    if not len(segments):
        return 0.0
    points = line[0] + np.linspace(0, 1, 21)[:, None] * direction
    delta = segments[:, 1] - segments[:, 0]
    fraction = np.clip(np.einsum('nmk,mk->nm', points[:, None] - segments[None, :, 0], delta)
                       / np.maximum(np.sum(delta * delta, axis=1), 1e-20), 0, 1)
    nearest = segments[None, :, 0] + fraction[:, :, None] * delta[None]
    distances = np.linalg.norm(points[:, None] - nearest, axis=-1)
    return float(np.mean(distances.min(axis=1) <= 0.25))
