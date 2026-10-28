"""Geometry and frozen-editing functions; no training or evaluation entry points."""

from __future__ import annotations

def kernel_0(arrays):
    pc, raw, ep = (arrays[k] for k in ['pc', 'raw', 'ep'])
    parser, balanced, pm = D.STACK['native'].parse_and_repair(raw, ep, pc, D.STACK)
    origin = pc.mean(0)
    scale = float(np.linalg.norm(pc - origin, axis=1).max())
    ev = dict(raw=raw - origin, pc=pc - origin, ep=ep - origin, parser=parser - origin, balanced=balanced - origin, origin=origin, scale=scale)
    actor, am = D.ACTOR.decode(ev)
    out, vm = D.API.decode(D.BUNDLE, ev, actor)
    return dict(parser=np.round(parser, 8), balanced=np.round(balanced, 8), actor=np.round(actor + origin, 8), candidate=np.round(out + origin, 8))

def kernel_1(arrays, distributions, base, origin, scale, parser, balanced):
    raw, _ = H.fallback(distributions)
    modeparts = []
    seen = set()
    for stage in [2, 1, 0]:
        keep = []
        for i, key in enumerate(zip(distributions[f'candidate_i_{stage}'], distributions[f'bin_{stage}'])):
            if key not in seen:
                seen.add(key)
                keep.append(i)
        modeparts.append(distributions[f'mode_{stage}'][keep])
    mode = np.concatenate(modeparts)
    assert len(raw) == len(mode)
    mean_source = H.decode(dict(arrays, raw=raw))['candidate']
    mode_source = H.decode(dict(arrays, raw=mode))['candidate']
    A = D.API
    g = A.c.e.verify.canonical(base - origin)
    oldlines = K.propose(mean_source - origin, g, raw - origin)
    proposed = K.propose(mode_source - origin, g, raw - origin)
    oldkeys = {R.key(line) for line in oldlines}
    new = np.asarray([line for line in proposed if R.key(line) not in oldkeys]).reshape(-1, 2, 3)
    lines = np.concatenate([oldlines, new])
    evidence = dict(raw=raw - origin, pc=arrays['pc'] - origin, ep=arrays['ep'] - origin, parser=parser, balanced=balanced, origin=origin, scale=scale)
    actions = [dict(family='diverse_add', remove=[], add=[line.tolist()]) for line in lines]
    X = np.c_[A.e.alg.features(evidence, g, actions, A.e.LEGACY, A.e.FINITE, A.e.roof), A.f.u.t.r.v.j.r.removed_features(evidence, g, actions)] if actions else np.empty((0, 174))
    assert X.shape == (len(lines), 174) and np.isfinite(X).all()
    probability = D.BUNDLE['pair']['model'].predict_proba(X)[:, 1] if len(lines) else np.array([])
    selected = [int(i) for i in np.argsort(-probability, kind='stable') if probability[i] > 0.55][:1]
    out = A.c.e.verify.canonical(np.r_[g, lines[selected]]) if selected else g.copy()
    world = np.round(out + origin, 8)
    baseworld = np.round(g + origin, 8)
    assert {R.key(x) for x in baseworld} <= {R.key(x) for x in world}
    assert np.array_equal(np.unique(baseworld.reshape(-1, 3), axis=0), np.unique(world.reshape(-1, 3), axis=0))
    assert len(world) - len(baseworld) in (0, 1)
    return (world, dict(old_candidates=len(oldlines), new_candidates=len(new), selected=selected, selected_new=bool(selected and selected[0] >= len(oldlines)), probability=probability[selected].tolist()), dict(lines=lines, X=X, probability=probability, mean_source=mean_source, mode_source=mode_source))

def kernel_2(lines):
    x = np.asarray(lines, float).reshape(-1, 2, 3)
    if not len(x):
        return x
    v, ix = np.unique(x.reshape(-1, 3), axis=0, return_inverse=True)
    e = np.unique(np.sort(ix.reshape(-1, 2), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return v[e]

def kernel_3(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = canonical(lines)
    v, inverse = np.unique(lines.reshape(-1, 3), axis=0, return_inverse=True)
    with path.open('w') as f:
        for point in v:
            f.write('v %.8f %.8f %.8f\n' % tuple(point))
        for a, b in inverse.reshape(-1, 2):
            f.write('l %d %d\n' % (a + 1, b + 1))

def kernel_8(evidence):
    candidate, stats = m.old_candidates(evidence)
    data = dict(candidate=candidate, stats=stats)
    prediction, operations = actor_module.actor(MODEL, evidence, data, 0.45, False)
    return (prediction, operations)

def kernel_10(bundle, evidence, actor_predelete):
    g, old = base_api.decode(bundle, evidence, actor_predelete)
    out, new = refine_graph(bundle, evidence, g)
    return (out, dict(supported_connection_stage=old, fit_guided_existing_connection=new, GT_used=False))

def kernel_16(z):
    seen = set(zip(z['candidate_i_2'].tolist(), z['bin_2'].tolist()))
    arrays = [z['raw_2']]
    rows = {2: len(z['raw_2']), 1: 0, 0: 0}
    keys = set(seen)
    for i in [1, 0]:
        ids = list(zip(z['candidate_i_%d' % i].tolist(), z['bin_%d' % i].tolist()))
        keep = []
        for n, key in enumerate(ids):
            if key not in seen:
                keep.append(n)
                seen.add(key)
        arrays.append(z['raw_%d' % i][keep])
        rows[i] = len(keep)
    out = np.concatenate(arrays)
    assert np.array_equal(out[:len(z['raw_2'])], z['raw_2'])
    assert len(out) == len(seen)
    return (out, dict(from_round2=rows[2], from_round1=rows[1], from_round0=rows[0], lineages=len(seen), final_raw_literal_prefix=True))

def kernel_17(source, g, raw):
    canon = D.API.c.e.verify.canonical
    verts = np.unique(g.reshape(-1, 3), axis=0)
    present = {R.key(line) for line in g}
    lines = []
    if len(verts) and len(source):
        dist = cdist(source.reshape(-1, 3), verts)
        near = dist.argmin(1).reshape(-1, 2)
        dd = dist.min(1).reshape(-1, 2)
        mask = (dd.max(1) <= 0.35) & (near[:, 0] != near[:, 1])
        candidates = canon(verts[near[mask]])
        for line in candidates:
            if np.linalg.norm(line[1] - line[0]) >= 0.2 and R.key(line) not in present and (R.support(line, raw) >= 0.65):
                lines.append(line)
    return canon(np.array(lines).reshape(-1, 2, 3))

def kernel_18(line):
    a = np.asarray(line).reshape(2, 3)
    return tuple(a.reshape(-1)) if tuple(a[0]) <= tuple(a[1]) else tuple(a[::-1].reshape(-1))

def kernel_19(lines):
    lines = np.asarray(lines, dtype=float).reshape(-1, 2, 3)
    if not len(lines):
        return lines.copy()
    v, inverse = np.unique(lines.reshape(-1, 3), axis=0, return_inverse=True)
    e = np.unique(np.sort(inverse.reshape(-1, 2), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return v[e]

def kernel_26(pred, raw, edge_points, steps, scale=None):
    out = pred.copy()
    log = []
    for step in steps:
        module = step['module']
        if module == 'geometry':
            out, meta = frozen_geometry(out, raw_lines=raw, edge_points=edge_points, config=step.get('config', {}))
        elif module == 'geometry_repair':
            out, meta = frozen_repair(out, raw_lines=raw, edge_points=edge_points, config=step.get('config', {}))
        elif module == 'augmentation':
            out, meta = frozen_augmentation(out, raw_lines=raw, edge_points=edge_points, config=step.get('config', {}), scale=scale)
        else:
            raise ValueError('Unsupported frozen geometry stage: ' + module)
        out = canonical(out)
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite repair output')
        log.append(dict(module=module, meta=meta))
    return (out, log)

def kernel_28(lines):
    lines = np.asarray(lines, dtype=float).reshape(-1, 2, 3)
    if not len(lines):
        return lines.copy()
    v, inverse = np.unique(lines.reshape(-1, 3), axis=0, return_inverse=True)
    e = np.unique(np.sort(inverse.reshape(-1, 2), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return v[e]

def kernel_40(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_41(bundle, s, d, threshold, topological_context):
    base = s['balanced']
    current = base.copy()
    c = d['candidate']
    st = d['stats']
    remaining = list(range(len(c)))
    chosen = []
    probs = []
    while remaining and len(chosen) < 6:
        prob = incremental.predict(bundle, dict(s, balanced=current), c[remaining], st[remaining])
        selected = None
        for pos in np.argsort(-prob, kind='stable'):
            if prob[pos] < threshold:
                break
            i = remaining[int(pos)]
            if incremental._redundant(c[i], current, False):
                continue
            selected = i
            probs.append(float(prob[pos]))
            break
        if selected is None:
            break
        chosen.append(selected)
        remaining.remove(selected)
        current = organize(base, c[chosen], s, 'shared_compete')[0] if topological_context else np.concatenate([base, c[chosen]])
    out, tm = organize(base, c[chosen], s, 'shared_compete')
    return (p.canonical(out), dict(selected=chosen, probabilities=probs, topology=tm, topology_feedback=topological_context, GT_used=False))

def kernel_42(bundle, evidence, current_supported_graph):
    g = d.canonical(current_supported_graph)
    native, _ = d.r.n.candidates(evidence['raw'], g)
    patches, _ = p.proposals(evidence, g, native)
    lines = f.candidates(g, patches)
    actions = [dict(family='diverse_add', remove=[], add=[line.tolist()]) for line in lines]
    X = np.c_[e.alg.features(evidence, g, actions, e.LEGACY, e.FINITE, e.roof), f.u.t.r.v.j.r.removed_features(evidence, g, actions)] if actions else np.empty((0, 174))
    prob = bundle['pair']['model'].predict_proba(X)[:, 1] if len(X) else np.array([])
    selected = [int(i) for i in np.argsort(-prob, kind='stable') if prob[i] > 0.55][:1]
    out = d.canonical(np.r_[g, lines[selected]]) if selected else g.copy()
    assert {d.key(line) for line in g} <= {d.key(line) for line in out}
    assert np.array_equal(np.unique(out.reshape(-1, 3), axis=0), np.unique(g.reshape(-1, 3), axis=0))
    return (out, dict(GT_used=False, threshold=0.55, max_add=1, candidates=len(lines), selected=selected, probability=prob[selected].tolist(), vertices_and_literal_edges_preserved=True, main_network_refiner_changed=False))

def kernel_44(bundle, evidence, actor_predelete):
    baseline, old = base_api.decode(bundle['base'], evidence, actor_predelete)
    out, extra = refine_graph(bundle, evidence, baseline)
    return (out, dict(residual_edit_stage=old, fixed_vertex_connection=extra, GT_used=False, main_network_and_refiner_changed=False))

def kernel_53(line, raw):
    d = line[1] - line[0]
    u = d / max(np.linalg.norm(d), 1e-20)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-20)
    good = (abs(ru @ u) >= np.cos(np.deg2rad(12))) & (rl > 0.15)
    points = line[0] + np.linspace(0, 1, 21)[:, None] * d
    D = points_to_lines(points, raw[good])
    return float((D.min(1) <= 0.25).mean()) if D.shape[1] else 0.0

def kernel_72(lines, raw_lines=None, edge_points=None, config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError('Unknown geometry refine keys: ' + str(sorted(unknown)))
        cfg.update(config)
    if cfg['position_weight'] <= 0 or cfg['raw_dedup_m'] <= 0 or cfg['max_move_m'] < 0:
        raise ValueError('Positive regularization/dedup and nonnegative movement required')
    original, raw = (_lines(lines), _lines(raw_lines))
    if not len(original):
        return (original, {'config': cfg, 'fitted_edges': 0, 'moved_vertices': 0, 'ground_truth_used': False})
    origin = original[0, 0].copy()
    local, raw = (original - origin, raw - origin)
    vertices, inverse = np.unique(local.reshape(-1, 3), axis=0, return_inverse=True)
    supplied_edges = inverse.reshape(-1, 2)
    edges = np.unique(np.sort(supplied_edges, axis=1), axis=0)
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError('Zero-length graph edges are unsupported; remove explicitly in IO')
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    incident = [[] for _ in vertices]
    fits, rejects, edge_logs = ([], {}, [])
    for k, (a, b) in enumerate(edges):
        incident[a].append(k)
        incident[b].append(k)
        fit, reason = _fit_edge(vertices[[a, b]], raw, cfg)
        fits.append(fit)
        rejects[reason] = rejects.get(reason, 0) + 1
        if fit:
            edge_logs.append({'edge': [int(a), int(b)], 'support_lines': fit['support_lines'], 'coverage': fit['coverage'], 'rms_m': fit['rms_m'], 'weight': fit['weight']})
    shifts = np.zeros_like(vertices)
    for i, edge_ids in enumerate(incident):
        if not any((fits[k] is not None for k in edge_ids)):
            continue
        matrix = cfg['position_weight'] * np.eye(3)
        rhs = np.zeros(3)
        for k in edge_ids:
            fit = fits[k]
            if fit is None:
                a, b = edges[k]
                direction = (vertices[b] - vertices[a]) / lengths[k]
                projector = np.eye(3) - np.outer(direction, direction)
                matrix += cfg['unsupported_line_weight'] * projector
                continue
            projector = np.eye(3) - np.outer(fit['direction'], fit['direction'])
            matrix += fit['weight'] * projector
            rhs += fit['weight'] * (projector @ (fit['point'] - vertices[i]))
        shift = np.linalg.solve(matrix, rhs)
        if len(edge_ids) == 1:
            direction = fits[edge_ids[0]]['direction']
            shift -= shift @ direction * direction
        limit = min(cfg['max_move_m'], cfg['max_move_edge_fraction'] * min(lengths[edge_ids]))
        magnitude = float(np.linalg.norm(shift))
        if magnitude > limit:
            shift *= limit / magnitude
        shifts[i] = shift
    old_distance = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
    separation = cfg['min_vertex_separation_fraction'] * old_distance
    protected = 0
    for _ in range(30):
        changed = vertices + shifts
        distance = np.linalg.norm(changed[:, None] - changed[None, :], axis=2)
        bad_pairs = np.triu(distance < separation, 1)
        new_lengths = np.linalg.norm(changed[edges[:, 1]] - changed[edges[:, 0]], axis=1)
        bad_edges = new_lengths < cfg['min_edge_length_fraction'] * lengths
        affected = set(np.flatnonzero(bad_pairs.any(0) | bad_pairs.any(1)).tolist())
        affected.update(edges[bad_edges].reshape(-1).tolist())
        if not affected:
            break
        shifts[list(affected)] *= 0.5
        protected += len(affected)
    else:
        shifts[:] = 0
    changed = vertices + shifts
    magnitudes = np.linalg.norm(shifts, axis=1)
    out = original.copy()
    updated = changed[inverse].reshape(-1, 2, 3) + origin
    moved_endpoint = (magnitudes[inverse] > 1e-12).reshape(-1, 2)
    out[moved_endpoint] = updated[moved_endpoint]
    return (out, {'config': cfg, 'input_edges': int(len(original)), 'output_edges': int(len(out)), 'vertices': int(len(vertices)), 'fitted_edges': int(sum((x is not None for x in fits))), 'moved_vertices': int(np.count_nonzero(magnitudes > 1e-12)), 'max_move_m': float(magnitudes.max()), 'mean_move_m': float(magnitudes.mean()), 'topology_protection_halvings': int(protected), 'edge_fit_results': rejects, 'fitted_edge_details': edge_logs, 'ground_truth_used': False, 'edge_points_used': False})

def kernel_73(lines, raw_lines=None, edge_points=None, config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError('Unknown config keys: ' + ', '.join(sorted(unknown)))
        cfg.update(config)
    if cfg['compress_mode'] not in ('all', 'bridge_vertices'):
        raise ValueError('compress_mode must be all or bridge_vertices')
    if cfg['bridge_matching'] not in ('greedy', 'mutual'):
        raise ValueError('bridge_matching must be greedy or mutual')
    lines, initial = base.repair(lines, config={'bridge': False, 'vertex_tolerance_m': cfg['vertex_tolerance_m']})
    meta = {'version': 'geometry_repair', 'config': cfg, 'counts': Counter(initial['counts']), 'input': initial['input'], 'warnings': [], 'log': []}
    raw = None if raw_lines is None else base._segments(raw_lines)
    if raw is not None:
        raw = raw[np.isfinite(raw).all(axis=(1, 2))]
    tree = None
    if edge_points is not None:
        points = np.asarray(edge_points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
    if len(lines) and cfg['overlap_merge']:
        lines = _merge_overlaps(lines, cfg, meta)
    if len(lines) and cfg['interior_weld']:
        lines = base._weld_interiors(lines, cfg, meta)
    if len(lines) and cfg['bridge']:
        lines = _bridge(lines, raw, tree, cfg, meta)
    if len(lines) and cfg['compress_degree2']:
        lines = _compress(lines, cfg, meta)
    meta['output'] = base._stats(lines, cfg['vertex_tolerance_m'])
    meta['counts'] = dict(meta['counts'])
    return (lines, meta)

def kernel_74(lines, raw_lines, edge_points, config=None, scale=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError('Unknown augmentation keys: ' + str(sorted(unknown)))
        cfg.update(config)
    if scale is None or scale <= 0:
        raise ValueError('Positive original point-cloud radius required for parser')
    candidates, parser_meta = candidate_graph(raw_lines, edge_points, cfg, scale)
    out, meta = _augment_from_candidates(lines, raw_lines, candidates, cfg, scale)
    meta['parser'] = parser_meta
    return (out, meta)

def kernel_86(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_87(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_89(base, added, evidence, mode=None):
    base = _lines(base)
    added = _lines(added)
    if mode == 'snap(old)':
        mode = 'snap'
    if mode not in ('snap', 'shared', 'shared_compete', 'anchored'):
        raise ValueError('Unknown topology mode: ' + str(mode))
    meta = dict(mode=mode, base_edges=len(base), input_added_edges=len(added), old_vertices_locked=True, old_edges_removed=0, old_vertices_merged=0, new_shared_nodes=0, old_anchored_nodes=0, endpoints_at_old_nodes=0, distinct_axis_or_span_merges_blocked=0, diameter_merges_blocked=0, max_cluster_original_diameter_m=0.0, moved_added_endpoints=0, removed_added_duplicates=0, compete_base_duplicates=0, compete_added_duplicates=0, max_endpoint_displacement_m=0.0, max_direction_change_deg=0.0, anchored_rejected=0, zero_new_vertices=mode == 'anchored')
    if not len(added):
        return (base, meta)
    vertices = np.unique(base.reshape(-1, 3), axis=0) if len(base) else np.empty((0, 3))
    if mode == 'anchored':
        anchored = []
        kept_original = []
        for e in added:
            result = candidate_anchor(e, base)
            if result is None:
                meta['anchored_rejected'] += 1
            else:
                anchored.append(result)
                kept_original.append(e)
        if not anchored:
            meta['output_added_edges'] = 0
            return (base, meta)
        added = np.asarray(kept_original).reshape(-1, 2, 3)
        organized = np.asarray(anchored).reshape(-1, 2, 3)
        ids = None
        clusters = None
    elif mode == 'snap':
        organized, moved = _snap_old(base, added, vertices)
        meta['moved_added_endpoints'] = moved
        ids = None
        clusters = None
    else:
        organized, ids, clusters = _shared(base, added, vertices, meta)
    keep = np.ones(len(added), dtype=bool)
    if mode == 'shared_compete':
        for i in range(len(added)):
            old_ids = [clusters[k]['old'] if k >= 0 else None for k in ids[i]]
            if all((k is not None for k in old_ids)):
                for e in base:
                    if not (np.max(np.abs(organized[i] - e)) <= 1e-10 or np.max(np.abs(organized[i] - e[::-1])) <= 1e-10):
                        continue
                    if _can_compete(added[i], e, organized[i], e):
                        keep[i] = False
                        meta['compete_base_duplicates'] += 1
                        break
            if not keep[i]:
                continue
            for j in range(i):
                if not keep[j] or set(ids[i]) != set(ids[j]):
                    continue
                if _can_compete(added[i], added[j], organized[i], organized[j]):
                    keep[i] = False
                    meta['compete_added_duplicates'] += 1
                    break
    displacement = np.linalg.norm(organized - added, axis=2)
    meta['moved_added_endpoints'] = int((displacement > 1e-10).sum())
    meta['max_endpoint_displacement_m'] = float(displacement.max(initial=0.0))
    meta['max_direction_change_deg'] = max([_angle(a, b) for a, b in zip(added, organized)] + [0.0])
    meta['removed_added_duplicates'] = int((~keep).sum())
    meta['output_added_edges'] = int(keep.sum())
    if any((not _valid(a, b) and (not np.array_equal(a, b)) for a, b in zip(added, organized))):
        raise AssertionError('Topology proposal exceeded displacement/direction constraints')
    return (np.concatenate([base, organized[keep]], axis=0), meta)

def kernel_90(model, evidence, candidate, stats):
    bundle = _bundle(model)
    c = _array(candidate)
    stats = _stats(stats, len(c))
    if not len(c):
        return np.empty(0)
    X = _FEATURE_MODULE.feature_matrix(_evidence(evidence), c, stats)
    classes = bundle['model'].classes_
    positive = np.flatnonzero(classes == 1)
    if not len(positive):
        return np.zeros(len(c))
    if len(classes) == 1:
        return np.ones(len(c))
    return bundle['model'].predict_proba(X)[:, int(positive[0])]

def kernel_91(edge, current, protect_parallel):
    if not len(current):
        return False
    close = _FEATURE_MODULE.pair_distance(edge[None], current)[0] < 0.35
    if not protect_parallel:
        return bool(close.any())
    others = current[close]
    if not len(others):
        return False
    vec = edge[1] - edge[0]
    length = np.linalg.norm(vec)
    if length <= 1e-12:
        return False
    u = vec / length
    ov = others[:, 1] - others[:, 0]
    ol = np.linalg.norm(ov, axis=1)
    ou = ov / np.maximum(ol[:, None], 1e-12)
    angle = np.abs(ou @ u) >= np.cos(np.deg2rad(2.5))
    other_to_axis = np.linalg.norm(np.cross(others - edge[0], u), axis=2).max(1)
    edge_to_other = np.linalg.norm(np.cross(edge[None] - others[:, None, 0], ou[:, None]), axis=2).max(1)
    same_axis = (other_to_axis <= 0.04) & (edge_to_other <= 0.04) & (ol > 1e-12)
    return bool((angle & same_axis).any())

def kernel_96(lines):
    return c.e.verify.canonical(np.asarray(lines).reshape(-1, 2, 3))

def kernel_99(raw, g):
    vertices = np.unique(g.reshape(-1, 3), axis=0)
    if not len(vertices):
        return (np.empty((0, 2, 3)), dict(raw=0, anchored=0, eligible=0, medoids=0))
    raw = np.asarray(raw)
    D = cdist(raw.reshape(-1, 3), vertices)
    near = D.argmin(1).reshape(-1, 2)
    dist = D.min(1).reshape(-1, 2)
    anchor = dist <= 0.35
    mask = (anchor.sum(1) == 1) & (np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1) > 0.15)
    lines = raw[mask].copy()
    aa = anchor[mask]
    nn = near[mask]
    lines[aa] = vertices[nn[aa]]
    anchored = len(lines)
    lines = c.e.verify.canonical(lines)
    good = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1) >= 0.2
    lines = lines[good]
    if len(lines):
        lines = lines[c.e.OLD.pair_distance(lines, g).min(1) > 0.35]
    eligible = len(lines)
    if not len(lines):
        return (lines, dict(raw=len(raw), anchored=anchored, eligible=eligible, medoids=0))
    votes = (c.e.OLD.pair_distance(lines, raw) <= 0.35).sum(1)
    pair = c.e.OLD.pair_distance(lines, lines)
    selected = []
    for i in np.argsort(-votes, kind='stable'):
        if selected and pair[i, selected].min() < 0.2:
            continue
        selected.append(int(i))
    return (lines[selected], dict(raw=len(raw), anchored=anchored, eligible=eligible, medoids=len(selected)))

def kernel_100(ev, g, native):
    vertices, ix, degree = e.LEGACY.degree(g)
    keys = {tuple(v): i for i, v in enumerate(vertices)}
    leaves = np.flatnonzero(degree == 1)
    raw = ev['raw']
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    unit = (raw[:, 1] - raw[:, 0]) / np.maximum(length[:, None], 1e-12)
    patches = []
    seen = set()
    groups = {}
    counts = dict(native_leaf_pairs=0, nonparallel_pairs=0, old_axes_missing=0, new_axes_missing=0, illconditioned=0, shift_or_length_rejected=0, fit_successes=0, raw_supported_fits=0)
    oldcache = {}
    near_cache = {}

    def axes(reference, leaf, exclude=()):
        L = np.linalg.norm(reference[1] - reference[0])
        u = (reference[1] - reference[0]) / max(L, 1e-12)
        lateral = np.linalg.norm(np.cross(raw - reference[0], u), axis=2).mean(1)
        sel = np.flatnonzero((abs(unit @ u) >= np.cos(np.deg2rad(12))) & (lateral <= 0.45) & (length > 0.3))
        if leaf not in near_cache:
            near_cache[leaf] = c.e.OLD.distances(vertices[leaf:leaf + 1], raw)[0]
        dist = near_cache[leaf]
        sel = sel[dist[sel] <= 0.8]
        if len(exclude):
            sel = sel[~np.isin(sel, exclude)]
        return sel[np.argsort(dist[sel] + lateral[sel], kind='stable')[:4]]

    def add(op, remove, lines, connector, meta):
        lines = d.canonical(np.asarray(lines).reshape(-1, 2, 3))
        signature = (tuple(remove), tuple((d.key(x) for x in lines)))
        if signature in seen:
            return
        patch = dict(family=op, remove=remove, add=lines.tolist(), connector=d.canonical(np.asarray(connector)[None])[0].tolist(), meta=meta)
        out = d.apply(g, patch)
        if len(out) != len(g) - len(remove) + len(lines) or np.array_equal(out, g):
            return
        seen.add(signature)
        patches.append(patch)
    for ni, line in enumerate(native):
        ends = [j for j in range(2) if tuple(line[j]) in keys]
        assert len(ends) == 1
        k = ends[0]
        anchor = keys[tuple(line[k])]
        point = line[1 - k]
        u = (point - vertices[anchor]) / np.linalg.norm(point - vertices[anchor])
        for leaf in leaves[np.linalg.norm(vertices[leaves] - point, axis=1) <= 1.0]:
            if leaf == anchor:
                continue
            edge = int(np.flatnonzero((ix == leaf).any(1))[0])
            other = int(ix[edge, 0] if ix[edge, 1] == leaf else ix[edge, 1])
            counts['native_leaf_pairs'] += 1
            if other == anchor:
                continue
            v = (vertices[leaf] - vertices[other]) / np.linalg.norm(vertices[leaf] - vertices[other])
            if abs(v @ u) >= np.cos(np.deg2rad(20)):
                continue
            counts['nonparallel_pairs'] += 1
            if int(leaf) not in oldcache:
                oldcache[int(leaf)] = axes(g[edge], int(leaf))
            old = oldcache[int(leaf)]
            if len(old) < 2:
                counts['old_axes_missing'] += 1
                continue
            new = axes(line, int(leaf), old)
            if len(new) < 2:
                counts['new_axes_missing'] += 1
                continue
            rr = np.r_[old, new]
            A = sum((np.eye(3) - np.outer(unit[j], unit[j]) for j in rr))
            b = sum(((np.eye(3) - np.outer(unit[j], unit[j])) @ raw[j].mean(0) for j in rr))
            if np.linalg.cond(A) > 10000.0:
                counts['illconditioned'] += 1
                continue
            Q = np.linalg.solve(A + 0.05 * np.eye(3), b + 0.05 * vertices[leaf])
            shift = float(np.linalg.norm(Q - vertices[leaf]))
            updated = np.array([vertices[other], Q])
            branch = np.array([vertices[anchor], Q])
            counts['fit_successes'] += 1
            if not 0.12 <= shift <= 0.9 or min(np.linalg.norm(updated[1] - updated[0]), np.linalg.norm(branch[1] - branch[0])) < 0.2:
                counts['shift_or_length_rejected'] += 1
                continue
            sc = [d.support(ev, z) for z in [updated, branch]]
            if min(sc) < 0.7:
                continue
            counts['raw_supported_fits'] += 1
            res = np.linalg.norm(np.cross(Q - raw[rr].mean(1), unit[rr]), axis=1)
            samples = g[edge, 0] + np.linspace(0, 1, 21)[:, None] * (g[edge, 1] - g[edge, 0])
            aligned = (abs(unit @ v) >= np.cos(np.deg2rad(12))) & (length > 0.15)
            witness = c.e.OLD.distances(samples, raw[aligned]).min(1) <= 0.25 if aligned.any() else np.zeros(21, bool)
            distance = c.e.OLD.distances(samples, updated[None])[:, 0]
            retained = int((distance[witness] <= 0.25).sum())
            meta = dict(target=int(leaf), degree=1, anchor=int(anchor), native_index=ni, old_geometry=False, old_axes=old.tolist(), new_axes=new.tolist(), shift=shift, fit_median_residual=float(np.median(res)), support_min=min(sc), support=sc, old_witnesses=int(witness.sum()), retained_old_witnesses=retained, old_witness_max_distance=float(distance[witness].max()) if witness.any() else 0.0)
            add('move_only', [edge], [updated], updated, meta)
            add('move_connect', [edge], [updated, branch], branch, meta)
            existing = d.canonical(np.array([vertices[anchor], vertices[leaf]])[None])[0]
            if d.support(ev, existing) >= 0.7:
                add('connect_old', [], [existing], existing, meta)
    counts['operations'] = {op: sum((p['family'] == op for p in patches)) for op in OPS}
    return (patches, counts)

def kernel_101(g, patches):
    vertices = np.unique(g.reshape(-1, 3), axis=0)
    present = {d.key(x) for x in g}
    lines = []
    for patch in patches:
        if patch['family'] != 'move_connect':
            continue
        meta = patch['meta']
        line = np.array([vertices[meta['anchor']], vertices[meta['target']]])
        if np.linalg.norm(line[1] - line[0]) >= 0.2:
            lines.append(line)
    lines = d.canonical(np.array(lines).reshape(-1, 2, 3))
    return np.array([x for x in lines if d.key(x) not in present]).reshape(-1, 2, 3)

def kernel_102(line):
    return tuple(np.asarray(line).reshape(-1))

def kernel_104(ev, graph, patches, legacy, finite, roof):
    base = np.c_[legacy.feature_matrix(ev, graph, patches), finite.features(ev, graph, patches, legacy)] if patches else np.empty((0, len(legacy.FEATURES) + len(finite.FEATURES)))
    if not patches:
        return np.empty((0, base.shape[1] + 54 + len(FAMILIES)))
    all_lines = np.concatenate([graph] + [np.asarray(p['add']) for p in patches])
    unique = legacy.e.verify.canonical(all_lines)
    X, _ = roof.roof_features(ev, unique)
    cache = {key(x): f for x, f in zip(unique, X)}
    rows = []
    for patch in patches:
        added = np.array([cache[key(x)] for x in patch['add']])
        removed = np.array([cache[key(graph[i])] for i in patch['remove']]) if patch['remove'] else np.zeros((1, 18))
        rows.append(np.r_[added.mean(0), added.max(0), removed.mean(0), [float(patch['family'] == f) for f in FAMILIES]])
    return np.c_[base, rows]

def kernel_114(ev, g, patches):
    raw = ev['raw']
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    vertices, ix, degree = e.LEGACY.degree(g)
    result = []
    for patch in patches:
        if not patch['remove']:
            result.append(np.zeros(12))
            continue
        old = g[patch['remove']]
        new = np.array(patch['add'])
        keep = np.ones(len(g), bool)
        keep[patch['remove']] = False
        other = g[keep]
        survive = np.r_[other.reshape(-1, 3), new.reshape(-1, 3)]
        lost = [x for x in np.unique(old.reshape(-1, 3), axis=0) if not (np.linalg.norm(survive - x, axis=1) < 1e-08).any()]
        if not lost:
            result.append(np.zeros(12))
            continue
        ref = new[np.linalg.norm(new[:, 1] - new[:, 0], axis=1).argmax()]
        u = ref[1] - ref[0]
        u /= max(np.linalg.norm(u), 1e-12)
        ou = (other[:, 1] - other[:, 0]) / np.maximum(np.linalg.norm(other[:, 1] - other[:, 0], axis=1)[:, None], 1e-12)
        branches = other[abs(ou @ u) < np.cos(np.radians(20))]
        rows = []
        for point in lost:
            aligned = (abs(ru @ u) > np.cos(np.radians(12))) & (np.linalg.norm(np.cross(raw - point, u), axis=2).max(1) <= 0.35)
            pr = (raw - point) @ u
            lo = pr.min(1)
            hi = pr.max(1)
            ends = aligned & (np.minimum(abs(lo), abs(hi)) <= 0.25)
            cross = aligned & (lo < -0.35) & (hi > 0.35)
            left = aligned & (abs(hi) <= 0.25) & (lo < -0.3)
            right = aligned & (abs(lo) <= 0.25) & (hi > 0.3)
            rd = c.e.OLD.distances(point[None], raw)[0]
            branch = (abs(ru @ u) < np.cos(np.radians(20))) & (rd <= 0.35) & (rl > 0.3)
            gd = c.e.OLD.distances(point[None], branches)
            rawend = c.e.OLD.nearest(point[None], raw.reshape(-1, 3))[0]
            ep = c.e.OLD.nearest(point[None], ev['ep'])[0]
            di = int(degree[np.linalg.norm(vertices - point, axis=1).argmin()])
            rows.append([min(5.0, float(rawend)), np.log1p(aligned.sum()), np.log1p(ends.sum()), np.log1p(cross.sum()), np.log1p(min(left.sum(), right.sum())), ends.sum() / (ends.sum() + cross.sum() + 1), np.log1p(branch.sum()), min(5.0, float(gd.min())) if gd.size else 5.0, min(5.0, float(ep)), np.log1p(np.median(rl[branch])) if branch.any() else 0.0, di])
        a = np.array(rows)
        agg = a.max(0)
        agg[[0, 7, 8]] = a[:, [0, 7, 8]].min(0)
        result.append(np.r_[len(lost), agg])
    return np.array(result).reshape(-1, 12)

def kernel_115(bundle, evidence, current_residual_graph):
    g = c.e.verify.canonical(current_residual_graph)
    pairs, keep, cover, votes, distance = v.pair_candidates(evidence, g)
    lines = pairs[keep]
    patches = [dict(family='diverse_add', remove=[], add=[line.tolist()]) for line in lines]
    X = np.c_[e.alg.features(evidence, g, patches, e.LEGACY, e.FINITE, e.roof), v.j.r.removed_features(evidence, g, patches)]
    p = bundle['pair']['model'].predict_proba(X)[:, 1] if len(X) else np.array([])
    selected = [int(k) for k in np.argsort(-p, kind='stable') if p[k] > 0.55][:1]
    out = c.e.verify.canonical(np.concatenate([g, lines[selected]])) if selected else g.copy()
    assert np.array_equal(np.unique(out.reshape(-1, 3), axis=0), np.unique(g.reshape(-1, 3), axis=0))
    assert {tuple(line.reshape(-1)) for line in g} <= {tuple(line.reshape(-1)) for line in out}
    return (out, dict(GT_used=False, threshold=0.55, max_add=1, candidates=len(lines), selected=selected, probability=p[selected].tolist(), vertices_exactly_preserved=True, literal_edges_exactly_preserved=True))

def kernel_117(bundle, evidence, actor_predelete):
    actor = api.c.e.verify.canonical(actor_predelete)
    if not len(actor):
        out, meta = api.e.API.decode(bundle['base'], evidence, actor)
        return (out, dict(GT_used=False, empty_actor=True, original_empty_behavior=meta))
    return api.decode(bundle, evidence, actor)

def kernel_129(points, lines):
    if not len(lines):
        return np.full((len(points), 0), 1000000.0)
    d = lines[:, 1] - lines[:, 0]
    t = np.clip(np.einsum('nmk,mk->nm', points[:, None] - lines[None, :, 0], d) / np.maximum((d * d).sum(1), 1e-20), 0, 1)
    return np.linalg.norm(points[:, None] - (lines[None, :, 0] + t[:, :, None] * d[None]), axis=-1)

def kernel_130(value):
    if value is None:
        return np.empty((0, 2, 3), dtype=float)
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        return np.empty((0, 2, 3), dtype=float)
    if a.ndim != 3 or a.shape[1:] != (2, 3) or (not np.isfinite(a).all()):
        raise ValueError('Expected finite (N,2,3) metre-coordinate segments')
    return a.copy()

def kernel_131(line, raw, cfg):
    a, b = line
    length = float(np.linalg.norm(b - a))
    u = (b - a) / length
    center = (a + b) / 2
    if not len(raw):
        return (None, 'no_raw')
    delta = raw[:, 1] - raw[:, 0]
    raw_length = np.linalg.norm(delta, axis=1)
    t = (raw - a) @ u
    denominator = t[:, 1] - t[:, 0]
    lo = np.maximum(np.minimum(t[:, 0], t[:, 1]), 0)
    hi = np.minimum(np.maximum(t[:, 0], t[:, 1]), length)
    overlap = hi - lo
    mask = (raw_length > 1e-08) & (np.abs(denominator) >= raw_length * np.cos(np.deg2rad(cfg['raw_angle_deg']))) & (overlap >= min(cfg['min_overlap_m'], 0.5 * length)) & (overlap >= cfg['min_overlap_fraction'] * np.minimum(length, np.abs(denominator)))
    idx = np.flatnonzero(mask)
    if not len(idx):
        return (None, 'no_direction_overlap')
    q0 = raw[idx, 0] + ((lo[idx] - t[idx, 0]) / denominator[idx])[:, None] * delta[idx]
    q1 = raw[idx, 0] + ((hi[idx] - t[idx, 0]) / denominator[idx])[:, None] * delta[idx]
    clipped = np.stack((q0, q1), axis=1)
    relative = clipped - center
    radial = relative - (relative @ u)[:, :, None] * u
    radius = np.linalg.norm(radial, axis=2).max(axis=1)
    inside = radius <= cfg['raw_radius_m']
    clipped, idx, radius = (clipped[inside], idx[inside], radius[inside])
    if len(clipped) < cfg['min_support_lines']:
        return (None, 'too_few_tube_supports')
    keys = np.round((clipped - center).reshape(-1, 6) / cfg['raw_dedup_m']).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    clipped, idx, radius = (clipped[first], idx[first], radius[first])
    if len(clipped) < cfg['min_support_lines']:
        return (None, 'too_few_independent_supports')
    order = np.lexsort((idx, radius))[:int(cfg['max_support_lines'])]
    clipped, idx = (clipped[order], idx[order])
    coverage = _coverage(list(zip(lo[idx], hi[idx])), length)
    if coverage < cfg['min_axial_coverage']:
        return (None, 'low_axial_coverage')
    samples = np.stack((clipped[:, 0], clipped.mean(axis=1), clipped[:, 1]), axis=1)
    weights = np.ones(len(clipped))
    direction = u.copy()
    fitted_center = np.median(clipped.mean(axis=1), axis=0)
    for _ in range(int(cfg['robust_iterations'])):
        offset = samples - fitted_center
        perp = offset - (offset @ direction)[:, :, None] * direction
        residual = np.sqrt(np.mean(np.sum(perp * perp, axis=2), axis=1))
        median = float(np.median(residual))
        scale = max(cfg['robust_floor_m'], 1.4826 * float(np.median(np.abs(residual - median))))
        weights = np.minimum(1.0, (median + 1.5 * scale) / np.maximum(residual, 1e-12))
        point_weights = np.repeat(weights / 3, 3)
        points = samples.reshape(-1, 3)
        fitted_center = np.average(points, axis=0, weights=point_weights)
        offset = points - fitted_center
        covariance = (offset * point_weights[:, None]).T @ offset
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        if direction @ u < 0:
            direction = -direction
    if direction @ u < np.cos(np.deg2rad(cfg['fit_max_angle_deg'])):
        return (None, 'fit_direction_changed')
    offset = samples - fitted_center
    perp = offset - (offset @ direction)[:, :, None] * direction
    rms = float(np.sqrt(np.average(np.mean(np.sum(perp * perp, axis=2), axis=1), weights=weights)))
    quality = float(coverage / (1.0 + (rms / max(cfg['robust_floor_m'], 0.05)) ** 2))
    return ({'point': fitted_center, 'direction': direction, 'weight': quality, 'support_lines': int(len(clipped)), 'coverage': coverage, 'rms_m': rms}, 'ok')

def kernel_135(lines, raw, tree, cfg, meta):
    stage = {'counts': Counter(), 'log': [], 'warnings': []}
    candidate_output = base._bridge(lines, raw, tree, cfg, stage)
    if not cfg['bridge_reject_crossings']:
        meta['counts'].update(stage['counts'])
        meta['log'].extend(stage['log'])
        meta['bridge_rejections'] = stage['bridge_rejections']
        return candidate_output
    output = list(lines)
    rejected = Counter(stage['bridge_rejections'])
    accepted = 0
    for entry in stage['log']:
        a, b = (np.asarray(entry['endpoint_i']), np.asarray(entry['endpoint_j']))
        conflict = _crossing(a, b, np.asarray(output).reshape(-1, 2, 3), cfg)
        if conflict is not None:
            rejected['third_edge_crossing'] += 1
            meta.setdefault('rejected_bridge_log', []).append(dict(entry, conflict=conflict))
            continue
        output.append(np.array([a, b]))
        meta['log'].append(entry)
        accepted += 1
    meta['counts'].update(stage['counts'])
    meta['counts']['bridges_added'] = accepted
    meta['bridge_rejections'] = dict(rejected)
    return base._canonical(np.asarray(output).reshape(-1, 2, 3), cfg['vertex_tolerance_m'])

def kernel_136(lines, cfg, meta):
    vertices, edges, adjacency = base._graph(lines, cfg['vertex_tolerance_m'])
    supports = {tuple(edge): vertices[edge].copy() for edge in edges}
    bridge_vertices = [np.asarray(row[key]) for row in meta['log'] if row['op'] == 'bridge' for key in ['endpoint_i', 'endpoint_j']]
    allowed = set()
    if cfg['compress_mode'] == 'bridge_vertices':
        for point in bridge_vertices:
            closest = int(np.argmin(np.linalg.norm(vertices - point, axis=1)))
            if np.linalg.norm(vertices[closest] - point) <= cfg['vertex_tolerance_m']:
                allowed.add(closest)
    protected = set()
    if cfg['compress_protect_branch_m'] > 0:
        for i, neighbors in enumerate(adjacency):
            if len(neighbors) != 2:
                continue
            a, b = sorted(neighbors)
            chord = vertices[b] - vertices[a]
            chord /= np.linalg.norm(chord)
            for c, d in edges:
                if i in (c, d) or c in neighbors or d in neighbors:
                    continue
                other = vertices[d] - vertices[c]
                length = np.linalg.norm(other)
                if base._angle(abs(chord @ (other / length))) < cfg['compress_protect_branch_angle_deg']:
                    continue
                t = np.clip((vertices[i] - vertices[c]) @ other / (length * length), 0, 1)
                distance = np.linalg.norm(vertices[i] - vertices[c] - t * other)
                if distance <= cfg['compress_protect_branch_m']:
                    protected.add(i)
                    break
    while True:
        chosen = None
        for i, neighbors in enumerate(adjacency):
            if len(neighbors) != 2 or i in protected:
                continue
            if cfg['compress_mode'] == 'bridge_vertices' and i not in allowed:
                continue
            a, b = sorted(neighbors)
            if b in adjacency[a]:
                continue
            va, vb = (vertices[a] - vertices[i], vertices[b] - vertices[i])
            angle = base._angle(-va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
            if angle > cfg['compress_angle_deg']:
                continue
            key1, key2 = (tuple(sorted((a, i))), tuple(sorted((i, b))))
            points = np.concatenate([supports[key1], supports[key2]])
            chord = vertices[b] - vertices[a]
            length = np.linalg.norm(chord)
            if length <= cfg['vertex_tolerance_m']:
                continue
            t, radius = base._radial(points, vertices[a], chord / length)
            if radius.max() > cfg['compress_lateral_m'] or t.min() < -1e-07 or t.max() > length + 1e-07:
                continue
            chosen = (i, a, b, key1, key2, points, angle)
            break
        if chosen is None:
            break
        i, a, b, key1, key2, points, angle = chosen
        adjacency[a].remove(i)
        adjacency[b].remove(i)
        adjacency[i].clear()
        adjacency[a].add(b)
        adjacency[b].add(a)
        del supports[key1], supports[key2]
        supports[a, b] = points
        meta['counts']['degree2_vertices_compressed'] += 1
        meta['log'].append({'op': 'compress_degree2', 'vertex': vertices[i].tolist(), 'angle_deg': angle})
    meta['counts']['compression_protected_vertices'] = len(protected)
    return base._canonical(np.asarray([vertices[list(e)] for e in sorted(supports)]).reshape(-1, 2, 3), cfg['vertex_tolerance_m'])

def kernel_138(lines, cfg, meta):
    vertices, edges, adjacency = base._graph(lines, cfg['vertex_tolerance_m'])
    lines = vertices[edges]
    delta = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    unit = delta / lengths[:, None]
    parent = np.arange(len(lines))
    cosine = np.cos(np.radians(cfg['overlap_angle_deg']))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if abs(unit[i] @ unit[j]) < cosine:
                continue
            t, radius = base._radial(lines[j], lines[i, 0], unit[i])
            if radius.max() > cfg['overlap_lateral_m']:
                continue
            overlap = min(lengths[i], t.max()) - max(0.0, t.min())
            if overlap < max(cfg['overlap_min_m'], cfg['overlap_min_fraction'] * min(lengths[i], lengths[j])):
                continue
            parent[find(j)] = find(i)
    groups = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    output = []
    for members in groups.values():
        if len(members) == 1:
            output.append(lines[members[0]])
            continue
        longest = max(members, key=lambda i: lengths[i])
        origin, direction = (lines[longest, 0], unit[longest])
        ids = np.unique(edges[members])
        position, radius = base._radial(vertices[ids], origin, direction)
        if radius.max() > cfg['overlap_lateral_m'] or any((abs(unit[i] @ direction) < cosine for i in members)):
            output.extend(lines[members])
            meta['counts']['overlap_ambiguous_groups_skipped'] += 1
            continue
        member_set = set(members)
        outside = {int(i) for k, edge in enumerate(edges) if k not in member_set for i in edge}
        keep = {int(ids[position.argmin()]), int(ids[position.argmax()])}
        keep.update((int(i) for i in ids if int(i) in outside))
        ordered = sorted(keep, key=lambda i: float((vertices[i] - origin) @ direction))
        if cfg['overlap_require_reduction'] and len(ordered) - 1 >= len(members):
            output.extend(lines[members])
            meta['counts']['overlap_no_reduction_skipped'] += 1
            continue
        output.extend((np.array([vertices[a], vertices[b]]) for a, b in zip(ordered[:-1], ordered[1:]) if np.linalg.norm(vertices[b] - vertices[a]) > cfg['vertex_tolerance_m']))
        meta['counts']['overlap_groups_merged'] += 1
        meta['counts']['overlap_lines_removed'] += len(members) - (len(ordered) - 1)
        meta['log'].append({'op': 'overlap_merge', 'input_segments': len(members), 'output_segments': len(ordered) - 1, 'retained_junctions': len(ordered) - 2, 'input_coordinates': lines[members].tolist()})
    return base._canonical(np.asarray(output).reshape(-1, 2, 3), cfg['vertex_tolerance_m'])

def kernel_140(lines, raw_lines=None, edge_points=None, config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError('Unknown repair config keys: ' + ', '.join(sorted(unknown)))
        cfg.update(config)
    if cfg['bridge_matching'] not in ('greedy', 'mutual'):
        raise ValueError('bridge_matching must be greedy or mutual.')
    supplied = _segments(lines)
    finite = np.isfinite(supplied).all(axis=(1, 2))
    nonzero = np.linalg.norm(supplied[:, 1] - supplied[:, 0], axis=1) > max(1e-10, cfg['vertex_tolerance_m'])
    lines = _canonical(supplied[finite & nonzero], cfg['vertex_tolerance_m'])
    meta = {'config': cfg, 'counts': Counter(), 'log': [], 'warnings': [], 'input': _stats(lines, cfg['vertex_tolerance_m'])}
    meta['counts']['input_rows'] = len(supplied)
    meta['counts']['invalid_or_zero_rows_removed'] = int((~(finite & nonzero)).sum())
    meta['counts']['duplicate_rows_removed'] = int((finite & nonzero).sum()) - len(lines)
    raw = None if raw_lines is None else _segments(raw_lines)
    if raw is not None:
        raw = raw[np.isfinite(raw).all(axis=(1, 2))]
    point_tree = None
    if edge_points is not None:
        points = np.asarray(edge_points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            from scipy.spatial import cKDTree
            point_tree = cKDTree(points)
    if cfg['overlap_merge'] and len(lines):
        lines = _merge_overlaps(lines, cfg, meta)
    if cfg['interior_weld'] and len(lines):
        lines = _weld_interiors(lines, cfg, meta)
    if cfg['bridge'] and len(lines):
        lines = _bridge(lines, raw, point_tree, cfg, meta)
    if cfg['compress_degree2'] and len(lines):
        lines = _compress(lines, cfg, meta)
    meta['output'] = _stats(lines, cfg['vertex_tolerance_m'])
    for key in ('overlap_groups_merged', 'overlap_lines_removed', 'overlap_ambiguous_groups_skipped', 'interior_welds', 'bridge_candidates', 'bridges_added', 'degree2_vertices_compressed'):
        meta['counts'].setdefault(key, 0)
    meta['counts'] = dict(meta['counts'])
    return (lines, meta)

def kernel_141(lines, tolerance):
    vertices, edges, adjacency = _graph(lines, tolerance)
    unseen = set(range(len(vertices)))
    components = 0
    while unseen:
        stack = [unseen.pop()]
        components += 1
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    degrees = np.array([len(a) for a in adjacency])
    return {'lines': len(edges), 'vertices': len(vertices), 'degree1': int((degrees == 1).sum()), 'degree2': int((degrees == 2).sum()), 'degree3plus': int((degrees >= 3).sum()), 'components': components}

def kernel_142(value):
    if value is None:
        return np.empty((0, 2, 3), dtype=float)
    result = np.asarray(value, dtype=float)
    if result.size == 0:
        return np.empty((0, 2, 3), dtype=float)
    if result.ndim != 3 or result.shape[1:] != (2, 3):
        raise ValueError('Segments must have shape (N, 2, 3).')
    return result.copy()

def kernel_143(lines, cfg, meta):
    for _ in range(int(cfg['weld_max_edits'])):
        vertices, edges, adjacency = _graph(lines, cfg['vertex_tolerance_m'])
        best = None
        for i in [j for j, neighbors in enumerate(adjacency) if len(neighbors) == 1]:
            neighbor = next(iter(adjacency[i]))
            tangent = vertices[i] - vertices[neighbor]
            length = np.linalg.norm(tangent)
            tangent /= length
            for edge_id, (a, b) in enumerate(edges):
                if i in (a, b) or neighbor in (a, b):
                    continue
                direction = vertices[b] - vertices[a]
                target_length = np.linalg.norm(direction)
                direction /= target_length
                t = float((vertices[i] - vertices[a]) @ direction)
                if not cfg['weld_interior_margin_m'] < t < target_length - cfg['weld_interior_margin_m']:
                    continue
                cross_angle = _angle(abs(tangent @ direction))
                if cross_angle < cfg['weld_min_cross_angle_deg']:
                    continue
                q = vertices[a] + t * direction
                distance = float(np.linalg.norm(q - vertices[i]))
                if distance > min(cfg['weld_distance_m'], cfg['weld_max_relative_move'] * length):
                    continue
                item = (distance, i, edge_id, q)
                if best is None or item[:3] < best[:3]:
                    best = item
        if best is None:
            break
        distance, i, edge_id, q = best
        p = vertices[i].copy()
        vertices[i] = q
        a, b = edges[edge_id]
        output = [vertices[edge] for k, edge in enumerate(edges) if k != edge_id]
        output.extend([np.array([vertices[a], q]), np.array([q, vertices[b]])])
        lines = _canonical(np.asarray(output), cfg['vertex_tolerance_m'])
        meta['counts']['interior_welds'] += 1
        meta['log'].append({'op': 'interior_weld', 'endpoint': p.tolist(), 'target': q.tolist(), 'move_m': distance})
    else:
        meta['warnings'].append('weld_max_edits reached; some proposals may remain.')
    return lines

def kernel_145(lines, raw_lines, candidates, cfg, scale):
    original, raw, candidates = (_lines(lines), _lines(raw_lines), _lines(candidates))
    if not len(original):
        return (original, {'added_edges': 0, 'reason': 'empty_existing_graph_no_anchors', 'ground_truth_used': False})
    existing_vertices = np.unique(original.reshape(-1, 3), axis=0)
    original_keys = {tuple(v) for v in existing_vertices}
    depth = {tuple(v): 0 for v in existing_vertices}
    degree, directions = (Counter(), {})
    for a, b in original:
        for p, q in ((a, b), (b, a)):
            key = tuple(p)
            degree[key] += 1
            directions.setdefault(key, []).append(q - p)
    rejected, proposals = (Counter(), [])
    vote_radius = cfg['raw_vote_radius_normalized'] * scale
    for candidate_id, line in enumerate(candidates):
        length = float(np.linalg.norm(line[1] - line[0]))
        if length < cfg['min_new_length_m']:
            rejected['short_candidate'] += 1
            continue
        covered, _ = _parallel_intervals(line, original, cfg['existing_coverage_radius_m'], cfg['existing_coverage_angle_deg'])
        covered_length = sum((hi - lo for lo, hi in covered))
        if 1 - covered_length / length < cfg['min_new_fraction']:
            rejected['already_covered'] += 1
            continue
        votes = _raw_votes(line, raw, vote_radius, cfg)
        if votes < cfg['raw_min_votes']:
            rejected['raw_votes'] += 1
            continue
        cursor, uncovered = (0.0, [])
        for lo, hi in covered:
            if lo > cursor:
                uncovered.append([cursor, lo])
            cursor = max(cursor, hi)
        if cursor < length:
            uncovered.append([cursor, length])
        for lo, hi in uncovered:
            if hi - lo < cfg['min_new_length_m']:
                continue
            q = np.array([line[0] if lo == 0 else line[0] + lo / length * (line[1] - line[0]), line[1] if hi == length else line[0] + hi / length * (line[1] - line[0])])
            coverage, _ = _parallel_intervals(q, raw, cfg['raw_radius_m'], cfg['raw_angle_deg'])
            raw_coverage = sum((y - x for x, y in coverage)) / (hi - lo)
            if raw_coverage < cfg['raw_min_coverage']:
                rejected['raw_coverage'] += 1
                continue
            proposals.append({'line': q, 'candidate_id': candidate_id, 'votes': votes, 'raw_coverage': raw_coverage, 'new_length_m': hi - lo})
    proposals.sort(key=lambda x: (-min(x['votes'], 8), -x['raw_coverage'], -x['new_length_m'], x['candidate_id']))
    added, operations, pending = ([], [], proposals)
    additions_per_vertex = Counter()
    budget = min(int(cfg['max_new_edges']), max(1, int(np.ceil(len(original) * cfg['max_new_edge_ratio']))))
    for hop in range(int(cfg['max_growth_hops'])):
        deferred = []
        for proposal in pending:
            if len(added) >= budget:
                rejected['scene_budget'] += 1
                continue
            source = proposal['line']
            q = source.copy()
            new_vertices = np.array([key for key in depth if key not in original_keys]).reshape(-1, 3)
            for k in range(2):
                dist = np.linalg.norm(existing_vertices - q[k], axis=1)
                nearest = int(dist.argmin())
                if dist[nearest] <= cfg['endpoint_snap_m']:
                    q[k] = existing_vertices[nearest]
                elif len(new_vertices):
                    dist = np.linalg.norm(new_vertices - q[k], axis=1)
                    nearest = int(dist.argmin())
                    if dist[nearest] <= cfg['new_node_snap_m']:
                        q[k] = new_vertices[nearest]
            keys = [tuple(p) for p in q]
            old_anchors = sum((key in original_keys for key in keys))
            node_depth = [depth.get(key, 999) for key in keys]
            if old_anchors < cfg['min_original_anchors'] or min(node_depth) >= cfg['max_growth_hops']:
                deferred.append(proposal)
                continue
            new_length = float(np.linalg.norm(q[1] - q[0]))
            if new_length < cfg['min_new_length_m'] or _angle(q[1] - q[0], source[1] - source[0]) > cfg['max_snap_turn_deg']:
                rejected['snap_geometry'] += 1
                continue
            current = np.concatenate([original, np.asarray(added).reshape(-1, 2, 3)])
            intervals, _ = _parallel_intervals(q, current, cfg['existing_coverage_radius_m'], cfg['existing_coverage_angle_deg'])
            if sum((b - a for a, b in intervals)) / new_length > 1 - cfg['min_new_fraction']:
                rejected['became_covered'] += 1
                continue
            coverage, _ = _parallel_intervals(q, raw, cfg['raw_radius_m'], cfg['raw_angle_deg'])
            raw_coverage = sum((b - a for a, b in coverage)) / new_length
            if raw_coverage < cfg['raw_min_coverage']:
                rejected['snapped_raw_coverage'] += 1
                continue
            if any((degree[key] >= cfg['max_degree'] or additions_per_vertex[key] >= cfg['max_additions_per_vertex'] for key in keys)):
                rejected['vertex_capacity'] += 1
                continue
            conflict = any((any((_angle(q[1 - k] - q[k], old) < cfg['direction_conflict_deg'] for old in directions.get(keys[k], []))) for k in range(2)))
            if conflict:
                rejected['direction_conflict'] += 1
                continue
            if _crosses(q, current, cfg['crossing_distance_m']):
                rejected['crossing_or_unshared_T'] += 1
                continue
            added.append(q)
            for k, key in enumerate(keys):
                depth[key] = min(depth.get(key, 999), min(node_depth) + 1)
                degree[key] += 1
                additions_per_vertex[key] += 1
                directions.setdefault(key, []).append(q[1 - k] - q[k])
            operations.append({'candidate_id': proposal['candidate_id'], 'line': q.tolist(), 'raw_votes': proposal['votes'], 'raw_coverage': raw_coverage, 'length_m': new_length, 'original_anchors': old_anchors, 'max_endpoint_depth': max((depth[key] for key in keys))})
        pending = deferred
    rejected['unanchored_or_path_depth'] += len(pending)
    out = np.concatenate([original, np.asarray(added).reshape(-1, 2, 3)])
    assert np.array_equal(out[:len(original)], original), 'Existing graph was modified'
    return (out, {'config': cfg, 'input_edges': len(original), 'parser_candidates': len(candidates), 'eligible_proposals': len(proposals), 'added_edges': len(added), 'added_length_m': float(sum((op['length_m'] for op in operations))), 'rejected': dict(rejected), 'operations': operations, 'existing_graph_unchanged': True, 'ground_truth_used': False})

def kernel_146(raw_lines, edge_points, cfg, scale):
    candidate, meta = parse_lines(raw_lines, edge_points, config={'cluster_eps': cfg['parser_cluster_eps'], 'attach_eps': cfg['parser_attach_eps'], 'max_snap': cfg['parser_max_snap'], 'symmetric_knn': True}, scale=scale)
    if cfg['refine_candidates'] and len(candidate):
        pass
        candidate, refinement = refine_vertices(candidate, raw_lines, edge_points, {'raw_radius_m': cfg['candidate_refine_radius_m']})
        meta['candidate_refinement'] = refinement
    return (candidate, meta)

def kernel_556(base, added, vertices):
    out = added.copy()
    moved = 0
    if not len(vertices):
        return (out, moved)
    for i, e in enumerate(added):
        limit = min(MAX_MOVE, 0.1 * np.linalg.norm(e[1] - e[0]))
        d = np.linalg.norm(e[:, None, :] - vertices[None, :, :], axis=2)
        k = d.argmin(axis=1)
        near = d[np.arange(2), k] <= limit
        proposal = e.copy()
        proposal[near] = vertices[k[near]]
        if _valid(e, proposal):
            out[i] = proposal
            moved += int((np.linalg.norm(proposal - e, axis=1) > 1e-10).sum())
    return (out, moved)

def kernel_557(original, candidate):
    length = float(np.linalg.norm(original[1] - original[0]))
    if length <= 1e-12:
        return False
    bound = min(MAX_MOVE, 0.1 * length)
    if np.linalg.norm(candidate - original, axis=1).max() > bound + 1e-10:
        return False
    newlen = float(np.linalg.norm(candidate[1] - candidate[0]))
    if newlen < min(0.15, 0.5 * length):
        return False
    c = float((candidate[1] - candidate[0]) @ (original[1] - original[0])) / (newlen * length)
    return c >= np.cos(np.radians(MAX_ANGLE_DEG)) - 1e-12

def kernel_558(candidate, base):
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (2, 3) or not np.isfinite(candidate).all():
        raise ValueError('Expected one finite 2x3 candidate')
    base = _lines(base)
    if not len(base):
        return None
    vertices = np.unique(base.reshape(-1, 3), axis=0)
    length = float(np.linalg.norm(candidate[1] - candidate[0]))
    limit = min(MAX_MOVE, 0.1 * length)
    distance = np.linalg.norm(candidate[:, None, :] - vertices[None, :, :], axis=2)
    left = np.where(distance[0] <= limit + 1e-10)[0]
    right = np.where(distance[1] <= limit + 1e-10)[0]
    choices = []
    for a in left:
        for b in right:
            if a == b:
                continue
            edge = vertices[[a, b]]
            if np.linalg.norm(edge[1] - edge[0]) <= 0.15 or not _valid(candidate, edge):
                continue
            choices.append((float(distance[0, a] + distance[1, b]), int(a), int(b)))
    if not choices:
        return None
    _, a, b = min(choices)
    return vertices[[a, b]].copy()

def kernel_559(base, added, vertices, meta):
    n = len(added)
    original = added.copy()
    out = added.copy()
    points = original.reshape(-1, 3)
    limits = np.minimum(MAX_MOVE, 0.1 * np.linalg.norm(original[:, 1] - original[:, 0], axis=1))
    old_incidence = [[] for _ in vertices]
    for e in base:
        for endpoint in e:
            matches = np.where(np.linalg.norm(vertices - endpoint, axis=1) <= 1e-10)[0]
            for node in matches:
                old_incidence[node].append(e)
    blockers = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i):
            blockers[i, j] = blockers[j, i] = _parallel_distinct(original[i], original[j]) or _distinct_span(original[i], original[j])
    clusters = []
    old_cluster = {}
    node_ids = np.full(2 * n, -1, dtype=int)

    def feasible(members, center):
        if max((np.linalg.norm(points[p] - center) for p in members)) > MAX_MOVE + 1e-10:
            return False
        touched = {p // 2 for p in members}
        for i in touched:
            cand = out[i].copy()
            for p in members:
                if p // 2 == i:
                    cand[p % 2] = center
            if not _valid(original[i], cand):
                return False
        return True

    def compatible(members, p, anchor=None):
        if any((q // 2 == p // 2 for q in members)):
            return False
        if any((blockers[q // 2, p // 2] for q in members)):
            meta['distinct_axis_or_span_merges_blocked'] += 1
            return False
        coords = points[members + [p]]
        if anchor is not None:
            coords = np.concatenate([coords, anchor[None]])
        if np.linalg.norm(coords[:, None] - coords[None, :], axis=2).max() > MAX_DIAMETER + 1e-10:
            meta['diameter_merges_blocked'] += 1
            return False
        return True
    order = np.lexsort((np.arange(2 * n), points[:, 2], points[:, 1], points[:, 0]))
    for p in order:
        i = p // 2
        limit = limits[i]
        if limit <= 1e-12:
            continue
        proposals = []
        if len(vertices):
            d = np.linalg.norm(vertices - points[p], axis=1)
            for old in np.where(d <= limit + 1e-10)[0]:
                if any((_parallel_distinct(original[i], e) or _distinct_span(original[i], e) for e in old_incidence[old])):
                    meta['distinct_axis_or_span_merges_blocked'] += 1
                    continue
                ci = old_cluster.get(int(old))
                members = [] if ci is None else clusters[ci]['members']
                if not compatible(members, int(p), vertices[old]):
                    continue
                if feasible(members + [int(p)], vertices[old]):
                    proposals.append((0, float(d[old]), int(old), ci, vertices[old], members + [int(p)]))
        for ci, cl in enumerate(clusters):
            if cl['old'] is not None:
                continue
            if np.linalg.norm(cl['center'] - points[p]) > MAX_DIAMETER:
                continue
            if not compatible(cl['members'], int(p)):
                continue
            members = cl['members'] + [int(p)]
            center, mean = _node_center(members, original, points)
            if not feasible(members, center):
                center = mean
            if feasible(members, center):
                proposals.append((1, float(np.linalg.norm(center - points[p])), ci, ci, center, members))
        if proposals:
            priority, distance, identity, ci, center, members = min(proposals, key=lambda x: x[:3])
            if ci is None:
                ci = len(clusters)
                clusters.append(dict(members=members, center=center.copy(), old=identity))
                old_cluster[identity] = ci
            else:
                clusters[ci]['members'] = members
                clusters[ci]['center'] = center.copy()
            for q in members:
                out[q // 2, q % 2] = center
                node_ids[q] = ci
        else:
            ci = len(clusters)
            clusters.append(dict(members=[int(p)], center=points[p].copy(), old=None))
            node_ids[p] = ci
    meta['new_shared_nodes'] = sum((cl['old'] is None and len(cl['members']) > 1 for cl in clusters))
    meta['old_anchored_nodes'] = sum((cl['old'] is not None for cl in clusters))
    meta['endpoints_at_old_nodes'] = sum((len(cl['members']) for cl in clusters if cl['old'] is not None))
    meta['max_cluster_original_diameter_m'] = max([float(np.linalg.norm(points[cl['members']][:, None] - points[cl['members']][None, :], axis=2).max()) for cl in clusters] + [0.0])
    return (out, node_ids.reshape(-1, 2), clusters)

def kernel_560(a):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return np.empty((0, 2, 3), dtype=float)
    if a.ndim != 3 or a.shape[1:] != (2, 3):
        raise ValueError('Expected finite local-metre Nx2x3 segments')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite segment coordinates')
    return a.copy()

def kernel_561(original_a, original_b, a, b):
    la = np.linalg.norm(original_a[1] - original_a[0])
    lb = np.linalg.norm(original_b[1] - original_b[0])
    if min(la, lb) <= 1e-12 or min(la, lb) / max(la, lb) < 0.95:
        return False
    if _parallel_distinct(original_a, original_b):
        return False
    if _angle(original_a, original_b) > 3.0 or _angle(a, b) > 3.0:
        return False
    if _finite_hausdorff(original_a, original_b) > COMPETE_ENDPOINT_M:
        return False
    direct = np.linalg.norm(original_a - original_b, axis=1).max()
    reverse = np.linalg.norm(original_a - original_b[::-1], axis=1).max()
    if min(direct, reverse) > COMPETE_ENDPOINT_M:
        return False
    return _finite_hausdorff(a, b) <= COMPETE_ENDPOINT_M

def kernel_562(a, b):
    va = a[1] - a[0]
    vb = b[1] - b[0]
    la = np.linalg.norm(va)
    lb = np.linalg.norm(vb)
    if min(la, lb) <= 1e-12:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(abs(va @ vb) / (la * lb), 0.0, 1.0))))

def kernel_565(lines):
    x = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite input lines')
    return x

def kernel_566(evidence):
    return {k: evidence[k] for k in _PREDICTION_FIELDS}

def kernel_567(stats, n):
    x = np.asarray(stats, dtype=np.float64).reshape(-1, 3)
    if len(x) != n or not np.isfinite(x).all():
        raise ValueError('Expected finite Nx3 candidate source statistics')
    return x

def kernel_568(model):
    if isinstance(model, (str, Path)):
        with Path(model).open('rb') as handle:
            model = pickle.load(handle)
    if model['features'] != FEATURES:
        raise ValueError('Feature schema mismatch')
    return model

def kernel_569(evidence, candidate, stats):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    ep = np.asarray(evidence['ep']).reshape(-1, 3)
    pc = np.asarray(evidence['pc']).reshape(-1, 3)
    scale = float(evidence['scale'])
    corners = base.reshape(-1, 3)
    ab = raw[:, 1] - raw[:, 0]
    rl = np.linalg.norm(ab, axis=1)
    ru = ab / np.maximum(rl[:, None], 1e-09)
    ep_tree = cKDTree(ep) if len(ep) else None
    pc_tree = cKDTree(pc) if len(pc) else None
    rows = []
    for edge, s in zip(candidate, stats):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-09)
        query = edge[0] + np.linspace(0, 1, 12)[:, None] * (edge[1] - edge[0])
        compatible = abs(ru @ u) > 0.966
        d = distances(query, raw[compatible])
        minimum = d.min(1) if d.shape[1] else np.full(12, 5.0)
        support = (d < 0.35).sum(1) if d.shape[1] else np.zeros(12)
        endpoint = nearest(edge, raw.reshape(-1, 3))
        ed = np.minimum(ep_tree.query(query)[0], 5) if ep_tree else np.full(12, 5.0)
        pd = np.minimum(pc_tree.query(query)[0], 5) if pc_tree else np.full(12, 5.0)
        graph = nearest(edge, corners)
        bd = distances(query, base)
        bm = bd.min(1) if bd.shape[1] else np.full(12, 5.0)
        rows.append([np.log1p(length), s[0], s[1], np.log1p(s[2]), (minimum < 0.18).mean(), (minimum < 0.35).mean(), np.log1p(support.min()), minimum.clip(0, 5).mean(), endpoint.mean(), (ed < 0.25).mean(), ed.mean(), (pd < 0.35).mean(), graph.mean(), graph.max(), bm.clip(0, 5).mean(), (bm < 0.35).mean(), pair_distance(edge[None], base).min() if len(base) else 5.0, abs(u[2]), np.log1p(scale)])
    return np.nan_to_num(np.asarray(rows).reshape(-1, len(FEATURES)), nan=5, posinf=5)

def kernel_570(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_572(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_574(graph):
    v, ix = np.unique(graph.reshape(-1, 3), axis=0, return_inverse=True)
    d = np.bincount(ix, minlength=len(v))
    return (v, ix.reshape(-1, 2), d)

def kernel_575(g, patch):
    mask = np.ones(len(g), bool)
    mask[patch['remove']] = False
    return canonical(np.r_[g[mask], np.asarray(patch['add']).reshape(-1, 2, 3)])

def kernel_576(points, segs):
    if not len(segs):
        return np.full((len(points), 0), np.inf)
    ab = segs[:, 1] - segs[:, 0]
    d = (ab * ab).sum(1)
    delta = points[:, None] - segs[None, :, 0]
    t = np.divide((delta * ab[None]).sum(2), d[None], out=np.zeros((len(points), len(segs))), where=d[None] > 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * ab[None], axis=2)

def kernel_577(ev, line):
    raw = ev['raw']
    L = np.linalg.norm(line[1] - line[0])
    u = (line[1] - line[0]) / max(L, 1e-12)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    mask = (abs(ru @ u) >= np.cos(np.deg2rad(12))) & (rl > 0.15)
    if not mask.any():
        return 0.0
    points = line[0] + np.linspace(0, 1, 21)[:, None] * (line[1] - line[0])
    return float((c.e.OLD.distances(points, raw[mask]).min(1) <= 0.25).mean())

def kernel_578(line):
    a = np.asarray(line).reshape(2, 3)
    return tuple(a.reshape(-1)) if tuple(a[0]) <= tuple(a[1]) else tuple(a[::-1].reshape(-1))

def kernel_582(graph):
    v, ix = np.unique(graph.reshape(-1, 3), axis=0, return_inverse=True)
    d = np.bincount(ix, minlength=len(v))
    return (v, ix.reshape(-1, 2), d)

def kernel_583(points, cloud, empty=None):
    return np.minimum(cKDTree(cloud).query(points)[0], 5.0) if len(cloud) else np.full(len(points), empty)

def kernel_586(ev, g):
    v = np.unique(g.reshape(-1, 3), axis=0)
    ii, jj = np.triu_indices(len(v), 1)
    allpairs = np.stack([v[ii], v[jj]], axis=1)
    present = {e.LEGACY.linekey(line) for line in g}
    length = np.linalg.norm(allpairs[:, 1] - allpairs[:, 0], axis=1)
    good = np.array([e.LEGACY.linekey(line) not in present for line in allpairs], dtype=bool) & (length >= 0.2)
    allpairs = allpairs[good]
    raw = np.asarray(ev['raw'])
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    cover = []
    votes = []
    pair = []
    for line in allpairs:
        u = (line[1] - line[0]) / np.linalg.norm(line[1] - line[0])
        aligned = (abs(ru @ u) >= np.cos(np.deg2rad(12))) & (rl > 0.15)
        q = line[0] + np.linspace(0, 1, 21)[:, None] * (line[1] - line[0])
        D = c.e.OLD.distances(q, raw[aligned])
        cv = float((D.min(1) <= 0.25).mean()) if D.shape[1] else 0.0
        pd = c.e.OLD.pair_distance(line[None], raw)[0]
        cover.append(cv)
        votes.append(int((pd <= 0.35).sum()))
        pair.append(float(pd.min()) if len(pd) else 1000000000.0)
    cover = np.array(cover)
    keep = np.flatnonzero(cover >= 0.65)
    return (allpairs, keep, cover, np.array(votes), np.array(pair))

def kernel_588(ev, graph, patches, legacy, finite, roof):
    base = np.c_[legacy.feature_matrix(ev, graph, patches), finite.features(ev, graph, patches, legacy)] if patches else np.empty((0, len(legacy.FEATURES) + len(finite.FEATURES)))
    if not patches:
        return np.empty((0, base.shape[1] + 54 + len(FAMILIES)))
    all_lines = np.concatenate([graph] + [np.asarray(p['add']) for p in patches])
    unique = legacy.e.verify.canonical(all_lines)
    X, _ = roof.roof_features(ev, unique)
    cache = {key(x): f for x, f in zip(unique, X)}
    rows = []
    for patch in patches:
        added = np.array([cache[key(x)] for x in patch['add']])
        removed = np.array([cache[key(graph[i])] for i in patch['remove']]) if patch['remove'] else np.zeros((1, 18))
        rows.append(np.r_[added.mean(0), added.max(0), removed.mean(0), [float(patch['family'] == f) for f in FAMILIES]])
    return np.c_[base, rows]

def kernel_594(ev, g, patches):
    raw = ev['raw']
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    vertices, ix, degree = e.LEGACY.degree(g)
    result = []
    for patch in patches:
        if not patch['remove']:
            result.append(np.zeros(12))
            continue
        old = g[patch['remove']]
        new = np.array(patch['add'])
        keep = np.ones(len(g), bool)
        keep[patch['remove']] = False
        other = g[keep]
        survive = np.r_[other.reshape(-1, 3), new.reshape(-1, 3)]
        lost = [x for x in np.unique(old.reshape(-1, 3), axis=0) if not (np.linalg.norm(survive - x, axis=1) < 1e-08).any()]
        if not lost:
            result.append(np.zeros(12))
            continue
        ref = new[np.linalg.norm(new[:, 1] - new[:, 0], axis=1).argmax()]
        u = ref[1] - ref[0]
        u /= max(np.linalg.norm(u), 1e-12)
        ou = (other[:, 1] - other[:, 0]) / np.maximum(np.linalg.norm(other[:, 1] - other[:, 0], axis=1)[:, None], 1e-12)
        branches = other[abs(ou @ u) < np.cos(np.radians(20))]
        rows = []
        for point in lost:
            aligned = (abs(ru @ u) > np.cos(np.radians(12))) & (np.linalg.norm(np.cross(raw - point, u), axis=2).max(1) <= 0.35)
            pr = (raw - point) @ u
            lo = pr.min(1)
            hi = pr.max(1)
            ends = aligned & (np.minimum(abs(lo), abs(hi)) <= 0.25)
            cross = aligned & (lo < -0.35) & (hi > 0.35)
            left = aligned & (abs(hi) <= 0.25) & (lo < -0.3)
            right = aligned & (abs(lo) <= 0.25) & (hi > 0.3)
            rd = c.e.OLD.distances(point[None], raw)[0]
            branch = (abs(ru @ u) < np.cos(np.radians(20))) & (rd <= 0.35) & (rl > 0.3)
            gd = c.e.OLD.distances(point[None], branches)
            rawend = c.e.OLD.nearest(point[None], raw.reshape(-1, 3))[0]
            ep = c.e.OLD.nearest(point[None], ev['ep'])[0]
            di = int(degree[np.linalg.norm(vertices - point, axis=1).argmin()])
            rows.append([min(5.0, float(rawend)), np.log1p(aligned.sum()), np.log1p(ends.sum()), np.log1p(cross.sum()), np.log1p(min(left.sum(), right.sum())), ends.sum() / (ends.sum() + cross.sum() + 1), np.log1p(branch.sum()), min(5.0, float(gd.min())) if gd.size else 5.0, min(5.0, float(ep)), np.log1p(np.median(rl[branch])) if branch.any() else 0.0, di])
        a = np.array(rows)
        agg = a.max(0)
        agg[[0, 7, 8]] = a[:, [0, 7, 8]].min(0)
        result.append(np.r_[len(lost), agg])
    return np.array(result).reshape(-1, 12)

def kernel_596(bundle, evidence, actor_predelete):
    cfg = bundle['config']
    assert cfg['cleaner'] == 'full75' and cfg['old_repair'] == 'keep_local_repair' and (cfg['mode'] == 'add_only')
    actor = c.e.verify.canonical(actor_predelete)
    X = e.residual_candidates.deletion.features(evidence, actor)
    score = bundle['base']['prune']['model'].predict_proba(X)[:, 1]
    removed = [int(i) for i in np.argsort(-score, kind='stable') if score[i] >= 0.75][:8]
    keep = np.ones(len(actor), bool)
    keep[removed] = False
    clean = c.e.verify.canonical(actor[keep])
    g, local_repair = e.API.repair(bundle['base']['repair'], evidence, clean)
    patches = e.alg.proposals(evidence, g, actor, c.e.OLD, e.residual_candidates.proposals, e.ADAPTER)
    xx = e.alg.features(evidence, g, patches, e.LEGACY, e.FINITE, e.roof)
    features = np.c_[xx, j.r.removed_features(evidence, g, patches)]
    out, ops = j.m.apply(g, features, patches, bundle['residual'], cfg)
    return (out, dict(GT_used=False, deleted=removed, local_repair=local_repair, residual=[dict(index=i, **patches[i]) for i in ops], main_refiner_changed=False))

def kernel_599(bundle, evidence, actor_graph):
    actor_graph = c.e.verify.canonical(actor_graph)
    clean, removed = prune(bundle['prune'], evidence, actor_graph)
    output, restored = repair(bundle['repair'], evidence, clean)
    return (output, dict(cleanup=removed, repair=restored, GT_used=False, old_delete_stage_replaced=True, network='frozen originalW', frame='local_metres'))

def kernel_1519(intervals, length):
    total, end = (0.0, -np.inf)
    for lo, hi in sorted(intervals):
        total += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return min(1.0, total / length)

def kernel_1549(a, b, lines, cfg):
    unit = b - a
    length = np.linalg.norm(unit)
    unit /= length
    for edge_id, (c, d) in enumerate(lines):
        other = d - c
        other_length = np.linalg.norm(other)
        if other_length < 1e-10:
            continue
        angle = base._angle(abs(unit @ (other / other_length)))
        if angle < cfg['bridge_crossing_min_angle_deg']:
            continue
        distance, s, t = _closest(a, b, c, d)
        if distance <= cfg['bridge_crossing_radius_m'] and cfg['bridge_crossing_margin_m'] < s * length < length - cfg['bridge_crossing_margin_m']:
            return {'edge_id': edge_id, 'distance_m': distance, 'bridge_t': s, 'other_t': t, 'cross_angle_deg': angle}
    return None

def kernel_1550(lines, raw_lines, point_tree, cfg, meta):
    vertices, edges, adjacency = _graph(lines, cfg['vertex_tolerance_m'])
    ends = [i for i, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    candidates = []
    rejected = Counter()
    for offset, i in enumerate(ends):
        ni = next(iter(adjacency[i]))
        ui = vertices[i] - vertices[ni]
        length_i = np.linalg.norm(ui)
        ui /= length_i
        for j in ends[offset + 1:]:
            if j in adjacency[i]:
                continue
            nj = next(iter(adjacency[j]))
            uj = vertices[j] - vertices[nj]
            length_j = np.linalg.norm(uj)
            uj /= length_j
            delta = vertices[j] - vertices[i]
            gap = float(np.linalg.norm(delta))
            if gap <= cfg['vertex_tolerance_m'] or gap > cfg['bridge_max_gap_m']:
                continue
            if min(length_i, length_j) < cfg['bridge_min_fragment_m']:
                rejected['short_fragment'] += 1
                continue
            relative_gap = gap / min(length_i, length_j)
            if relative_gap > cfg['bridge_max_gap_ratio']:
                rejected['relative_gap'] += 1
                continue
            direction = delta / gap
            ai, aj = (_angle(ui @ direction), _angle(-uj @ direction))
            if max(ai, aj) > cfg['bridge_angle_deg']:
                rejected['facing_angle'] += 1
                continue
            opposition = _angle(-ui @ uj)
            if opposition > cfg['bridge_opposition_deg']:
                rejected['opposition'] += 1
                continue
            lateral = gap * max(math.sin(math.radians(ai)), math.sin(math.radians(aj)))
            if lateral > cfg['bridge_lateral_m']:
                rejected['lateral'] += 1
                continue
            existing = _interval_coverage(vertices[i], vertices[j], lines, cfg['bridge_existing_radius_m'], cfg['bridge_existing_direction_deg'])
            if existing >= cfg['bridge_existing_coverage_reject']:
                rejected['already_covered'] += 1
                continue
            raw_coverage = None
            if raw_lines is not None:
                raw_coverage = _interval_coverage(vertices[i], vertices[j], raw_lines, cfg['bridge_raw_radius_m'], cfg['bridge_raw_direction_deg'])
            if cfg['bridge_raw_min_coverage'] > 0 and (raw_coverage is None or raw_coverage < cfg['bridge_raw_min_coverage']):
                rejected['raw_coverage'] += 1
                continue
            point_coverage = None
            if point_tree is not None:
                count = max(9, int(math.ceil(gap / max(0.02, cfg['bridge_point_radius_m'] / 2))) + 1)
                samples = vertices[i] + np.linspace(0, 1, count)[:, None] * delta
                distances, _ = point_tree.query(samples)
                point_coverage = float(np.mean(distances <= cfg['bridge_point_radius_m']))
            if cfg['bridge_point_min_coverage'] > 0 and (point_coverage is None or point_coverage < cfg['bridge_point_min_coverage']):
                rejected['point_coverage'] += 1
                continue
            score = gap / cfg['bridge_max_gap_m'] + max(ai, aj) / max(cfg['bridge_angle_deg'], 1e-09) + lateral / max(cfg['bridge_lateral_m'], 1e-09) + 0.25 * relative_gap
            if raw_coverage is not None:
                score += cfg['bridge_raw_score_weight'] * max(0, 1 - raw_coverage / max(cfg['bridge_raw_soft_coverage'], 1e-09))
            if point_coverage is not None:
                score += cfg['bridge_point_score_weight'] * (1 - point_coverage)
            candidates.append({'i': i, 'j': j, 'score': float(score), 'distance_m': gap, 'angle_i_deg': ai, 'angle_j_deg': aj, 'opposition_deg': opposition, 'lateral_m': lateral, 'relative_gap': float(relative_gap), 'raw_coverage': raw_coverage, 'point_coverage': point_coverage, 'existing_coverage': existing})
    candidates.sort(key=lambda c: (c['score'], c['distance_m'], c['i'], c['j']))
    best_for_end = {}
    for index, c in enumerate(candidates):
        for i in (c['i'], c['j']):
            best_for_end.setdefault(i, index)
    used, additions = (set(), [])
    for index, candidate in enumerate(candidates):
        i, j = (candidate['i'], candidate['j'])
        if i in used or j in used:
            rejected['matching_conflict'] += 1
            continue
        if cfg['bridge_matching'] == 'mutual' and (best_for_end[i] != index or best_for_end[j] != index):
            rejected['not_mutual'] += 1
            continue
        used.update([i, j])
        additions.append(vertices[[i, j]])
        meta['log'].append(dict(op='bridge', endpoint_i=vertices[i].tolist(), endpoint_j=vertices[j].tolist(), **candidate))
    meta['counts']['bridge_candidates'] = len(candidates)
    meta['counts']['bridges_added'] = len(additions)
    meta['bridge_rejections'] = dict(rejected)
    if additions:
        lines = _canonical(np.concatenate([lines, np.asarray(additions)]), cfg['vertex_tolerance_m'])
    return lines

def kernel_1551(lines, tolerance):
    vertices, edges, _ = _graph(lines, tolerance)
    return vertices[edges]

def kernel_1552(lines, tolerance):
    if not len(lines):
        return (np.empty((0, 3)), np.empty((0, 2), dtype=int), [])
    flat = lines.reshape(-1, 3)
    if tolerance > 0:
        keys = np.round(flat / tolerance).astype(np.int64)
    else:
        keys = flat
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vertices = flat[first]
    edges = np.unique(np.sort(inverse.reshape(-1, 2), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    used = np.unique(edges)
    remap = np.full(len(vertices), -1, dtype=int)
    remap[used] = np.arange(len(used))
    vertices, edges = (vertices[used], remap[edges])
    adjacency = [set() for _ in vertices]
    for a, b in edges:
        adjacency[a].add(int(b))
        adjacency[b].add(int(a))
    return (vertices, edges, adjacency)

def kernel_1553(dot):
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))

def kernel_1554(points, origin, direction):
    delta = points - origin
    along = delta @ direction
    return (along, np.linalg.norm(delta - along[..., None] * direction, axis=-1))

def kernel_1626(lines, cfg, meta):
    vertices, edges, adjacency = _graph(lines, cfg['vertex_tolerance_m'])
    lines = vertices[edges]
    delta = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    unit = delta / lengths[:, None]
    parent = np.arange(len(lines))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if abs(unit[i] @ unit[j]) < np.cos(np.radians(cfg['overlap_angle_deg'])):
                continue
            t, radius = _radial(lines[j], lines[i, 0], unit[i])
            if radius.max() > cfg['overlap_lateral_m']:
                continue
            overlap = min(lengths[i], t.max()) - max(0.0, t.min())
            if overlap < cfg['overlap_min_m']:
                continue
            parent[find(j)] = find(i)
    groups = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    output = []
    for members in groups.values():
        if len(members) == 1:
            output.append(lines[members[0]])
            continue
        longest = max(members, key=lambda i: lengths[i])
        origin, direction = (lines[longest, 0], unit[longest])
        ids = np.unique(edges[members])
        position, radius = _radial(vertices[ids], origin, direction)
        if radius.max() > cfg['overlap_lateral_m'] or any((abs(unit[i] @ direction) < np.cos(np.radians(cfg['overlap_angle_deg'])) for i in members)):
            output.extend(lines[members])
            meta['counts']['overlap_ambiguous_groups_skipped'] += 1
            continue
        member_set = set(members)
        incident_outside = set()
        for edge_id, (a, b) in enumerate(edges):
            if edge_id not in member_set:
                incident_outside.update([int(a), int(b)])
        keep = {int(ids[position.argmin()]), int(ids[position.argmax()])}
        keep.update((int(i) for i in ids if int(i) in incident_outside))
        ordered = sorted(keep, key=lambda i: float((vertices[i] - origin) @ direction))
        output.extend((np.array([vertices[a], vertices[b]]) for a, b in zip(ordered[:-1], ordered[1:]) if np.linalg.norm(vertices[b] - vertices[a]) > cfg['vertex_tolerance_m']))
        meta['counts']['overlap_groups_merged'] += 1
        meta['counts']['overlap_lines_removed'] += len(members) - (len(ordered) - 1)
        meta['log'].append({'op': 'overlap_merge', 'input_segments': len(members), 'output_segments': len(ordered) - 1, 'retained_junctions': len(ordered) - 2})
    return _canonical(np.asarray(output).reshape(-1, 2, 3), cfg['vertex_tolerance_m'])

def kernel_1627(lines, cfg, meta):
    vertices, edges, adjacency = _graph(lines, cfg['vertex_tolerance_m'])
    supports = {tuple(edge): vertices[edge].copy() for edge in edges}
    while True:
        chosen = None
        for i, neighbors in enumerate(adjacency):
            if len(neighbors) != 2:
                continue
            a, b = sorted(neighbors)
            if b in adjacency[a]:
                continue
            va, vb = (vertices[a] - vertices[i], vertices[b] - vertices[i])
            angle = _angle(-va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
            if angle > cfg['compress_angle_deg']:
                continue
            key1, key2 = (tuple(sorted((a, i))), tuple(sorted((i, b))))
            points = np.concatenate([supports[key1], supports[key2]])
            chord = vertices[b] - vertices[a]
            length = np.linalg.norm(chord)
            if length <= cfg['vertex_tolerance_m']:
                continue
            t, radius = _radial(points, vertices[a], chord / length)
            if radius.max() > cfg['compress_lateral_m'] or t.min() < -1e-07 or t.max() > length + 1e-07:
                continue
            chosen = (i, a, b, key1, key2, points, angle)
            break
        if chosen is None:
            break
        i, a, b, key1, key2, points, angle = chosen
        adjacency[a].remove(i)
        adjacency[b].remove(i)
        adjacency[i].clear()
        adjacency[a].add(b)
        adjacency[b].add(a)
        del supports[key1], supports[key2]
        supports[a, b] = points
        meta['counts']['degree2_vertices_compressed'] += 1
        meta['log'].append({'op': 'compress_degree2', 'vertex': vertices[i].tolist(), 'angle_deg': angle})
    return _canonical(np.asarray([vertices[list(edge)] for edge in sorted(supports)]).reshape(-1, 2, 3), cfg['vertex_tolerance_m'])

def kernel_1630(value):
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        return np.empty((0, 2, 3))
    if a.ndim != 3 or a.shape[1:] != (2, 3) or (not np.isfinite(a).all()):
        raise ValueError('Expected finite (N,2,3) metre-coordinate lines')
    return a.copy()

def kernel_1631(line, other, tolerance):
    p, q = line
    L = np.linalg.norm(q - p)
    if not len(other) or L <= 2 * tolerance:
        return False
    u = (q - p) / L
    for a, b in other:
        W = np.linalg.norm(b - a)
        if W < 1e-10:
            continue
        w = (b - a) / W
        c = float(u @ w)
        denominator = 1 - c * c
        if denominator < 1e-08:
            continue
        delta = a - p
        du, dw = (float(delta @ u), float(delta @ w))
        t, s = ((du - c * dw) / denominator, (c * du - dw) / denominator)
        miss = np.linalg.norm(p + t * u - a - s * w)
        if tolerance < t < L - tolerance and tolerance < s < W - tolerance and (miss < tolerance):
            return True
        if tolerance < s < W - tolerance and miss < tolerance and (min(abs(t), abs(t - L)) < tolerance):
            return True
    return False

def kernel_1632(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15), -1, 1))))

def kernel_1633(line, raw, radius, cfg):
    if not len(raw):
        return 0
    u = line[1] - line[0]
    delta = raw[:, 1] - raw[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    cosine = abs(delta @ u) / np.maximum(lengths * np.linalg.norm(u), 1e-15)
    forward = np.linalg.norm(raw - line, axis=2).max(1)
    backward = np.linalg.norm(raw[:, ::-1] - line, axis=2).max(1)
    keep = (np.minimum(forward, backward) <= radius) & (cosine >= np.cos(np.deg2rad(cfg['raw_angle_deg'])))
    rows = raw[keep].copy()
    if not len(rows):
        return 0
    reverse = (rows[:, 1] - rows[:, 0]) @ u < 0
    rows[reverse] = rows[reverse, ::-1]
    keys = np.round((rows - line.mean(0)).reshape(-1, 6) / cfg['raw_dedup_m']).astype(np.int64)
    return int(len(np.unique(keys, axis=0)))

def kernel_1634(line, other, radius, angle):
    a, b = line
    length = float(np.linalg.norm(b - a))
    if not len(other) or length <= 1e-10:
        return ([], np.empty(0, dtype=int))
    u = (b - a) / length
    delta = other[:, 1] - other[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    t = (other - a) @ u
    dt = t[:, 1] - t[:, 0]
    lo = np.maximum(np.minimum(t[:, 0], t[:, 1]), 0.0)
    hi = np.minimum(np.maximum(t[:, 0], t[:, 1]), length)
    eligible = (lengths > 1e-10) & (abs(dt) >= lengths * np.cos(np.deg2rad(angle))) & (hi - lo > 1e-07)
    idx = np.flatnonzero(eligible)
    if not len(idx):
        return ([], idx)
    q0 = other[idx, 0] + ((lo[idx] - t[idx, 0]) / dt[idx])[:, None] * delta[idx]
    q1 = other[idx, 0] + ((hi[idx] - t[idx, 0]) / dt[idx])[:, None] * delta[idx]
    q = np.stack((q0 - a, q1 - a), axis=1)
    perp = q - (q @ u)[:, :, None] * u
    idx = idx[np.linalg.norm(perp, axis=2).max(1) <= radius]
    return (_merge(list(zip(lo[idx], hi[idx]))), idx)

def kernel_1635(raw_lines, edge_points, config=None, scale=None):
    cfg = dict(config or {})
    symmetric = cfg.pop('symmetric_knn', False)
    scheme = replace(POSTPROCESS_SCHEME, **cfg)
    if scale is None or scale <= 0:
        raise ValueError('Original point-cloud radius is required')
    out, meta = linecloud_to_wireframe_variant(np.asarray(raw_lines) / scale, np.asarray(edge_points) / scale, scheme, symmetric_knn=symmetric)
    return (out * scale, dict(meta, scale_m=float(scale), symmetric_knn=bool(symmetric)))

def kernel_1688(members, original, points):
    mean = points[members].mean(axis=0)
    matrix = np.zeros((3, 3))
    rhs = np.zeros(3)
    directions = []
    for p in members:
        e = original[p // 2]
        v = e[1] - e[0]
        length = np.linalg.norm(v)
        if length <= 1e-12:
            continue
        u = v / length
        if any((abs(u @ old) > 0.985 for old in directions)):
            continue
        directions.append(u)
        project = np.eye(3) - np.outer(u, u)
        matrix += project
        rhs += project @ points[p]
    if len(directions) >= 2 and np.linalg.cond(matrix) < 100:
        center = np.linalg.solve(matrix + 0.05 * np.eye(3), rhs + 0.05 * mean)
        if np.linalg.norm(center - mean) <= MAX_DIAMETER:
            return (center, mean)
    return (mean, mean)

def kernel_1689(a, b):
    la = np.linalg.norm(a[1] - a[0])
    lb = np.linalg.norm(b[1] - b[0])
    if min(la, lb) < 1e-09 or _angle(a, b) > MAX_ANGLE_DEG:
        return False
    if min(la, lb) / max(la, lb) >= 0.95:
        return False
    u = (a[1] - a[0]) / la
    t = (b - a[0]) @ u
    overlap = max(0.0, min(la, t.max()) - max(0.0, t.min()))
    lateral = np.linalg.norm(b - a[0] - t[:, None] * u, axis=1)
    return bool(overlap / min(la, lb) >= 0.8 and lateral.max() <= PARALLEL_SEPARATION_M and (_finite_hausdorff(a, b) > COMPETE_ENDPOINT_M))

def kernel_1690(a, b):
    la = np.linalg.norm(a[1] - a[0])
    lb = np.linalg.norm(b[1] - b[0])
    if min(la, lb) < 1e-09 or _angle(a, b) > MAX_ANGLE_DEG:
        return False
    if min(la, lb) / max(la, lb) < 0.8:
        return False
    u = (a[1] - a[0]) / la
    t = (b - a[0]) @ u
    overlap = max(0.0, min(la, t.max()) - max(0.0, t.min()))
    if overlap / min(la, lb) < 0.8:
        return False
    lateral = np.linalg.norm(b - a[0] - t[:, None] * u, axis=1)
    return bool(lateral.min() > PARALLEL_SEPARATION_M)

def kernel_1691(a, b):
    return float(max(_point_distance(a, b).max(), _point_distance(b, a).max()))

def kernel_1694(points, cloud, empty=None):
    return np.minimum(cKDTree(cloud).query(points)[0], 5.0) if len(cloud) else np.full(len(points), empty)

def kernel_1695(points, segs):
    if not len(segs):
        return np.full((len(points), 0), np.inf)
    ab = segs[:, 1] - segs[:, 0]
    d = (ab * ab).sum(1)
    delta = points[:, None] - segs[None, :, 0]
    t = np.divide((delta * ab[None]).sum(2), d[None], out=np.zeros((len(points), len(segs))), where=d[None] > 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * ab[None], axis=2)

def kernel_1696(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_1697(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_1703(evidence, candidate, stats):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    ep = np.asarray(evidence['ep']).reshape(-1, 3)
    pc = np.asarray(evidence['pc']).reshape(-1, 3)
    scale = float(evidence['scale'])
    corners = base.reshape(-1, 3)
    ab = raw[:, 1] - raw[:, 0]
    rl = np.linalg.norm(ab, axis=1)
    ru = ab / np.maximum(rl[:, None], 1e-09)
    ep_tree = cKDTree(ep) if len(ep) else None
    pc_tree = cKDTree(pc) if len(pc) else None
    rows = []
    for edge, s in zip(candidate, stats):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-09)
        query = edge[0] + np.linspace(0, 1, 12)[:, None] * (edge[1] - edge[0])
        compatible = abs(ru @ u) > 0.966
        d = distances(query, raw[compatible])
        minimum = d.min(1) if d.shape[1] else np.full(12, 5.0)
        support = (d < 0.35).sum(1) if d.shape[1] else np.zeros(12)
        endpoint = nearest(edge, raw.reshape(-1, 3))
        ed = np.minimum(ep_tree.query(query)[0], 5) if ep_tree else np.full(12, 5.0)
        pd = np.minimum(pc_tree.query(query)[0], 5) if pc_tree else np.full(12, 5.0)
        graph = nearest(edge, corners)
        bd = distances(query, base)
        bm = bd.min(1) if bd.shape[1] else np.full(12, 5.0)
        rows.append([np.log1p(length), s[0], s[1], np.log1p(s[2]), (minimum < 0.18).mean(), (minimum < 0.35).mean(), np.log1p(support.min()), minimum.clip(0, 5).mean(), endpoint.mean(), (ed < 0.25).mean(), ed.mean(), (pd < 0.35).mean(), graph.mean(), graph.max(), bm.clip(0, 5).mean(), (bm < 0.35).mean(), pair_distance(edge[None], base).min() if len(base) else 5.0, abs(u[2]), np.log1p(scale)])
    return np.nan_to_num(np.asarray(rows).reshape(-1, len(FEATURES)), nan=5, posinf=5)

def kernel_1704(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_1705(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_1714(edge):
    x = np.asarray(edge).reshape(2, 3)
    if tuple(x[0]) > tuple(x[1]):
        x = x[::-1]
    return tuple(x.reshape(-1))

def kernel_1715(line):
    a = np.asarray(line).reshape(2, 3)
    return tuple(a.reshape(-1)) if tuple(a[0]) <= tuple(a[1]) else tuple(a[::-1].reshape(-1))

def kernel_1719(graph):
    v, ix = np.unique(graph.reshape(-1, 3), axis=0, return_inverse=True)
    d = np.bincount(ix, minlength=len(v))
    return (v, ix.reshape(-1, 2), d)

def kernel_1723(ev, graph):
    return np.c_[c.e.alg.features(ev, graph), f.finite_features(ev, graph), f.roof_features(ev, graph)[0]]

def kernel_1724(bundle, evidence, graph):
    return LOCAL_REPAIR_API.decode(bundle, evidence, graph)

def kernel_1726(ev, graph, actor, old, legacy, coverpool):
    patches = legacy(ev, graph, actor)
    seen = {(tuple(p['remove']), tuple((key(x) for x in np.asarray(p['add'])))) for p in patches}
    v, ix, degree = coverpool.degree(graph)
    raw = np.asarray(ev['raw'])
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    gl = np.linalg.norm(graph[:, 1] - graph[:, 0], axis=1)
    gu = (graph[:, 1] - graph[:, 0]) / np.maximum(gl[:, None], 1e-12)

    def support(edge):
        L = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(L, 1e-12)
        good = (abs(ru @ u) >= np.cos(np.deg2rad(12))) & (rl > 0.15)
        if not good.any():
            return 0.0
        q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
        dd = old.distances(q, raw[good])
        return float((dd.min(1) <= 0.25).mean())

    def add(family, remove, lines):
        lines = coverpool.canonical(np.asarray(lines).reshape(-1, 2, 3))
        remove = sorted(set(map(int, remove)))
        if not len(lines) or np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1).min() < 0.15:
            return
        patch = dict(family=family, remove=remove, add=lines.tolist())
        signature = (tuple(remove), tuple((key(x) for x in lines)))
        if signature in seen:
            return
        pred = coverpool.apply(graph, patch)
        if len(pred) != len(graph) - len(remove) + len(lines):
            return
        seen.add(signature)
        patches.append(patch)
    empty = np.empty((0, 2, 3))
    pool, stats, _ = coverpool.candidates(dict(ev, balanced=empty), 'coverage')
    if len(pool):
        feat = old.feature_matrix(dict(ev, balanced=graph), pool, stats)
        length = np.linalg.norm(pool[:, 1] - pool[:, 0], axis=1)
        pair = old.pair_distance(pool, graph)
        eligible = (feat[:, 4] >= 0.65) & (length >= 0.2) & (pair.min(1) > 0.03 if len(graph) else True)
        pool = pool[eligible]
        stats = stats[eligible]
        feat = feat[eligible]
        chosen = []
        for b in range(4):
            ids = np.flatnonzero(np.searchsorted([1.0, 2.0, 5.0], np.linalg.norm(pool[:, 1] - pool[:, 0], axis=1), side='right') == b)
            if not len(ids):
                continue
            selected, _ = coverpool.spread(pool[ids], stats[ids], min(24, len(ids)), float(ev['scale']))
            chosen.extend(ids[selected].tolist())
        pool = pool[chosen]
    else:
        pool = np.empty((0, 2, 3))
    tree = cKDTree(v) if len(v) else None
    for candidate in pool:
        add('diverse_add', [], [candidate])
        line = candidate.copy()
        if tree:
            dist, near = tree.query(line)
            for k in range(2):
                if dist[k] <= 0.35:
                    line[k] = v[near[k]]
            add('diverse_add', [], [line])
        if len(graph):
            u = (candidate[1] - candidate[0]) / max(np.linalg.norm(candidate[1] - candidate[0]), 1e-12)
            for end in range(2):
                t = ((line[end] - graph[:, 0]) * gu).sum(1)
                project = graph[:, 0] + t[:, None] * gu
                distance = np.linalg.norm(project - line[end], axis=1)
                good = np.flatnonzero((distance <= 0.25) & (t > 0.2) & (t < gl - 0.2) & (abs(gu @ u) < np.cos(np.deg2rad(20))))
                for parent in good[np.argsort(distance[good], kind='stable')[:1]]:
                    node = project[parent]
                    child = line.copy()
                    child[end] = node
                    if support(child) >= 0.7:
                        add('junction_add_split', [parent], [[graph[parent, 0], node], [node, graph[parent, 1]], child])
    leaves = [(i, k) for i in range(len(graph)) for k in range(2) if degree[ix[i, k]] == 1]
    scored = []
    for a, (i, ki) in enumerate(leaves):
        for j, kj in leaves[a + 1:]:
            if i == j or abs(gu[i] @ gu[j]) >= np.cos(np.deg2rad(20)):
                continue
            if np.linalg.norm(graph[i, ki] - graph[j, kj]) > 2.0:
                continue
            A = np.stack([gu[i], -gu[j]], axis=1)
            ts = np.linalg.lstsq(A, graph[j, 0] - graph[i, 0], rcond=None)[0]
            p = graph[i, 0] + ts[0] * gu[i]
            q = graph[j, 0] + ts[1] * gu[j]
            if np.linalg.norm(p - q) > 0.2:
                continue
            point = (p + q) / 2
            shift = max(np.linalg.norm(point - graph[i, ki]), np.linalg.norm(point - graph[j, kj]))
            if not 0.05 <= shift <= 1.0:
                continue
            new = graph[[i, j]].copy()
            new[0, ki] = point
            new[1, kj] = point
            if min((support(x) for x in new)) >= 0.7:
                scored.append((shift, [i, j], new))
    for _, remove, lines in sorted(scored, key=lambda x: x[0])[:24]:
        add('corner_intersection', remove, lines)
    diverse = [p for p in patches if p['family'] == 'diverse_add']
    buckets = [[] for _ in range(4)]
    for patch in diverse:
        edge = np.asarray(patch['add'])[0]
        b = int(np.searchsorted([1.0, 2.0, 5.0], np.linalg.norm(edge[1] - edge[0]), side='right'))
        buckets[b].append(patch)
    kept = []
    for i in range(max([len(x) for x in buckets] or [0])):
        for bucket in buckets:
            if i < len(bucket):
                kept.append(bucket[i])
            if len(kept) == 72:
                break
        if len(kept) == 72:
            break
    patches = [p for p in patches if p['family'] != 'diverse_add'] + kept
    counts = {f: 0 for f in FAMILIES}
    out = []
    for patch in patches:
        family = patch['family']
        if family in counts:
            limit = 72 if family == 'diverse_add' else 32
            if counts[family] >= limit:
                continue
            counts[family] += 1
        out.append(patch)
    return out

def kernel_1727(ev, graph, actor):
    patches = [p for p in ALG.proposals(ev, graph) if p['family'] != 'junction_refit'] if len(graph) else []
    present = {ALG.linekey(x) for x in graph}
    added = {tuple((ALG.linekey(x) for x in np.asarray(p['add']))) for p in patches if not p['remove']}
    for line in actor:
        key = ALG.linekey(line)
        if key in present or (key,) in added:
            continue
        patches.append(dict(family='add_single', remove=[], add=[line.tolist()], source='preprune_actor'))
        added.add((key,))
    return patches

def kernel_1729(ev, graph, patches, legacy, finite, roof):
    base = np.c_[legacy.feature_matrix(ev, graph, patches), finite.features(ev, graph, patches, legacy)] if patches else np.empty((0, len(legacy.FEATURES) + len(finite.FEATURES)))
    if not patches:
        return np.empty((0, base.shape[1] + 54 + len(FAMILIES)))
    all_lines = np.concatenate([graph] + [np.asarray(p['add']) for p in patches])
    unique = legacy.e.verify.canonical(all_lines)
    X, _ = roof.roof_features(ev, unique)
    cache = {key(x): f for x, f in zip(unique, X)}
    rows = []
    for patch in patches:
        added = np.array([cache[key(x)] for x in patch['add']])
        removed = np.array([cache[key(graph[i])] for i in patch['remove']]) if patch['remove'] else np.zeros((1, 18))
        rows.append(np.r_[added.mean(0), added.max(0), removed.mean(0), [float(patch['family'] == f) for f in FAMILIES]])
    return np.c_[base, rows]

def kernel_1734(g, X, patches, bundle, cfg):
    values = bundle['model'].predict_proba(X)[:, list(bundle['model'].classes_).index(1)] if len(X) else np.array([])
    if cfg['mode'] == 'add_only':
        values[[i for i, p in enumerate(patches) if p['remove']]] = -1000000000.0
    return e.residual_candidates.select_patches(g, patches, values, cfg['threshold'], cfg['budget'])

def kernel_1736(ev, g, patches):
    raw = ev['raw']
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    vertices, ix, degree = e.LEGACY.degree(g)
    result = []
    for patch in patches:
        if not patch['remove']:
            result.append(np.zeros(12))
            continue
        old = g[patch['remove']]
        new = np.array(patch['add'])
        keep = np.ones(len(g), bool)
        keep[patch['remove']] = False
        other = g[keep]
        survive = np.r_[other.reshape(-1, 3), new.reshape(-1, 3)]
        lost = [x for x in np.unique(old.reshape(-1, 3), axis=0) if not (np.linalg.norm(survive - x, axis=1) < 1e-08).any()]
        if not lost:
            result.append(np.zeros(12))
            continue
        ref = new[np.linalg.norm(new[:, 1] - new[:, 0], axis=1).argmax()]
        u = ref[1] - ref[0]
        u /= max(np.linalg.norm(u), 1e-12)
        ou = (other[:, 1] - other[:, 0]) / np.maximum(np.linalg.norm(other[:, 1] - other[:, 0], axis=1)[:, None], 1e-12)
        branches = other[abs(ou @ u) < np.cos(np.radians(20))]
        rows = []
        for point in lost:
            aligned = (abs(ru @ u) > np.cos(np.radians(12))) & (np.linalg.norm(np.cross(raw - point, u), axis=2).max(1) <= 0.35)
            pr = (raw - point) @ u
            lo = pr.min(1)
            hi = pr.max(1)
            ends = aligned & (np.minimum(abs(lo), abs(hi)) <= 0.25)
            cross = aligned & (lo < -0.35) & (hi > 0.35)
            left = aligned & (abs(hi) <= 0.25) & (lo < -0.3)
            right = aligned & (abs(lo) <= 0.25) & (hi > 0.3)
            rd = c.e.OLD.distances(point[None], raw)[0]
            branch = (abs(ru @ u) < np.cos(np.radians(20))) & (rd <= 0.35) & (rl > 0.3)
            gd = c.e.OLD.distances(point[None], branches)
            rawend = c.e.OLD.nearest(point[None], raw.reshape(-1, 3))[0]
            ep = c.e.OLD.nearest(point[None], ev['ep'])[0]
            di = int(degree[np.linalg.norm(vertices - point, axis=1).argmin()])
            rows.append([min(5.0, float(rawend)), np.log1p(aligned.sum()), np.log1p(ends.sum()), np.log1p(cross.sum()), np.log1p(min(left.sum(), right.sum())), ends.sum() / (ends.sum() + cross.sum() + 1), np.log1p(branch.sum()), min(5.0, float(gd.min())) if gd.size else 5.0, min(5.0, float(ep)), np.log1p(np.median(rl[branch])) if branch.any() else 0.0, di])
        a = np.array(rows)
        agg = a.max(0)
        agg[[0, 7, 8]] = a[:, [0, 7, 8]].min(0)
        result.append(np.r_[len(lost), agg])
    return np.array(result).reshape(-1, 12)

def kernel_1737(bundle, evidence, graph):
    graph = c.e.verify.canonical(graph)
    return prune_from_features(bundle, graph, d.features(evidence, graph))

def kernel_2645(a, b, c, d):
    u, v, w = (b - a, d - c, a - c)
    aa, bb, cc = (u @ u, u @ v, v @ v)
    dd, ee = (u @ w, v @ w)
    options = []
    if aa > 1e-16:
        options.extend([(float(np.clip(-dd / aa, 0, 1)), 0.0), (float(np.clip((bb - dd) / aa, 0, 1)), 1.0)])
    if cc > 1e-16:
        options.extend([(0.0, float(np.clip(ee / cc, 0, 1))), (1.0, float(np.clip((ee + bb) / cc, 0, 1)))])
    denominator = aa * cc - bb * bb
    if denominator > 1e-12 * max(aa * cc, 1e-12):
        s, t = ((bb * ee - cc * dd) / denominator, (aa * ee - bb * dd) / denominator)
        if 0 <= s <= 1 and 0 <= t <= 1:
            options.append((float(s), float(t)))
    if not options:
        return (float(np.linalg.norm(a - c)), 0.0, 0.0)
    return min(((float(np.linalg.norm(a + s * u - c - t * v)), s, t) for s, t in options))

def kernel_2646(a, b, lines, radius, direction_degrees):
    gap_length = float(np.linalg.norm(b - a))
    if len(lines) == 0 or gap_length <= 1e-10:
        return 0.0
    unit = (b - a) / gap_length
    delta = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    parallel = np.abs(delta @ unit) >= np.cos(np.radians(direction_degrees)) * lengths
    t, r = _radial(lines, a, unit)
    lo, hi = (np.minimum(t[:, 0], t[:, 1]), np.maximum(t[:, 0], t[:, 1]))
    mask = parallel & (lengths > 1e-10) & (r.max(axis=1) <= radius) & (hi > 0) & (lo < gap_length)
    intervals = sorted(zip(np.maximum(lo[mask], 0).tolist(), np.minimum(hi[mask], gap_length).tolist()))
    total, end = (0.0, -math.inf)
    for low, high in intervals:
        total += max(0.0, high - max(low, end))
        end = max(end, high)
    return float(min(1.0, total / gap_length))

def kernel_2648(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([float(lo), float(hi)])
    return out

def kernel_2650(lines, edge_points, scheme, symmetric_knn=None):
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
    if lines.size == 0:
        return (np.empty((0, 2, 3), dtype=np.float64), {'num_clusters': 0, 'num_candidates': 0})
    vec = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(vec, axis=1)
    valid_len = length >= scheme.min_graph_len
    lines = lines[valid_len]
    vec = vec[valid_len]
    length = length[valid_len]
    if len(lines) == 0:
        return (np.empty((0, 2, 3), dtype=np.float64), {'num_clusters': 0, 'num_candidates': 0})
    endpoints = lines.reshape(-1, 3)
    labels = DBSCAN(eps=scheme.cluster_eps, min_samples=scheme.min_samples).fit_predict(endpoints)
    centers = cluster_centers(endpoints, labels, scheme.center_method)
    if len(centers) < 2:
        return (np.empty((0, 2, 3), dtype=np.float64), {'num_clusters': int(len(centers)), 'num_candidates': 0})
    dist_to_centers = cdist(endpoints, centers)
    nearest = dist_to_centers.argmin(axis=1)
    nearest_dist = dist_to_centers[np.arange(len(endpoints)), nearest]
    nearest[nearest_dist > scheme.attach_eps] = -1
    line_clusters = nearest.reshape(-1, 2)
    line_snap = nearest_dist.reshape(-1, 2)
    candidates = {}
    for idx, (ca, cb) in enumerate(line_clusters):
        if ca < 0 or cb < 0 or ca == cb:
            continue
        a, b = sorted((int(ca), int(cb)))
        key = (a, b)
        cand = candidates.setdefault(key, {'votes': 0, 'snap_sum': 0.0, 'consistency_sum': 0.0, 'proj_min': math.inf, 'proj_max': -math.inf})
        ga, gb = (centers[a], centers[b])
        gv = gb - ga
        gl = np.linalg.norm(gv)
        if gl <= 1e-12:
            continue
        gu = gv / gl
        lu = vec[idx] / max(length[idx], 1e-12)
        consistency = abs(float(np.dot(lu, gu)))
        if consistency < scheme.min_consistency:
            continue
        pts = lines[idx]
        proj = (pts - ga.reshape(1, 3)) @ gu / gl
        cand['votes'] += 1
        cand['snap_sum'] += float(line_snap[idx].mean())
        cand['consistency_sum'] += consistency
        cand['proj_min'] = min(cand['proj_min'], float(proj.min()))
        cand['proj_max'] = max(cand['proj_max'], float(proj.max()))
    if scheme.path_pair_max_len > 0 and scheme.path_support_radius > 0 and (edge_points.size != 0):
        center_dist = cdist(centers, centers)
        neighbor_sets = None
        if symmetric_knn and scheme.path_knn > 0:
            neighbor_sets = [set(np.argsort(center_dist[a])[1:scheme.path_knn + 1].tolist()) for a in range(len(centers))]
            for a in range(len(centers)):
                for b in tuple(neighbor_sets[a]):
                    neighbor_sets[b].add(a)
        for a in range(len(centers)):
            order = np.argsort(center_dist[a])
            if neighbor_sets is not None:
                order = sorted(neighbor_sets[a])
            elif scheme.path_knn > 0:
                order = order[1:scheme.path_knn + 1]
            else:
                order = order[1:]
            for b in order:
                if b <= a:
                    continue
                gl = float(center_dist[a, b])
                if gl < scheme.min_graph_len or gl > scheme.path_pair_max_len:
                    continue
                key = (a, int(b))
                if key in candidates and candidates[key]['votes'] >= scheme.min_votes:
                    continue
                support, edge_cov = point_segment_stats(edge_points, centers[a], centers[b], scheme.path_support_radius)
                if support < scheme.path_min_support or edge_cov < scheme.path_min_coverage:
                    continue
                cand = candidates.setdefault(key, {'votes': 0, 'snap_sum': 0.0, 'consistency_sum': 0.0, 'proj_min': math.inf, 'proj_max': -math.inf})
                cand['path_support'] = max(int(cand.get('path_support', 0)), support)
                cand['path_coverage'] = max(float(cand.get('path_coverage', 0.0)), edge_cov)
    scored = []
    for (a, b), cand in candidates.items():
        votes = cand['votes']
        has_path = 'path_support' in cand
        if votes < scheme.min_votes and (not has_path):
            continue
        ga, gb = (centers[a], centers[b])
        gl = float(np.linalg.norm(gb - ga))
        if gl < scheme.min_graph_len:
            continue
        snap = cand['snap_sum'] / max(votes, 1) if votes > 0 else 0.0
        if snap > scheme.max_snap:
            continue
        consistency = cand['consistency_sum'] / max(votes, 1) if votes > 0 else 1.0
        direct_coverage = max(0.0, min(1.0, cand['proj_max']) - max(0.0, cand['proj_min']))
        support, edge_coverage = point_segment_stats(edge_points, ga, gb, scheme.edge_support_radius)
        if has_path:
            support = max(support, int(cand.get('path_support', 0)))
            edge_coverage = max(edge_coverage, float(cand.get('path_coverage', 0.0)))
        coverage = max(direct_coverage, edge_coverage)
        if support < scheme.min_edge_support:
            continue
        support_norm = min(1.0, support / max(4.0, gl / max(scheme.edge_support_radius, 1e-06)))
        score = scheme.score_vote_weight * math.log1p(votes) + consistency + 0.5 * coverage + scheme.score_support_weight * support_norm - scheme.score_snap_weight * (snap / max(scheme.attach_eps, 1e-06)) + scheme.path_score_weight * float(has_path) * coverage
        scored.append({'edge': (a, b), 'score': score, 'votes': votes, 'snap': snap, 'consistency': consistency, 'support': support, 'length': gl})
    scored.sort(key=lambda item: item['score'], reverse=True)
    chosen = []
    degrees = np.zeros(len(centers), dtype=np.int64)
    directions_by_vertex = [[] for _ in range(len(centers))]
    cos_sep = math.cos(math.radians(scheme.angle_sep_deg))
    chosen_set = set()
    for item in scored:
        a, b = item['edge']
        if (a, b) in chosen_set:
            continue
        if degrees[a] >= scheme.max_degree or degrees[b] >= scheme.max_degree:
            continue
        va = centers[b] - centers[a]
        vb = -va
        norm = np.linalg.norm(va)
        if norm <= 1e-12:
            continue
        da = va / norm
        db = vb / norm
        if any((float(np.dot(da, old)) > cos_sep for old in directions_by_vertex[a])):
            continue
        if any((float(np.dot(db, old)) > cos_sep for old in directions_by_vertex[b])):
            continue
        chosen.append((a, b))
        chosen_set.add((a, b))
        degrees[a] += 1
        degrees[b] += 1
        directions_by_vertex[a].append(da)
        directions_by_vertex[b].append(db)
    if scheme.repair_degree > 0:
        repair_cos_sep = math.cos(math.radians(scheme.repair_angle_sep_deg or scheme.angle_sep_deg))
        for item in scored:
            if item['score'] < scheme.repair_min_score:
                continue
            a, b = item['edge']
            if (a, b) in chosen_set:
                continue
            if degrees[a] >= scheme.repair_degree and degrees[b] >= scheme.repair_degree:
                continue
            if degrees[a] >= scheme.max_degree or degrees[b] >= scheme.max_degree:
                continue
            va = centers[b] - centers[a]
            vb = -va
            norm = np.linalg.norm(va)
            if norm <= 1e-12:
                continue
            da = va / norm
            db = vb / norm
            if any((float(np.dot(da, old)) > repair_cos_sep for old in directions_by_vertex[a])):
                continue
            if any((float(np.dot(db, old)) > repair_cos_sep for old in directions_by_vertex[b])):
                continue
            chosen.append((a, b))
            chosen_set.add((a, b))
            degrees[a] += 1
            degrees[b] += 1
            directions_by_vertex[a].append(da)
            directions_by_vertex[b].append(db)
    if not chosen:
        out = np.empty((0, 2, 3), dtype=np.float64)
    else:
        out = np.asarray([[centers[a], centers[b]] for a, b in chosen], dtype=np.float64)
        if scheme.output_repeats > 1:
            out = np.repeat(out, scheme.output_repeats, axis=0)
    meta = {'num_clusters': int(len(centers)), 'num_candidates': int(len(scored)), 'num_chosen': int(len(chosen))}
    return (out, meta)

def kernel_2682(points, edge):
    v = edge[1] - edge[0]
    den = float(v @ v)
    t = np.clip((np.asarray(points) - edge[0]) @ v / max(den, 1e-12), 0.0, 1.0)
    return np.linalg.norm(np.asarray(points) - edge[0] - t[..., None] * v, axis=-1)

def kernel_2711(ev, graph):
    graph = np.asarray(graph, float).reshape(-1, 2, 3)
    n = len(graph)
    if not n:
        return np.empty((0, len(FEATURES)))
    x = old.feature_matrix(dict(ev, balanced=np.empty((0, 2, 3))), graph, np.zeros((n, 3)))[:, RAW_INDEX]
    verts, ix = np.unique(graph.reshape(-1, 3), axis=0, return_inverse=True)
    degs = np.bincount(ix, minlength=len(verts))
    ix = ix.reshape(-1, 2)
    raw = old.lines(ev['raw'])
    balanced = old.lines(ev['balanced'])
    extras = []
    for i, e in enumerate(graph):
        L = np.linalg.norm(e[1] - e[0])
        u = (e[1] - e[0]) / max(L, 1e-10)
        others = graph[np.arange(n) != i]
        query = e[0] + np.linspace(0, 1, 21)[:, None] * (e[1] - e[0])
        degree = degs[ix[i]]
        rawpair = float(np.min(old.pair_distance(e[None], raw))) if len(raw) else 5.0
        inherited = float(len(balanced) > 0 and np.min(old.pair_distance(e[None], balanced)) < 1e-07)
        if len(others):
            v = others[:, 1] - others[:, 0]
            length = np.linalg.norm(v, axis=1)
            direction = v / np.maximum(length[:, None], 1e-10)
            aligned = np.abs(direction @ u) >= np.cos(np.radians(5.0))
            D = old.distances(query, others)
            cover = float((D[:, aligned].min(1) <= 0.12).mean()) if aligned.any() else 0.0
            dm = np.minimum(D.min(1), 5).mean()
            pair = float(min(5, old.pair_distance(e[None], others).min()))
            end = old.nearest(e, others.reshape(-1, 3))
            projection = (others - e[0]) @ u
            lateral = np.linalg.norm(np.cross(others - e[0], u), axis=2)
            branch = ((np.abs(direction @ u) < np.cos(np.radians(20))) & (length >= 0.3) & ((projection > 0.05 * L) & (projection < 0.95 * L) & (lateral <= 0.15)).any(1)).sum()
        else:
            cover = 0.0
            dm = pair = 5.0
            end = np.array([5.0, 5.0])
            branch = 0
        extras.append([min(5, rawpair), pair, cover, dm, float(end.min()), float(end.max()), np.log1p(degree.min()), np.log1p(degree.max()), float((degree == 1).all()), inherited, np.log1p(branch)])
    return np.nan_to_num(np.c_[x, extras], nan=5.0, posinf=5.0, neginf=-5.0)

def kernel_2712(ev, lines):
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    u = (raw[:, 1] - raw[:, 0]) / np.maximum(length[:, None], 1e-10)
    pair = c.e.OLD.pair_distance(lines, raw)
    out = []
    for i, edge in enumerate(lines):
        L = np.linalg.norm(edge[1] - edge[0])
        direction = (edge[1] - edge[0]) / max(L, 1e-10)
        order = np.sort(pair[i])
        j = int(pair[i].argmin())
        aligned = abs(u @ direction) >= np.cos(np.radians(15))
        single = 0.0
        if aligned.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = c.e.OLD.distances(q, raw[aligned])
            single = float((D <= 0.2).mean(0).max())
        out.append([min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), np.log1p((pair[i] <= 0.35).sum()), np.log1p((pair[i] <= 1).sum()), float(np.log(max(length[j], 0.01) / max(L, 0.01))), single])
    return np.array(out).reshape(-1, len(FINITE_NAMES))

def kernel_2713(ev, lines):
    pc = np.asarray(ev['pc']).reshape(-1, 3)
    tree = cKDTree(pc[:, :2])
    out = []
    refit = []
    for i, edge in enumerate(lines):
        x, p = roof(edge, pc, tree)
        out.append(x)
        if p is not None:
            refit.append((i, p))
    return (np.array(out).reshape(-1, len(ROOF_NAMES)), refit)

def kernel_2715(bundle, evidence, current_graph):
    graph = provider.verify.canonical(current_graph)
    patches = [p for p in algorithm.proposals(evidence, graph) if p['family'] != 'junction_refit']
    if not patches:
        return (graph.copy(), dict(operations=[], GT_used=False, score_threshold=0.05, max_patches=3))
    X = np.c_[algorithm.feature_matrix(evidence, graph, patches), finite.features(evidence, graph, patches, algorithm)]
    value = bundle['model'].predict(X)
    removed = set()
    added = []
    operations = []
    for i in np.argsort(-value, kind='stable'):
        if value[i] <= 0.05:
            break
        patch = patches[int(i)]
        if removed & set(patch['remove']):
            continue
        lines = np.array(patch['add']).reshape(-1, 2, 3)
        if added and provider.OLD.pair_distance(lines, np.concatenate(added)).min() < 0.2:
            continue
        removed.update(patch['remove'])
        added.append(lines)
        operations.append(dict(index=int(i), score=float(value[i]), **patch))
        if len(operations) >= 3:
            break
    out = algorithm.apply(graph, dict(remove=sorted(removed), add=np.concatenate(added))) if operations else graph.copy()
    return (out, dict(operations=operations, GT_used=False, score_threshold=0.05, max_patches=3))

def kernel_2716(line):
    a = np.asarray(line).reshape(2, 3)
    return tuple(a.reshape(-1)) if tuple(a[0]) <= tuple(a[1]) else tuple(a[::-1].reshape(-1))

def kernel_2718(edge):
    x = np.asarray(edge).reshape(2, 3)
    if tuple(x[0]) > tuple(x[1]):
        x = x[::-1]
    return tuple(x.reshape(-1))

def kernel_2719(ev, graph):
    graph = e.verify.canonical(graph)
    v, ix, deg = degree(graph)
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    gl = np.linalg.norm(graph[:, 1] - graph[:, 0], axis=1)
    gu = (graph[:, 1] - graph[:, 0]) / np.maximum(gl[:, None], 1e-12)
    pool, stats = e.OLD.candidates(ev)
    patches = []
    seen = set()

    def support(edge, radius=0.25):
        L = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(L, 1e-12)
        select = (abs(ru @ u) >= np.cos(np.radians(12.0))) & (rl > 0.15)
        if not select.any():
            return (0.0, 5.0)
        query = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
        d = e.OLD.distances(query, raw[select]).min(1)
        return (float((d <= radius).mean()), float(np.minimum(d, 5).mean()))

    def add(family, remove, segments):
        segments = e.verify.canonical(np.asarray(segments).reshape(-1, 2, 3))
        remove = sorted(set(map(int, remove)))
        if not len(segments) or np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).min() < 0.15:
            return
        patch = dict(family=family, remove=remove, add=segments.tolist())
        key = (tuple(remove), tuple((linekey(x) for x in segments)))
        if key in seen:
            return
        pred = apply(graph, patch)
        if e.verify.keys(pred) == e.verify.keys(graph):
            return
        if len(pred) != len(graph) - len(remove) + len(segments):
            return
        seen.add(key)
        patches.append(patch)
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if abs(gu[i] @ gu[j]) < np.cos(np.radians(8)):
                continue
            ends = np.r_[graph[i], graph[j]]
            project = (ends - graph[i, 0]) @ gu[i]
            if np.linalg.norm(np.cross(ends - graph[i, 0], gu[i]), axis=1).max() > 0.18:
                continue
            pi = project[:2]
            pj = project[2:]
            gap = max(pi.min(), pj.min()) - min(pi.max(), pj.max())
            if gap > 0.6:
                continue
            new = ends[[int(project.argmin()), int(project.argmax())]]
            L = np.linalg.norm(new[1] - new[0])
            if L < max(gl[i], gl[j]) + 0.2:
                continue
            remaining = np.delete(graph, [i, j], axis=0).reshape(-1, 3)
            internal = [x for x in ends if not any((np.array_equal(x, y) for y in new))]
            if any((any((np.array_equal(x, y) for y in remaining)) for x in internal)):
                continue
            cov, dist = support(new)
            if cov >= 0.85:
                add('merge_fragments', [i, j], [new])
    for vi in np.flatnonzero((deg >= 2) & (deg <= 3)):
        incident = np.flatnonzero((ix == vi).any(1))
        axes = []
        for i in incident:
            lateral = np.linalg.norm(np.cross(raw - graph[i, 0], gu[i]), axis=2).mean(1)
            selected = np.flatnonzero((abs(ru @ gu[i]) >= np.cos(np.radians(12))) & (lateral <= 0.45) & (rl > 0.3))
            if not len(selected):
                axes = []
                break
            proximity = e.OLD.distances(v[vi:vi + 1], raw[selected])[0]
            selected = selected[np.argsort(proximity + lateral[selected], kind='stable')[:4]]
            previous = len(axes)
            for j in selected:
                if e.OLD.distances(v[vi:vi + 1], raw[j:j + 1])[0, 0] <= 0.8:
                    axes.append((ru[j], raw[j].mean(0)))
            if len(axes) == previous:
                axes = []
                break
        if len(axes) < 2:
            continue
        A = sum((np.eye(3) - np.outer(u, u) for u, p in axes))
        b = sum(((np.eye(3) - np.outer(u, u)) @ p for u, p in axes))
        if np.linalg.cond(A) > 10000.0:
            continue
        point = np.linalg.solve(A + 0.05 * np.eye(3), b + 0.05 * v[vi])
        shift = np.linalg.norm(point - v[vi])
        if not 0.12 <= shift <= 0.9:
            continue
        segments = graph[incident].copy()
        for k, i in enumerate(incident):
            segments[k, ix[i] == vi] = point
        if min((support(x)[0] for x in segments)) >= 0.7:
            add('junction_refit', incident, segments)
    for i, edge in enumerate(graph):
        if gl[i] < 0.6:
            continue
        projection = (v - edge[0]) @ gu[i]
        lateral = np.linalg.norm(np.cross(v - edge[0], gu[i]), axis=1)
        eligible = np.flatnonzero((projection > 0.2) & (projection < gl[i] - 0.2) & (lateral <= 0.18))
        for vi in eligible[np.argsort(lateral[eligible], kind='stable')[:3]]:
            incident = np.flatnonzero((ix == vi).any(1))
            if i in incident or not len(incident) or (not (abs(gu[incident] @ gu[i]) < np.cos(np.radians(20))).any()):
                continue
            segs = np.array([[edge[0], v[vi]], [v[vi], edge[1]]])
            if min((support(x)[0] for x in segs)) >= 0.8:
                add('branch_split', [i], segs)
    rawends = raw.reshape(-1, 3)
    endpoint_owner = np.repeat(np.arange(len(raw)), 2)
    for i, edge in enumerate(graph):
        if gl[i] < 0.8:
            continue
        t = (rawends - edge[0]) @ gu[i]
        lat = np.linalg.norm(np.cross(rawends - edge[0], gu[i]), axis=1)
        eligible = np.flatnonzero((t > 0.25) & (t < gl[i] - 0.25) & (lat >= 0.12) & (lat <= 0.6) & (abs(ru[endpoint_owner] @ gu[i]) < np.cos(np.radians(12))))
        chosen = []
        for j in eligible[np.argsort(lat[eligible], kind='stable')]:
            point = rawends[j]
            if chosen and np.linalg.norm(np.array(chosen) - point, axis=1).min() < 0.25:
                continue
            segs = np.array([[edge[0], point], [point, edge[1]]])
            vec = segs[:, 1] - segs[:, 0]
            length = np.linalg.norm(vec, axis=1)
            if length.min() < 0.25:
                continue
            if min((support(x, 0.2)[0] for x in segs)) < 0.8:
                continue
            add('raw_kink', [i], segs)
            chosen.append(point)
            if len(chosen) >= 2:
                break
    if len(pool) and len(v):
        tree = cKDTree(v)
        distance, nearest = tree.query(pool.reshape(-1, 3))
        distance = distance.reshape(-1, 2)
        nearest = nearest.reshape(-1, 2)
        plen = np.linalg.norm(pool[:, 1] - pool[:, 0], axis=1)
        poolx = e.OLD.feature_matrix(dict(ev, balanced=graph), pool, stats)
        qualified = (poolx[:, 4] >= 0.7) & (poolx[:, 15] <= 0.7) & (plen >= 0.2)
        selected = []
        bins = np.searchsorted([1.0, 2.0, 5.0], plen, side='right')
        for b in range(4):
            js = np.flatnonzero(qualified & (bins == b))
            selected.extend(js[np.argsort(-stats[js, 2], kind='stable')[:8]].tolist())
        for j in selected:
            seg = pool[j].copy()
            for k in range(2):
                if distance[j, k] <= 0.18:
                    seg[k] = v[nearest[j, k]]
            add('add_single', [], [seg])
        pairs = []
        for i in selected:
            for j in selected:
                if j <= i:
                    continue
                d = np.linalg.norm(pool[i, :, None] - pool[j, None, :], axis=2)
                a, b = np.unravel_index(d.argmin(), d.shape)
                if d[a, b] > 0.25:
                    continue
                ai = 1 - a
                bj = 1 - b
                if max(distance[i, ai], distance[j, bj]) > 0.35 or nearest[i, ai] == nearest[j, bj]:
                    continue
                point = (pool[i, a] + pool[j, b]) / 2
                A = v[nearest[i, ai]]
                B = v[nearest[j, bj]]
                segments = np.array([[A, point], [point, B]])
                deltas = segments[:, 1] - segments[:, 0]
                lens = np.linalg.norm(deltas, axis=1)
                if lens.min() < 0.25:
                    continue
                cos = abs(deltas[0] @ deltas[1] / (lens[0] * lens[1]))
                if cos > np.cos(np.radians(12)):
                    continue
                if min((support(x, 0.2)[0] for x in segments)) < 0.8:
                    continue
                pairs.append((float(d[a, b] + distance[i, ai] + distance[j, bj]), segments))
        for _, segs in sorted(pairs, key=lambda x: x[0])[:16]:
            add('add_connected_pair', [], segs)
    counts = {k: 0 for k in FAMILIES}
    result = []
    for patch in patches:
        family = patch['family']
        if counts[family] >= 24:
            continue
        counts[family] += 1
        result.append(patch)
    return result

def kernel_2720(graph):
    v, ix = np.unique(graph.reshape(-1, 3), axis=0, return_inverse=True)
    d = np.bincount(ix, minlength=len(v))
    return (v, ix.reshape(-1, 2), d)

def kernel_2721(graph, patch):
    keep = np.ones(len(graph), bool)
    keep[patch['remove']] = False
    return e.verify.canonical(np.r_[graph[keep], np.asarray(patch['add']).reshape(-1, 2, 3)])

def kernel_2722(evidence, mode=None):
    if mode not in ('old', 'coverage'):
        raise ValueError('pool mode must be old or coverage')
    raw = _lines(evidence['raw'])
    base = _lines(evidence['balanced'])
    parser = _lines(evidence['parser'])
    scale = float(evidence['scale'])
    initial_raw = len(raw)
    good = np.isfinite(raw).all((1, 2)) & (np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1) > 0.15)
    raw = raw[good]
    dm = _pair(raw, raw)
    old_trace = _old_trace(raw, base, parser, dm)
    if mode == 'old':
        c, s = _old().candidates(evidence)
        assert len(c) == old_trace['post_dedup']
        return (c, s, dict(mode=mode, raw_input=initial_raw, raw_after_length_gate=len(raw), old=old_trace, output=len(c), capacity_limit='historical raw96 plus parser/unions', uses_labels=False))
    counts = (dm < 0.35).sum(1)
    used = np.zeros(len(raw), bool)
    representatives = []
    rstats = []
    medoid_changed = 0
    for i in np.argsort(-counts, kind='stable'):
        if used[i]:
            continue
        eligible = (dm[i] < 0.45) & ~used & _same_line(raw[i], raw)
        group = np.flatnonzero(eligible)
        group = group if len(group) else np.array([i])
        used[group] = True
        local = dm[np.ix_(group, group)]
        medoid = int(group[np.argmin(local.sum(1))])
        medoid_changed += medoid != i
        representatives.append(raw[medoid])
        rstats.append([0.0, 0.0, float(len(group))])
    representatives = _lines(representatives)
    rstats = np.asarray(rstats, float).reshape(-1, 3)
    representatives, rstats, raw_base_removed, raw_dup_removed = _dedup(representatives, rstats, base)
    raw_ids, raw_capacity = _spread_select(representatives, rstats, 160, scale)
    selected_raw = representatives[raw_ids]
    selected_stats = rstats[raw_ids]
    extras = []
    estats = []
    parser_missing = 0
    union_count = 0
    for edge in parser:
        if np.isfinite(edge).all() and np.linalg.norm(edge[1] - edge[0]) > 0.15:
            close = _pair(edge[None], base)[0] < 0.35 if len(base) else np.zeros(0, bool)
            if not np.any(close & _same_line(edge, base, angle_deg=2.5, lateral_m=0.04)):
                extras.append(edge)
                estats.append([1.0, 0.0, 1.0])
                parser_missing += 1
    u, length = _directions(selected_raw)
    for i, edge in enumerate(selected_raw):
        aligned = _same_line(edge, selected_raw, angle_deg=3.0, lateral_m=0.055)
        projection = (selected_raw - edge[0]) @ u[i]
        lo = projection.min(1)
        hi = projection.max(1)
        js = np.flatnonzero(aligned & (lo <= length[i] + 0.15) & (hi >= -0.15))
        js = js[js > i]
        if not len(js):
            continue
        j = int(js[np.argmax(selected_stats[js, 2])])
        lower = min(0.0, float(lo[j]))
        upper = max(float(length[i]), float(hi[j]))
        span = upper - lower
        if span > max(length[i], length[j]) + 0.15 and span <= 1.5 * max(length[i], length[j]) + 0.15:
            extras.append(np.stack([edge[0] + lower * u[i], edge[0] + upper * u[i]]))
            estats.append([0.0, 1.0, float(selected_stats[i, 2] + selected_stats[j, 2])])
            union_count += 1
    extras = _lines(extras)
    estats = np.asarray(estats, float).reshape(-1, 3)
    extras, estats, extra_base_removed, extra_dup_removed = _dedup(extras, estats, base)
    extra_limit = min(32, len(extras))
    extra_ids, extra_capacity = _spread_select(extras, estats, extra_limit, scale)
    raw_limit = 192 - len(extra_ids)
    raw_ids, raw_final_capacity = _spread_select(representatives, rstats, raw_limit, scale)
    pool = np.concatenate([representatives[raw_ids], extras[extra_ids]])
    stats = np.concatenate([rstats[raw_ids], estats[extra_ids]])
    pool, stats, final_base_removed, final_dup_removed = _dedup(pool, stats, base)
    assert len(pool) <= 192 and pool.shape == (len(stats), 2, 3) and np.isfinite(pool).all()
    meta = dict(mode=mode, raw_input=initial_raw, raw_after_length_gate=len(raw), old=old_trace, new=dict(strict_geometric_groups=int(used.sum() > 0 and len(rstats) + raw_base_removed + raw_dup_removed), medoid_changed=medoid_changed, representatives_after_baseline_gate=len(representatives), raw_baseline_removed=raw_base_removed, raw_representative_dedup_removed=raw_dup_removed, raw_union_seed_count=len(selected_raw), raw_capacity=raw_final_capacity, parser_missing_candidates=parser_missing, finite_pair_unions=union_count, extra_baseline_removed=extra_base_removed, extra_representative_dedup_removed=extra_dup_removed, extra_capacity=extra_capacity, final_cross_source_dedup_removed=final_dup_removed, output=len(pool), source_raw=int((stats[:, 0] == 0).sum() - (stats[:, 1] == 1).sum()), source_parser=int((stats[:, 0] == 1).sum()), source_union=int((stats[:, 1] == 1).sum())), max_candidates=192, stats_columns=['source_parser', 'source_union', 'geometric_response_count'], uses_labels=False)
    return (pool, stats, meta)

def kernel_2723(pool, stats, limit, scale):
    if len(pool) <= limit:
        return (np.arange(len(pool), dtype=int), dict(bucket_count=0, capacity_removed=0))
    u, length = _directions(pool)
    major = np.argmax(abs(u), axis=1)
    sign = np.where(u[np.arange(len(u)), major] < 0, -1.0, 1.0)
    u = u * sign[:, None]
    grid = float(np.clip(0.2 * scale, 1.5, 4.0))
    cells = np.floor(pool.mean(1) / grid).astype(int)
    az = np.floor((np.arctan2(u[:, 1], u[:, 0]) + np.pi) / (np.pi / 4)).astype(int)
    el = np.floor((np.arcsin(np.clip(u[:, 2], -1, 1)) + np.pi / 2) / (np.pi / 6)).astype(int)
    buckets = {}
    for i, key in enumerate(zip(cells[:, 0], cells[:, 1], cells[:, 2], az, el)):
        buckets.setdefault(tuple((int(v) for v in key)), []).append(i)
    weight = np.log1p(stats[:, 2]) + 0.1 * np.log1p(length)
    for key, ids in buckets.items():
        ids.sort(key=lambda i: (-weight[i], i))
    keys = sorted(buckets, key=lambda key: (-weight[buckets[key][0]], key))
    selected = []
    round_index = 0
    while len(selected) < limit:
        took = False
        for key in keys:
            if round_index < len(buckets[key]):
                selected.append(buckets[key][round_index])
                took = True
            if len(selected) == limit:
                break
        if not took:
            break
        round_index += 1
    return (np.array(selected, int), dict(bucket_count=len(buckets), capacity_removed=len(pool) - len(selected), grid_m=grid))

def kernel_2724(graph, patches, values, threshold, budget):
    removed = set()
    added = []
    chosen = []
    for i in np.argsort(-values, kind='stable'):
        if values[i] <= threshold:
            break
        patch = patches[int(i)]
        if removed & set(patch['remove']):
            continue
        new = np.array(patch['add'])
        if added and c.e.OLD.pair_distance(new, np.concatenate(added)).min() < 0.2:
            continue
        removed.update(patch['remove'])
        added.append(new)
        chosen.append(int(i))
        if len(chosen) >= budget:
            break
    output = ALG.apply(graph, dict(remove=sorted(removed), add=np.concatenate(added))) if chosen else graph.copy()
    return (output, chosen)

def kernel_2725(bundle, graph, X):
    scores = bundle['model'].predict_proba(X)[:, list(bundle['model'].classes_).index(1)] if len(X) else np.array([])
    chosen = [int(i) for i in np.argsort(-scores, kind='stable') if scores[i] >= 0.8][:8]
    keep = np.ones(len(graph), bool)
    keep[chosen] = False
    return (c.e.verify.canonical(graph[keep]), dict(deleted=chosen, scores=scores[chosen].tolist()))

def kernel_4110(points, a, b, radius, bins=None):
    if points.size == 0:
        return (0, 0.0)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return (0, 0.0)
    t = (points - a.reshape(1, 3)) @ ab / denom
    mask = (t >= 0.0) & (t <= 1.0)
    if not np.any(mask):
        return (0, 0.0)
    proj = a.reshape(1, 3) + t[mask].reshape(-1, 1) * ab.reshape(1, 3)
    d = np.linalg.norm(points[mask] - proj, axis=1)
    inlier_t = t[mask][d <= radius]
    count = int(len(inlier_t))
    if count == 0:
        return (0, 0.0)
    bin_idx = np.clip((inlier_t * bins).astype(np.int64), 0, bins - 1)
    coverage = len(set(bin_idx.tolist())) / float(bins)
    return (count, coverage)

def kernel_4112(points, labels, method=None):
    centers = []
    for label in sorted(set(labels) - {-1}):
        pts = points[labels == label]
        if method == 'mean':
            centers.append(np.mean(pts, axis=0))
        else:
            centers.append(np.median(pts, axis=0))
    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64)

def kernel_4129(edge, pc, tree):
    vector = edge[1] - edge[0]
    xylen = np.linalg.norm(vector[:2])
    out = np.zeros(len(ROOF_NAMES))
    out[6:8] = 1.0
    out[9:15] = 2.0
    if xylen < 0.25:
        return (out, None)
    center = edge.mean(0)
    u = vector[:2] / xylen
    n = np.array([-u[1], u[0]])
    near = pc[tree.query_ball_point(center[:2], xylen / 2 + 1.5)]
    delta = near[:, :2] - edge[0, :2]
    along = delta @ u
    lat = delta @ n
    within = (along >= -0.2) & (along <= xylen + 0.2) & (abs(lat) >= 0.15) & (abs(lat) <= 1.2)
    side = [within & (lat < 0), within & (lat > 0)]
    counts = np.array([x.sum() for x in side])
    fits = [plane(near[s], center) for s in side]
    valid = np.array([x is not None for x in fits])
    cov = np.array([len(np.unique(np.clip((along[s] / xylen * 6).astype(int), 0, 5))) / 6 for s in side])
    res = np.array([min(1.0, x['mad']) if x else 1.0 for x in fits])
    errors = []
    slopes = []
    for f in fits:
        if f:
            z = (edge[:, :2] - center[:2]) @ f['v'][:2] + f['v'][2] + center[2]
            errors.append(min(2.0, float(np.abs(z - edge[:, 2]).mean())))
            slopes.append(min(2.0, float(abs(f['v'][:2] @ u - vector[2] / xylen))))
        else:
            errors.append(2.0)
            slopes.append(2.0)
    out[:8] = [1.0, np.log1p(counts.min() / max(xylen, 0.5)), np.log1p(counts.max() / max(xylen, 0.5)), counts.min() / max(1, counts.max()), valid.min(), valid.max(), res.min(), res.max()]
    out[10:14] = [min(errors), max(errors), min(slopes), max(slopes)]
    out[15:] = [cov.min(), cov.max(), min(5.0, float(np.ptp(near[within, 2]))) if within.any() else 0.0]
    refit = None
    if valid.all():
        a, b = fits
        angle = float(np.arccos(np.clip(a['normal'] @ b['normal'], -1.0, 1.0)))
        out[8] = angle
        out[9] = min(2.0, abs(a['v'][2] - b['v'][2]))
        direction = np.cross(a['normal'], b['normal'])
        dl = np.linalg.norm(direction)
        if dl > 1e-05:
            direction /= dl
            old = vector / max(np.linalg.norm(vector), 1e-10)
            turn = float(np.arccos(np.clip(abs(direction @ old), 0, 1)))
            out[14] = turn
            if direction @ old < 0:
                direction = -direction
            if np.radians(8) <= angle <= np.radians(80) and turn <= np.radians(12) and (res.max() <= 0.15) and (cov.min() >= 0.5):
                A = np.array([[-a['v'][0], -a['v'][1], 1.0], [-b['v'][0], -b['v'][1], 1.0], old])
                rhs = np.array([a['v'][2], b['v'][2], 0.0])
                if np.linalg.cond(A) < 100:
                    point = np.linalg.solve(A, rhs) + center
                    proj = ((edge - point) @ direction)[:, None]
                    proposal = point + proj * direction
                    displacement = np.linalg.norm(proposal - edge, axis=1).max()
                    if 0.08 <= displacement <= 0.6 and np.linalg.norm(proposal[1] - proposal[0]) > 0.25:
                        refit = proposal
    return (out, refit)

def kernel_4132(lines):
    x = np.asarray(lines, float).reshape(-1, 2, 3)
    if not len(x):
        return x
    v, ix = np.unique(x.reshape(-1, 3), axis=0, return_inverse=True)
    e = np.unique(np.sort(ix.reshape(-1, 2), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return v[e]

def kernel_4133(ev, graph, patches):
    graph = e.verify.canonical(graph)
    v, ix, d = degree(graph)
    base_leaf = int((d == 1).sum())
    base_branch = int((d >= 3).sum())
    rows = []
    for patch in patches:
        remove = patch['remove']
        added = np.array(patch['add'])
        keep = np.ones(len(graph), bool)
        keep[remove] = False
        context = dict(ev, balanced=graph[keep])
        newx = e.OLD.feature_matrix(context, added, np.zeros((len(added), 3)))
        oldx = e.OLD.feature_matrix(context, graph[remove], np.zeros((len(remove), 3))) if remove else np.zeros((1, len(e.OLD.FEATURES)))
        pred = apply(graph, patch)
        nv, _, nd = degree(pred)
        oldlength = float(np.linalg.norm(graph[remove, 1] - graph[remove, 0], axis=1).sum()) if remove else 0.0
        newlength = float(np.linalg.norm(added[:, 1] - added[:, 0], axis=1).sum())
        degrees = d[ix[remove]].reshape(-1) if remove else np.array([0.0])
        pair = e.OLD.pair_distance(added, graph[remove]) if remove else np.zeros((len(added), 1))
        extra = [len(remove), len(added), len(pred) - len(graph), np.log1p(oldlength), np.log1p(newlength), newlength / max(oldlength, 0.15), len(nv) - len(v), int((nd == 1).sum()) - base_leaf, int((nd >= 3).sum()) - base_branch, float(degrees.min()), float(degrees.max()), float(newx[:, 4].min()), float(newx[:, 7].max()), float(pair.min(1).max())]
        rows.append(np.r_[newx.mean(0), newx.min(0), oldx.mean(0), extra, [float(patch['family'] == k) for k in FAMILIES]])
    X = np.array(rows).reshape(-1, len(FEATURES))
    assert np.isfinite(X).all()
    return X

def kernel_4134(ev, graph, patches, alg):
    e = alg.e
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    union = e.verify.canonical(np.concatenate([graph] + [np.array(p['add']).reshape(-1, 2, 3) for p in patches])) if patches else graph
    pair = e.OLD.pair_distance(union, raw)
    cache = {}
    for i, edge in enumerate(union):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-12)
        closest = int(pair[i].argmin()) if len(raw) else 0
        order = np.sort(pair[i]) if len(raw) else np.array([5.0])
        select = abs(ru @ u) >= np.cos(np.radians(15))
        if select.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = e.OLD.distances(q, raw[select])
            single = float((D <= 0.2).mean(0).max())
        else:
            single = 0.0
        cache[alg.linekey(edge)] = [min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), float(np.log1p((pair[i] <= 0.35).sum())), float(np.log1p((pair[i] <= 1.0).sum())), float(np.log(max(rl[closest], 0.01) / max(length, 0.01))) if len(raw) else 0.0, single]
    result = []
    for patch in patches:
        added = np.array(patch['add'])
        remove = patch['remove']
        newx = np.array([cache[alg.linekey(x)] for x in added])
        oldx = np.array([cache[alg.linekey(x)] for x in graph[remove]]) if remove else np.zeros((1, len(BASE_NAMES)))
        vertices, ii = np.unique(added.reshape(-1, 3), axis=0, return_inverse=True)
        deg = np.bincount(ii, minlength=len(vertices))
        junction = vertices[deg >= 2]
        if not len(junction):
            junction = vertices
        old = graph[remove] if remove else added
        longest = int(np.linalg.norm(old[:, 1] - old[:, 0], axis=1).argmax())
        reference = old[longest]
        u = reference[1] - reference[0]
        u = u / max(np.linalg.norm(u), 1e-12)
        aligned = (abs(ru @ u) >= np.cos(np.radians(12))) & (np.linalg.norm(np.cross(raw - reference[0], u), axis=2).max(1) <= 0.35)
        projection = (raw - reference[0]) @ u
        lo = projection.min(1)
        hi = projection.max(1)
        keep = np.ones(len(graph), bool)
        keep[remove] = False
        other = graph[keep]
        if len(other):
            ou = (other[:, 1] - other[:, 0]) / np.maximum(np.linalg.norm(other[:, 1] - other[:, 0], axis=1)[:, None], 1e-12)
            otherbranch = other[abs(ou @ u) < np.cos(np.radians(20))]
        else:
            otherbranch = other
        jj = []
        for point in junction:
            end = e.OLD.nearest(point[None], raw.reshape(-1, 3))[0]
            distance = e.OLD.distances(point[None], raw)[0]
            branch = (abs(ru @ u) < np.cos(np.radians(20))) & (distance <= 0.2) & (rl > 0.3)
            t = (point - reference[0]) @ u
            endnear = aligned & (np.minimum(abs(lo - t), abs(hi - t)) <= 0.25)
            crossing = aligned & (lo < t - 0.35) & (hi > t + 0.35)
            left = aligned & (abs(hi - t) <= 0.25) & (lo < t - 0.3)
            right = aligned & (abs(lo - t) <= 0.25) & (hi > t + 0.3)
            gd = e.OLD.distances(point[None], otherbranch)
            jj.append([float(end), float(np.log1p(branch.sum())), float(np.log1p(np.median(rl[branch]))) if branch.any() else 0.0, float(np.log1p(endnear.sum())), float(np.log1p(crossing.sum())), float(np.log1p(min(left.sum(), right.sum()))), float(endnear.sum() / (endnear.sum() + crossing.sum() + 1)), float(e.OLD.nearest(point[None], ev['ep'])[0]), float(e.OLD.nearest(point[None], ev['pc'])[0]), float(min(5.0, gd.min())) if gd.size else 5.0])
        result.append(np.r_[newx.mean(0), newx.max(0), oldx.mean(0), np.mean(jj, axis=0)])
    X = np.array(result).reshape(-1, len(FEATURES))
    assert np.isfinite(X).all()
    return X

def kernel_4136(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_4138(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_4139(evidence, candidate, stats):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    ep = np.asarray(evidence['ep']).reshape(-1, 3)
    pc = np.asarray(evidence['pc']).reshape(-1, 3)
    scale = float(evidence['scale'])
    corners = base.reshape(-1, 3)
    ab = raw[:, 1] - raw[:, 0]
    rl = np.linalg.norm(ab, axis=1)
    ru = ab / np.maximum(rl[:, None], 1e-09)
    ep_tree = cKDTree(ep) if len(ep) else None
    pc_tree = cKDTree(pc) if len(pc) else None
    rows = []
    for edge, s in zip(candidate, stats):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-09)
        query = edge[0] + np.linspace(0, 1, 12)[:, None] * (edge[1] - edge[0])
        compatible = abs(ru @ u) > 0.966
        d = distances(query, raw[compatible])
        minimum = d.min(1) if d.shape[1] else np.full(12, 5.0)
        support = (d < 0.35).sum(1) if d.shape[1] else np.zeros(12)
        endpoint = nearest(edge, raw.reshape(-1, 3))
        ed = np.minimum(ep_tree.query(query)[0], 5) if ep_tree else np.full(12, 5.0)
        pd = np.minimum(pc_tree.query(query)[0], 5) if pc_tree else np.full(12, 5.0)
        graph = nearest(edge, corners)
        bd = distances(query, base)
        bm = bd.min(1) if bd.shape[1] else np.full(12, 5.0)
        rows.append([np.log1p(length), s[0], s[1], np.log1p(s[2]), (minimum < 0.18).mean(), (minimum < 0.35).mean(), np.log1p(support.min()), minimum.clip(0, 5).mean(), endpoint.mean(), (ed < 0.25).mean(), ed.mean(), (pd < 0.35).mean(), graph.mean(), graph.max(), bm.clip(0, 5).mean(), (bm < 0.35).mean(), pair_distance(edge[None], base).min() if len(base) else 5.0, abs(u[2]), np.log1p(scale)])
    return np.nan_to_num(np.asarray(rows).reshape(-1, len(FEATURES)), nan=5, posinf=5)

def kernel_4140(x):
    return {tuple(e) for e in np.round(canonical(x).reshape(-1, 6), 7)}

def kernel_4141(points, segs):
    if not len(segs):
        return np.full((len(points), 0), np.inf)
    ab = segs[:, 1] - segs[:, 0]
    d = (ab * ab).sum(1)
    delta = points[:, None] - segs[None, :, 0]
    t = np.divide((delta * ab[None]).sum(2), d[None], out=np.zeros((len(points), len(segs))), where=d[None] > 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * ab[None], axis=2)

def kernel_4142():
    return _OLD

def kernel_4143(a, b):
    result = np.empty((len(a), len(b)))
    for start in range(0, len(a), 128):
        result[start:start + 128] = _old().pair_distance(a[start:start + 128], b)
    return result

def kernel_4144(value):
    return np.asarray(value, dtype=np.float64).reshape(-1, 2, 3)

def kernel_4145(pool, stats, base):
    chosen = []
    selected_stats = []
    base_removed = 0
    candidate_removed = 0
    for edge, s in zip(pool, stats):
        if len(base):
            close = _pair(edge[None], base)[0] < 0.35
            if np.any(close & _same_line(edge, base, angle_deg=2.5, lateral_m=0.04)):
                base_removed += 1
                continue
        if chosen:
            existing = np.asarray(chosen)
            close = _pair(edge[None], existing)[0] < 0.2
            if np.any(close & _same_line(edge, existing, angle_deg=2.5, lateral_m=0.04)):
                candidate_removed += 1
                continue
        chosen.append(edge)
        selected_stats.append(s)
    return (_lines(chosen), np.asarray(selected_stats, float).reshape(-1, 3), base_removed, candidate_removed)

def kernel_4146(lines):
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    return ((lines[:, 1] - lines[:, 0]) / np.maximum(lengths[:, None], 1e-12), lengths)

def kernel_4147(reference, others, angle_deg=None, lateral_m=None):
    if not len(others):
        return np.zeros(0, bool)
    u, L = _directions(reference.reshape(1, 2, 3))
    v, _ = _directions(others)
    u = u[0]
    lateral = np.linalg.norm(np.cross(others - reference[0], u), axis=-1).max(1)
    reverse = np.linalg.norm(np.cross(reference[None] - others[:, None, 0], v[:, None]), axis=-1).max(1)
    return (abs(v @ u) >= np.cos(np.deg2rad(angle_deg))) & (lateral <= lateral_m) & (reverse <= lateral_m)

def kernel_4148(raw, base, parser, dm):
    if not len(raw):
        return dict(raw_group_total=0, raw_groups_selected=0, raw_groups_capacity_dropped=0, raw_responses_not_represented_at96=0, parser_missing_candidates=0, union_candidates=0, pre_baseline_gate=0, baseline_gate_removed=0, post_baseline_gate=0, representative_dedup_removed=0, post_dedup=0)
    counts = (dm < 0.35).sum(1)
    used = np.zeros(len(raw), bool)
    groups = []
    indices = []
    for i in np.argsort(-counts, kind='stable'):
        if used[i]:
            continue
        group = np.flatnonzero((dm[i] < 0.45) & ~used)
        used[group] = True
        groups.append(group)
        indices.append(int(i))
    selected = indices[:96]
    original = raw[selected]
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    u = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-12)
    missing = [edge for edge in parser if not len(base) or _pair(edge[None], base).min() > 0.25]
    proposed = list(original) + missing
    union_count = 0
    for i, edge in enumerate(original):
        projected = (original - edge[0]) @ u[i]
        lo = projected.min(1)
        hi = projected.max(1)
        mask = (abs(u @ u[i]) > 0.99) & (np.linalg.norm(np.cross(original - edge[0], u[i]), axis=-1).max(1) < 0.18) & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                proposed.append(np.stack([edge[0] + lower * u[i], edge[0] + upper * u[i]]))
                union_count += 1
    proposed = np.asarray(proposed).reshape(-1, 2, 3)
    keep = _pair(proposed, base).min(1) > 0.35 if len(base) else np.ones(len(proposed), bool)
    remaining = proposed[keep]
    chosen = []
    for edge in remaining:
        if chosen and _pair(edge[None], np.asarray(chosen)).min() < 0.2:
            continue
        chosen.append(edge)
    return dict(raw_group_total=len(groups), raw_groups_selected=min(96, len(groups)), raw_groups_capacity_dropped=max(0, len(groups) - 96), raw_responses_not_represented_at96=int(len(raw) - sum((len(g) for g in groups[:96]))), parser_missing_candidates=len(missing), union_candidates=union_count, pre_baseline_gate=len(proposed), baseline_gate_removed=int((~keep).sum()), post_baseline_gate=len(remaining), representative_dedup_removed=len(remaining) - len(chosen), post_dedup=len(chosen))

def kernel_4192(points, center):
    if len(points) < 6:
        return None
    local = points - center
    xy = local[:, :2]
    A = np.c_[xy, np.ones(len(xy))]
    if np.linalg.cond(A) > 100:
        return None
    v = np.linalg.lstsq(A, local[:, 2], rcond=None)[0]
    for _ in range(2):
        error = abs(A @ v - local[:, 2])
        cut = max(0.08, float(np.quantile(error, 0.75)))
        good = error <= cut
        if good.sum() < 6 or np.linalg.cond(A[good]) > 100:
            break
        v = np.linalg.lstsq(A[good], local[good, 2], rcond=None)[0]
    error = abs(A @ v - local[:, 2])
    mad = float(np.median(error))
    normal = np.array([-v[0], -v[1], 1.0])
    normal /= np.linalg.norm(normal)
    return dict(v=v, mad=mad, normal=normal)

def kernel_4197(points, cloud, empty=None):
    return np.minimum(cKDTree(cloud).query(points)[0], 5.0) if len(cloud) else np.full(len(points), empty)

def kernel_4198(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_4314(points, cloud, empty=None):
    return np.minimum(cKDTree(cloud).query(points)[0], 5.0) if len(cloud) else np.full(len(points), empty)

def kernel_4315(points, segs):
    if not len(segs):
        return np.full((len(points), 0), np.inf)
    ab = segs[:, 1] - segs[:, 0]
    d = (ab * ab).sum(1)
    delta = points[:, None] - segs[None, :, 0]
    t = np.divide((delta * ab[None]).sum(2), d[None], out=np.zeros((len(points), len(segs))), where=d[None] > 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * ab[None], axis=2)

def kernel_4316(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_4317(evidence, candidate, stats):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    ep = np.asarray(evidence['ep']).reshape(-1, 3)
    pc = np.asarray(evidence['pc']).reshape(-1, 3)
    scale = float(evidence['scale'])
    corners = base.reshape(-1, 3)
    ab = raw[:, 1] - raw[:, 0]
    rl = np.linalg.norm(ab, axis=1)
    ru = ab / np.maximum(rl[:, None], 1e-09)
    ep_tree = cKDTree(ep) if len(ep) else None
    pc_tree = cKDTree(pc) if len(pc) else None
    rows = []
    for edge, s in zip(candidate, stats):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-09)
        query = edge[0] + np.linspace(0, 1, 12)[:, None] * (edge[1] - edge[0])
        compatible = abs(ru @ u) > 0.966
        d = distances(query, raw[compatible])
        minimum = d.min(1) if d.shape[1] else np.full(12, 5.0)
        support = (d < 0.35).sum(1) if d.shape[1] else np.zeros(12)
        endpoint = nearest(edge, raw.reshape(-1, 3))
        ed = np.minimum(ep_tree.query(query)[0], 5) if ep_tree else np.full(12, 5.0)
        pd = np.minimum(pc_tree.query(query)[0], 5) if pc_tree else np.full(12, 5.0)
        graph = nearest(edge, corners)
        bd = distances(query, base)
        bm = bd.min(1) if bd.shape[1] else np.full(12, 5.0)
        rows.append([np.log1p(length), s[0], s[1], np.log1p(s[2]), (minimum < 0.18).mean(), (minimum < 0.35).mean(), np.log1p(support.min()), minimum.clip(0, 5).mean(), endpoint.mean(), (ed < 0.25).mean(), ed.mean(), (pd < 0.35).mean(), graph.mean(), graph.max(), bm.clip(0, 5).mean(), (bm < 0.35).mean(), pair_distance(edge[None], base).min() if len(base) else 5.0, abs(u[2]), np.log1p(scale)])
    return np.nan_to_num(np.asarray(rows).reshape(-1, len(FEATURES)), nan=5, posinf=5)

def kernel_4318(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_4319(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_4326(lines):
    if not len(lines):
        return (np.empty((0, 3)), np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64))
    vertices, inv = np.unique(lines.reshape(-1, 3), axis=0, return_inverse=True)
    edges = np.sort(inv.reshape(-1, 2), axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    degree = np.bincount(edges.reshape(-1), minlength=len(vertices))
    return (vertices, edges, degree)

def kernel_4328(raw_world, edge_world, pc_world, stack):
    pc_world = np.round(np.asarray(pc_world, np.float64).reshape(-1, 3), 8)
    raw_world = np.round(np.asarray(raw_world, np.float64).reshape(-1, 2, 3), 8)
    edge_world = np.round(np.asarray(edge_world, np.float64).reshape(-1, 3), 8)
    center = pc_world.mean(0)
    scale = float(np.linalg.norm(pc_world - center, axis=1).max())
    if scale <= 1e-09:
        scale = 1.0
    post = stack['post']
    normalized, parser_log = post.linecloud_to_wireframe((raw_world - center) / scale, (edge_world - center) / scale, stack['scheme'])
    parser_world = stack['canonical'](np.round(np.asarray(normalized).reshape(-1, 2, 3) * scale + center, 8))
    origin = parser_world.reshape(-1, 3).mean(0) if len(parser_world) else np.zeros(3)
    fixed, operations = stack['pipeline'](parser_world - origin, raw_world - origin, edge_world - origin, stack['balanced']['steps'], scale=scale)
    fixed_world = stack['canonical'](np.round(fixed + origin, 8))
    return (parser_world, fixed_world, {'parser': parser_log, 'balanced': operations, 'parser_pc_radius_m': scale})

def kernel_4330(graph, patch):
    keep = np.ones(len(graph), bool)
    keep[patch['remove']] = False
    return e.verify.canonical(np.r_[graph[keep], np.asarray(patch['add']).reshape(-1, 2, 3)])

def kernel_4332(ev, graph, patches):
    graph = e.verify.canonical(graph)
    v, ix, d = degree(graph)
    base_leaf = int((d == 1).sum())
    base_branch = int((d >= 3).sum())
    rows = []
    for patch in patches:
        remove = patch['remove']
        added = np.array(patch['add'])
        keep = np.ones(len(graph), bool)
        keep[remove] = False
        context = dict(ev, balanced=graph[keep])
        newx = e.OLD.feature_matrix(context, added, np.zeros((len(added), 3)))
        oldx = e.OLD.feature_matrix(context, graph[remove], np.zeros((len(remove), 3))) if remove else np.zeros((1, len(e.OLD.FEATURES)))
        pred = apply(graph, patch)
        nv, _, nd = degree(pred)
        oldlength = float(np.linalg.norm(graph[remove, 1] - graph[remove, 0], axis=1).sum()) if remove else 0.0
        newlength = float(np.linalg.norm(added[:, 1] - added[:, 0], axis=1).sum())
        degrees = d[ix[remove]].reshape(-1) if remove else np.array([0.0])
        pair = e.OLD.pair_distance(added, graph[remove]) if remove else np.zeros((len(added), 1))
        extra = [len(remove), len(added), len(pred) - len(graph), np.log1p(oldlength), np.log1p(newlength), newlength / max(oldlength, 0.15), len(nv) - len(v), int((nd == 1).sum()) - base_leaf, int((nd >= 3).sum()) - base_branch, float(degrees.min()), float(degrees.max()), float(newx[:, 4].min()), float(newx[:, 7].max()), float(pair.min(1).max())]
        rows.append(np.r_[newx.mean(0), newx.min(0), oldx.mean(0), extra, [float(patch['family'] == k) for k in FAMILIES]])
    X = np.array(rows).reshape(-1, len(FEATURES))
    assert np.isfinite(X).all()
    return X

def kernel_4333(edge):
    x = np.asarray(edge).reshape(2, 3)
    if tuple(x[0]) > tuple(x[1]):
        x = x[::-1]
    return tuple(x.reshape(-1))

def kernel_4334(ev, graph):
    graph = e.verify.canonical(graph)
    v, ix, deg = degree(graph)
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    gl = np.linalg.norm(graph[:, 1] - graph[:, 0], axis=1)
    gu = (graph[:, 1] - graph[:, 0]) / np.maximum(gl[:, None], 1e-12)
    pool, stats = e.OLD.candidates(ev)
    patches = []
    seen = set()

    def support(edge, radius=0.25):
        L = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(L, 1e-12)
        select = (abs(ru @ u) >= np.cos(np.radians(12.0))) & (rl > 0.15)
        if not select.any():
            return (0.0, 5.0)
        query = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
        d = e.OLD.distances(query, raw[select]).min(1)
        return (float((d <= radius).mean()), float(np.minimum(d, 5).mean()))

    def add(family, remove, segments):
        segments = e.verify.canonical(np.asarray(segments).reshape(-1, 2, 3))
        remove = sorted(set(map(int, remove)))
        if not len(segments) or np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).min() < 0.15:
            return
        patch = dict(family=family, remove=remove, add=segments.tolist())
        key = (tuple(remove), tuple((linekey(x) for x in segments)))
        if key in seen:
            return
        pred = apply(graph, patch)
        if e.verify.keys(pred) == e.verify.keys(graph):
            return
        if len(pred) != len(graph) - len(remove) + len(segments):
            return
        seen.add(key)
        patches.append(patch)
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if abs(gu[i] @ gu[j]) < np.cos(np.radians(8)):
                continue
            ends = np.r_[graph[i], graph[j]]
            project = (ends - graph[i, 0]) @ gu[i]
            if np.linalg.norm(np.cross(ends - graph[i, 0], gu[i]), axis=1).max() > 0.18:
                continue
            pi = project[:2]
            pj = project[2:]
            gap = max(pi.min(), pj.min()) - min(pi.max(), pj.max())
            if gap > 0.6:
                continue
            new = ends[[int(project.argmin()), int(project.argmax())]]
            L = np.linalg.norm(new[1] - new[0])
            if L < max(gl[i], gl[j]) + 0.2:
                continue
            remaining = np.delete(graph, [i, j], axis=0).reshape(-1, 3)
            internal = [x for x in ends if not any((np.array_equal(x, y) for y in new))]
            if any((any((np.array_equal(x, y) for y in remaining)) for x in internal)):
                continue
            cov, dist = support(new)
            if cov >= 0.85:
                add('merge_fragments', [i, j], [new])
    for vi in np.flatnonzero((deg >= 2) & (deg <= 3)):
        incident = np.flatnonzero((ix == vi).any(1))
        axes = []
        for i in incident:
            lateral = np.linalg.norm(np.cross(raw - graph[i, 0], gu[i]), axis=2).mean(1)
            selected = np.flatnonzero((abs(ru @ gu[i]) >= np.cos(np.radians(12))) & (lateral <= 0.45) & (rl > 0.3))
            if not len(selected):
                axes = []
                break
            proximity = e.OLD.distances(v[vi:vi + 1], raw[selected])[0]
            selected = selected[np.argsort(proximity + lateral[selected], kind='stable')[:4]]
            previous = len(axes)
            for j in selected:
                if e.OLD.distances(v[vi:vi + 1], raw[j:j + 1])[0, 0] <= 0.8:
                    axes.append((ru[j], raw[j].mean(0)))
            if len(axes) == previous:
                axes = []
                break
        if len(axes) < 2:
            continue
        A = sum((np.eye(3) - np.outer(u, u) for u, p in axes))
        b = sum(((np.eye(3) - np.outer(u, u)) @ p for u, p in axes))
        if np.linalg.cond(A) > 10000.0:
            continue
        point = np.linalg.solve(A + 0.05 * np.eye(3), b + 0.05 * v[vi])
        shift = np.linalg.norm(point - v[vi])
        if not 0.12 <= shift <= 0.9:
            continue
        segments = graph[incident].copy()
        for k, i in enumerate(incident):
            segments[k, ix[i] == vi] = point
        if min((support(x)[0] for x in segments)) >= 0.7:
            add('junction_refit', incident, segments)
    for i, edge in enumerate(graph):
        if gl[i] < 0.6:
            continue
        projection = (v - edge[0]) @ gu[i]
        lateral = np.linalg.norm(np.cross(v - edge[0], gu[i]), axis=1)
        eligible = np.flatnonzero((projection > 0.2) & (projection < gl[i] - 0.2) & (lateral <= 0.18))
        for vi in eligible[np.argsort(lateral[eligible], kind='stable')[:3]]:
            incident = np.flatnonzero((ix == vi).any(1))
            if i in incident or not len(incident) or (not (abs(gu[incident] @ gu[i]) < np.cos(np.radians(20))).any()):
                continue
            segs = np.array([[edge[0], v[vi]], [v[vi], edge[1]]])
            if min((support(x)[0] for x in segs)) >= 0.8:
                add('branch_split', [i], segs)
    rawends = raw.reshape(-1, 3)
    endpoint_owner = np.repeat(np.arange(len(raw)), 2)
    for i, edge in enumerate(graph):
        if gl[i] < 0.8:
            continue
        t = (rawends - edge[0]) @ gu[i]
        lat = np.linalg.norm(np.cross(rawends - edge[0], gu[i]), axis=1)
        eligible = np.flatnonzero((t > 0.25) & (t < gl[i] - 0.25) & (lat >= 0.12) & (lat <= 0.6) & (abs(ru[endpoint_owner] @ gu[i]) < np.cos(np.radians(12))))
        chosen = []
        for j in eligible[np.argsort(lat[eligible], kind='stable')]:
            point = rawends[j]
            if chosen and np.linalg.norm(np.array(chosen) - point, axis=1).min() < 0.25:
                continue
            segs = np.array([[edge[0], point], [point, edge[1]]])
            vec = segs[:, 1] - segs[:, 0]
            length = np.linalg.norm(vec, axis=1)
            if length.min() < 0.25:
                continue
            if min((support(x, 0.2)[0] for x in segs)) < 0.8:
                continue
            add('raw_kink', [i], segs)
            chosen.append(point)
            if len(chosen) >= 2:
                break
    if len(pool) and len(v):
        tree = cKDTree(v)
        distance, nearest = tree.query(pool.reshape(-1, 3))
        distance = distance.reshape(-1, 2)
        nearest = nearest.reshape(-1, 2)
        plen = np.linalg.norm(pool[:, 1] - pool[:, 0], axis=1)
        poolx = e.OLD.feature_matrix(dict(ev, balanced=graph), pool, stats)
        qualified = (poolx[:, 4] >= 0.7) & (poolx[:, 15] <= 0.7) & (plen >= 0.2)
        selected = []
        bins = np.searchsorted([1.0, 2.0, 5.0], plen, side='right')
        for b in range(4):
            js = np.flatnonzero(qualified & (bins == b))
            selected.extend(js[np.argsort(-stats[js, 2], kind='stable')[:8]].tolist())
        for j in selected:
            seg = pool[j].copy()
            for k in range(2):
                if distance[j, k] <= 0.18:
                    seg[k] = v[nearest[j, k]]
            add('add_single', [], [seg])
        pairs = []
        for i in selected:
            for j in selected:
                if j <= i:
                    continue
                d = np.linalg.norm(pool[i, :, None] - pool[j, None, :], axis=2)
                a, b = np.unravel_index(d.argmin(), d.shape)
                if d[a, b] > 0.25:
                    continue
                ai = 1 - a
                bj = 1 - b
                if max(distance[i, ai], distance[j, bj]) > 0.35 or nearest[i, ai] == nearest[j, bj]:
                    continue
                point = (pool[i, a] + pool[j, b]) / 2
                A = v[nearest[i, ai]]
                B = v[nearest[j, bj]]
                segments = np.array([[A, point], [point, B]])
                deltas = segments[:, 1] - segments[:, 0]
                lens = np.linalg.norm(deltas, axis=1)
                if lens.min() < 0.25:
                    continue
                cos = abs(deltas[0] @ deltas[1] / (lens[0] * lens[1]))
                if cos > np.cos(np.radians(12)):
                    continue
                if min((support(x, 0.2)[0] for x in segments)) < 0.8:
                    continue
                pairs.append((float(d[a, b] + distance[i, ai] + distance[j, bj]), segments))
        for _, segs in sorted(pairs, key=lambda x: x[0])[:16]:
            add('add_connected_pair', [], segs)
    counts = {k: 0 for k in FAMILIES}
    result = []
    for patch in patches:
        family = patch['family']
        if counts[family] >= 24:
            continue
        counts[family] += 1
        result.append(patch)
    return result

def kernel_4336(ev, graph, patches, alg):
    e = alg.e
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    union = e.verify.canonical(np.concatenate([graph] + [np.array(p['add']).reshape(-1, 2, 3) for p in patches])) if patches else graph
    pair = e.OLD.pair_distance(union, raw)
    cache = {}
    for i, edge in enumerate(union):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-12)
        closest = int(pair[i].argmin()) if len(raw) else 0
        order = np.sort(pair[i]) if len(raw) else np.array([5.0])
        select = abs(ru @ u) >= np.cos(np.radians(15))
        if select.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = e.OLD.distances(q, raw[select])
            single = float((D <= 0.2).mean(0).max())
        else:
            single = 0.0
        cache[alg.linekey(edge)] = [min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), float(np.log1p((pair[i] <= 0.35).sum())), float(np.log1p((pair[i] <= 1.0).sum())), float(np.log(max(rl[closest], 0.01) / max(length, 0.01))) if len(raw) else 0.0, single]
    result = []
    for patch in patches:
        added = np.array(patch['add'])
        remove = patch['remove']
        newx = np.array([cache[alg.linekey(x)] for x in added])
        oldx = np.array([cache[alg.linekey(x)] for x in graph[remove]]) if remove else np.zeros((1, len(BASE_NAMES)))
        vertices, ii = np.unique(added.reshape(-1, 3), axis=0, return_inverse=True)
        deg = np.bincount(ii, minlength=len(vertices))
        junction = vertices[deg >= 2]
        if not len(junction):
            junction = vertices
        old = graph[remove] if remove else added
        longest = int(np.linalg.norm(old[:, 1] - old[:, 0], axis=1).argmax())
        reference = old[longest]
        u = reference[1] - reference[0]
        u = u / max(np.linalg.norm(u), 1e-12)
        aligned = (abs(ru @ u) >= np.cos(np.radians(12))) & (np.linalg.norm(np.cross(raw - reference[0], u), axis=2).max(1) <= 0.35)
        projection = (raw - reference[0]) @ u
        lo = projection.min(1)
        hi = projection.max(1)
        keep = np.ones(len(graph), bool)
        keep[remove] = False
        other = graph[keep]
        if len(other):
            ou = (other[:, 1] - other[:, 0]) / np.maximum(np.linalg.norm(other[:, 1] - other[:, 0], axis=1)[:, None], 1e-12)
            otherbranch = other[abs(ou @ u) < np.cos(np.radians(20))]
        else:
            otherbranch = other
        jj = []
        for point in junction:
            end = e.OLD.nearest(point[None], raw.reshape(-1, 3))[0]
            distance = e.OLD.distances(point[None], raw)[0]
            branch = (abs(ru @ u) < np.cos(np.radians(20))) & (distance <= 0.2) & (rl > 0.3)
            t = (point - reference[0]) @ u
            endnear = aligned & (np.minimum(abs(lo - t), abs(hi - t)) <= 0.25)
            crossing = aligned & (lo < t - 0.35) & (hi > t + 0.35)
            left = aligned & (abs(hi - t) <= 0.25) & (lo < t - 0.3)
            right = aligned & (abs(lo - t) <= 0.25) & (hi > t + 0.3)
            gd = e.OLD.distances(point[None], otherbranch)
            jj.append([float(end), float(np.log1p(branch.sum())), float(np.log1p(np.median(rl[branch]))) if branch.any() else 0.0, float(np.log1p(endnear.sum())), float(np.log1p(crossing.sum())), float(np.log1p(min(left.sum(), right.sum()))), float(endnear.sum() / (endnear.sum() + crossing.sum() + 1)), float(e.OLD.nearest(point[None], ev['ep'])[0]), float(e.OLD.nearest(point[None], ev['pc'])[0]), float(min(5.0, gd.min())) if gd.size else 5.0])
        result.append(np.r_[newx.mean(0), newx.max(0), oldx.mean(0), np.mean(jj, axis=0)])
    X = np.array(result).reshape(-1, len(FEATURES))
    assert np.isfinite(X).all()
    return X

def kernel_4338(ev, lines):
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    u = (raw[:, 1] - raw[:, 0]) / np.maximum(length[:, None], 1e-10)
    pair = c.e.OLD.pair_distance(lines, raw)
    out = []
    for i, edge in enumerate(lines):
        L = np.linalg.norm(edge[1] - edge[0])
        direction = (edge[1] - edge[0]) / max(L, 1e-10)
        order = np.sort(pair[i])
        j = int(pair[i].argmin())
        aligned = abs(u @ direction) >= np.cos(np.radians(15))
        single = 0.0
        if aligned.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = c.e.OLD.distances(q, raw[aligned])
            single = float((D <= 0.2).mean(0).max())
        out.append([min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), np.log1p((pair[i] <= 0.35).sum()), np.log1p((pair[i] <= 1).sum()), float(np.log(max(length[j], 0.01) / max(L, 0.01))), single])
    return np.array(out).reshape(-1, len(FINITE_NAMES))

def kernel_4339(edge, pc, tree):
    vector = edge[1] - edge[0]
    xylen = np.linalg.norm(vector[:2])
    out = np.zeros(len(ROOF_NAMES))
    out[6:8] = 1.0
    out[9:15] = 2.0
    if xylen < 0.25:
        return (out, None)
    center = edge.mean(0)
    u = vector[:2] / xylen
    n = np.array([-u[1], u[0]])
    near = pc[tree.query_ball_point(center[:2], xylen / 2 + 1.5)]
    delta = near[:, :2] - edge[0, :2]
    along = delta @ u
    lat = delta @ n
    within = (along >= -0.2) & (along <= xylen + 0.2) & (abs(lat) >= 0.15) & (abs(lat) <= 1.2)
    side = [within & (lat < 0), within & (lat > 0)]
    counts = np.array([x.sum() for x in side])
    fits = [plane(near[s], center) for s in side]
    valid = np.array([x is not None for x in fits])
    cov = np.array([len(np.unique(np.clip((along[s] / xylen * 6).astype(int), 0, 5))) / 6 for s in side])
    res = np.array([min(1.0, x['mad']) if x else 1.0 for x in fits])
    errors = []
    slopes = []
    for f in fits:
        if f:
            z = (edge[:, :2] - center[:2]) @ f['v'][:2] + f['v'][2] + center[2]
            errors.append(min(2.0, float(np.abs(z - edge[:, 2]).mean())))
            slopes.append(min(2.0, float(abs(f['v'][:2] @ u - vector[2] / xylen))))
        else:
            errors.append(2.0)
            slopes.append(2.0)
    out[:8] = [1.0, np.log1p(counts.min() / max(xylen, 0.5)), np.log1p(counts.max() / max(xylen, 0.5)), counts.min() / max(1, counts.max()), valid.min(), valid.max(), res.min(), res.max()]
    out[10:14] = [min(errors), max(errors), min(slopes), max(slopes)]
    out[15:] = [cov.min(), cov.max(), min(5.0, float(np.ptp(near[within, 2]))) if within.any() else 0.0]
    refit = None
    if valid.all():
        a, b = fits
        angle = float(np.arccos(np.clip(a['normal'] @ b['normal'], -1.0, 1.0)))
        out[8] = angle
        out[9] = min(2.0, abs(a['v'][2] - b['v'][2]))
        direction = np.cross(a['normal'], b['normal'])
        dl = np.linalg.norm(direction)
        if dl > 1e-05:
            direction /= dl
            old = vector / max(np.linalg.norm(vector), 1e-10)
            turn = float(np.arccos(np.clip(abs(direction @ old), 0, 1)))
            out[14] = turn
            if direction @ old < 0:
                direction = -direction
            if np.radians(8) <= angle <= np.radians(80) and turn <= np.radians(12) and (res.max() <= 0.15) and (cov.min() >= 0.5):
                A = np.array([[-a['v'][0], -a['v'][1], 1.0], [-b['v'][0], -b['v'][1], 1.0], old])
                rhs = np.array([a['v'][2], b['v'][2], 0.0])
                if np.linalg.cond(A) < 100:
                    point = np.linalg.solve(A, rhs) + center
                    proj = ((edge - point) @ direction)[:, None]
                    proposal = point + proj * direction
                    displacement = np.linalg.norm(proposal - edge, axis=1).max()
                    if 0.08 <= displacement <= 0.6 and np.linalg.norm(proposal[1] - proposal[0]) > 0.25:
                        refit = proposal
    return (out, refit)

def kernel_4340(ev, lines):
    pc = np.asarray(ev['pc']).reshape(-1, 3)
    tree = cKDTree(pc[:, :2])
    out = []
    refit = []
    for i, edge in enumerate(lines):
        x, p = roof(edge, pc, tree)
        out.append(x)
        if p is not None:
            refit.append((i, p))
    return (np.array(out).reshape(-1, len(ROOF_NAMES)), refit)

def kernel_4342(graph, patch):
    keep = np.ones(len(graph), bool)
    keep[patch['remove']] = False
    return e.verify.canonical(np.r_[graph[keep], np.asarray(patch['add']).reshape(-1, 2, 3)])

def kernel_4343(graph):
    v, ix = np.unique(graph.reshape(-1, 3), axis=0, return_inverse=True)
    d = np.bincount(ix, minlength=len(v))
    return (v, ix.reshape(-1, 2), d)

def kernel_4345(ev, graph, patches):
    graph = e.verify.canonical(graph)
    v, ix, d = degree(graph)
    base_leaf = int((d == 1).sum())
    base_branch = int((d >= 3).sum())
    rows = []
    for patch in patches:
        remove = patch['remove']
        added = np.array(patch['add'])
        keep = np.ones(len(graph), bool)
        keep[remove] = False
        context = dict(ev, balanced=graph[keep])
        newx = e.OLD.feature_matrix(context, added, np.zeros((len(added), 3)))
        oldx = e.OLD.feature_matrix(context, graph[remove], np.zeros((len(remove), 3))) if remove else np.zeros((1, len(e.OLD.FEATURES)))
        pred = apply(graph, patch)
        nv, _, nd = degree(pred)
        oldlength = float(np.linalg.norm(graph[remove, 1] - graph[remove, 0], axis=1).sum()) if remove else 0.0
        newlength = float(np.linalg.norm(added[:, 1] - added[:, 0], axis=1).sum())
        degrees = d[ix[remove]].reshape(-1) if remove else np.array([0.0])
        pair = e.OLD.pair_distance(added, graph[remove]) if remove else np.zeros((len(added), 1))
        extra = [len(remove), len(added), len(pred) - len(graph), np.log1p(oldlength), np.log1p(newlength), newlength / max(oldlength, 0.15), len(nv) - len(v), int((nd == 1).sum()) - base_leaf, int((nd >= 3).sum()) - base_branch, float(degrees.min()), float(degrees.max()), float(newx[:, 4].min()), float(newx[:, 7].max()), float(pair.min(1).max())]
        rows.append(np.r_[newx.mean(0), newx.min(0), oldx.mean(0), extra, [float(patch['family'] == k) for k in FAMILIES]])
    X = np.array(rows).reshape(-1, len(FEATURES))
    assert np.isfinite(X).all()
    return X

def kernel_4346(ev, graph):
    graph = e.verify.canonical(graph)
    v, ix, deg = degree(graph)
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    gl = np.linalg.norm(graph[:, 1] - graph[:, 0], axis=1)
    gu = (graph[:, 1] - graph[:, 0]) / np.maximum(gl[:, None], 1e-12)
    pool, stats = e.OLD.candidates(ev)
    patches = []
    seen = set()

    def support(edge, radius=0.25):
        L = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(L, 1e-12)
        select = (abs(ru @ u) >= np.cos(np.radians(12.0))) & (rl > 0.15)
        if not select.any():
            return (0.0, 5.0)
        query = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
        d = e.OLD.distances(query, raw[select]).min(1)
        return (float((d <= radius).mean()), float(np.minimum(d, 5).mean()))

    def add(family, remove, segments):
        segments = e.verify.canonical(np.asarray(segments).reshape(-1, 2, 3))
        remove = sorted(set(map(int, remove)))
        if not len(segments) or np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).min() < 0.15:
            return
        patch = dict(family=family, remove=remove, add=segments.tolist())
        key = (tuple(remove), tuple((linekey(x) for x in segments)))
        if key in seen:
            return
        pred = apply(graph, patch)
        if e.verify.keys(pred) == e.verify.keys(graph):
            return
        if len(pred) != len(graph) - len(remove) + len(segments):
            return
        seen.add(key)
        patches.append(patch)
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if abs(gu[i] @ gu[j]) < np.cos(np.radians(8)):
                continue
            ends = np.r_[graph[i], graph[j]]
            project = (ends - graph[i, 0]) @ gu[i]
            if np.linalg.norm(np.cross(ends - graph[i, 0], gu[i]), axis=1).max() > 0.18:
                continue
            pi = project[:2]
            pj = project[2:]
            gap = max(pi.min(), pj.min()) - min(pi.max(), pj.max())
            if gap > 0.6:
                continue
            new = ends[[int(project.argmin()), int(project.argmax())]]
            L = np.linalg.norm(new[1] - new[0])
            if L < max(gl[i], gl[j]) + 0.2:
                continue
            remaining = np.delete(graph, [i, j], axis=0).reshape(-1, 3)
            internal = [x for x in ends if not any((np.array_equal(x, y) for y in new))]
            if any((any((np.array_equal(x, y) for y in remaining)) for x in internal)):
                continue
            cov, dist = support(new)
            if cov >= 0.85:
                add('merge_fragments', [i, j], [new])
    for vi in np.flatnonzero((deg >= 2) & (deg <= 3)):
        incident = np.flatnonzero((ix == vi).any(1))
        axes = []
        for i in incident:
            lateral = np.linalg.norm(np.cross(raw - graph[i, 0], gu[i]), axis=2).mean(1)
            selected = np.flatnonzero((abs(ru @ gu[i]) >= np.cos(np.radians(12))) & (lateral <= 0.45) & (rl > 0.3))
            if not len(selected):
                axes = []
                break
            proximity = e.OLD.distances(v[vi:vi + 1], raw[selected])[0]
            selected = selected[np.argsort(proximity + lateral[selected], kind='stable')[:4]]
            previous = len(axes)
            for j in selected:
                if e.OLD.distances(v[vi:vi + 1], raw[j:j + 1])[0, 0] <= 0.8:
                    axes.append((ru[j], raw[j].mean(0)))
            if len(axes) == previous:
                axes = []
                break
        if len(axes) < 2:
            continue
        A = sum((np.eye(3) - np.outer(u, u) for u, p in axes))
        b = sum(((np.eye(3) - np.outer(u, u)) @ p for u, p in axes))
        if np.linalg.cond(A) > 10000.0:
            continue
        point = np.linalg.solve(A + 0.05 * np.eye(3), b + 0.05 * v[vi])
        shift = np.linalg.norm(point - v[vi])
        if not 0.12 <= shift <= 0.9:
            continue
        segments = graph[incident].copy()
        for k, i in enumerate(incident):
            segments[k, ix[i] == vi] = point
        if min((support(x)[0] for x in segments)) >= 0.7:
            add('junction_refit', incident, segments)
    for i, edge in enumerate(graph):
        if gl[i] < 0.6:
            continue
        projection = (v - edge[0]) @ gu[i]
        lateral = np.linalg.norm(np.cross(v - edge[0], gu[i]), axis=1)
        eligible = np.flatnonzero((projection > 0.2) & (projection < gl[i] - 0.2) & (lateral <= 0.18))
        for vi in eligible[np.argsort(lateral[eligible], kind='stable')[:3]]:
            incident = np.flatnonzero((ix == vi).any(1))
            if i in incident or not len(incident) or (not (abs(gu[incident] @ gu[i]) < np.cos(np.radians(20))).any()):
                continue
            segs = np.array([[edge[0], v[vi]], [v[vi], edge[1]]])
            if min((support(x)[0] for x in segs)) >= 0.8:
                add('branch_split', [i], segs)
    rawends = raw.reshape(-1, 3)
    endpoint_owner = np.repeat(np.arange(len(raw)), 2)
    for i, edge in enumerate(graph):
        if gl[i] < 0.8:
            continue
        t = (rawends - edge[0]) @ gu[i]
        lat = np.linalg.norm(np.cross(rawends - edge[0], gu[i]), axis=1)
        eligible = np.flatnonzero((t > 0.25) & (t < gl[i] - 0.25) & (lat >= 0.12) & (lat <= 0.6) & (abs(ru[endpoint_owner] @ gu[i]) < np.cos(np.radians(12))))
        chosen = []
        for j in eligible[np.argsort(lat[eligible], kind='stable')]:
            point = rawends[j]
            if chosen and np.linalg.norm(np.array(chosen) - point, axis=1).min() < 0.25:
                continue
            segs = np.array([[edge[0], point], [point, edge[1]]])
            vec = segs[:, 1] - segs[:, 0]
            length = np.linalg.norm(vec, axis=1)
            if length.min() < 0.25:
                continue
            if min((support(x, 0.2)[0] for x in segs)) < 0.8:
                continue
            add('raw_kink', [i], segs)
            chosen.append(point)
            if len(chosen) >= 2:
                break
    if len(pool) and len(v):
        tree = cKDTree(v)
        distance, nearest = tree.query(pool.reshape(-1, 3))
        distance = distance.reshape(-1, 2)
        nearest = nearest.reshape(-1, 2)
        plen = np.linalg.norm(pool[:, 1] - pool[:, 0], axis=1)
        poolx = e.OLD.feature_matrix(dict(ev, balanced=graph), pool, stats)
        qualified = (poolx[:, 4] >= 0.7) & (poolx[:, 15] <= 0.7) & (plen >= 0.2)
        selected = []
        bins = np.searchsorted([1.0, 2.0, 5.0], plen, side='right')
        for b in range(4):
            js = np.flatnonzero(qualified & (bins == b))
            selected.extend(js[np.argsort(-stats[js, 2], kind='stable')[:8]].tolist())
        for j in selected:
            seg = pool[j].copy()
            for k in range(2):
                if distance[j, k] <= 0.18:
                    seg[k] = v[nearest[j, k]]
            add('add_single', [], [seg])
        pairs = []
        for i in selected:
            for j in selected:
                if j <= i:
                    continue
                d = np.linalg.norm(pool[i, :, None] - pool[j, None, :], axis=2)
                a, b = np.unravel_index(d.argmin(), d.shape)
                if d[a, b] > 0.25:
                    continue
                ai = 1 - a
                bj = 1 - b
                if max(distance[i, ai], distance[j, bj]) > 0.35 or nearest[i, ai] == nearest[j, bj]:
                    continue
                point = (pool[i, a] + pool[j, b]) / 2
                A = v[nearest[i, ai]]
                B = v[nearest[j, bj]]
                segments = np.array([[A, point], [point, B]])
                deltas = segments[:, 1] - segments[:, 0]
                lens = np.linalg.norm(deltas, axis=1)
                if lens.min() < 0.25:
                    continue
                cos = abs(deltas[0] @ deltas[1] / (lens[0] * lens[1]))
                if cos > np.cos(np.radians(12)):
                    continue
                if min((support(x, 0.2)[0] for x in segments)) < 0.8:
                    continue
                pairs.append((float(d[a, b] + distance[i, ai] + distance[j, bj]), segments))
        for _, segs in sorted(pairs, key=lambda x: x[0])[:16]:
            add('add_connected_pair', [], segs)
    counts = {k: 0 for k in FAMILIES}
    result = []
    for patch in patches:
        family = patch['family']
        if counts[family] >= 24:
            continue
        counts[family] += 1
        result.append(patch)
    return result

def kernel_4348(ev, graph, patches, alg):
    e = alg.e
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    rl = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    ru = (raw[:, 1] - raw[:, 0]) / np.maximum(rl[:, None], 1e-12)
    union = e.verify.canonical(np.concatenate([graph] + [np.array(p['add']).reshape(-1, 2, 3) for p in patches])) if patches else graph
    pair = e.OLD.pair_distance(union, raw)
    cache = {}
    for i, edge in enumerate(union):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-12)
        closest = int(pair[i].argmin()) if len(raw) else 0
        order = np.sort(pair[i]) if len(raw) else np.array([5.0])
        select = abs(ru @ u) >= np.cos(np.radians(15))
        if select.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = e.OLD.distances(q, raw[select])
            single = float((D <= 0.2).mean(0).max())
        else:
            single = 0.0
        cache[alg.linekey(edge)] = [min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), float(np.log1p((pair[i] <= 0.35).sum())), float(np.log1p((pair[i] <= 1.0).sum())), float(np.log(max(rl[closest], 0.01) / max(length, 0.01))) if len(raw) else 0.0, single]
    result = []
    for patch in patches:
        added = np.array(patch['add'])
        remove = patch['remove']
        newx = np.array([cache[alg.linekey(x)] for x in added])
        oldx = np.array([cache[alg.linekey(x)] for x in graph[remove]]) if remove else np.zeros((1, len(BASE_NAMES)))
        vertices, ii = np.unique(added.reshape(-1, 3), axis=0, return_inverse=True)
        deg = np.bincount(ii, minlength=len(vertices))
        junction = vertices[deg >= 2]
        if not len(junction):
            junction = vertices
        old = graph[remove] if remove else added
        longest = int(np.linalg.norm(old[:, 1] - old[:, 0], axis=1).argmax())
        reference = old[longest]
        u = reference[1] - reference[0]
        u = u / max(np.linalg.norm(u), 1e-12)
        aligned = (abs(ru @ u) >= np.cos(np.radians(12))) & (np.linalg.norm(np.cross(raw - reference[0], u), axis=2).max(1) <= 0.35)
        projection = (raw - reference[0]) @ u
        lo = projection.min(1)
        hi = projection.max(1)
        keep = np.ones(len(graph), bool)
        keep[remove] = False
        other = graph[keep]
        if len(other):
            ou = (other[:, 1] - other[:, 0]) / np.maximum(np.linalg.norm(other[:, 1] - other[:, 0], axis=1)[:, None], 1e-12)
            otherbranch = other[abs(ou @ u) < np.cos(np.radians(20))]
        else:
            otherbranch = other
        jj = []
        for point in junction:
            end = e.OLD.nearest(point[None], raw.reshape(-1, 3))[0]
            distance = e.OLD.distances(point[None], raw)[0]
            branch = (abs(ru @ u) < np.cos(np.radians(20))) & (distance <= 0.2) & (rl > 0.3)
            t = (point - reference[0]) @ u
            endnear = aligned & (np.minimum(abs(lo - t), abs(hi - t)) <= 0.25)
            crossing = aligned & (lo < t - 0.35) & (hi > t + 0.35)
            left = aligned & (abs(hi - t) <= 0.25) & (lo < t - 0.3)
            right = aligned & (abs(lo - t) <= 0.25) & (hi > t + 0.3)
            gd = e.OLD.distances(point[None], otherbranch)
            jj.append([float(end), float(np.log1p(branch.sum())), float(np.log1p(np.median(rl[branch]))) if branch.any() else 0.0, float(np.log1p(endnear.sum())), float(np.log1p(crossing.sum())), float(np.log1p(min(left.sum(), right.sum()))), float(endnear.sum() / (endnear.sum() + crossing.sum() + 1)), float(e.OLD.nearest(point[None], ev['ep'])[0]), float(e.OLD.nearest(point[None], ev['pc'])[0]), float(min(5.0, gd.min())) if gd.size else 5.0])
        result.append(np.r_[newx.mean(0), newx.max(0), oldx.mean(0), np.mean(jj, axis=0)])
    X = np.array(result).reshape(-1, len(FEATURES))
    assert np.isfinite(X).all()
    return X

def kernel_4350(ev, lines):
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    u = (raw[:, 1] - raw[:, 0]) / np.maximum(length[:, None], 1e-10)
    pair = c.e.OLD.pair_distance(lines, raw)
    out = []
    for i, edge in enumerate(lines):
        L = np.linalg.norm(edge[1] - edge[0])
        direction = (edge[1] - edge[0]) / max(L, 1e-10)
        order = np.sort(pair[i])
        j = int(pair[i].argmin())
        aligned = abs(u @ direction) >= np.cos(np.radians(15))
        single = 0.0
        if aligned.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = c.e.OLD.distances(q, raw[aligned])
            single = float((D <= 0.2).mean(0).max())
        out.append([min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), np.log1p((pair[i] <= 0.35).sum()), np.log1p((pair[i] <= 1).sum()), float(np.log(max(length[j], 0.01) / max(L, 0.01))), single])
    return np.array(out).reshape(-1, len(FINITE_NAMES))

def kernel_4351(edge, pc, tree):
    vector = edge[1] - edge[0]
    xylen = np.linalg.norm(vector[:2])
    out = np.zeros(len(ROOF_NAMES))
    out[6:8] = 1.0
    out[9:15] = 2.0
    if xylen < 0.25:
        return (out, None)
    center = edge.mean(0)
    u = vector[:2] / xylen
    n = np.array([-u[1], u[0]])
    near = pc[tree.query_ball_point(center[:2], xylen / 2 + 1.5)]
    delta = near[:, :2] - edge[0, :2]
    along = delta @ u
    lat = delta @ n
    within = (along >= -0.2) & (along <= xylen + 0.2) & (abs(lat) >= 0.15) & (abs(lat) <= 1.2)
    side = [within & (lat < 0), within & (lat > 0)]
    counts = np.array([x.sum() for x in side])
    fits = [plane(near[s], center) for s in side]
    valid = np.array([x is not None for x in fits])
    cov = np.array([len(np.unique(np.clip((along[s] / xylen * 6).astype(int), 0, 5))) / 6 for s in side])
    res = np.array([min(1.0, x['mad']) if x else 1.0 for x in fits])
    errors = []
    slopes = []
    for f in fits:
        if f:
            z = (edge[:, :2] - center[:2]) @ f['v'][:2] + f['v'][2] + center[2]
            errors.append(min(2.0, float(np.abs(z - edge[:, 2]).mean())))
            slopes.append(min(2.0, float(abs(f['v'][:2] @ u - vector[2] / xylen))))
        else:
            errors.append(2.0)
            slopes.append(2.0)
    out[:8] = [1.0, np.log1p(counts.min() / max(xylen, 0.5)), np.log1p(counts.max() / max(xylen, 0.5)), counts.min() / max(1, counts.max()), valid.min(), valid.max(), res.min(), res.max()]
    out[10:14] = [min(errors), max(errors), min(slopes), max(slopes)]
    out[15:] = [cov.min(), cov.max(), min(5.0, float(np.ptp(near[within, 2]))) if within.any() else 0.0]
    refit = None
    if valid.all():
        a, b = fits
        angle = float(np.arccos(np.clip(a['normal'] @ b['normal'], -1.0, 1.0)))
        out[8] = angle
        out[9] = min(2.0, abs(a['v'][2] - b['v'][2]))
        direction = np.cross(a['normal'], b['normal'])
        dl = np.linalg.norm(direction)
        if dl > 1e-05:
            direction /= dl
            old = vector / max(np.linalg.norm(vector), 1e-10)
            turn = float(np.arccos(np.clip(abs(direction @ old), 0, 1)))
            out[14] = turn
            if direction @ old < 0:
                direction = -direction
            if np.radians(8) <= angle <= np.radians(80) and turn <= np.radians(12) and (res.max() <= 0.15) and (cov.min() >= 0.5):
                A = np.array([[-a['v'][0], -a['v'][1], 1.0], [-b['v'][0], -b['v'][1], 1.0], old])
                rhs = np.array([a['v'][2], b['v'][2], 0.0])
                if np.linalg.cond(A) < 100:
                    point = np.linalg.solve(A, rhs) + center
                    proj = ((edge - point) @ direction)[:, None]
                    proposal = point + proj * direction
                    displacement = np.linalg.norm(proposal - edge, axis=1).max()
                    if 0.08 <= displacement <= 0.6 and np.linalg.norm(proposal[1] - proposal[0]) > 0.25:
                        refit = proposal
    return (out, refit)

def kernel_4352(ev, lines):
    pc = np.asarray(ev['pc']).reshape(-1, 3)
    tree = cKDTree(pc[:, :2])
    out = []
    refit = []
    for i, edge in enumerate(lines):
        x, p = roof(edge, pc, tree)
        out.append(x)
        if p is not None:
            refit.append((i, p))
    return (np.array(out).reshape(-1, len(ROOF_NAMES)), refit)

def kernel_4354(ev, lines):
    raw = np.asarray(ev['raw']).reshape(-1, 2, 3)
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    u = (raw[:, 1] - raw[:, 0]) / np.maximum(length[:, None], 1e-10)
    pair = c.e.OLD.pair_distance(lines, raw)
    out = []
    for i, edge in enumerate(lines):
        L = np.linalg.norm(edge[1] - edge[0])
        direction = (edge[1] - edge[0]) / max(L, 1e-10)
        order = np.sort(pair[i])
        j = int(pair[i].argmin())
        aligned = abs(u @ direction) >= np.cos(np.radians(15))
        single = 0.0
        if aligned.any():
            q = edge[0] + np.linspace(0, 1, 21)[:, None] * (edge[1] - edge[0])
            D = c.e.OLD.distances(q, raw[aligned])
            single = float((D <= 0.2).mean(0).max())
        out.append([min(5.0, float(order[0])), min(5.0, float(order[:3].mean())), np.log1p((pair[i] <= 0.35).sum()), np.log1p((pair[i] <= 1).sum()), float(np.log(max(length[j], 0.01) / max(L, 0.01))), single])
    return np.array(out).reshape(-1, len(FINITE_NAMES))

def kernel_4355(edge, pc, tree):
    vector = edge[1] - edge[0]
    xylen = np.linalg.norm(vector[:2])
    out = np.zeros(len(ROOF_NAMES))
    out[6:8] = 1.0
    out[9:15] = 2.0
    if xylen < 0.25:
        return (out, None)
    center = edge.mean(0)
    u = vector[:2] / xylen
    n = np.array([-u[1], u[0]])
    near = pc[tree.query_ball_point(center[:2], xylen / 2 + 1.5)]
    delta = near[:, :2] - edge[0, :2]
    along = delta @ u
    lat = delta @ n
    within = (along >= -0.2) & (along <= xylen + 0.2) & (abs(lat) >= 0.15) & (abs(lat) <= 1.2)
    side = [within & (lat < 0), within & (lat > 0)]
    counts = np.array([x.sum() for x in side])
    fits = [plane(near[s], center) for s in side]
    valid = np.array([x is not None for x in fits])
    cov = np.array([len(np.unique(np.clip((along[s] / xylen * 6).astype(int), 0, 5))) / 6 for s in side])
    res = np.array([min(1.0, x['mad']) if x else 1.0 for x in fits])
    errors = []
    slopes = []
    for f in fits:
        if f:
            z = (edge[:, :2] - center[:2]) @ f['v'][:2] + f['v'][2] + center[2]
            errors.append(min(2.0, float(np.abs(z - edge[:, 2]).mean())))
            slopes.append(min(2.0, float(abs(f['v'][:2] @ u - vector[2] / xylen))))
        else:
            errors.append(2.0)
            slopes.append(2.0)
    out[:8] = [1.0, np.log1p(counts.min() / max(xylen, 0.5)), np.log1p(counts.max() / max(xylen, 0.5)), counts.min() / max(1, counts.max()), valid.min(), valid.max(), res.min(), res.max()]
    out[10:14] = [min(errors), max(errors), min(slopes), max(slopes)]
    out[15:] = [cov.min(), cov.max(), min(5.0, float(np.ptp(near[within, 2]))) if within.any() else 0.0]
    refit = None
    if valid.all():
        a, b = fits
        angle = float(np.arccos(np.clip(a['normal'] @ b['normal'], -1.0, 1.0)))
        out[8] = angle
        out[9] = min(2.0, abs(a['v'][2] - b['v'][2]))
        direction = np.cross(a['normal'], b['normal'])
        dl = np.linalg.norm(direction)
        if dl > 1e-05:
            direction /= dl
            old = vector / max(np.linalg.norm(vector), 1e-10)
            turn = float(np.arccos(np.clip(abs(direction @ old), 0, 1)))
            out[14] = turn
            if direction @ old < 0:
                direction = -direction
            if np.radians(8) <= angle <= np.radians(80) and turn <= np.radians(12) and (res.max() <= 0.15) and (cov.min() >= 0.5):
                A = np.array([[-a['v'][0], -a['v'][1], 1.0], [-b['v'][0], -b['v'][1], 1.0], old])
                rhs = np.array([a['v'][2], b['v'][2], 0.0])
                if np.linalg.cond(A) < 100:
                    point = np.linalg.solve(A, rhs) + center
                    proj = ((edge - point) @ direction)[:, None]
                    proposal = point + proj * direction
                    displacement = np.linalg.norm(proposal - edge, axis=1).max()
                    if 0.08 <= displacement <= 0.6 and np.linalg.norm(proposal[1] - proposal[0]) > 0.25:
                        refit = proposal
    return (out, refit)

def kernel_4356(ev, lines):
    pc = np.asarray(ev['pc']).reshape(-1, 3)
    tree = cKDTree(pc[:, :2])
    out = []
    refit = []
    for i, edge in enumerate(lines):
        x, p = roof(edge, pc, tree)
        out.append(x)
        if p is not None:
            refit.append((i, p))
    return (np.array(out).reshape(-1, len(ROOF_NAMES)), refit)

def kernel_4421(lines):
    x = np.asarray(lines, float).reshape(-1, 2, 3)
    if not len(x):
        return x
    v, ix = np.unique(x.reshape(-1, 3), axis=0, return_inverse=True)
    e = np.unique(np.sort(ix.reshape(-1, 2), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return v[e]

def kernel_4424(evidence, candidate, stats):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    ep = np.asarray(evidence['ep']).reshape(-1, 3)
    pc = np.asarray(evidence['pc']).reshape(-1, 3)
    scale = float(evidence['scale'])
    corners = base.reshape(-1, 3)
    ab = raw[:, 1] - raw[:, 0]
    rl = np.linalg.norm(ab, axis=1)
    ru = ab / np.maximum(rl[:, None], 1e-09)
    ep_tree = cKDTree(ep) if len(ep) else None
    pc_tree = cKDTree(pc) if len(pc) else None
    rows = []
    for edge, s in zip(candidate, stats):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-09)
        query = edge[0] + np.linspace(0, 1, 12)[:, None] * (edge[1] - edge[0])
        compatible = abs(ru @ u) > 0.966
        d = distances(query, raw[compatible])
        minimum = d.min(1) if d.shape[1] else np.full(12, 5.0)
        support = (d < 0.35).sum(1) if d.shape[1] else np.zeros(12)
        endpoint = nearest(edge, raw.reshape(-1, 3))
        ed = np.minimum(ep_tree.query(query)[0], 5) if ep_tree else np.full(12, 5.0)
        pd = np.minimum(pc_tree.query(query)[0], 5) if pc_tree else np.full(12, 5.0)
        graph = nearest(edge, corners)
        bd = distances(query, base)
        bm = bd.min(1) if bd.shape[1] else np.full(12, 5.0)
        rows.append([np.log1p(length), s[0], s[1], np.log1p(s[2]), (minimum < 0.18).mean(), (minimum < 0.35).mean(), np.log1p(support.min()), minimum.clip(0, 5).mean(), endpoint.mean(), (ed < 0.25).mean(), ed.mean(), (pd < 0.35).mean(), graph.mean(), graph.max(), bm.clip(0, 5).mean(), (bm < 0.35).mean(), pair_distance(edge[None], base).min() if len(base) else 5.0, abs(u[2]), np.log1p(scale)])
    return np.nan_to_num(np.asarray(rows).reshape(-1, len(FEATURES)), nan=5, posinf=5)

def kernel_4425(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_4427(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_4428(x):
    return {tuple(e) for e in np.round(canonical(x).reshape(-1, 6), 7)}

def kernel_4429(points, segs):
    if not len(segs):
        return np.full((len(points), 0), np.inf)
    ab = segs[:, 1] - segs[:, 0]
    d = (ab * ab).sum(1)
    delta = points[:, None] - segs[None, :, 0]
    t = np.divide((delta * ab[None]).sum(2), d[None], out=np.zeros((len(points), len(segs))), where=d[None] > 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * ab[None], axis=2)

def kernel_4459(points, center):
    if len(points) < 6:
        return None
    local = points - center
    xy = local[:, :2]
    A = np.c_[xy, np.ones(len(xy))]
    if np.linalg.cond(A) > 100:
        return None
    v = np.linalg.lstsq(A, local[:, 2], rcond=None)[0]
    for _ in range(2):
        error = abs(A @ v - local[:, 2])
        cut = max(0.08, float(np.quantile(error, 0.75)))
        good = error <= cut
        if good.sum() < 6 or np.linalg.cond(A[good]) > 100:
            break
        v = np.linalg.lstsq(A[good], local[good, 2], rcond=None)[0]
    error = abs(A @ v - local[:, 2])
    mad = float(np.median(error))
    normal = np.array([-v[0], -v[1], 1.0])
    normal /= np.linalg.norm(normal)
    return dict(v=v, mad=mad, normal=normal)

def kernel_4524(lines):
    x = np.asarray(lines, float).reshape(-1, 2, 3)
    if not len(x):
        return x
    v, ix = np.unique(x.reshape(-1, 3), axis=0, return_inverse=True)
    e = np.unique(np.sort(ix.reshape(-1, 2), axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return v[e]

def kernel_4527(evidence, candidate, stats):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    ep = np.asarray(evidence['ep']).reshape(-1, 3)
    pc = np.asarray(evidence['pc']).reshape(-1, 3)
    scale = float(evidence['scale'])
    corners = base.reshape(-1, 3)
    ab = raw[:, 1] - raw[:, 0]
    rl = np.linalg.norm(ab, axis=1)
    ru = ab / np.maximum(rl[:, None], 1e-09)
    ep_tree = cKDTree(ep) if len(ep) else None
    pc_tree = cKDTree(pc) if len(pc) else None
    rows = []
    for edge, s in zip(candidate, stats):
        length = np.linalg.norm(edge[1] - edge[0])
        u = (edge[1] - edge[0]) / max(length, 1e-09)
        query = edge[0] + np.linspace(0, 1, 12)[:, None] * (edge[1] - edge[0])
        compatible = abs(ru @ u) > 0.966
        d = distances(query, raw[compatible])
        minimum = d.min(1) if d.shape[1] else np.full(12, 5.0)
        support = (d < 0.35).sum(1) if d.shape[1] else np.zeros(12)
        endpoint = nearest(edge, raw.reshape(-1, 3))
        ed = np.minimum(ep_tree.query(query)[0], 5) if ep_tree else np.full(12, 5.0)
        pd = np.minimum(pc_tree.query(query)[0], 5) if pc_tree else np.full(12, 5.0)
        graph = nearest(edge, corners)
        bd = distances(query, base)
        bm = bd.min(1) if bd.shape[1] else np.full(12, 5.0)
        rows.append([np.log1p(length), s[0], s[1], np.log1p(s[2]), (minimum < 0.18).mean(), (minimum < 0.35).mean(), np.log1p(support.min()), minimum.clip(0, 5).mean(), endpoint.mean(), (ed < 0.25).mean(), ed.mean(), (pd < 0.35).mean(), graph.mean(), graph.max(), bm.clip(0, 5).mean(), (bm < 0.35).mean(), pair_distance(edge[None], base).min() if len(base) else 5.0, abs(u[2]), np.log1p(scale)])
    return np.nan_to_num(np.asarray(rows).reshape(-1, len(FEATURES)), nan=5, posinf=5)

def kernel_4528(a, b):
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), np.inf)
    direct = np.linalg.norm(a[:, None] - b[None], axis=-1).max(-1)
    reverse = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).max(-1)
    return np.minimum(direct, reverse)

def kernel_4530(evidence):
    raw = lines(evidence['raw'])
    base = lines(evidence['balanced'])
    parser = lines(evidence['parser'])
    length = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    raw = raw[length > 0.15]
    length = length[length > 0.15]
    if not len(raw):
        return (np.empty((0, 2, 3)), np.empty((0, 3)))
    dm = pair_distance(raw, raw)
    counts = (dm < 0.35).sum(1)
    order = np.argsort(-counts, kind='stable')
    used = np.zeros(len(raw), bool)
    reps = []
    stats = []
    groups = []
    for i in order:
        if used[i]:
            continue
        group = np.where((dm[i] < 0.45) & ~used)[0]
        used[group] = True
        reps.append(raw[i])
        stats.append((0, 0, len(group)))
        groups.append(group)
        if len(reps) >= 96:
            break
    for edge in parser:
        if not len(base) or pair_distance(edge[None], base).min() > 0.25:
            reps.append(edge)
            stats.append((1, 0, 1))
    original = np.array(reps[:len(groups)])
    lens = np.linalg.norm(original[:, 1] - original[:, 0], axis=1)
    directions = (original[:, 1] - original[:, 0]) / np.maximum(lens[:, None], 1e-09)
    for i, edge in enumerate(original):
        u = directions[i]
        compatible = abs(directions @ u) > 0.99
        lateral = np.linalg.norm(np.cross(original - edge[0], u), axis=-1).max(1) < 0.18
        projected = (original - edge[0]) @ u
        lo = projected.min(1)
        hi = projected.max(1)
        mask = compatible & lateral & (lo < lens[i] + 0.3) & (hi > -0.3)
        js = np.flatnonzero(mask)
        if len(js) > 1:
            lower = float(lo[js].min())
            upper = float(hi[js].max())
            if upper - lower > lens[i] + 0.3 and upper - lower < 1.8 * lens[i] + 0.5:
                reps.append(np.stack([edge[0] + lower * u, edge[0] + upper * u]))
                stats.append((0, 1, len(js)))
    candidate = np.asarray(reps).reshape(-1, 2, 3)
    stats = np.asarray(stats, float)
    keep = pair_distance(candidate, base).min(1) > 0.35 if len(base) else np.ones(len(candidate), bool)
    chosen = []
    for i in np.where(keep)[0]:
        if chosen and pair_distance(candidate[i:i + 1], candidate[chosen]).min() < 0.2:
            continue
        chosen.append(i)
    return (candidate[chosen], stats[chosen])

def kernel_4531(x):
    return {tuple(e) for e in np.round(canonical(x).reshape(-1, 6), 7)}

def kernel_4532(points, segs):
    if not len(segs):
        return np.full((len(points), 0), np.inf)
    ab = segs[:, 1] - segs[:, 0]
    d = (ab * ab).sum(1)
    delta = points[:, None] - segs[None, :, 0]
    t = np.divide((delta * ab[None]).sum(2), d[None], out=np.zeros((len(points), len(segs))), where=d[None] > 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(delta - t[:, :, None] * ab[None], axis=2)

def kernel_4554(points, center):
    if len(points) < 6:
        return None
    local = points - center
    xy = local[:, :2]
    A = np.c_[xy, np.ones(len(xy))]
    if np.linalg.cond(A) > 100:
        return None
    v = np.linalg.lstsq(A, local[:, 2], rcond=None)[0]
    for _ in range(2):
        error = abs(A @ v - local[:, 2])
        cut = max(0.08, float(np.quantile(error, 0.75)))
        good = error <= cut
        if good.sum() < 6 or np.linalg.cond(A[good]) > 100:
            break
        v = np.linalg.lstsq(A[good], local[good, 2], rcond=None)[0]
    error = abs(A @ v - local[:, 2])
    mad = float(np.median(error))
    normal = np.array([-v[0], -v[1], 1.0])
    normal /= np.linalg.norm(normal)
    return dict(v=v, mad=mad, normal=normal)

def kernel_4557(points, center):
    if len(points) < 6:
        return None
    local = points - center
    xy = local[:, :2]
    A = np.c_[xy, np.ones(len(xy))]
    if np.linalg.cond(A) > 100:
        return None
    v = np.linalg.lstsq(A, local[:, 2], rcond=None)[0]
    for _ in range(2):
        error = abs(A @ v - local[:, 2])
        cut = max(0.08, float(np.quantile(error, 0.75)))
        good = error <= cut
        if good.sum() < 6 or np.linalg.cond(A[good]) > 100:
            break
        v = np.linalg.lstsq(A[good], local[good, 2], rcond=None)[0]
    error = abs(A @ v - local[:, 2])
    mad = float(np.median(error))
    normal = np.array([-v[0], -v[1], 1.0])
    normal /= np.linalg.norm(normal)
    return dict(v=v, mad=mad, normal=normal)

def kernel_4558(points, cloud, empty=None):
    return np.minimum(cKDTree(cloud).query(points)[0], 5.0) if len(cloud) else np.full(len(points), empty)

def kernel_4559(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_4560(points, cloud, empty=None):
    return np.minimum(cKDTree(cloud).query(points)[0], 5.0) if len(cloud) else np.full(len(points), empty)

def kernel_4561(x):
    return np.asarray(x, dtype=float).reshape(-1, 2, 3)

def kernel_4566(lines, edge_points, scheme):
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
    if lines.size == 0:
        return (np.empty((0, 2, 3), dtype=np.float64), {'num_clusters': 0, 'num_candidates': 0})
    vec = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(vec, axis=1)
    valid_len = length >= scheme.min_graph_len
    lines = lines[valid_len]
    vec = vec[valid_len]
    length = length[valid_len]
    if len(lines) == 0:
        return (np.empty((0, 2, 3), dtype=np.float64), {'num_clusters': 0, 'num_candidates': 0})
    endpoints = lines.reshape(-1, 3)
    labels = DBSCAN(eps=scheme.cluster_eps, min_samples=scheme.min_samples).fit_predict(endpoints)
    centers = cluster_centers(endpoints, labels, scheme.center_method)
    if len(centers) < 2:
        return (np.empty((0, 2, 3), dtype=np.float64), {'num_clusters': int(len(centers)), 'num_candidates': 0})
    dist_to_centers = cdist(endpoints, centers)
    nearest = dist_to_centers.argmin(axis=1)
    nearest_dist = dist_to_centers[np.arange(len(endpoints)), nearest]
    nearest[nearest_dist > scheme.attach_eps] = -1
    line_clusters = nearest.reshape(-1, 2)
    line_snap = nearest_dist.reshape(-1, 2)
    candidates = {}
    for idx, (ca, cb) in enumerate(line_clusters):
        if ca < 0 or cb < 0 or ca == cb:
            continue
        a, b = sorted((int(ca), int(cb)))
        key = (a, b)
        cand = candidates.setdefault(key, {'votes': 0, 'snap_sum': 0.0, 'consistency_sum': 0.0, 'proj_min': math.inf, 'proj_max': -math.inf})
        ga, gb = (centers[a], centers[b])
        gv = gb - ga
        gl = np.linalg.norm(gv)
        if gl <= 1e-12:
            continue
        gu = gv / gl
        lu = vec[idx] / max(length[idx], 1e-12)
        consistency = abs(float(np.dot(lu, gu)))
        if consistency < scheme.min_consistency:
            continue
        pts = lines[idx]
        proj = (pts - ga.reshape(1, 3)) @ gu / gl
        cand['votes'] += 1
        cand['snap_sum'] += float(line_snap[idx].mean())
        cand['consistency_sum'] += consistency
        cand['proj_min'] = min(cand['proj_min'], float(proj.min()))
        cand['proj_max'] = max(cand['proj_max'], float(proj.max()))
    if scheme.path_pair_max_len > 0 and scheme.path_support_radius > 0 and (edge_points.size != 0):
        center_dist = cdist(centers, centers)
        for a in range(len(centers)):
            order = np.argsort(center_dist[a])
            if scheme.path_knn > 0:
                order = order[1:scheme.path_knn + 1]
            else:
                order = order[1:]
            for b in order:
                if b <= a:
                    continue
                gl = float(center_dist[a, b])
                if gl < scheme.min_graph_len or gl > scheme.path_pair_max_len:
                    continue
                key = (a, int(b))
                if key in candidates and candidates[key]['votes'] >= scheme.min_votes:
                    continue
                support, edge_cov = point_segment_stats(edge_points, centers[a], centers[b], scheme.path_support_radius)
                if support < scheme.path_min_support or edge_cov < scheme.path_min_coverage:
                    continue
                cand = candidates.setdefault(key, {'votes': 0, 'snap_sum': 0.0, 'consistency_sum': 0.0, 'proj_min': math.inf, 'proj_max': -math.inf})
                cand['path_support'] = max(int(cand.get('path_support', 0)), support)
                cand['path_coverage'] = max(float(cand.get('path_coverage', 0.0)), edge_cov)
    scored = []
    for (a, b), cand in candidates.items():
        votes = cand['votes']
        has_path = 'path_support' in cand
        if votes < scheme.min_votes and (not has_path):
            continue
        ga, gb = (centers[a], centers[b])
        gl = float(np.linalg.norm(gb - ga))
        if gl < scheme.min_graph_len:
            continue
        snap = cand['snap_sum'] / max(votes, 1) if votes > 0 else 0.0
        if snap > scheme.max_snap:
            continue
        consistency = cand['consistency_sum'] / max(votes, 1) if votes > 0 else 1.0
        direct_coverage = max(0.0, min(1.0, cand['proj_max']) - max(0.0, cand['proj_min']))
        support, edge_coverage = point_segment_stats(edge_points, ga, gb, scheme.edge_support_radius)
        if has_path:
            support = max(support, int(cand.get('path_support', 0)))
            edge_coverage = max(edge_coverage, float(cand.get('path_coverage', 0.0)))
        coverage = max(direct_coverage, edge_coverage)
        if support < scheme.min_edge_support:
            continue
        support_norm = min(1.0, support / max(4.0, gl / max(scheme.edge_support_radius, 1e-06)))
        score = scheme.score_vote_weight * math.log1p(votes) + consistency + 0.5 * coverage + scheme.score_support_weight * support_norm - scheme.score_snap_weight * (snap / max(scheme.attach_eps, 1e-06)) + scheme.path_score_weight * float(has_path) * coverage
        scored.append({'edge': (a, b), 'score': score, 'votes': votes, 'snap': snap, 'consistency': consistency, 'support': support, 'length': gl})
    scored.sort(key=lambda item: item['score'], reverse=True)
    chosen = []
    degrees = np.zeros(len(centers), dtype=np.int64)
    directions_by_vertex = [[] for _ in range(len(centers))]
    cos_sep = math.cos(math.radians(scheme.angle_sep_deg))
    chosen_set = set()
    for item in scored:
        a, b = item['edge']
        if (a, b) in chosen_set:
            continue
        if degrees[a] >= scheme.max_degree or degrees[b] >= scheme.max_degree:
            continue
        va = centers[b] - centers[a]
        vb = -va
        norm = np.linalg.norm(va)
        if norm <= 1e-12:
            continue
        da = va / norm
        db = vb / norm
        if any((float(np.dot(da, old)) > cos_sep for old in directions_by_vertex[a])):
            continue
        if any((float(np.dot(db, old)) > cos_sep for old in directions_by_vertex[b])):
            continue
        chosen.append((a, b))
        chosen_set.add((a, b))
        degrees[a] += 1
        degrees[b] += 1
        directions_by_vertex[a].append(da)
        directions_by_vertex[b].append(db)
    if scheme.repair_degree > 0:
        repair_cos_sep = math.cos(math.radians(scheme.repair_angle_sep_deg or scheme.angle_sep_deg))
        for item in scored:
            if item['score'] < scheme.repair_min_score:
                continue
            a, b = item['edge']
            if (a, b) in chosen_set:
                continue
            if degrees[a] >= scheme.repair_degree and degrees[b] >= scheme.repair_degree:
                continue
            if degrees[a] >= scheme.max_degree or degrees[b] >= scheme.max_degree:
                continue
            va = centers[b] - centers[a]
            vb = -va
            norm = np.linalg.norm(va)
            if norm <= 1e-12:
                continue
            da = va / norm
            db = vb / norm
            if any((float(np.dot(da, old)) > repair_cos_sep for old in directions_by_vertex[a])):
                continue
            if any((float(np.dot(db, old)) > repair_cos_sep for old in directions_by_vertex[b])):
                continue
            chosen.append((a, b))
            chosen_set.add((a, b))
            degrees[a] += 1
            degrees[b] += 1
            directions_by_vertex[a].append(da)
            directions_by_vertex[b].append(db)
    if not chosen:
        out = np.empty((0, 2, 3), dtype=np.float64)
    else:
        out = np.asarray([[centers[a], centers[b]] for a, b in chosen], dtype=np.float64)
        if scheme.output_repeats > 1:
            out = np.repeat(out, scheme.output_repeats, axis=0)
    meta = {'num_clusters': int(len(centers)), 'num_candidates': int(len(scored)), 'num_chosen': int(len(chosen))}
    return (out, meta)
