"""Read-only recording of the existing three-round network prediction.

This module runs in the network environment. CPU graph reconstruction runs in
its separate frozen sklearn environment. No target annotations are required.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


class EvidenceRecorder:
    def __init__(self, model):
        self.model = model
        self.captured = []
        self.handle = None

    def __enter__(self):
        def hook(module, args, kwargs, output):
            if args:
                raise ValueError('Expected keyword-based line refinement call')
            self.captured.append({
                **{k: kwargs[k].detach().cpu().clone() for k in ('A', 'pred_dir', 'L_coarse')},
                **{k: output[k].detach().cpu().clone() for k in
                   ('endpoint_logits', 'L_ext', 'L_refined', 'delta_B_perp')},
            })
            # Returning None guarantees that the forward result is unchanged.
        self.handle = self.model.line_refiner.register_forward_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        self.handle = None

    def arrays(self, prediction, geometry, real_pc, to_world):
        captured = self.captured
        if len(captured) != len(prediction['history_stage']) or len(captured) % 2:
            raise ValueError('Refiner calls and prediction histories are misaligned')
        if len(captured) > 6:
            raise ValueError('The frozen reconstruction expects exactly three configured rounds')
        mappings, modal = [], []
        for j, record in enumerate(captured):
            ci, cb = torch.where(prediction['history_mask'][j])
            ci, cb = ci.cpu(), cb.cpu()
            if len(ci) != len(record['A']):
                raise ValueError('Response identity mismatch')
            mappings.append({(int(i), int(b)): k for k, (i, b) in enumerate(zip(ci, cb))})
            bins = record['endpoint_logits'].shape[1]
            probability = torch.softmax(record['endpoint_logits'], dim=-1)
            mean_distance = probability @ torch.linspace(0, 1, bins) * record['L_ext']
            if not torch.allclose(mean_distance, record['L_refined'], atol=2e-7, rtol=2e-6):
                raise ValueError('Endpoint distribution no longer matches the frozen refiner')
            reconstructed = record['A'] + record['L_refined'][:, None] * record['pred_dir'] + record['delta_B_perp']
            history = prediction['history_B' if j % 2 == 0 else 'history_A'][j].detach().cpu()[ci, cb]
            if not torch.equal(reconstructed, history):
                raise ValueError('Read-only endpoint reconstruction mismatch')
            length = torch.argmax(record['endpoint_logits'], dim=-1).to(record['L_ext'].dtype) / (bins - 1) * record['L_ext']
            modal.append(record['A'] + length[:, None] * record['pred_dir'] + record['delta_B_perp'])
        distributions = {}
        device = prediction['edge_mask'].device
        for stage in range(3):
            j = 2 * stage + 1
            if j >= len(captured):
                distributions.update({f'raw_{stage}': np.empty((0, 2, 3)),
                                      f'mode_{stage}': np.empty((0, 2, 3)),
                                      f'candidate_i_{stage}': np.empty(0, dtype=np.int64),
                                      f'bin_{stage}': np.empty(0, dtype=np.int64)})
                continue
            ci, cb = torch.where(prediction['history_mask'][j] & (prediction['history_round_scores'][j] > .75))
            a = to_world(prediction['history_A'][j][ci, cb], geometry).cpu().numpy()
            b = to_world(prediction['history_B'][j][ci, cb], geometry).cpu().numpy()
            keep = np.linalg.norm(b - a, axis=1) >= .15
            ci, cb = ci.cpu()[keep], cb.cpu()[keep]
            distributions[f'raw_{stage}'] = np.round(np.stack([a[keep], b[keep]], axis=1).astype(np.float64), 8)
            fi = np.array([mappings[j - 1][(int(i), int(k))] for i, k in zip(ci, cb)], dtype=int)
            bi = np.array([mappings[j][(int(i), int(k))] for i, k in zip(ci, cb)], dtype=int)
            ma = to_world(modal[j][bi].to(device), geometry).cpu().numpy()
            mb = to_world(modal[j - 1][fi].to(device), geometry).cpu().numpy()
            distributions[f'mode_{stage}'] = np.round(np.stack([ma, mb], axis=1).astype(np.float64), 8)
            distributions[f'candidate_i_{stage}'] = ci.numpy()
            distributions[f'bin_{stage}'] = cb.numpy()
        pc = np.round(np.asarray(real_pc, dtype=np.float64), 8)
        mask = torch.sigmoid(prediction['edge_mask']).detach().cpu().numpy().reshape(-1) > .5
        arrays = dict(pc=pc, ep=pc[mask], raw=distributions['raw_2'])
        return arrays, distributions

    def save(self, folder, prediction, geometry, real_pc, to_world):
        arrays, distributions = self.arrays(prediction, geometry, real_pc, to_world)
        folder = Path(folder)
        paths = [folder / 'input.npz', folder / 'distributions.npz', folder / 'evidence.json']
        if any(p.exists() for p in paths):
            raise FileExistsError('Evidence already exists; refusing overwrite')
        np.savez_compressed(paths[0], **arrays)
        np.savez_compressed(paths[1], **distributions)
        paths[2].write_text(json.dumps(dict(
            schema='three_round_endpoint_evidence', metric_coordinates=True,
            stage_score_threshold=.75, minimum_length_m=.15, edge_threshold=.5,
            input_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            distributions_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
            read_only_hook=True), indent=2) + '\n')
