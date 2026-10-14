"""Descriptive pilot figures from previously validated, frozen native bundles.

Uses the separate plot-lock.txt environment. It cannot select checkpoints,
train models, qualify results, or alter experimental evidence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'fixed-r8': '#2463a6', 'fixed-r12': '#d77b18', 'adalora-12-to-8': '#138578'}
LABELS = {'fixed-r8': 'Fixed LoRA r8', 'fixed-r12': 'Fixed LoRA r12',
          'adalora-12-to-8': 'AdaLoRA 12 to 8 (persistent masks)'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render(repo, report_path, destination):
    repo, report_path, destination = Path(repo).resolve(), Path(report_path), Path(destination)
    report = json.loads(report_path.read_text())
    if report['stage'] != 'PILOT' or report['qualification_status'] != 'NOT_EVALUATED':
        raise ValueError('descriptive unqualified pilot report required')
    if not report['runs']:
        raise ValueError('no validated native training artifacts to plot')
    destination.mkdir(parents=True, exist_ok=False)
    runs = []
    for run in report['runs']:
        path = (repo/run['artifact_path']).resolve()
        if not path.is_relative_to(repo) or digest(path/'artifact_inventory.json') != run['artifact_inventory_sha256']:
            raise ValueError('frozen inventory identity mismatch')
        inventory = json.loads((path/'artifact_inventory.json').read_text())['files']
        for name in ('checkpoint_selection.json', 'rank_history.jsonl', 'result.json'):
            if digest(path/name) != inventory[name]['sha256']:
                raise ValueError('frozen plot input changed '+name)
        selection = json.loads((path/'checkpoint_selection.json').read_text())
        rows = [json.loads(line) for line in (path/'rank_history.jsonl').read_text().splitlines()]
        runs.append((run, selection, rows))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = (('complete_pair_accuracy', 'Complete counterfactual pairs', .90),
               ('worst_family_accuracy', 'Worst public family accuracy', .80),
               ('unauthorized_proposals', 'Label-based action proposal errors', None))
    for run, selection, rows in runs:
        history, method = selection['history'], run['method']
        steps = [h['step'] for h in history]
        for axis, (metric, title, target) in zip(axes, metrics):
            values = [h['metrics'][metric] for h in history]
            axis.plot(steps, values, '-o', markersize=3, color=COLORS[method], label=LABELS[method])
            selected = next(h for h in history if h['step'] == selection['best_step'])
            axis.scatter([selection['best_step']], [selected['metrics'][metric]], marker='*',
                         s=100, color=COLORS[method], zorder=5)
            if target is not None:
                axis.set_ylim(0, 1.04)
        accounting = run['resource_accounting']
        if accounting['sample_exposures'] != 16000:
            raise ValueError('full-pilot sample exposure mismatch')
    for axis, (metric, title, target) in zip(axes, metrics):
        axis.set_title(title)
        axis.set_xlabel('Completed optimizer updates')
        axis.grid(alpha=.2)
        if target is not None:
            axis.axhline(target, ls='--', color='#777777', lw=1, label='Public screening target')
            axis.set_ylabel('Fraction correct')
        else:
            axis.set_ylim(bottom=0)
            axis.set_ylabel('Count (expected ASK/REFUSE, proposed CHEAP/FAST)')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Public engineering pilot, seed 60606 — excluded from the five-seed study', fontsize=12)
    fig.text(.5, .015, 'Stars: development-selected checkpoints. Terminal snapshots remain separate. Correlated public diagnostics; no independent qualification.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(destination/'pilot-learning-curves.png', dpi=170)
    fig.savefig(destination/'pilot-learning-curves.svg')
    plt.close(fig)
    for run, selection, rows in runs:
        if run['method'] != 'adalora-12-to-8':
            continue
        steps = (100, 250, 500, 800, 1000)
        by_step = {r['completed_optimizer_step']: r for r in rows}
        targets = sorted(by_step[100]['active_rank_by_layer'], key=lambda t: int(t.split('layers.')[1].split('.')[0]))
        matrix = np.array([[by_step[s]['active_rank_by_layer'][t] for s in steps] for t in targets])
        fig, axis = plt.subplots(figsize=(7, 8))
        display = axis.imshow(matrix, vmin=0, vmax=12, cmap='viridis', aspect='auto')
        axis.set_xticks(range(len(steps)), steps)
        axis.set_yticks(range(len(targets)), [t.replace('model.layers.', 'Layer ').replace('.attn.Wqkv', '') for t in targets])
        axis.set_xlabel('Completed optimizer updates')
        axis.set_title('AdaLoRA active rank by projection — engineering pilot')
        fig.colorbar(display, ax=axis, label='Active singular components')
        for i in range(len(targets)):
            for j in range(len(steps)):
                axis.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                          color='white' if matrix[i, j] < 7 else 'black')
        fig.tight_layout()
        fig.savefig(destination/'pilot-adalora-ranks.png', dpi=170)
        fig.savefig(destination/'pilot-adalora-ranks.svg')
        plt.close(fig)
        with (destination/'pilot-adalora-ranks.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['target', *steps])
            writer.writerows([[t, *matrix[i].tolist()] for i, t in enumerate(targets)])
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot([r['completed_optimizer_step'] for r in rows],
                  [r['aggregate_active_rank'] for r in rows], color=COLORS[run['method']], label='Observed active rank')
        axis.axhline(176, color='#777777', ls='--', label='Final target ceiling: 176')
        axis.axhline(264, color='#aaaaaa', ls=':', label='Stored allocated rank: 264')
        axis.set(xlabel='Completed optimizer updates', ylabel='Aggregate rank',
                 title='Rank masking changes active capacity, not stored tensor dimensions')
        axis.legend(fontsize=8); axis.grid(alpha=.2); fig.tight_layout()
        fig.savefig(destination/'pilot-rank-budget.png', dpi=170)
        fig.savefig(destination/'pilot-rank-budget.svg')
        plt.close(fig)
    metadata = {'schema': 'saal-adaptive-pilot-figures-v1', 'report_sha256': digest(report_path),
        'plotting_versions': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'methods_present': [r['method'] for r, _, _ in runs],
        'scope': 'Descriptive public engineering pilot, excluded seed; no significance, retention or independent-generalization claim.',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
        'files': {p.name: digest(p) for p in sorted(destination.iterdir())}}
    (destination/'figure-inventory.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('repo'); p.add_argument('report'); p.add_argument('destination')
    args = p.parse_args()
    print(json.dumps(render(args.repo, args.report, args.destination), indent=2))
