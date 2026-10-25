"""CPU-only follow-up; source predictions/choices are verified, never overwritten."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np

from climate_emulator.config import load_config
from climate_emulator.data.loader import discover, load_scenario, convert_targets
from climate_emulator.data.members import complete_period_members
from climate_emulator.data.preprocessing import collect_samples, Preprocessor
from climate_emulator.evaluation.metrics import period, common_arrays
from climate_emulator.evaluation.predictions import prediction_array, load_predictions
from climate_emulator.models.pattern_scaling import PatternScaling
from climate_emulator.models.ridge import RidgeEmulator
from climate_emulator.utils import file_hash, object_hash, read_json, write_json
from .math import ALPHAS, WEIGHTS, ORIGINAL_ALPHAS, ORIGINAL_WEIGHTS, summary, scores, objective, select, complementarity

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('spatial_rmse', 'global_rmse', 'nrmse_total')
BASELINES = ('lps', 'ridge', 'rf', 'cnn_lstm')


def safe_verify(root, artifacts):
    root = root.resolve()
    for name, digest in artifacts.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Changed/missing artifact: {name}')


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'Empty table: {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix('.tmp.csv')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def marker(directory):
    artifacts = {p.relative_to(directory).as_posix(): file_hash(p) for p in directory.rglob('*')
                 if p.is_file() and p != directory / 'complete.json' and '.tmp' not in p.name}
    write_json(directory / 'complete.json', dict(artifacts=artifacts))


def complete(directory):
    path = directory / 'complete.json'
    if not path.exists():
        return False
    safe_verify(directory, read_json(path)['artifacts'])
    return True


def choice(settings):
    i = settings['selected_index']
    if any(not np.isclose(w, float(j == i), atol=1e-12) for j, w in enumerate(settings['weights'])):
        raise ValueError('Expected one selected scalar-blend candidate')
    spec = settings['specs'][i]
    if spec.get('name') == 'lps':
        return dict(alpha=None, weight=0.)
    if spec.get('family') != 'linear_blend':
        raise ValueError('Unexpected selected family')
    return dict(alpha=float(spec['alpha']), weight=float(spec['ridge_weight']))


def feature_view(data, features):
    # Preserve identical targets/masks, forcing histories and support diagnostics.
    return data if features == 'full' else replace(data, features=data.features[:, :2])


def fit_branches(training, validation):
    result = {'lps': PatternScaling().fit(training).predict(validation)}
    for features in ('full', 'ghg'):
        train, valid = feature_view(training, features), feature_view(validation, features)
        for alpha in ALPHAS:
            result[f'{features}_{alpha:g}'] = RidgeEmulator(alpha).fit(train).predict(valid)
    return result


def period_summaries(predictions, data):
    positions = [i for i, (_, year) in enumerate(data.keys) if 2080 <= year <= 2100]
    if [data.keys[i][1] for i in positions] != list(range(2080, 2101)):
        raise ValueError('Missing validation years')
    mask = data.mask
    w = data.weights[mask] / data.weights[mask].sum()
    truth = data.target[positions][:, mask]
    return {name: summary(value[positions][:, mask], truth, w) for name, value in predictions.items()}, int(mask.sum())


def verify_inner_reproduction(archived, loaded, cells):
    # Reproduce every registered candidate objective, not only the selected one.
    observed = []
    lps = [r['lps'] for r in loaded]
    for spec in archived['specs']:
        if spec.get('name') == 'lps':
            value = objective(lps, lps, 0., cells)
        else:
            branches = [r[f"full_{spec['alpha']:g}"] for r in loaded]
            value = objective(lps, branches, spec['ridge_weight'], cells)
        observed.append(value)
    for i, value in enumerate(observed):
        # Archived entries are [objective, index].
        recorded = next(float(x[0]) for x in archived['candidate_objectives'] if int(x[1]) == i)
        if not np.isclose(recorded, value, rtol=2e-5, atol=2e-8):
            raise ValueError(f'Inner objective reproduction failed for candidate {i}: {value} vs {recorded}')
    chosen = min(range(len(observed)), key=lambda i: (observed[i], i))
    if chosen != archived['selected_index']:
        # Tiny BLAS differences at a numerical tie are allowed, but frozen choice is retained.
        if not np.isclose(observed[chosen], observed[archived['selected_index']], rtol=1e-10, atol=1e-12):
            raise ValueError('Archived choice is not reproduced')
    return max(abs(observed[i] - next(float(x[0]) for x in archived['candidate_objectives'] if int(x[1]) == i)) for i in range(len(observed)))


def inner_case(base, source, output, task, target):
    directory = output / 'inner' / task['name'] / target
    directory.mkdir(parents=True, exist_ok=True)
    if complete(directory):
        print(f"Reuse inner {task['name']}/{target}", flush=True)
        return
    archived = read_json(source / 'selection' / task['name'] / f'{target}__scalar_blend.json')
    helds = [s for s in task['train'] if s.startswith('ssp')]
    loaded, cell_counts, provenance = [], [], []
    for held in helds:
        fitting = [s for s in task['train'] if s != held]
        if set(fitting) & {held, 'ssp585'}:
            raise ValueError('Scenario target leakage')
        fold = directory / held
        fold.mkdir(exist_ok=True)
        if not complete(fold):
            print(f"Inner {task['name']}/{target}: hold out {held}; fit {fitting}", flush=True)
            raw = collect_samples(Path(base['data']['root']), fitting, target, base)
            prep = Preprocessor(base).fit(raw)
            train = prep.transform(raw)
            vr = collect_samples(Path(base['data']['root']), [held], target, base)
            valid = prep.transform(vr)
            predictions = fit_branches(train, valid)
            values, cells = period_summaries(predictions, valid)
            np.savez_compressed(fold / 'summaries.npz', **values)
            write_json(fold / 'provenance.json', dict(held=held, fitting=fitting, cells=cells,
                train_keys_sha256=object_hash(raw.keys), validation_keys_sha256=object_hash(vr.keys),
                target=target, mask_sha256=object_hash(valid.mask)))
            marker(fold)
            del raw, prep, train, vr, valid, predictions
            gc.collect()
        with np.load(fold / 'summaries.npz', allow_pickle=False) as data:
            loaded.append({k: data[k] for k in data.files})
        prov = read_json(fold / 'provenance.json')
        cell_counts.append(prov['cells'])
        provenance.append(prov)
    # Different inner training masks would need separate per-fold cell offsets.
    if len(set(cell_counts)) != 1 or len({p['mask_sha256'] for p in provenance}) != 1:
        raise ValueError('Inner folds require the same valid-cell mask in this follow-up')
    cells = cell_counts[0]
    maximum_difference = verify_inner_reproduction(archived, loaded, cells)
    lps = [r['lps'] for r in loaded]
    ghg = select(lps, {a: [r[f'ghg_{a:g}'] for r in loaded] for a in ORIGINAL_ALPHAS}, cells)
    surface = []
    for features in ('full', 'ghg'):
        for alpha in ALPHAS:
            for weight in WEIGHTS:
                surface.append(dict(features=features, alpha=alpha, weight=weight,
                    selection_objective=objective(lps, [r[f'{features}_{alpha:g}'] for r in loaded], weight, cells),
                    original_grid=alpha in ORIGINAL_ALPHAS and weight in ORIGINAL_WEIGHTS))
    csv_write(directory / 'training_sensitivity.csv', surface)
    write_json(directory / 'decisions.json', dict(frozen_full=choice(archived), training_selected_ghg=ghg,
        fold_provenance=provenance, original_objective_max_abs_difference=maximum_difference,
        note='GHG control selected on original grid only. Expanded sensitivity grid does not change the main method.'))
    marker(directory)


def archived_scores(source, task, target, model):
    with (source / 'metrics' / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        return {r['metric']: float(r['value']) for r in csv.DictReader(stream)
                if r['task'] == task and r['target'] == target and r['scenario'] == 'ssp585'
                and r['model'] == model and r['reference'] == 'ensemble_mean' and r['metric'] in METRICS}


def compare(actual, expected, label):
    if set(expected) != set(METRICS):
        raise ValueError(f'Missing archived metrics: {label}')
    for name in METRICS:
        if actual[name] is None or not np.isclose(actual[name], expected[name], rtol=2e-5, atol=2e-8):
            raise ValueError(f'{label} reproduction failed: {name}: {actual[name]} vs {expected[name]}')


def final_case(base, source, output, task, target):
    directory = output / 'final' / task['name'] / target
    directory.mkdir(parents=True, exist_ok=True)
    if complete(directory):
        print(f"Reuse final {task['name']}/{target}", flush=True)
        return
    decisions = read_json(output / 'inner' / task['name'] / target / 'decisions.json')
    frozen, ghg = decisions['frozen_full'], decisions['training_selected_ghg']
    # All choices have already been frozen before any final-case outer scoring.
    print(f"Final exploratory diagnostics {task['name']}/{target}", flush=True)
    raw = collect_samples(Path(base['data']['root']), task['train'], target, base)
    prep = Preprocessor(base).fit(raw)
    train = prep.transform(raw)
    vr = collect_samples(Path(base['data']['root']), ['ssp585'], target, base)
    valid = prep.transform(vr)
    predictions = fit_branches(train, valid)
    reference = convert_targets(load_scenario(base['data']['root'], 'ssp585').targets,
                                base['data']['missing_unit_assumptions'])[target]
    members = complete_period_members(period(reference, [2080, 2100]))
    reference = members.mean('member', skipna=False)
    # Archive field and mask define the original scoring mask; reject silent reordering.
    archive_pred, _, archive_mask, _ = load_predictions(source / 'runs' / task['name'] / target / 'scalar_blend' / 'ssp585.nc')
    if not np.array_equal(archive_mask, valid.mask):
        raise ValueError('Archived/fitted mask mismatch')
    current = period(prediction_array(predictions['lps'], valid, 'ssp585'), [2080, 2100])
    truth, _, weights = common_arrays(reference, current, valid.mask)
    cells = truth.shape[1]
    if cells != base['data']['expected_grid'][0] * base['data']['expected_grid'][1]:
        raise ValueError('Incomplete valid grid; refusing a changed benchmark')
    denominator = abs(float(truth.mean(axis=0) @ weights))
    if denominator <= 1e-10:
        raise ValueError('Undefined normalization')
    values, fields = {}, {}
    for name, prediction in predictions.items():
        array = period(prediction_array(prediction, valid, 'ssp585'), [2080, 2100])
        _, field, _ = common_arrays(reference, array, valid.mask)
        fields[name] = field
        values[name] = summary(field, truth, weights)
    # Pure LPS means the ridge branch is not identifiable from the selected model.
    # Alpha=100 is a declared diagnostic reference in that case, not a selected branch.
    alpha = frozen['alpha'] if frozen['alpha'] is not None else 100.
    full_ridge = values[f'full_{alpha:g}']
    selected = (1-frozen['weight'])*values['lps'] + frozen['weight']*full_ridge
    variants = {'lps': values['lps'], 'full_frozen_blend': selected,
                'ghg_same_settings_blend': (1-frozen['weight'])*values['lps']+frozen['weight']*values[f'ghg_{alpha:g}'],
                'full_ridge_same_alpha': full_ridge, 'ghg_ridge_same_alpha': values[f'ghg_{alpha:g}']}
    ga = ghg['alpha'] if ghg['alpha'] is not None else 100.
    variants['ghg_training_selected_blend'] = (1-ghg['weight'])*values['lps']+ghg['weight']*values[f'ghg_{ga:g}']
    compare(scores(selected, cells, denominator), archived_scores(source, task['name'], target, 'scalar_blend'), 'selected blend')
    compare(scores(values['lps'], cells, denominator), archived_scores(source, task['name'], target, 'lps'), 'LPS')
    reconstructed = (1-frozen['weight'])*fields['lps']+frozen['weight']*fields[f'full_{alpha:g}']
    _, saved_field, _ = common_arrays(reference, period(archive_pred, [2080, 2100]), valid.mask)
    if not np.allclose(saved_field, reconstructed, rtol=2e-5, atol=2e-7):
        raise ValueError('Archived blend predictions not reproduced')
    results = [dict(variant=name, **scores(value, cells, denominator)) for name, value in variants.items()]
    csv_write(directory / 'aerosol_controls.csv', results)
    surfaces = [dict(features=features, alpha=a, weight=w, outer_exploratory=True,
        **scores((1-w)*values['lps']+w*values[f'{features}_{a:g}'], cells, denominator))
        for features in ('full', 'ghg') for a in ALPHAS for w in WEIGHTS]
    csv_write(directory / 'outer_sensitivity.csv', surfaces)
    diagnostic = complementarity(values['lps'], full_ridge, frozen['weight'], cells)
    write_json(directory / 'complementarity.json', dict(**diagnostic, selected_alpha=frozen['alpha'],
        diagnostic_ridge_alpha=alpha, weight=frozen['weight'], ridge_branch_active=frozen['weight'] > 0,
        available_members=int(members.sizes['member']), normalization_denominator=denominator,
        valid_grid_cells=cells, years=21, frozen_decisions=decisions,
        archive_prediction_max_abs_difference=float(np.max(abs(saved_field-reconstructed))),
        interpretation='Error cosine is not Pearson correlation. Oracle weight uses outer labels for diagnosis only; never deploy it.'))
    # Maps and annual means sufficient for paper figures, without retaining every sensitivity prediction.
    np.savez_compressed(directory / 'figure_data.npz', lat=valid.lat, lon=valid.lon, mask=valid.mask,
        years=np.arange(2080, 2101), area_weights=weights,
        reference_mean_map=truth.mean(axis=0), lps_mean_map=fields['lps'].mean(axis=0),
        ridge_mean_map=fields[f'full_{alpha:g}'].mean(axis=0), blend_mean_map=reconstructed.mean(axis=0),
        reference_global_series=truth@weights, lps_global_series=fields['lps']@weights,
        ridge_global_series=fields[f'full_{alpha:g}']@weights, blend_global_series=reconstructed@weights,
        reference=truth, lps=fields['lps'], ridge=fields[f'full_{alpha:g}'], blend=reconstructed)
    comparisons = []
    for row in results:
        for baseline in BASELINES:
            original = archived_scores(source, task['name'], target, baseline)
            if set(original) != set(METRICS):
                raise ValueError('Missing named baseline')
            for metric in METRICS:
                comparisons.append(dict(variant=row['variant'], baseline=baseline, metric=metric,
                    variant_score=row[metric], baseline_score=original[metric],
                    reduction_percent=100*(1-row[metric]/original[metric])))
    csv_write(directory / 'each_baseline_comparison.csv', comparisons)
    marker(directory)


def aggregate(output, tasks):
    tables = {name: [] for name in ('aerosol_controls', 'outer_sensitivity', 'training_sensitivity', 'each_baseline_comparison')}
    diagnostics = []
    for task in tasks:
        for target in ('tas', 'pr'):
            inner = output / 'inner' / task['name'] / target
            final = output / 'final' / task['name'] / target
            if not complete(inner) or not complete(final):
                raise ValueError('Cannot aggregate incomplete cases')
            for name in tables:
                location = inner if name == 'training_sensitivity' else final
                with (location / (name+'.csv')).open(newline='', encoding='utf-8') as stream:
                    tables[name] += [dict(task=task['name'], target=target, **r) for r in csv.DictReader(stream)]
            diag = read_json(final / 'complementarity.json')
            diagnostics.append(dict(task=task['name'], target=target,
                **{k:v for k,v in diag.items() if not isinstance(v, (dict,list))}))
    for name, rows in tables.items():
        csv_write(output / (name+'.csv'), rows)
    csv_write(output / 'branch_complementarity.csv', diagnostics)
    lines = ['# Branch analysis: completed outputs', '',
        'These are post-development diagnostics on reused SSP585 data, not independent confirmation.', '',
        '- `branch_complementarity.csv`: weighted error cosine, exact blend-error decomposition and diagnostic oracle weight.',
        '- `aerosol_controls.csv`: full frozen blend, GHG-only fixed-settings control, GHG-only training-selected control, and matched-alpha ridge branches.',
        '- `training_sensitivity.csv`: inner-fold objective surfaces; original choices remain fixed.',
        '- `outer_sensitivity.csv`: exploratory outer-score surfaces; do not select a replacement method from them.',
        '- `each_baseline_comparison.csv`: explicit LPS, ridge, RF and CNN–LSTM comparisons for every control and metric.',
        '- `final/*/*/figure_data.npz`: reference/branch/blend fields, map means and annual global series for plotting.', '',
        'GHG-only controls retain the identical sample keys and masks. They remove BC and SO2 jointly and do not isolate individual aerosol species or establish a causal physical effect.',
        'If the frozen weight is zero, no ridge branch contributes. Its alpha=100 branch diagnostics are explicitly counterfactual.',
        'Complementarity can arise with positively aligned errors; a negative cross term is not required for error reduction.',
        'Compare fixed-settings controls to isolate the input removal at shared alpha/w; compare separately selected controls to compare deployable selection pipelines.', '',
        '## Frozen-blend reproduction and branch diagnostics', '',
        '| Pool | Target | Weight | Error cosine | Blend spatial MSE |', '|---|---|---:|---:|---:|']
    for row in diagnostics:
        cosine = row['spatial_error_cosine']
        lines.append(f"| {row['task']} | {row['target']} | {row['weight']:.2f} | {cosine if cosine is not None else 'undefined'} | {row['blend_spatial_mse']:.8g} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def acquire_lock(output, recover=False):
    path = output / '.runner.lock'
    if recover and path.exists():
        old = read_json(path)
        if platform.system() != 'Linux' or old['host'] != socket.gethostname():
            raise RuntimeError('Automatic stale-lock recovery requires Linux and the same host')
        try:
            os.kill(int(old['pid']), 0)
        except ProcessLookupError:
            path.unlink()
        else:
            raise RuntimeError('Recorded process exists; refusing lock recovery')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), stream)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/publication-coverage-v2-full-gatefix'))
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('results/publication-branch-analysis-v1'))
    parser.add_argument('--base-config', type=Path, default=ROOT/'configs/core.yaml')
    parser.add_argument('--stage', choices=['plan', 'preflight', 'all'], default='all')
    parser.add_argument('--recover-stale-lock', action='store_true')
    args = parser.parse_args(argv)
    plan = dict(pools=4, targets=['tas','pr'], inner_fold_jobs=18, final_jobs=8,
                alpha_grid=ALPHAS, weight_grid=WEIGHTS, variants=['full','ghg_only'],
                ridge_fits=26*len(ALPHAS)*2, lps_fits=26, gpu_required=False,
                selection='Archived full blend frozen; GHG control selected on original grid only',
                resume='Completed fold/case checkpoints reused; interrupted fold repeats',
                independence='Exploratory follow-up; no new independent evidence')
    if args.stage == 'plan':
        print(json.dumps(plan, indent=2)); return
    source, data, output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be separate from the source run')
    if not output.name.startswith('publication-branch-analysis'):
        raise ValueError('Use a dedicated publication-branch-analysis* output directory')
    status = read_json(source / 'status.json')
    if status['status'] != 'complete':
        raise ValueError('Original coverage run must be complete')
    freeze = read_json(source / 'protocol_lock.json')
    safe_verify(source, freeze['artifacts'])
    tasks = [t for t in read_json(source/'plan.json')['tasks'] if t['name'].startswith('coverage_')]
    expected = {('ssp126','ssp245'), ('ssp126','ssp370'), ('ssp245','ssp370'), ('ssp126','ssp245','ssp370')}
    if len(tasks) != 4 or {tuple(sorted(s for s in t['train'] if s != 'historical')) for t in tasks} != expected:
        raise ValueError('Expected exact four-pool coverage design')
    if any(t['test'] != ['ssp585'] or 'historical' not in t['train'] for t in tasks):
        raise ValueError('Wrong held-out scenario')
    for task in tasks:
        for target in ('tas','pr'):
            run = source/'runs'/task['name']/target/'scalar_blend'
            record = read_json(run/'complete.json')
            rel = (run/'ssp585.nc').relative_to(source).as_posix()
            safe_verify(source, {rel: record['artifacts'][rel]})
    raw = discover(data)
    input_hashes = {}
    for scenario in ('historical','ssp126','ssp245','ssp370','ssp585'):
        for kind in ('inputs','targets'):
            path = raw[scenario][kind]
            print(f'Fingerprint data {scenario}/{kind}', flush=True)
            input_hashes[f'{scenario}/{kind}'] = file_hash(path)
    base = load_config(args.base_config)
    base['data']['root'] = str(data)
    packages = {p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','xarray']}
    code = {p.relative_to(ROOT).as_posix():file_hash(p) for folder in ('branch_study','src') for p in sorted((ROOT/folder).rglob('*.py'))}
    identity = dict(source_freeze=file_hash(source/'protocol_lock.json'),
        source_metrics=file_hash(source/'metrics/metrics.csv'), inputs=input_hashes,
        config={k:v for k,v in base.items() if not k.startswith('_')}, code=code, packages=packages, plan=plan)
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output, args.recover_stale_lock)
    start = time.time()
    try:
        receipt = output/'identity.json'
        if receipt.exists() and read_json(receipt) != identity:
            raise ValueError('Inputs, code, configuration or runtime changed. Use a fresh output directory.')
        write_json(receipt, identity)
        write_json(output/'plan.json', plan)
        if args.stage == 'preflight':
            print('Preflight passed; no regression fits started.'); return
        write_json(output/'status.json', dict(status='running', stage='inner', pid=os.getpid()))
        for task in tasks:
            for target in ('tas','pr'):
                inner_case(base, source, output, task, target)
        frozen = {f"{t['name']}/{v}":file_hash(output/'inner'/t['name']/v/'decisions.json') for t in tasks for v in ('tas','pr')}
        path = output/'decisions_freeze.json'
        if path.exists() and read_json(path) != frozen:
            raise ValueError('Frozen decisions changed')
        write_json(path, frozen)
        write_json(output/'status.json', dict(status='running', stage='outer_diagnostics', pid=os.getpid()))
        for task in tasks:
            for target in ('tas','pr'):
                final_case(base, source, output, task, target)
                gc.collect()
        aggregate(output, tasks)
        write_json(output/'status.json', dict(status='complete', stage='all', seconds_this_invocation=time.time()-start))
        print(f'All three analyses complete: {output}', flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='failed', error=repr(exc), seconds_this_invocation=time.time()-start))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
