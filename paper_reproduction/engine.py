from copy import deepcopy
from pathlib import Path
import joblib
import numpy as np
from climate_emulator.config import load_config,resolve_path
from climate_emulator.data.preprocessing import collect_samples,Preprocessor
from climate_emulator.models.pattern_scaling import PatternScaling
from climate_emulator.models.ridge import RidgeEmulator
from climate_emulator.training.experiment import model_candidates,make_model,fit_with_deadline
from climate_emulator.evaluation.predictions import save_predictions
from climate_emulator.utils import object_hash,read_json,write_json,file_hash
from .common import PROJECT,new_or_same,records,verify,freeze,verify_frozen
from .protocol import folds,error_summary
SELECTION_CONTROLS=('scalar_blend',)

def plan(settings):
    base=load_config(settings['base_config'])
    tasks=[]
    for pool in [('ssp126','ssp245'),('ssp126','ssp370'),('ssp245','ssp370'),('ssp126','ssp245','ssp370')]:
        tasks.append(dict(name='coverage_'+'_'.join(pool),train=['historical',*pool],test=['ssp585'],future_scenarios=len(pool)))
    from .selection_controls import grid
    specs={'scalar_blend':[dict(family='baseline',name='lps'),*grid(settings['selector'],base['models']['ridge']['alphas'])]}
    specs.update({name:baseline_specs(name,base,settings) for name in settings['baselines']})
    return dict(tasks=tasks,models=list(specs),specs=specs,targets=['tas','pr'])

def baseline_specs(name, base, settings):
    if name == 'lps':
        return [dict(family='baseline', name='lps')]
    if name == 'cnn_lstm':
        epochs = settings['cnn_epochs']
        if not epochs or len(set(epochs)) != len(epochs) or any(type(e) is not int or e < 1 for e in epochs):
            raise ValueError('Distinct positive cnn_epochs required')
        return [dict(family='baseline', name=name, epochs=e) for e in epochs]
    if name not in ('ridge', 'rf'):
        raise ValueError(f'Unsupported baseline {name}')
    return [dict(family='baseline', name=name, parameters=p) for p in model_candidates(name, base)]


def fit_one(spec, raw, data, prep, cache, base, settings, directory):
    directory.mkdir(parents=True, exist_ok=True)
    if spec['family'] == 'linear_blend':
        from .selection_controls import LinearBlend
        return dict(kind='baseline', models=[LinearBlend(spec['alpha'], spec['ridge_weight']).fit(data)])
    if spec['family'] != 'baseline':
        raise ValueError('Only named baselines and scalar blend are supported')
    name = spec['name']
    if name == 'lps':
        return dict(kind='baseline', models=[PatternScaling().fit(data)])
    if name == 'ridge':
        return dict(kind='baseline', models=[RidgeEmulator(**spec['parameters']).fit(data)])
    seeds = base['training']['seeds'] if name == 'cnn_lstm' else base['training']['seeds'][:1]
    models = []
    for seed in seeds:
        if name == 'cnn_lstm':
            import tensorflow as tf
            tf.keras.backend.clear_session()
        model = make_model(name, base, seed, spec.get('parameters', {}), spec.get('epochs'), directory / f'seed_{seed}')
        model = fit_with_deadline(model, data, None, name, base, directory / f'seed_{seed}')
        models.append(model)
    return dict(kind='baseline', models=models)


def predict(bundle, raw, data, prep, features=None):
    values = []
    for model in bundle['models']:
        values.append(model.predict(data))
    return np.mean(values, axis=0)


def selection(base, settings, output, identity, checkpoint=lambda: None):
    study = plan(settings)
    new_or_same(output / 'plan.json', study)
    paths = []
    root = resolve_path(base, base['data']['root'])
    for task in study['tasks']:
        for target in study['targets']:
            # Every pipeline is tuned using the same held-training scenarios.
            choices = {}
            for name in study['models']:
                destination = output / 'selection' / task['name'] / f'{target}__{name}.json'
                marker = destination.with_suffix('.complete.json')
                if marker.exists():
                    record = read_json(marker)
                    if record['fingerprint'] != identity:
                        raise ValueError('Selection checkpoint identity changed')
                    verify(output, record['artifacts'])
                    paths += [destination, marker]
                else:
                    choices[name] = (destination, marker, [], [])
            for held, fitting in folds(task) if choices else []:
                checkpoint()
                print(f"Route 1 selection {task['name']}/{target}, hold out {held}", flush=True)
                raw = collect_samples(root, fitting, target, base)
                if set(s for s, _ in raw.keys) & (set(task['test']) | {held}):
                    raise ValueError('Outer/inner target leakage in coverage training')
                prep = Preprocessor(base).fit(raw)
                data = prep.transform(raw)
                noise,report,cache=None,{},None
                vr = collect_samples(root, [held], target, base)
                vd = prep.transform(vr)
                features = None
                lps_summary = error_summary(PatternScaling().fit(data).predict(vd), vd, [2080, 2100])
                for name, (_, _, summaries, evidence) in choices.items():
                    summaries_this_fold = []
                    for i, spec in enumerate(study['specs'][name]):
                        checkpoint()
                        print(f'  {name} candidate {i+1}/{len(study["specs"][name])}', flush=True)
                        fitted = fit_one(spec, raw, data, prep, cache, base, settings,
                                         output / 'work' / task['name'] / target / held / name / str(i))
                        summaries_this_fold.append(error_summary(predict(fitted, vr, vd, prep, features), vd, [2080, 2100]))
                        del fitted
                    if name in settings['baselines'] and name != 'lps':
                        summaries_this_fold.insert(0, lps_summary)
                    summaries.append((np.stack([x[0] for x in summaries_this_fold]),
                                      np.stack([x[1] for x in summaries_this_fold])))
                    evidence.append(dict(held=held, fitting=fitting, training_key_hash=object_hash(raw.keys),
                                         mask_hash=object_hash(data.mask), noise=report))
                del raw, data, prep, noise, cache, vr, vd, features
            for name, (destination, marker, summaries, evidence) in choices.items():
                specs = study['specs'][name]
                if name in settings['baselines'] or name in SELECTION_CONTROLS:
                    if name in settings['baselines'] and name != 'lps':
                        specs = [dict(family='baseline', name='lps'), *specs]
                    # Select baseline hyperparameters; LPS is normalization only,
                    # never substituted for a named ridge/RF/CNN comparator.
                    from .selection_controls import choose_single
                    first = 1 if name in settings['baselines'] and name != 'lps' else 0
                    chosen = choose_single(summaries, first, settings['v4']['worst_fold_weight'])
                else:
                    raise ValueError('Unsupported pipeline')
                chosen['fold_candidate_spatial_rmse'] = [[float(np.linalg.norm(x)) for x in s[0]] for s in summaries]
                chosen['fold_candidate_global_rmse'] = [[float(np.sqrt(np.mean(x*x))) for x in s[1]] for s in summaries]
                write_json(destination, dict(fingerprint=identity, task=task, target=target, model=name,
                    specs=specs, folds=evidence, **chosen))
                write_json(marker, dict(fingerprint=identity, artifacts=records(output, [destination])))
                paths += [destination, marker]
    freeze(output, identity, [output / 'plan.json', *paths])


def final(base, settings, output, identity, checkpoint=lambda: None):
    verify_frozen(output, identity)
    study = read_json(output / 'plan.json')
    root = resolve_path(base, base['data']['root'])
    for task in study['tasks']:
        for target in study['targets']:
            for name in study['models']:
                checkpoint()
                directory = output / 'runs' / task['name'] / target / name
                marker = directory / 'complete.json'
                if marker.exists():
                    record = read_json(marker)
                    if record['fingerprint'] != identity:
                        raise ValueError('Final checkpoint identity changed')
                    verify(output, record['artifacts'])
                    continue
                print(f"Route 1 final {task['name']}/{target}/{name}", flush=True)
                choice = read_json(output / 'selection' / task['name'] / f'{target}__{name}.json')
                raw = collect_samples(root, task['train'], target, base)
                prep = Preprocessor(base).fit(raw)
                data = prep.transform(raw)
                noise,noise_report,cache=None,{},None
                fitted, saved, tracked = [], [], []
                for i, (weight, spec) in enumerate(zip(choice['weights'], choice['specs'])):
                    if weight <= 0:
                        continue
                    checkpoint()
                    bundle = fit_one(spec, raw, data, prep, cache, base, settings, directory / 'fit' / str(i))
                    fitted.append((weight, bundle))
                    if spec.get('name') == 'cnn_lstm':
                        model_paths = []
                        for seed, model in zip(base['training']['seeds'], bundle['models']):
                            path = directory / f'cnn_seed_{seed}'
                            model.save(path)
                            tracked += sorted(p for p in path.rglob('*') if p.is_file())
                            model_paths.append(path.relative_to(output).as_posix())
                        saved.append((weight, dict(kind='cnn_paths', paths=model_paths)))
                    else:
                        saved.append((weight, bundle))
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint_path = directory / 'model.joblib'
                joblib.dump(dict(preprocessor=prep, models=saved, selection=choice, noise=noise_report,
                                 fingerprint=identity), checkpoint_path)
                tracked.append(checkpoint_path)
                del raw, data, cache, noise
                for scenario in task['test']:
                    vr = collect_samples(root, [scenario], target, base)
                    vd = prep.transform(vr)
                    values = sum(w*predict(b, vr, vd, prep) for w, b in fitted)
                    path = directory / f'{scenario}.nc'
                    save_predictions(path, values, vd, scenario,
                        dict(task=task['name'], target=target, scenario=scenario, model=name, fingerprint=identity))
                    tracked.append(path)
                write_json(marker, dict(fingerprint=identity, artifacts=records(output, tracked)))


def reload_predict(path, raw, data):
    saved = joblib.load(path)
    bundles = []
    root = Path(path).resolve().parents[4]
    for weight, bundle in saved['models']:
        if bundle['kind'] == 'cnn_paths':
            from climate_emulator.models.cnn_lstm import CNNLSTMEmulator
            bundle = dict(kind='baseline', models=[CNNLSTMEmulator.load(root / p) for p in bundle['paths']])
        bundles.append((weight, bundle))
    return sum(w*predict(b, raw, data, saved['preprocessor']) for w, b in bundles)

