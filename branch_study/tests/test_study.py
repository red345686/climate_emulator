from copy import deepcopy
import csv
from pathlib import Path

import numpy as np
import pytest

from branch_study import runner
from branch_study.math import summary, scores, objective, select, complementarity, ORIGINAL_ALPHAS, ORIGINAL_WEIGHTS
from climate_emulator.data.preprocessing import collect_samples, Preprocessor
from climate_emulator.models.pattern_scaling import PatternScaling
from climate_emulator.models.ridge import RidgeEmulator
from climate_emulator.evaluation.predictions import save_predictions
from climate_emulator.utils import write_json, file_hash, read_json


def test_error_identity_and_cancellation():
    # Opposite branch errors cancel exactly at w=.5.
    truth = np.ones((21, 4))
    weights = np.array([.1,.2,.3,.4])
    left = summary(truth+2, truth, weights)
    right = summary(truth-2, truth, weights)
    result = complementarity(left, right, .5, 4)
    assert result['spatial_error_cosine'] == pytest.approx(-1)
    assert result['blend_spatial_mse'] == 0
    assert result['outer_oracle_weight_diagnostic_only'] == .5
    assert result['identity_absolute_error'] < 1e-12
    assert scores(.5*left+.5*right, 4, 2)['nrmse_total'] == 0
    assert scores(left, 4, 0)['nrmse_total'] is None


def test_selection_is_original_grid_only_and_deterministic():
    lps = [np.r_[1., np.zeros(21)]]
    branches = {a: [np.r_[-1., np.zeros(21)]] for a in ORIGINAL_ALPHAS}
    result = select(lps, branches, 1)
    assert result['alpha'] == .1 and result['weight'] == .5
    assert objective(lps, branches[.1], .5, 1) == 0
    equal = select(lps, {a:lps for a in ORIGINAL_ALPHAS}, 1)
    assert equal['alpha'] is None and equal['weight'] == 0


def synthetic_source(base, rawroot, source):
    pools = [('ssp126','ssp245'), ('ssp126','ssp370'), ('ssp245','ssp370'), ('ssp126','ssp245','ssp370')]
    tasks = [dict(name='coverage_'+'_'.join(pool), train=['historical',*pool], test=['ssp585']) for pool in pools]
    source.mkdir()
    write_json(source/'plan.json', dict(tasks=tasks))
    write_json(source/'status.json', dict(status='complete'))
    tracked = [source/'plan.json']
    metrics = []
    for task in tasks:
        for target in ('tas','pr'):
            specs = [dict(family='baseline',name='lps')]+[dict(family='linear_blend',alpha=a,ridge_weight=w) for a in ORIGINAL_ALPHAS for w in ORIGINAL_WEIGHTS]
            lps, branches, provenance = [], {a:[] for a in ORIGINAL_ALPHAS}, []
            for held in task['train'][1:]:
                fitting = [s for s in task['train'] if s != held]
                raw = collect_samples(rawroot, fitting, target, base)
                prep = Preprocessor(base).fit(raw)
                train = prep.transform(raw)
                vr = collect_samples(rawroot, [held], target, base)
                valid = prep.transform(vr)
                values = {'lps':PatternScaling().fit(train).predict(valid)}
                values.update({f'full_{a:g}':RidgeEmulator(a).fit(train).predict(valid) for a in ORIGINAL_ALPHAS})
                stats,cells = runner.period_summaries(values, valid)
                lps.append(stats['lps'])
                for a in ORIGINAL_ALPHAS:
                    branches[a].append(stats[f'full_{a:g}'])
            objs = [objective(lps,lps,0.,cells)]+[objective(lps,branches[a],w,cells) for a in ORIGINAL_ALPHAS for w in ORIGINAL_WEIGHTS]
            selected = min(range(len(objs)),key=lambda i:(objs[i],i))
            weights = [float(i==selected) for i in range(len(objs))]
            choice = dict(task=task,target=target,specs=specs,selected_index=selected,weights=weights,
                          candidate_objectives=[[v,i] for i,v in enumerate(objs)])
            path = source/'selection'/task['name']/f'{target}__scalar_blend.json'
            write_json(path,choice); tracked.append(path)
            raw=collect_samples(rawroot,task['train'],target,base)
            prep=Preprocessor(base).fit(raw); train=prep.transform(raw)
            vr=collect_samples(rawroot,['ssp585'],target,base); valid=prep.transform(vr)
            lp=PatternScaling().fit(train).predict(valid)
            spec=specs[selected]
            w=spec.get('ridge_weight',0.)
            prediction=lp if w==0 else (1-w)*lp+w*RidgeEmulator(spec['alpha']).fit(train).predict(valid)
            path=source/'runs'/task['name']/target/'scalar_blend'/'ssp585.nc'
            save_predictions(path,prediction,valid,'ssp585',dict(target=target))
            write_json(path.parent/'complete.json',dict(artifacts={path.relative_to(source).as_posix():file_hash(path)}))
            stats,cells=runner.period_summaries({'blend':prediction,'lps':lp},valid)
            positions=[i for i,(_,y) in enumerate(valid.keys) if 2080<=y<=2100]
            ww=valid.weights[valid.mask];ww=ww/ww.sum()
            denominator=abs(float(valid.target[positions][:,valid.mask].mean(axis=0)@ww))
            for model in ('scalar_blend',*runner.BASELINES):
                scored=scores(stats['blend' if model=='scalar_blend' else 'lps'],cells,denominator)
                for metric,value in scored.items():
                    metrics.append(dict(task=task['name'],target=target,scenario='ssp585',model=model,
                                        reference='ensemble_mean',metric=metric,value=value))
    (source/'metrics').mkdir()
    runner.csv_write(source/'metrics/metrics.csv',metrics)
    write_json(source/'protocol_lock.json',dict(artifacts={p.relative_to(source).as_posix():file_hash(p) for p in tracked}))
    return tasks


def test_end_to_end_resume_freeze_baselines_and_tampering(config,dataset,tmp_path,monkeypatch):
    source=tmp_path/'source';output=tmp_path/'publication-branch-analysis-test'
    tasks=synthetic_source(config,dataset,source)
    # Keep the synthetic test on the complete original alpha set plus one neighbor.
    monkeypatch.setattr(runner,'ALPHAS',[.03,*ORIGINAL_ALPHAS])
    monkeypatch.setattr(runner,'WEIGHTS',ORIGINAL_WEIGHTS)
    monkeypatch.setattr(runner,'load_config',lambda _:deepcopy(config))
    arguments=['--source',str(source),'--data',str(dataset),'--output',str(output)]
    runner.main(arguments)
    assert read_json(output/'status.json')['status']=='complete'
    with (output/'each_baseline_comparison.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==8*6*4*3
    assert {r['baseline'] for r in rows}==set(runner.BASELINES)
    with (output/'training_sensitivity.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==8*2*5*5
    decisions=read_json(output/'decisions_freeze.json')
    timestamps={p:p.stat().st_mtime_ns for p in output.glob('final/*/*/complete.json')}
    runner.main(arguments)
    assert read_json(output/'decisions_freeze.json')==decisions
    assert all(p.stat().st_mtime_ns==t for p,t in timestamps.items())
    fold=next(output.glob('inner/*/*/ssp*/summaries.npz'))
    fold.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Changed/missing artifact'):
        runner.main(arguments)


def test_lock_refuses_existing_and_remote(tmp_path,monkeypatch):
    output=tmp_path/'out';output.mkdir()
    lock=runner.acquire_lock(output)
    with pytest.raises(FileExistsError): runner.acquire_lock(output)
    record=read_json(lock);record['host']='unrelated-host';write_json(lock,record)
    with pytest.raises(RuntimeError,match='same host'):runner.acquire_lock(output,recover=True)


def test_output_must_not_overlap_source(tmp_path):
    source=tmp_path/'publication-branch-analysis-test';source.mkdir()
    with pytest.raises(ValueError,match='separate'):
        runner.main(['--source',str(source),'--output',str(source)])
