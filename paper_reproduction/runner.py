"""Paper-only baseline/blend training, auditing and direct raw-target scoring."""
import argparse,csv,gc,importlib.util,json,os,time
from pathlib import Path
import numpy as np
from climate_emulator.config import load_config
from climate_emulator.data.loader import load_scenario,convert_targets
from climate_emulator.data.members import complete_period_members
from climate_emulator.evaluation.metrics import period,benchmark_metrics,common_arrays
from climate_emulator.evaluation.predictions import load_predictions
from climate_emulator.training.experiment import preflight,environment_versions
from climate_emulator.utils import read_json,write_json,file_hash,object_hash
from branch_study.runner import acquire_lock,csv_write
from . import engine
from .common import records,new_or_same,verify_frozen,verify
from .selection_controls import choose_single
ROOT=Path(__file__).resolve().parents[1]
MODELS=['lps','ridge','rf','cnn_lstm','scalar_blend']

def audit_reference(reference,output=None):
    """Replay all 40 original named-baseline/blend selections without legacy code."""
    if read_json(reference/'status.json')['status']!='complete':raise ValueError('Incomplete reference')
    verify(reference,read_json(reference/'protocol_lock.json')['artifacts'])
    tasks=[t for t in read_json(reference/'plan.json')['tasks'] if t['name'].startswith('coverage_')]
    checked=[]
    for task in tasks:
        for target in ['tas','pr']:
            for model in MODELS:
                d=read_json(reference/'selection'/task['name']/f'{target}__{model}.json')
                spatial=d['fold_candidate_spatial_rmse']
                # choose_single uses spatial norms only; preserve the recorded norms.
                summaries=[(np.asarray(v)[:,None],np.zeros((len(v),21))) for v in spatial]
                first=1 if model in ['ridge','rf','cnn_lstm'] else 0
                actual=choose_single(summaries,first,.25)
                chosen=d['selected_index']
                if actual['selected_index']!=chosen:
                    best=min(float(x[0]) for x in actual['candidate_objectives'])
                    frozen=next(float(x[0]) for x in actual['candidate_objectives'] if int(x[1])==chosen)
                    if not np.isclose(best,frozen,rtol=1e-12,atol=1e-12):raise ValueError('Selection replay differs')
                for score,index in actual['candidate_objectives']:
                    old=next(float(x[0]) for x in d['candidate_objectives'] if int(x[1])==index)
                    if not np.isclose(score,old,rtol=1e-12,atol=1e-12):raise ValueError('Objective differs')
                checked.append(dict(task=task['name'],target=target,model=model,selected_index=chosen))
    if len(checked)!=40:raise ValueError('Expected 40 original selections')
    if output:
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'original_selection_replay.json',dict(checked=checked,reference=str(reference),
            scope='Saved selection statistics replay; does not establish equivalence of new fits.'))
    print('Original selection replay: 40/40 passed',flush=True)
    return checked

def evaluate(base,settings,out,identity,reference=None):
    verify_frozen(out,identity)
    rows=[];comparison=[]
    oldrows=[]
    if reference:
        with (reference/'metrics/metrics.csv').open(newline='') as f:oldrows=list(csv.DictReader(f))
    for task in engine.plan(settings)['tasks']:
        for target in ['tas','pr']:
            ref=convert_targets(load_scenario(base['data']['root'],'ssp585').targets,base['data']['missing_unit_assumptions'])[target]
            members=complete_period_members(period(ref,[2080,2100]));truth=members.mean('member',skipna=False)
            masks=[]
            for model in MODELS:
                folder=out/'runs'/task['name']/target/model
                receipt=read_json(folder/'complete.json')
                verify(out,receipt['artifacts'])
                pred,_,mask,_=load_predictions(folder/'ssp585.nc')
                masks.append(mask)
                if not np.array_equal(mask,masks[0]):raise ValueError('Different model scoring masks')
                metric=benchmark_metrics(truth,period(pred,[2080,2100]),mask)
                if metric['valid_grid_cells']!=int(np.prod(base['data']['expected_grid'])):raise ValueError('Incomplete grid')
                for name in ['spatial_rmse','global_rmse','nrmse_total']:
                    if metric[name] is None:raise ValueError('Undefined metric')
                    rows.append(dict(task=task['name'],target=target,scenario='ssp585',model=model,
                        reference='ensemble_mean',metric=name,value=metric[name]))
                    if reference:
                        old=next(float(r['value']) for r in oldrows if all(r[k]==rows[-1][k] for k in ['task','target','scenario','model','reference','metric']))
                        delta=metric[name]-old
                        comparison.append(dict(task=task['name'],target=target,model=model,metric=name,
                            original=old,current=metric[name],difference=delta,matches=bool(np.isclose(metric[name],old,rtol=2e-5,atol=2e-8))))
    (out/'metrics').mkdir(exist_ok=True)
    csv_write(out/'metrics/metrics.csv',rows)
    if comparison:csv_write(out/'metrics/original_comparison.csv',comparison)
    write_json(out/'metrics/evaluation_status.json',dict(status='complete',reference='complete SSP585 member mean',
        all_original_metrics_match=all(r['matches'] for r in comparison) if comparison else None,
        note='Inspect any mismatch before replacing published numbers; neural fits can vary across runtimes.'))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','audit-reference','preflight','selection','final','evaluate','all'],default='plan')
    p.add_argument('--data',type=Path,default=Path('../data/raw'))
    p.add_argument('--output',type=Path,default=Path('../results/publication-paper-minimal-v1'))
    p.add_argument('--reference',type=Path)
    p.add_argument('--recover-stale-lock',action='store_true')
    args=p.parse_args(argv)
    if args.stage=='audit-reference':
        if not args.reference:p.error('--reference required')
        audit_reference(args.reference.resolve(),args.output.resolve());return
    config=ROOT/'configs/core.yaml'
    base=load_config(config)
    settings=dict(base_config=str(config),baselines=MODELS[:-1],cnn_epochs=[20,40,80],
        selector=dict(scalar_weights=[0.,.25,.5,.75,1.]),v4=dict(worst_fold_weight=.25))
    # v4 is only the legacy field name for objective coefficient .25, not a model import.
    plan=engine.plan(settings)
    if args.stage=='plan':
        print(json.dumps(dict(**plan,legacy_model_folders_required=False,previous_runs_required=False),indent=2));return
    data,out=args.data.resolve(),args.output.resolve()
    reference=args.reference.resolve() if args.reference else None
    if (out==data or out.is_relative_to(data) or data.is_relative_to(out) or
        reference and (out==reference or out.is_relative_to(reference) or reference.is_relative_to(out))):
        raise ValueError('Raw data, reference and output must be separate')
    if not out.name.startswith('publication-paper-minimal'):raise ValueError('Use publication-paper-minimal* output')
    for key in ['root','archives','manifests','processed']:
        base['data'][key]=str(data if key=='root' else data.parent/key)
    if importlib.util.find_spec('tensorflow') is None:raise ImportError('TensorFlow required to reproduce CNN-LSTM baseline')
    provenance=preflight(base)
    identity=dict(config={k:v for k,v in base.items() if not k.startswith('_')},provenance=provenance,
        runtime=environment_versions(),code={p.relative_to(ROOT).as_posix():file_hash(p) for d in ['src','paper_reproduction'] for p in sorted((ROOT/d).rglob('*.py'))},
        plan=plan,reference_metrics=file_hash(reference/'metrics/metrics.csv') if reference else None)
    out.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(out,args.recover_stale_lock)
    try:
        new_or_same(out/'experiment.json',identity)
        if reference:audit_reference(reference,out)
        if args.stage=='preflight':print('Preflight passed; no fits started');return
        write_json(out/'status.json',dict(status='running',stage=args.stage,pid=os.getpid()))
        if args.stage in ['selection','all']:engine.selection(base,settings,out,identity);gc.collect()
        if args.stage in ['final','all']:engine.final(base,settings,out,identity);gc.collect()
        if args.stage in ['evaluate','all']:evaluate(base,settings,out,identity,reference)
        write_json(out/'status.json',dict(status='complete',stage=args.stage))
        print('Paper-only stage complete: '+str(out),flush=True)
    except BaseException as exc:
        write_json(out/'status.json',dict(status='failed',stage=args.stage,error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
