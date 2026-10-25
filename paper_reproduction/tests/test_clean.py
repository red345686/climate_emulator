from pathlib import Path
from copy import deepcopy
import ast,csv
import numpy as np
import pytest
from paper_reproduction import engine,runner
from paper_reproduction.selection_controls import LinearBlend
from climate_emulator.models.pattern_scaling import PatternScaling
from climate_emulator.models.ridge import RidgeEmulator
from climate_emulator.utils import read_json,file_hash

def test_relocated_audit_preserves_hashes_and_rejects_ambiguity(tmp_path):
    from climate_emulator.training.experiment import relocated_audit_records
    f=tmp_path/'inputs_historical.nc';f.write_bytes(b'original-data')
    report={'files':[{'file':r'C:\old-server\data\raw\inputs_historical.nc','sha256':file_hash(f)}]}
    result=relocated_audit_records(report,{'historical':{'inputs':f}})
    assert result[f.resolve()]['sha256']==file_hash(f)
    assert report['files'][0]['file'].startswith('C:')
    other=tmp_path/'other'/'inputs_historical.nc'
    with pytest.raises(ValueError,match='one current raw file'):
        relocated_audit_records(report,{'a':{'inputs':f},'b':{'inputs':other}})
    f.write_bytes(b'changed-data')
    assert result[f.resolve()]['sha256']!=file_hash(f)

def test_no_legacy_imports():
    root=Path(__file__).resolve().parents[2]
    for folder in ['paper_reproduction','branch_study','src']:
        for p in (root/folder).rglob('*.py'):
            for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
                if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('improved_','publication_routes'))
                if isinstance(node,ast.Import):assert not any(a.name.startswith(('improved_','publication_routes')) for a in node.names)

def test_blend_is_original_formula(prepared):
    data,_,_=prepared
    expected=.75*PatternScaling().fit(data).predict(data)+.25*RidgeEmulator(10.).fit(data).predict(data)
    assert np.allclose(LinearBlend(10.,.25).fit(data).predict(data),expected,equal_nan=True,atol=1e-12)

def test_classical_training_scoring_resume(tmp_path,dataset,config,monkeypatch):
    cfg=deepcopy(config);cfg['data']['root']=str(dataset)
    settings=dict(base_config='unused',baselines=['lps','ridge'],cnn_epochs=[20,40,80],
        selector=dict(scalar_weights=[0.,.25,.5,.75,1.]),v4=dict(worst_fold_weight=.25))
    monkeypatch.setattr(engine,'load_config',lambda _:cfg)
    plan=engine.plan(settings);plan['tasks']=plan['tasks'][:1]
    monkeypatch.setattr(engine,'plan',lambda _:plan)
    monkeypatch.setattr(runner,'MODELS',['lps','ridge','scalar_blend'])
    output=tmp_path/'run';output.mkdir()
    identity={'test':'synthetic-classical'}
    engine.selection(cfg,settings,output,identity)
    frozen=file_hash(output/'protocol_lock.json')
    engine.final(cfg,settings,output,identity)
    runner.evaluate(cfg,settings,output,identity)
    assert read_json(output/'metrics/evaluation_status.json')['status']=='complete'
    with (output/'metrics/metrics.csv').open() as f:assert len(list(csv.DictReader(f)))==18
    times={p:p.stat().st_mtime_ns for p in output.glob('runs/*/*/*/complete.json')}
    engine.selection(cfg,settings,output,identity);engine.final(cfg,settings,output,identity)
    assert file_hash(output/'protocol_lock.json')==frozen
    assert all(p.stat().st_mtime_ns==t for p,t in times.items())
    path=next(output.glob('runs/*/*/*/ssp585.nc'));path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact'):runner.evaluate(cfg,settings,output,identity)
