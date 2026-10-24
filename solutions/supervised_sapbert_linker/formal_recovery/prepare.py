"""Freeze authorized formal source/quality plans; reuse exact native CPU bytes."""
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023')
B=R/'research/supervised-linker-readiness-20261024';D=B/'formal-recovery-v1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'


def ref(p):
    p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    D.mkdir(exist_ok=True)
    old=json.loads((S.parent/'engineering/execution-plan.json').read_text())
    cpu=B/'probe-v2/collected/cpu-complete.json';native=json.loads(cpu.read_text())
    probe=B/'probe-v2/collected/probe-output/complete.json';actual=json.loads(probe.read_text())
    assert actual['status']=='PASS_REAL_ENGINEERING_PROBE_DISCARD_STATE'
    def inner(p):
        main=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='main')
        return {x.name:ast.dump(x,include_attributes=False) for x in main.body if isinstance(x,ast.FunctionDef)}
    a=inner(S.parent/'engineering/probe.py');b=inner(S/'train.py')
    assert all(a[k]==b[k] for k in ['inputs','encode','run_batch'])
    contract=dict(status='PASS_SOURCE_AST_REUSE_OF_ACTUAL_NEURAL_CORE',functions=['inputs','encode','run_batch'],probe=ref(probe),
        source=ref(S/'train.py'),native_cpu=ref(cpu),no_new_model_probe=True)
    write(S/'source-contract.json',contract)
    plan=dict(schema='supervised-sapbert-fresh-formal-recovery-v1',authorization='Root approved single600+TERM30 full3epoch219+4bank+final durablestate+303; sole F1 quality after full freeze',
        candidate_plan=ref(S.parent/'candidate-plan.json'),budget_seconds=600,term_grace_seconds=30,absolute_term_deadline_unix=1792857849.943514,
        numerical_amendment=dict(init_scale_from=65536,init_scale_to=4096,only_other_change='Before-batch identity and finite-gradient fault metadata; no silent skipped Adam',failed_formal='formal-v1:73 completeupdates+partialepoch2; no resumable checkpoint',root_authorized=True),
        launch=dict(prebound_validity_seconds=300,check_at=['entry','before_Popen'],exclusive='job/exclusive-launch.json'),
        model=old['model'],model_files=old['model_files'],python=old['python'],gpu_uuid=old['gpu_uuid'],environment=old['environment'],
        native_plan_sha256=native['plan']['sha256'],data_sha256=native['data']['sha256'],
        source_files={p.name:ref(p) for p in [S/'common.py',S/'train.py',S/'driver.py']},
        epochs=3,updates=219,batch=16,tail=8,warmup=11,seed=20261024,lr=1e-5,temperature=.05,
        decoder='Final fullpublicbank argmax first row on exact numerical tie; ambiguous alias retained old ID; no threshold',
        fresh_from_public_base=True,probe_state_forbidden=True,checkpoint='Only finalepoch3 fullmodel/optimizer/schedule/scaler/RNG ACK; no epoch choice',
        growth_limit=dict(max_candidates=46,max_native_length=27,observed_probe_peak_bytes=actual['peak_allocated_bytes'],
            measured_cost_estimate_seconds=actual['estimated_formal_seconds'],formal_budget_seconds=600),
        exact_native_contract=ref(cpu),source_contract=ref(S/'source-contract.json'))
    write(S/'formal-plan.json',plan)
    rel='research/supervised-sapbert-linker/formal-recovery-v1';remote=P+'/'+rel
    members={p.name:p.read_bytes() for p in [S/'common.py',S/'train.py',S/'driver.py',S/'formal-plan.json']}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in members.items():z.writestr(name,data)
    (D/'package.zip').write_bytes(buf.getvalue())
    package=dict(mount='1593f159-19bc-41ca-8b4c-2e7455f5db99',relative=rel,remote=remote,archive=ref(D/'package.zip'),plan=ref(S/'formal-plan.json'),
        members={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in members.items()},
        reuse_files={'data.json.gz':native['data'],'cpu-complete.json':{**ref(cpu),'path':P+'/research/supervised-sapbert-linker/probe-v2/cpu-complete.json'}})
    write(D/'package.json',package)
    ready=json.loads((B/'owner-v1/supply.json').read_text());refs=ready['resources']
    quality=dict(schema='supervised-sapbert-only-complete-F1-endpoint-v1',collected=str(D/'collected'),output=str(D/'quality-v1'),
        refs=dict(baseline=refs['strong_baseline'],baseline_metrics=ref(R/'research/lightgbm-selector-readiness/formal-control/fold1/lightgbm.metrics.json'),
            input=refs['F1_input'],eligible=ref(B/'owner-v1/F1-eligible.json'),
            gold=dict(path='/workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl',sha256='665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3')),
        evaluator_dir='/workspace/CHIP2026_PatientPheX/scripts',hpo='/workspace/patientphex-data/hp.obo',
        source=ref(S/'quality.py'),baseline_score=.7286975364690524,planned_new_v2=1,old_v2=0,zero_changes='byteexact prediction/0newv2',
        formula='Only422 eligible IDs replace with final Sap fullbank top1; protected891 and all spans/text/order/relations retained',
        secondary_grounded_quote_application='Not executed or part of this endpoint; root may freeze independent follow-up later')
    write(S/'quality-plan.json',quality)
    print(json.dumps(dict(package=ref(D/'package.zip'),plan=ref(S/'formal-plan.json'),quality_plan=ref(S/'quality-plan.json'))))


if __name__=='__main__':main()
