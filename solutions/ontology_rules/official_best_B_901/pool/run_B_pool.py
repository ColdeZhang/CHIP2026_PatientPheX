from pathlib import Path
import sys,json,subprocess,time,datetime,hashlib
D=Path(__file__).parent;O=D.parent;R=O.parents[2];S=R/'research/official-901-B/old-pool-cpu';W=Path('/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/official_best_B_901/pool')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(D/n).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
mode=sys.argv[1]
if mode=='bind':
 assert not (D/'B-plan.json').exists() and not (D/'B-prepared').exists()
 args={'root-contract':R/'official-highest-901-B-20261024/input-contract.json','template':D/'B-plan.template.json','source':W/'build_pool.py','base-manifest':S/'rules-table/manifest.json','crf-manifest':S/'B-crf/manifest.json','sap-lexical-manifest':S/'sap-table/manifest.json','sap-scored-manifest':S/'actual-run/collected/B-scored/manifest.json','sap-scored-candidates':S/'actual-run/collected/B-scored/scored-candidates.jsonl.gz','batch-plan':O/'batch-plan.json','output-plan':D/'B-plan.json','output-dir':D/'B-prepared'}
 for k,m in [('pubmed','pubmedbert'),('bioformer','bioformer'),('biobert','biobert')]:args[k+'-cache']=O/'results'/m/'terminal/native-B901-final-cache'
 command=[sys.executable,str(W/'bind_B_plan.py')]+sum([['--'+k,str(v)] for k,v in args.items()],[])
 dump('binding-entry.json',dict(at=now(),source_sha256=sha(__file__),command=command,earlier_owner_CLI_typo='One initial direct invocation used a wrong Bioformer cache directory; stopped before writing B plan/pool, no data/model mutation. This entry uses one common root.'))
 p=subprocess.run(command,capture_output=True,text=True);dump('binding-result.json',dict(at=now(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr;print(p.stdout.strip())
elif mode=='build':
 assert (D/'B-plan.json').exists() and not (D/'B-prepared').exists()
 with (D/'build-start-intent.json').open('x') as f:json.dump(dict(at=now(),plan_sha256=sha(D/'B-plan.json'),source_sha256=sha(W/'build_pool.py'),CPU_cap_seconds=120,TERM_grace_seconds=5),f)
 command=['timeout','--signal=TERM','--kill-after=5s','120s',sys.executable,str(W/'build_pool.py'),'--plan',str(D/'B-plan.json')];start=time.monotonic()
 with (D/'B-build.stdout').open('w') as out,(D/'B-build.stderr').open('w') as err:p=subprocess.run(command,stdout=out,stderr=err,timeout=130)
 dump('build-execution.json',dict(at=now(),command=command,returncode=p.returncode,elapsed_seconds=time.monotonic()-start,plan_sha256=sha(D/'B-plan.json'),source_sha256=sha(W/'build_pool.py'),no_model_fit_predict_or_quality=True));assert p.returncode==0,(D/'B-build.stderr').read_text();print((D/'B-build.stdout').read_text())
else:raise AssertionError(mode)
