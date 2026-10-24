"""After an actual explicit fence, bind one unchanged normalization with finite admission."""
import json,time,datetime
from pathlib import Path
from prepare import S,D,P,ref,read,save
oldroot=P+'/copy-grounded-decode-v1/F0-proposer-v1/normalize-v1';newroot=P+'/copy-grounded-decode-v1/F0-proposer-v1/normalize-recovery-v3'
fence=read(D/'unknown-fence-v3/complete.json');assert fence['actual']['status']=='FENCED_BEFORE_MODEL' and fence['actual']['value']['schema']=='FENCED_UNKNOWN_LAUNCH'
p=read(S/'normalize-plan-essential-v2.json');original=ref(S/'normalize-plan-essential-v2.json')
def move(v):
 if isinstance(v,str):return v.replace(oldroot,newroot)
 if isinstance(v,list):return [move(x) for x in v]
 if isinstance(v,dict):return {move(k):move(x) for k,x in v.items()}
 return v
p=move(p);p['operations_source']=ref(S/'remote_recovery_v3.py');p['collection_source']=ref(S/'collect_recovery_v3.py');p['binding_source']=ref(__file__)
p['payload_sources']['job_driver.py']=ref(S/'job_driver_admission_v3.py');p['payload_sources']['normalize_entities.py']=ref(S/'normalize_entities_admission_v3.py');p['payload_sources']['admission_v3.py']=ref(S/'admission_v3.py')
p['runner']=p['payload_sources']['normalize_entities.py'];p['files'].update({newroot+'/'+k:{n:v[n] for n in ['bytes','sha256']} for k,v in p['payload_sources'].items()})
p['reuse_packages']=[ref(D/'normalize-v1/package.json')];p['input_sources'].update(original_plan=original,original_UNKNOWN=ref(D/'normalize-v1/transport/start-1792855867731525732.error.json'),explicit_fence=ref(D/'unknown-fence-v3/complete.json'))
now=time.time();p.update(launch_plan_frozen_unix=now,launch_not_after_unix=now+300,launch_not_after_utc=datetime.datetime.fromtimestamp(now+300,datetime.timezone.utc).isoformat(),recovery='Independent namespace after explicit original late-start tombstone; unchanged296 queries/inputs/weights/policy, no original inference rerun',authorization='Root explicitly authorized fenced UNKNOWN recovery. Plan/request/driver/normalizer all enforce five-minute start validity; unique actual child retains180+30 runtime budget')
p['environment']['COPY_PROPOSER_LAUNCH_VALID_UNTIL']=str(now+300)
assert p['budget_seconds']==180 and p['grace_seconds']==30 and p['necessary_query_count']==296
save(S/'normalize-recovery-plan-v3.json',p);print(json.dumps(ref(S/'normalize-recovery-plan-v3.json')))
