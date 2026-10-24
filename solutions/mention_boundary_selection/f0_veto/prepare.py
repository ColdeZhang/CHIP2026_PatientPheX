"""Complete fixed F0 options with original public Top8; no model or labels."""
import argparse,json,hashlib
from pathlib import Path
from schema import SYSTEM

def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();d=a.directory
    plan=json.loads((Path(__file__).parent/'plan.json').read_text())
    for k in ['pending','cached_top8','queries']:
        x=plan['input_refs'][k];assert ref(Path(x['path']))==x
    pending=[json.loads(x) for x in (d/'pending.jsonl').read_text().splitlines()]
    old=json.loads((d/'cached-top8.json').read_text());new=json.loads((d/'query-v1-collected/actual/results.json').read_text())
    complete=json.loads((d/'query-v1-collected/actual/complete.json').read_text())
    assert complete['queries']==373 and complete['long_unique_queries']==0
    assert ref(d/'query-v1-collected/actual/results.json')['sha256']==complete['results_sha256']
    bank={}
    for x in old+new:
        assert x['quote'] not in bank and len(x['candidates'])==8
        assert x['status']=='TOP8_PUBLIC_DISTINCT_HPO';bank[x['quote']]=x
    assert len(bank)==538 and {x['quote'] for x in pending}==set(bank)
    names_path=Path('/workspace/patientphex-runs/round2-20261023/research/mention-boundary-selection-v1/prepared-v1/public-names.json')
    assert ref(names_path)['sha256']=='f7a4052f2bf89511ec5d509b778aba687d22d90c9374b2b726680c35f12f7142'
    names=json.loads(names_path.read_text());canonical={v:k for k,v in names.items()};rows=[]
    for row in pending:
        options=bank[row['quote']]['candidates'];assert len({c['identifier'] for c in options})==8
        candidates=[{'code':f'C{i+1:02d}','identifier':c['identifier'],'name':canonical[c['identifier']],'score':c['score']} for i,c in enumerate(options)]
        rendered='\n'.join(c['code']+' | '+c['identifier']+' | '+c['name'] for c in candidates)
        x={k:row[k] for k in ['row_id','pmc_id','quote','offset','length']};x['candidates']=candidates
        x['messages']=[{'role':'system','content':SYSTEM},{'role':'user','content':row['user_body']+'\n\nCandidate options (retrieval rank):\n'+rendered}]
        assert 'answer' not in x and 'target_status' not in x;rows.append(x)
    assert len(rows)==1085 and len({x['row_id'] for x in rows})==1085
    out=d/'prepared-v1';out.mkdir(exist_ok=False)
    p=out/'held.jsonl';p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    m={'status':'COMPLETE_FIXED1085_F0_NO_GOLD_INPUTS','held_rows':1085,'documents':16,'patients':47,
       'files':{'held':ref(p)},'public_names':ref(names_path),'source':ref(Path(__file__)),
       'scope':'Every original strongF0 span, original +/-256 context and public Top8; no selected predicted HPO used as an option override',
       'cached_top8':ref(d/'cached-top8.json'),'new_top8':ref(d/'query-v1-collected/actual/results.json'),
       'parent_schema':ref(Path(__file__).parent/'schema.py'),'no_gold_model_fit_quality':True}
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))

if __name__=='__main__':main()
