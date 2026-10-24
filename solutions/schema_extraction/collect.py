"""Create a bounded evidence archive with per-member hashes; never archive assets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);ap.add_argument('--selected-only',action='store_true');args=ap.parse_args()
    root=Path(args.run);complete=json.loads((root/'complete.json').read_text());assert complete['status']=='COMPLETE_NATIVE_CANDIDATE_EXTRACTION'
    files=sorted(p for p in root.iterdir() if p.is_file() and p.suffix=='.json' and (not args.selected_only or p.name.endswith('.mapped.json') or p.name in ['complete.json','inventory.json','input-census.json']))
    assert len(list(root.glob('*.raw.json')))==len(list(root.glob('*.mapped.json')))==len(list(root.glob('*.candidates.json')))==complete['windows']
    refs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.name)
        z.writestr('member-index.json',json.dumps(refs,indent=2)+'\n')
    manifest=dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),members=len(refs),selected_only=args.selected_only,run_complete=complete,raw_bytes=sum(r['bytes'] for r in refs.values()),member_index=refs)
    out.with_suffix(out.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='member_index'}))
if __name__=='__main__':main()
