"""Read original formal root markers (driver writes them outside output/).

The frozen training/deployment source is untouched. Only the observation string's
root-marker lookup is corrected; no pack/deploy/start entry is exposed here.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import formal_remote as original

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);a=ap.parse_args()
 assert original.OBSERVE.count('value=small_json(output/name)')==2
 original.OBSERVE=original.OBSERVE.replace('value=small_json(output/name)','value=small_json(root/name)',1)
 sys.argv=[original.__file__,'observe','--directory',a.directory]
 original.main()
 p=Path(__file__).resolve();print(json.dumps(dict(actual_readonly_source=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),training_or_launch_modified=False)))
if __name__=='__main__':main()
