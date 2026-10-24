"""One synthetic stdlib metadata repair check; no original CPU/model replay."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-source',type=Path,required=True);args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));started=time.monotonic();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    evidence.fsync_chain(out,out.parent)
    def function(source):
        node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='json_default')
        return ast.get_source_segment(source,node)
    legacy=function(args.legacy_source.read_text());new=function(Path(evidence.__file__).read_text())
    assert legacy==new
    payload=dict(missing_keys=set(),unexpected_keys={'z.key','a.key'},mismatched_keys=frozenset(),
                 error_msgs=[],nested=[{'values':frozenset({'b','a'})}],mixed={1,'x'})
    evidence.save(out/'synthetic-loading.json',payload)
    result=json.loads((out/'synthetic-loading.json').read_text())
    assert result==dict(missing_keys=[],unexpected_keys=['a.key','z.key'],mismatched_keys=[],error_msgs=[],
                        nested=[{'values':['a','b']}],mixed=sorted({1,'x'},key=repr))
    unknown=False
    try:evidence.save(out/'unknown.json',{'value':object()})
    except TypeError as error:
        assert str(error)=='Unsupported evidence value type: object';unknown=True
    assert unknown and not (out/'unknown.json').exists() and (out/'unknown.json.tmp').is_file()
    nonfinite=False
    try:evidence.save(out/'nonfinite.json',{'value':float('nan')})
    except ValueError:nonfinite=True
    assert nonfinite and not (out/'nonfinite.json').exists()
    assert not any(k in sys.modules for k in ('torch','fla','triton','transformers'))
    evidence.save(out/'result.json',dict(status='PASS_SYNTHETIC_METADATA_SERIALIZATION_REPAIR',
        elapsed_seconds=time.monotonic()-started,at_utc=datetime.now(timezone.utc).isoformat(),
        source=evidence.ref(__file__),repaired_evidence=evidence.ref(evidence.__file__),legacy=evidence.ref(args.legacy_source),
        copied_function_source_exact=True,copied_function_sha256=hashlib.sha256(legacy.encode()).hexdigest(),
        empty_set=True,nonempty_set=True,frozenset=True,nested_set=True,mixed_sorted_repr=True,
        unknown_still_TypeError=True,allow_nan_false_preserved=True,original_failed_loading_recovered=False,
        original_CPU_input_contract_reexecuted=False,GPU_imported=False,model_forward=False,gold_read=False))
    print(json.dumps(evidence.ref(out/'result.json')))


if __name__=='__main__':main()
