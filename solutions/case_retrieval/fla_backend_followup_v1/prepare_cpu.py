"""One <=60s stdlib contract: pure inputs, exact index map, NaN persistence."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback
from evidence import checked,fsync_chain,jsha,persist_then_observe,positions,ref,save,softmax


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));started=time.monotonic()
    plan=json.loads(args.plan.read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fsync_chain(out,out.parent);save(out/'intent.json',dict(plan=ref(args.plan),source=ref(__file__),cpu_hard_seconds=60))
    result=dict(status='CPU_RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),model_loaded=False,
                GPU_imported=False,gold_read=False,quality_calls=0)
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        assert plan['execution_ready'] is False and plan['scope']=='A_ONLY_NO_AUTOMATIC_B_C'
        for name,expected in plan['sources'].items():
            actual=ref(Path(__file__).parent/name)
            assert all(actual[k]==expected[k] for k in ('bytes','sha256'))
        sources=plan['CPU_references'];actual={name:ref(checked(record)) for name,record in sources.items()}
        synthetic=json.loads(Path(sources['synthetic']['path']).read_text())[0]
        encoded=[json.loads(line) for line in Path(sources['encoded']['path']).read_text().splitlines()]
        audit=json.loads(Path(sources['audit']['path']).read_text());row=encoded[19]
        assert len(synthetic['input_ids'])==128 and len(synthetic['teacher_forced_ids'])==8
        assert row['index']==19 and row['row_id']=='12455887:relations:40' and len(row['input_ids'])==14390
        assert jsha(row['input_ids'])==audit[19]['reverse']['input_ids_sha256']==plan['long_input_ids_sha256']
        selected=positions(14390)
        assert selected==plan['long_positions'] and [list(range(14390))[i] for i in selected]==selected
        vector=[-3.,0.,1.,4.];shifted=[x+17.25 for x in vector]
        assert max(abs(a-b) for a,b in zip(softmax(vector),softmax(shifted)))<1e-14
        centered=lambda v:[x-sum(v)/len(v) for x in v]
        assert centered(vector)==centered(shifted)
        # Same writer/observer ordering used by actual tensors, no torch imported.
        path=out/'synthetic-nonfinite.bin';budget=dict(used_bytes=0,limit_bytes=1024)
        observed=[]
        def observe(saved):
            assert saved.is_file() and saved.with_suffix('.bin.raw.json').is_file()
            value=struct.unpack('<d',saved.read_bytes())[0];observed.append(True)
            return dict(finite=math.isfinite(value),statistics=None)
        caught=False
        try:persist_then_observe(path,lambda h:h.write(struct.pack('<d',float('nan'))),observe,budget,dict(kind='synthetic_nan'))
        except AssertionError as error:
            assert str(error)=='Nonfinite evidence retained before failure';caught=True
        assert caught and observed==[True] and path.with_suffix('.bin.observation.json').is_file()
        assert math.isnan(struct.unpack('<d',path.read_bytes())[0])
        inputs=dict(synthetic128=dict(input_ids=synthetic['input_ids'],positions=list(range(128))),
                    fulltext14390=dict(input_ids=row['input_ids'],row_id=row['row_id'],original_index=19,positions=selected),
                    teacher_forced_ids=synthetic['teacher_forced_ids'])
        save(out/'fixed-inputs.json',inputs)
        assert not any(name=='torch' or name=='transformers' or name=='triton' or name=='fla' or name.startswith('fla.') for name in sys.modules)
        result.update(status='PASS_CPU_A_ONLY_INPUT_INDEX_AND_SAVE_ORDER',references=actual,fixed_inputs=ref(out/'fixed-inputs.json'),
            synthetic128_input_ids_sha256=jsha(synthetic['input_ids']),long_input_ids_sha256=jsha(row['input_ids']),
            long_positions=selected,teacher_forced_ids_sha256=jsha(synthetic['teacher_forced_ids']),
            constant_shift_softmax_and_centered_exact=True,actual_long_index_map_exact=True,
            nonfinite_raw_saved_before_observation_and_assert=True,native_modeling_math_reused_not_reexecuted=plan['head_review'],
            planned_raw_FP32_bytes=667484160,raw_limit_bytes=1073741824)
    except BaseException as error:
        result.update(status='FAILED_CPU_A_PREPARATION_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save(out/'result.json',result);fsync_chain(out,out.parent)
        print(json.dumps(dict(status=result['status'],result=ref(out/'result.json'))))


if __name__=='__main__':main()
