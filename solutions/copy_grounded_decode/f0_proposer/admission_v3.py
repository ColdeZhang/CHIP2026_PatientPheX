"""Finite wall-clock admission before process/model launch; no GPU imports."""
import os,time
from pathlib import Path
from raw_store import publish_json

def admit(stage,expected=None):
 deadline=float(os.environ['COPY_PROPOSER_LAUNCH_VALID_UNTIL']);now=time.time()
 if expected is not None:assert deadline==expected
 if now>deadline:
  publish_json(Path.cwd()/'admission-rejected.json',dict(status='EXPIRED_BEFORE_MODEL_ADMISSION',stage=stage,at_unix=now,launch_not_after_unix=deadline,no_model_started_by_this_check=True),os.environ['PATIENTPHEX_DURABILITY_ROOT'])
  raise RuntimeError('Frozen launch validity expired; do not load model or retry')
 return deadline
