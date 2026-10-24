"""Small recorded MCP calls. UNKNOWN never retries a mutation/start."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client


def call(directory, name, arguments, tag):
    d=Path(directory)/'transport';d.mkdir(parents=True,exist_ok=True)
    p=d/(str(time.time_ns())+'-'+tag)
    p.with_suffix('.intent.json').write_text(json.dumps({'name':name,'arguments':arguments},indent=2)+'\n')
    c=Client();c.initialize()
    try:r=c.request('tools/call',{'name':name,'arguments':arguments})
    except Exception as e:
        p.with_suffix('.unknown.json').write_text(json.dumps({'error_type':type(e).__name__,'action':'Recover exact original intent; do not repeat start'}));raise
    p.with_suffix('.response.json').write_text(json.dumps(r,indent=2)+'\n')
    if r.get('isError'):raise RuntimeError(json.dumps(r)[:1000])
    v=r.get('structuredContent')
    if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
    return v
