#!/usr/bin/env python3
"""Download pinned ModelScope bytes with verified HTTP ranges and final source hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--workers',type=int,default=12);ap.add_argument('--chunk-mib',type=int,default=64)
    args=ap.parse_args();source=json.loads(Path(args.manifest).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    chunks=out/'.chunks';chunks.mkdir(exist_ok=True);block=args.chunk_mib*1024*1024;tasks=[];plans=[];began=time.time()
    for item in source['files']:
        name=item['name'];final=out/name;partial=out/(name+'.part')
        if final.exists() and final.stat().st_size==item['size'] and digest(final)==item['sha256']:continue
        have=partial.stat().st_size if partial.exists() else 0
        if have>item['size']:raise RuntimeError('Existing partial exceeds expected size')
        pieces=[]
        for start in range(have,item['size'],block):
            end=min(start+block,item['size'])-1;part=chunks/(name+f'.{start}.{end}')
            task={'name':name,'start':start,'end':end,'path':str(part)};tasks.append(task);pieces.append(task)
        plans.append((item,partial,final,pieces))
    def download(task):
        path=Path(task['path']);wanted=task['end']-task['start']+1
        if path.exists() and path.stat().st_size==wanted:return task
        url='https://modelscope.cn/api/v1/models/'+source['model']+'/repo?'+urllib.parse.urlencode({'Revision':source['modelscope_revision'],'FilePath':task['name']})
        for attempt in range(5):
            try:
                request=urllib.request.Request(url,headers={'Range':f"bytes={task['start']}-{task['end']}"})
                with urllib.request.urlopen(request,timeout=90) as response:
                    expected=f"bytes {task['start']}-{task['end']}/"
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(expected):raise RuntimeError('HTTP byte range mismatch')
                    with path.open('wb') as handle:
                        copied=0
                        while chunk:=response.read(4*1024*1024):
                            copied+=len(chunk)
                            if copied>wanted:raise RuntimeError('range exceeded')
                            handle.write(chunk)
                if path.stat().st_size!=wanted:raise RuntimeError('incomplete range')
                return task
            except Exception as exc:
                print(json.dumps({'name':task['name'],'start':task['start'],'attempt':attempt,'error':type(exc).__name__}),flush=True)
                if attempt==4:raise
                time.sleep(2)
    complete=0;bytes_done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task in pool.map(download,tasks):
            complete+=1;bytes_done+=task['end']-task['start']+1
            if complete%8==0:print(json.dumps({'chunks_complete':complete,'chunks_total':len(tasks),'downloaded_bytes':bytes_done,'elapsed_seconds':round(time.time()-began,1)}),flush=True)
    results=[]
    for item,partial,final,pieces in plans:
        with partial.open('ab') as handle:
            for task in pieces:
                assert handle.tell()==task['start'],'append range mismatch'
                with Path(task['path']).open('rb') as part:shutil.copyfileobj(part,handle)
        assert partial.stat().st_size==item['size'],'final size mismatch'
        value=digest(partial)
        if value!=item['sha256']:raise RuntimeError('Final source checksum mismatch')
        partial.replace(final)
        for task in pieces:Path(task['path']).unlink()
        results.append({'name':item['name'],'size':item['size'],'sha256':value})
        print(json.dumps(results[-1]),flush=True)
    (out/'download-complete.json').write_text(json.dumps({'source':source,'checked':results,'elapsed_seconds':time.time()-began},indent=2)+'\n')
    print('DOWNLOAD_COMPLETE',flush=True)

if __name__=='__main__':main()
