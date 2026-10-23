import urllib.request,urllib.parse,html,re,hashlib,json,time,os
from pathlib import Path
B=Path(__file__).parent;t=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
url='https://drive.usercontent.google.com/download?id=1AwRnaB5RruFUEdMkKohZmTlD4ILCkQ_z&export=download';expected=434159420;out=B/'bert_model_max_triple.pkl';assert not out.exists()
with urllib.request.urlopen(url,timeout=30) as r:
 ct=r.headers.get('Content-Type','');data=r.read(32769)
assert ct.startswith('text/html') and len(data)<=32768
txt=data.decode();action=html.unescape(re.search(r'<form[^>]+action="([^"]+)"',txt)[1]);fields=dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',txt));assert action.startswith('https://drive.usercontent.google.com/download')
url2=action+'?'+urllib.parse.urlencode({k:html.unescape(v) for k,v in fields.items()});h=hashlib.sha256();n=0;first=bytearray();tmp=out.with_suffix('.download')
with urllib.request.urlopen(url2,timeout=30) as r,open(tmp,'wb') as f:
 assert r.headers.get('Content-Type','').startswith('application/octet-stream')
 while True:
  b=r.read(1048576)
  if not b:break
  n+=len(b);assert n<=expected;h.update(b);f.write(b)
  if len(first)<131072:first.extend(b[:131072-len(first)])
assert n==expected;prefix_sha=hashlib.sha256(first).hexdigest();assert prefix_sha=='131c2ddab9054f7ae6e343d3d4f91e1c91b306269b9998ccbff51f9d6513b360'
tmp.replace(out);d={'scope':'Only official public pair checkpoint; no CNN/fasttext/Stanza/second encoder','official_source_id':'1AwRnaB5RruFUEdMkKohZmTlD4ILCkQ_z','official_source_url':url,'source_repo_commit':'195d5df2ad9c6265e1de523f139694c7646eee29','bytes':n,'sha256':h.hexdigest(),'prefix_131072_sha256':prefix_sha,'seconds':time.monotonic()-t,'git_commit_does_not_cryptographically_pin_Drive_binary':True,'checkpoint_path':str(out)};(B/'download-manifest.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
