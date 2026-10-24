"""Small, durable MCP controller for this isolated DocRE namespace."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import time
import zipfile


def result_value(result):
    if result.get('isError'):
        raise RuntimeError(json.dumps(result, ensure_ascii=False)[:2000])
    if 'structuredContent' in result:
        return result['structuredContent']
    for content in result.get('content', []):
        if content.get('type') == 'text':
            return json.loads(content['text'])
    raise ValueError('No structured MCP result')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['deploy', 'cpu', 'read', 'run'])
    ap.add_argument('--mcp-client-dir', required=True)
    ap.add_argument('--mount', required=True)
    ap.add_argument('--local', required=True)
    ap.add_argument('--remote', required=True)
    ap.add_argument('--path')
    ap.add_argument('--command-file')
    args = ap.parse_args()
    sys.path.insert(0, args.mcp_client_dir)
    from mcp_client import Client
    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=True)
    client = Client()
    client.initialize()
    ident = str(time.time_ns())
    history = local / 'transport'
    history.mkdir(exist_ok=True)
    if args.action == 'deploy':
        members = {}
        for p in sorted(Path(__file__).parent.glob('*.py')):
            members['source/' + p.name] = p.read_bytes()
        for p in sorted((local / 'prepared').glob('*.json*')):
            members['prepared/' + p.name] = p.read_bytes()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, body in members.items():
                z.writestr(name, body)
        blob = archive.getvalue()
        (local / 'deployment.zip').write_bytes(blob)
        manifest = {name: {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} for name, body in members.items()}
        (local / 'deployment-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        call = {'name': 'write_file', 'arguments': {'mount_id': args.mount,
                'path': args.remote + '/deployment.zip', 'content': base64.b64encode(blob).decode(),
                'encoding': 'base64', 'create_parents': True}}
        value = result_value(client.request('tools/call', call))
        (history / f'deploy-{ident}.response.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps({'archive_bytes': len(blob), 'archive_sha256': hashlib.sha256(blob).hexdigest(),
                          'members': len(members), 'remote_response': value}))
        return
    if args.action == 'read':
        call = {'name': 'read_file', 'arguments': {'mount_id': args.mount, 'path': args.remote + '/' + args.path,
                                                  'limit': 1048576}}
    else:
        command = Path(args.command_file).read_text()
        call = {'name': 'start_process', 'arguments': {'mount_id': args.mount, 'command': command,
                'env': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}}}
        if args.action == 'run':
            call['arguments'].pop('env')
    # Record the unique operation before it can be dispatched. UNKNOWN outcomes
    # must be inspected, never retried as a fresh start by this controller.
    (history / f'{args.action}-{ident}.intent.json').write_text(json.dumps(call, indent=2) + '\n')
    try:
        value = result_value(client.request('tools/call', call))
    except BaseException as exc:
        (history / f'{args.action}-{ident}.error.json').write_text(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:2000]}) + '\n')
        raise
    (history / f'{args.action}-{ident}.response.json').write_text(json.dumps(value, indent=2) + '\n')
    if args.action == 'read':
        body = value.get('content', '')
        if value.get('encoding') == 'base64':
            body = base64.b64decode(body)
        elif isinstance(body, str):
            body = body.encode()
        else:
            raise ValueError('Unsupported file content type')
        path = local / 'collected' / args.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        print(json.dumps({'path': str(path), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}))
    else:
        print(json.dumps({key: value[key] for key in ['id', 'pid', 'started_at', 'running', 'mount_id']
                          if key in value}, ensure_ascii=False))


if __name__ == '__main__':
    main()
