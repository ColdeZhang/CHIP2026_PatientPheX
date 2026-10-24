"""Pure checks plus full-state fingerprints; no network, model, or evaluator entry."""
from pathlib import Path
import hashlib,json,math,os,re,time

STATE_KEYS={'completed_epochs','global_step','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng','optimizer_parameter_names'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def check_deadline(deadline):
    if not math.isfinite(deadline) or time.time()>=deadline:raise TimeoutError('Original absolute training/export deadline reached')

def complete_epoch_directories(output):
    return sorted((p for p in Path(output).iterdir() if p.is_dir() and not p.is_symlink() and re.fullmatch(r'epoch-[0-9]{3}',p.name)),key=lambda p:p.name)

def validate_state_structure(state,completed,names):
    if set(state)!=STATE_KEYS:raise ValueError('Complete optimizer/scheduler/Python/NumPy/Torch/CUDA state required')
    if completed!=1 or state['completed_epochs']!=completed or state['global_step']!=16*completed:raise ValueError('Resume only exact first complete epoch16')
    if state['optimizer_parameter_names']!=list(names):raise ValueError('Optimizer parameter ordering differs')
    if state['scheduler']['last_epoch']!=16 or state['scheduler']['_step_count']!=17:raise ValueError('Full scheduler/warmup state differs')
    optimizer=state['optimizer']
    if len(optimizer['state'])!=len(names) or len(optimizer['param_groups'])!=1:raise ValueError('Complete Adam state inventory differs')
    if optimizer['param_groups'][0]['params']!=list(range(len(names))):raise ValueError('Adam parameter IDs differ')
    if not isinstance(state['cuda_rng'],list) or len(state['cuda_rng'])!=1:raise ValueError('One original CUDA RNG state required')
    for item in optimizer['state'].values():
        if set(item)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Incomplete Adam step/moments')
    return True

def tree_digest(value,torch=None):
    """Hash logical value/dtype/shape, ignoring device; copy one tensor at a time."""
    h=hashlib.sha256()
    def put(x):
        if torch is not None and isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous();h.update(b'tensor\0'+str(y.dtype).encode()+repr(tuple(y.shape)).encode()+b'\0');h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes());return
        if type(x).__module__=='numpy' and type(x).__name__=='ndarray':
            h.update(b'ndarray\0'+str(x.dtype).encode()+repr(tuple(x.shape)).encode()+b'\0');h.update(x.tobytes(order='C'));return
        if isinstance(x,dict):
            h.update(b'dict\0')
            for key in sorted(x,key=lambda k:(type(k).__name__,repr(k))):put(key);put(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode()+b'\0'+str(len(x)).encode()+b'\0')
            for part in x:put(part)
        elif x is None or type(x) in (bool,int,float,str):h.update(type(x).__name__.encode()+b'\0'+repr(x).encode()+b'\0')
        else:raise TypeError('Unsupported state fingerprint '+type(x).__name__)
    put(value);return h.hexdigest()

def validate_export_ack(path,output,checkpoint,metadata,trainer_sha):
    output=Path(output);checkpoint=Path(checkpoint);ack=json.loads(Path(path).read_text())
    expected_name='export-ack-epoch-001.json'
    if Path(path).resolve()!= (output/expected_name).resolve():raise ValueError('ACK must use own output root non-checkpoint basename')
    if ack.get('status')!='persistent_checkpoint_verified' or ack.get('completed_epochs')!=1 or ack.get('global_step')!=16:raise ValueError('Persistent first-epoch ACK required')
    if ack.get('trainer_sha256')!=trainer_sha or ack.get('source_checkpoint_directory')!=str(checkpoint):raise ValueError('ACK source identity differs')
    if ack.get('checkpoint_sha256')!=sha(checkpoint/'checkpoint.json'):raise ValueError('ACK checkpoint SHA differs')
    if ack.get('persistent_mount_id')!='5a9955da-5240-41cc-9e52-05d3d8da8a3c':raise ValueError('ACK is not on expected persistent mount')
    expected_directory='/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen35-fulltext32-indexed-formal-v1/checkpoints/epoch-001'
    if ack.get('persistent_directory')!=expected_directory:raise ValueError('ACK persistent namespace differs')
    if not re.fullmatch('[0-9a-f]{64}',ack.get('transport_manifest_sha256','')):raise ValueError('ACK transport manifest SHA missing')
    for key in ['fsync_all_files','reopened_final_filename_SHA_all_files','destination_directories_fsynced']:
        if ack.get(key) is not True:raise ValueError('ACK lacks durable contract: '+key)
    actual={str(p.relative_to(checkpoint)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(checkpoint.rglob('*')) if p.is_file()}
    if set(actual)!=set(metadata['files'])|{'checkpoint.json'}:raise ValueError('Checkpoint inventory differs before resume')
    files=ack.get('files',{})
    if not set(actual)<=set(files) or '_run/run_manifest.json' not in files:raise ValueError('ACK does not cover full state+manifest')
    for name,record in files.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe ACK member')
        source=output/Path(*relative.parts[1:]) if relative.parts[0]=='_run' else checkpoint/relative
        if not source.is_file() or source.is_symlink() or record!={'bytes':source.stat().st_size,'sha256':sha(source)}:raise ValueError('ACK source member SHA differs: '+name)
    return {'status':'PASS_PERSISTENT_ACK_BINDS_COMPLETE_FIRST_EPOCH_STATE', 'ack_sha256':sha(path),'checkpoint_sha256':sha(checkpoint/'checkpoint.json'),'members':len(files),'persistent_directory':expected_directory,'completed_epochs':1,'global_step':16}

def local_publish_checkpoint(staging,destination,output):
    """Fsync/reopen own SHM files; explicitly does not claim persistent durability."""
    staging=Path(staging);destination=Path(destination);output=Path(output)
    files=[p for p in sorted(staging.rglob('*')) if p.is_file()]
    if any(p.is_symlink() for p in staging.rglob('*')):raise ValueError('Checkpoint symlinks forbidden')
    before={str(p.relative_to(staging)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    for p in files:
        with p.open('rb') as f:os.fsync(f.fileno())
    directories=[p for p in staging.rglob('*') if p.is_dir()]+[staging]
    for p in sorted(directories,key=lambda x:-len(x.parts)):
        fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    staging.replace(destination)
    fd=os.open(str(output),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    after={str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(destination.rglob('*')) if p.is_file()}
    if before!=after:raise ValueError('Local checkpoint changed across publication')
    return {'status':'SHM_LOCAL_COMPLETE_FSYNC_REOPEN_SHA_NOT_PERSISTENT','checkpoint':str(destination),'files':after,'persistent_export_ack_still_required':True}
