"""Verify the actual private FLA bindings; never switch either GDN to native."""
import importlib.metadata as metadata
import inspect
from pathlib import Path
import zipfile
from common import checked, ref
import legacy_runner as legacy


def binding_graph(fn,seen=None):
    seen=set() if seen is None else seen
    if id(fn) in seen or not inspect.isfunction(fn):return []
    seen.add(id(fn));out=[dict(name=fn.__qualname__,module=fn.__module__,file=inspect.getsourcefile(fn))]
    for cell in fn.__closure__ or ():
        try:value=cell.cell_contents
        except ValueError:continue
        if inspect.isfunction(value):out+=binding_graph(value,seen)
    if hasattr(fn,'__wrapped__'):out+=binding_graph(fn.__wrapped__,seen)
    return out


def verify_fla_backend(plan,q):
    import triton
    assert triton.__version__=='3.3.1'
    assert metadata.version('fla-core')=='0.6.0' and metadata.version('einops')=='0.8.1'
    names=('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule')
    fused={name:getattr(q,name) for name in names}
    graph={name:binding_graph(fn) for name,fn in fused.items()}
    assert all(any(item['module'].startswith('fla.ops.gated_delta_rule') for item in graph[name]) for name in names)
    assert all(inspect.unwrap(fused[name]).__module__==q.__name__ for name in names)
    for item in [item for branch in graph.values() for item in branch]:
        if item['file'] and item['file'].endswith('/integrations/hub_kernels.py'):
            assert legacy.digest(item['file'])=='a4875051ee4f23a723fe769e94b6ad3030284983a02dba03ecf25ec22d114510'
    conv={name:binding_graph(getattr(q,name)) for name in ('causal_conv1d_fn','causal_conv1d_update')}
    assert not any(item['module'].startswith('causal_conv1d') for branch in conv.values() for item in branch)
    installed={}
    for package,record in [('fla-core',plan['FLA_wheel']),('einops',plan['einops_wheel'])]:
        dist=metadata.distribution(package);verified={}
        with zipfile.ZipFile(checked(record)) as wheel:
            for entry in wheel.namelist():
                if entry.endswith('.py'):
                    path=Path(dist.locate_file(entry));assert path.read_bytes()==wheel.read(entry)
                    verified[entry]=ref(path)
        assert verified;installed[package]=verified
    evidence=dict(fused=graph,conv=conv,installed_sources=installed,
        binding_source_files={item['file']:ref(item['file']) for branch in [*graph.values(),*conv.values()] for item in branch if item['file']},
        wheels={'fla-core':plan['FLA_wheel'],'einops':plan['einops_wheel']},
        runtime=dict(triton=triton.__version__,fla_core=metadata.version('fla-core'),einops=metadata.version('einops')),
        both_GDN_globals_fused=True,conv_native=True,no_native_comparison_calls=True,
        independent_quality_variant_not_numerical_equivalence=True)
    return fused,evidence
