"""Small-batch tensor-core PQ2 path: numerics, determinism, graph updates and dispatch scope."""
import pytest
import torch
from vllm_gguf_plugin.triton import prism, pq2_int_gemv as integer, pq2_small_mmq as small
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(8,6),
                              reason='SM86 CUDA required')
CASES=sorted((m,n,k) for m,table in small.GEOMETRY_BY_M.items() for n,k in table)
_WEIGHTS={}


def _weight(n,k):
    if (n,k) not in _WEIGHTS:
        torch.manual_seed(90)
        raw=torch.randint(0,256,(n,k//128,34),device='cuda',dtype=torch.uint8)  # includes unused code 3
        scales=(torch.rand(n,k//128,device='cuda')*0.03+0.001).half()
        raw[:,:,:2]=scales.view(torch.uint8).reshape(n,k//128,2)
        _WEIGHTS[(n,k)]=prepare_pq2_layout(raw.reshape(n,-1))
    return _WEIGHTS[(n,k)]


def _rel_rms(a,b):
    return float((a.float()-b.float()).norm()/b.float().norm())


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(prism,'_EXPERIMENTAL_INT_GEMV',True)
    monkeypatch.setattr(small,'_SMALL_MMQ',True)
    monkeypatch.setattr(small,'_BATCH8_SMALL_MMQ',False)


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('m,n,k',CASES)
def test_matches_q8_integer_and_is_deterministic(enabled,dtype,m,n,k):
    w=_weight(n,k);x=torch.randn(m,k,device='cuda',dtype=dtype)
    expected=integer.pq2_int_gemv(x,w,prepared=True,chained=False)  # same Q8 arithmetic
    first=prism.pq2_matmul(x,w,prepared=True)
    assert _rel_rms(first,expected)<2e-4
    for _ in range(2):torch.testing.assert_close(prism.pq2_matmul(x,w,prepared=True),first,atol=0,rtol=0)


@pytest.mark.parametrize('m',[9,12,15])
def test_rows_between_eight_and_sixteen_use_the_sixteen_table(enabled,m):
    if 16 not in small.GEOMETRY_BY_M:pytest.skip('no M=16 table')
    n,k=next(iter(small.GEOMETRY_BY_M[16]))
    w=_weight(n,k);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    assert small.geometry_for(m,n,k)==small.GEOMETRY_BY_M[16][(n,k)]
    assert _rel_rms(prism.pq2_matmul(x,w,prepared=True),integer.pq2_int_gemv(x,w,prepared=True,chained=False))<2e-4


@pytest.mark.parametrize('m,n,k',[c for c in CASES if c[0] in (4,8,16)])
def test_graph_replay_follows_input_updates(enabled,m,n,k):
    w=_weight(n,k);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    for _ in range(3):prism.pq2_matmul(x,w,prepared=True)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=prism.pq2_matmul(x,w,prepared=True)
    x.mul_(0.5).add_(0.03)
    expected=prism.pq2_matmul(x,w,prepared=True)
    graph.replay()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_dispatch_scope(monkeypatch):
    calls=[]
    real=small.pq2_small_mmq
    monkeypatch.setattr(small,'pq2_small_mmq',lambda x,w:calls.append(x.shape[0]) or real(x,w))
    m8=sorted(small.GEOMETRY_BY_M[8])[0];w=_weight(*m8);k=m8[1]
    def run(m,dtype=torch.bfloat16,prepared=True):
        prism.pq2_matmul(torch.randn(m,k,device='cuda',dtype=dtype),w,prepared=prepared)
    monkeypatch.setattr(small,'_SMALL_MMQ',False);monkeypatch.setattr(small,'_BATCH8_SMALL_MMQ',False)
    for m in (1,4,8,16):run(m)
    assert calls==[]
    monkeypatch.setattr(small,'_BATCH8_SMALL_MMQ',True)
    for m in (1,4,8,16):run(m)
    assert calls==[8]                              # narrow switch: M = 8 only
    calls.clear();monkeypatch.setattr(small,'_BATCH8_SMALL_MMQ',False);monkeypatch.setattr(small,'_SMALL_MMQ',True)
    expected=[m for m in range(1,18) if small.geometry_for(m,*m8) is not None and 2<=m<=16]
    for m in range(1,18):run(m)
    assert calls==expected and 1 not in calls and 17 not in calls
    calls.clear();run(8,dtype=torch.float32);assert calls==[]  # floating-point activations excluded


def test_resolved_settings_enter_compile_identity(monkeypatch):
    import vllm.envs as envs
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors
    _register_pq2_compile_factors()
    for enabled in (False,True):
        monkeypatch.setattr(small,'_SMALL_MMQ',enabled);monkeypatch.setattr(small,'_BATCH8_SMALL_MMQ',not enabled)
        assert envs.environment_variables['GGUF_PQ2_SMALL_MMQ']()==enabled
        assert envs.environment_variables['GGUF_PQ2_BATCH8_SMALL_MMQ']()==(not enabled)
    assert envs.environment_variables['GGUF_PQ2_BATCH8_SMALL_MMQ_VERSION']()==small._VERSION


@pytest.mark.parametrize('m',[5,6,7,9,12,16])
def test_small_float_output_tile_is_bit_exact(monkeypatch,m):
    n,k=5120,6144;w=_weight(n,k);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    monkeypatch.setattr(small,'_SMALL_MMQ',False);monkeypatch.setattr(small,'_BATCH8_SMALL_MMQ',False)
    monkeypatch.setattr(prism,'_SMALL_FLOAT_OUTPUT',False)
    expected=prism.pq2_matmul(x,w,prepared=True)
    monkeypatch.setattr(prism,'_SMALL_FLOAT_OUTPUT',True)
    torch.testing.assert_close(prism.pq2_matmul(x,w,prepared=True),expected,atol=0,rtol=0)
    import vllm.envs as envs
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors
    _register_pq2_compile_factors()
    assert envs.environment_variables['GGUF_PQ2_SMALL_FLOAT_OUTPUT']() is True
