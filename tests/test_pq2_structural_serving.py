"""Production dispatch must preserve numerical results and graph input updates."""
import pytest
import torch
from vllm_gguf_plugin.triton import prism, pq2_int_gemv as integer
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('prepared',[False,True])
def test_float_output_tile_exact_and_graph(monkeypatch,dtype,prepared):
    n,k=5120,6144
    torch.manual_seed(82)
    raw=torch.randint(0,256,(n,k//128,34),device='cuda',dtype=torch.uint8)
    scales=torch.full((n,k//128),0.03125,device='cuda',dtype=torch.float16)
    raw[:,:,:2]=scales.view(torch.uint8).reshape(n,k//128,2)
    w=raw.reshape(n,-1)
    if prepared:w=prepare_pq2_layout(w)
    x=torch.randn(8,k,device='cuda',dtype=dtype)
    monkeypatch.setattr(prism,'_EXPERIMENTAL_INT_GEMV',False)
    monkeypatch.setattr(prism,'_SMALL_FLOAT_OUTPUT',False)
    expected=prism.pq2_matmul(x,w,prepared=prepared)
    monkeypatch.setattr(prism,'_SMALL_FLOAT_OUTPUT',True)
    for _ in range(3):actual=prism.pq2_matmul(x,w,prepared=prepared)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=prism.pq2_matmul(x,w,prepared=prepared)
    x.mul_(0.5).add_(0.03)
    monkeypatch.setattr(prism,'_SMALL_FLOAT_OUTPUT',False)
    expected=prism.pq2_matmul(x,w,prepared=prepared)
    graph.replay()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_chained_default_scope_and_explicit_override(monkeypatch):
    observed=[]
    class Probe:
        def __getitem__(self,grid):
            def run(*args,**kwargs):observed.append(args[12])
            return run
    monkeypatch.setattr(integer,'_int_gemv',Probe())
    monkeypatch.setattr(integer,'_BATCH8_CHAINED',True)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(8,6))
    w=torch.zeros(9,34,device='cuda',dtype=torch.uint8)
    for m in (1,4,8,16):
        x=torch.ones(m,128,device='cuda',dtype=torch.bfloat16)
        integer.pq2_int_gemv(x,w)
    assert observed==[False,False,True,False]
    integer.pq2_int_gemv(torch.ones(8,128,device='cuda',dtype=torch.bfloat16),w,chained=False)
    assert observed[-1] is False
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(9,0))
    integer.pq2_int_gemv(torch.ones(8,128,device='cuda',dtype=torch.bfloat16),w)
    assert observed[-1] is False


def test_resolved_settings_enter_compile_identity(monkeypatch):
    import vllm.envs as envs
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors
    _register_pq2_compile_factors()
    for chain,small in ((False,False),(True,True)):
        monkeypatch.setattr(integer,'_BATCH8_CHAINED',chain)
        monkeypatch.setattr(prism,'_SMALL_FLOAT_OUTPUT',small)
        assert envs.environment_variables['GGUF_PQ2_BATCH8_CHAINED']()==chain
        assert envs.environment_variables['GGUF_PQ2_SMALL_FLOAT_OUTPUT']()==small
    for removed in ('GGUF_PQ2_BATCH8_FLOAT_OUTPUT_BM','GGUF_PQ2_INT_GEMV_VARIANT','GGUF_PQ2_INT_GEMV_DECODE',
                    'GGUF_PQ2_INT_GEMV_CHAINED','GGUF_PQ2_MMQ_GROUP','GGUF_PQ2_BATCH8_SMALL_MMQ'):
        assert removed not in envs.environment_variables
