"""New production path: exact numerics, graph updates, and safe fallbacks."""
import pytest
import torch
from vllm_gguf_plugin.triton import pq2_int_gemv as integer
from vllm_gguf_plugin.triton.pq2_layout import prepare_pq2_layout

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_single_exact_and_graph(monkeypatch,dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    n,k=5120,6144
    torch.manual_seed(86)
    raw=torch.randint(0,256,(n,k//128,34),device='cuda',dtype=torch.uint8)
    scales=torch.full((n,k//128),.03125,device='cuda',dtype=torch.float16)
    raw[:,:,:2]=scales.view(torch.uint8).reshape(n,k//128,2)
    w=prepare_pq2_layout(raw.reshape(n,-1));x=torch.randn(1,k,device='cuda',dtype=dtype)
    monkeypatch.setattr(integer,'_SINGLE_GEMV',False)
    expected=integer.pq2_int_gemv(x,w,prepared=True,chained=False)
    monkeypatch.setattr(integer,'_SINGLE_GEMV',True)
    for _ in range(3):actual=integer.pq2_int_gemv(x,w,prepared=True,chained=False)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=integer.pq2_int_gemv(x,w,prepared=True,chained=False)
    x.mul_(.7).add_(.02)
    monkeypatch.setattr(integer,'_SINGLE_GEMV',False)
    expected=integer.pq2_int_gemv(x,w,prepared=True,chained=False)
    graph.replay();torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_single_fallbacks(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from vllm_gguf_plugin.triton import pq2_single_gemv as single
    calls=[]
    class Probe:
        def __getitem__(self,grid):
            return lambda *a,**kw:calls.append('generic')
    monkeypatch.setattr(integer,'_int_gemv',Probe())
    monkeypatch.setattr(single,'launch_single',lambda *a,**kw:calls.append('single'))
    monkeypatch.setattr(integer,'_SINGLE_GEMV',True)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(8,6))
    n,k=5120,6144
    raw=torch.zeros(n,k//128*34,device='cuda',dtype=torch.uint8)
    prepared=prepare_pq2_layout(raw)
    for m,dtype,layout,decode,chained,gated in (
        (1,torch.bfloat16,True,'prmt',False,False),
        (2,torch.bfloat16,True,'prmt',False,False),
        (1,torch.float32,True,'prmt',False,False),
        (1,torch.bfloat16,False,'prmt',False,False),
        (1,torch.bfloat16,True,'shift',False,False),
        (1,torch.bfloat16,True,'prmt',True,False),
        (1,torch.bfloat16,True,'prmt',False,True)):
        x=torch.ones(m,k,device='cuda',dtype=dtype)
        q=torch.zeros(m,k,device='cuda',dtype=torch.int8)
        s=torch.ones(m,k//128,device='cuda',dtype=torch.float32)
        integer.pq2_int_gemv(x,prepared if layout else raw,prepared=layout,decode=decode,
                              chained=chained,gated=gated,quantized=(q,s))
    assert calls==['single']+['generic']*6
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(9,0))
    integer.pq2_int_gemv(x,prepared,prepared=True,decode='prmt',chained=False,quantized=(q,s))
    assert calls[-1]=='generic'


def test_single_resolved_compile_identity(monkeypatch):
    import vllm.envs as envs
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors
    _register_pq2_compile_factors()
    for enabled in (False,True):
        monkeypatch.setattr(integer,'_SINGLE_GEMV',enabled)
        assert envs.environment_variables['GGUF_PQ2_SINGLE_GEMV']()==enabled
    assert envs.environment_variables['GGUF_PQ2_INT_GEMV_VERSION']()=='batch8-chained-single-byte-default-v4'


def test_promoted_defaults_in_fresh_process():
    import json
    import os
    import subprocess
    import sys

    env=dict(os.environ)
    for key in ('GGUF_PQ2_SINGLE_GEMV','GGUF_PQ2_INT_GEMV_VARIANT','GGUF_PQ2_INT_GEMV_DECODE'):
        env.pop(key,None)
    code='from vllm_gguf_plugin.triton import pq2_int_gemv as m; import json; print(json.dumps([m._SINGLE_GEMV,m._VARIANT,m._DECODE]))'
    output=subprocess.check_output([sys.executable,'-c',code],env=env,text=True)
    assert json.loads(output.strip().splitlines()[-1])==[True,'shape-tuned-v2','prmt']
    env.update(GGUF_PQ2_SINGLE_GEMV='0',GGUF_PQ2_INT_GEMV_VARIANT='shape-tuned')
    output=subprocess.check_output([sys.executable,'-c',code],env=env,text=True)
    assert json.loads(output.strip().splitlines()[-1])==[False,'shape-tuned','prmt']
