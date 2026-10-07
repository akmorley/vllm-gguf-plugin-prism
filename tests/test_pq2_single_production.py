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


def _single_attention_inputs():
    cache = torch.empty((2, 784, 4, 512), device="cuda", dtype=torch.bfloat16)
    query = torch.empty((1, 24, 512), device="cuda", dtype=torch.bfloat16)
    q = query[..., :256]
    return dict(
        q=q,
        k=cache[..., :256],
        v=cache[..., 256:],
        out=torch.empty_like(q),
        cu_seqlens_q=torch.empty(2, device="cuda", dtype=torch.int32),
        seqused_k=torch.empty(1, device="cuda", dtype=torch.int32),
        block_table=torch.empty((1, 45), device="cuda", dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=34816,
        causal=True,
        fa_version=2,
    )


def _single_attention_spy():
    from functools import wraps
    from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func

    calls = []
    marker = object()

    @wraps(flash_attn_varlen_func)
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    return original, calls, marker


@pytest.mark.parametrize(
    "reason",
    [
        "prefill",
        "batch",
        "fp16",
        "sm90",
        "lse",
        "softcap",
        "window",
        "context_parallel",
        "auxiliary",
        "noncausal",
        "cpu_metadata",
        "inner_stride",
    ],
)
def test_single_attention_preserves_fa2_for_unsupported_calls(monkeypatch, reason):
    """Unsupported attention features must reach the original FA2 unchanged."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vllm_gguf_plugin import attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    original, calls, marker = _single_attention_spy()
    dispatch = attention._make_single_decode_dispatch(original, 32)
    with FakeTensorMode():
        args = _single_attention_inputs()
        if reason in ("prefill", "batch"):
            args["q"] = torch.empty((2, 24, 256), device="cuda", dtype=torch.bfloat16)
            args["max_seqlen_q"] = 2 if reason == "prefill" else 1
        elif reason == "fp16":
            args["q"] = args["q"].to(torch.float16)
        elif reason == "sm90":
            monkeypatch.setattr(
                torch.cuda, "get_device_capability", lambda device: (9, 0)
            )
        elif reason == "lse":
            args["return_softmax_lse"] = True
        elif reason == "softcap":
            args["softcap"] = 30.0
        elif reason == "window":
            args["window_size"] = (1024, 0)
        elif reason == "context_parallel":
            args["cp_world_size"] = 2
        elif reason == "auxiliary":
            args["s_aux"] = torch.empty(24, device="cuda")
        elif reason == "noncausal":
            args["causal"] = False
        elif reason == "cpu_metadata":
            args["seqused_k"] = torch.empty(1, device="cpu", dtype=torch.int32)
        elif reason == "inner_stride":
            args["q"] = torch.empty((1, 24, 512), device="cuda", dtype=torch.bfloat16)[
                ..., ::2
            ]
        assert dispatch(**args) is marker
        assert len(calls) == 1
        assert all(calls[0][1][key] is value for key, value in args.items())


def test_single_attention_scratch_is_not_shared_by_calls(monkeypatch):
    """Independent calls and captures must not alias mutable global scratch."""
    from types import SimpleNamespace
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vllm_gguf_plugin import attention
    import vllm.v1.attention.ops.triton_unified_attention as ops

    launches = []
    monkeypatch.setattr(
        ops, "unified_attention", lambda *args, **kw: launches.append((args, kw))
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    stream = [100]
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=stream[0]),
    )
    original, calls, _ = _single_attention_spy()
    dispatch = attention._make_single_decode_dispatch(original, 32)
    with FakeTensorMode():
        args = _single_attention_inputs()
        assert dispatch(**args) is args["out"]
        stream[0] = 101
        assert dispatch(**args) is args["out"]
        stream[0] = 100
        assert dispatch(**args) is args["out"]
        assert not calls
        assert (
            launches[0][1]["softmax_segm_output"]
            is not launches[1][1]["softmax_segm_output"]
        )
        assert (
            launches[0][1]["softmax_segm_output"]
            is not launches[2][1]["softmax_segm_output"]
        )
        assert all(row[1]["num_par_softmax_segments"] == 32 for row in launches)
        assert all(row[0][0] is args["q"] for row in launches)


def test_single_attention_install_is_opt_in_and_idempotent(monkeypatch):
    from vllm_gguf_plugin import attention
    import vllm.v1.attention.backends.flash_attn as backend

    original = backend.flash_attn_varlen_func
    monkeypatch.setattr(attention, "_SINGLE_ATTN_BACKEND", "fa2")
    attention.install_single_decode_attention()
    assert backend.flash_attn_varlen_func is original
    monkeypatch.setattr(attention, "_SINGLE_ATTN_BACKEND", "triton")
    monkeypatch.setattr(attention, "_SINGLE_ATTN_SEGMENTS", 32)
    monkeypatch.setattr(backend, "flash_attn_varlen_func", original)
    attention.install_single_decode_attention()
    installed = backend.flash_attn_varlen_func
    attention.install_single_decode_attention()
    assert backend.flash_attn_varlen_func is installed
    assert installed._gguf_single_attention_segments == 32


def test_single_attention_changes_compilation_identity(monkeypatch):
    import vllm.envs as envs
    from vllm_gguf_plugin import attention
    from vllm_gguf_plugin.plugin import _register_pq2_compile_factors

    _register_pq2_compile_factors()
    monkeypatch.setattr(attention, "_SINGLE_ATTN_BACKEND", "fa2")
    monkeypatch.setattr(attention, "_SINGLE_ATTN_SEGMENTS", 32)
    baseline = envs.compile_factors()
    monkeypatch.setattr(attention, "_SINGLE_ATTN_BACKEND", "triton")
    monkeypatch.setattr(attention, "_SINGLE_ATTN_SEGMENTS", 64)
    candidate = envs.compile_factors()
    assert (
        baseline["GGUF_PQ2_SINGLE_ATTN_BACKEND"]
        != candidate["GGUF_PQ2_SINGLE_ATTN_BACKEND"]
    )
    assert (
        baseline["GGUF_PQ2_SINGLE_ATTN_SEGMENTS"]
        != candidate["GGUF_PQ2_SINGLE_ATTN_SEGMENTS"]
    )
    assert candidate["GGUF_PQ2_SINGLE_ATTN_VERSION"] == attention._SINGLE_ATTN_VERSION


@torch.inference_mode()
def test_single_attention_paged_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("SM86 required")
    from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
    from vllm_gguf_plugin.attention import _make_single_decode_dispatch

    torch.manual_seed(184)
    cache = torch.randn((45, 784, 4, 512), device="cuda", dtype=torch.bfloat16)
    k, v = cache[..., :256], cache[..., 256:]
    q = torch.randn((1, 24, 512), device="cuda", dtype=torch.bfloat16)[..., :256]
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    used = torch.tensor([4096], device="cuda", dtype=torch.int32)
    table = torch.randperm(45, device="cuda").int()[None, :]
    out = torch.empty_like(q)
    arguments = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        max_seqlen_k=34816,
        seqused_k=used,
        block_table=table,
        causal=True,
        softmax_scale=0.0625,
        fa_version=2,
    )
    for segments in (32, 64):
        dispatch = _make_single_decode_dispatch(flash_attn_varlen_func, segments)
        for _ in range(3):
            dispatch(**arguments, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dispatch(**arguments, out=out)
        for length in (4096, 16384, 32768):
            used.fill_(length)
            q.mul_(0.97).add_(0.02)
            expected = flash_attn_varlen_func(**arguments).clone()
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            pages = table[0, : (length + 783) // 784].long()
            keys = k.index_select(0, pages).reshape(-1, 4, 256)[:length]
            values = v.index_select(0, pages).reshape(-1, 4, 256)[:length]
            truth = (
                (
                    q[0].reshape(4, 6, 256).float()
                    @ keys.permute(1, 2, 0).float()
                    * 0.0625
                ).softmax(-1)
                @ values.permute(1, 0, 2).float()
            ).reshape(1, 24, 256)
            assert torch.isfinite(out).all()
            for reference in (expected, truth):
                relative_rms = (
                    (out.float() - reference.float()).square().mean()
                    / reference.float().square().mean()
                ).sqrt()
                assert relative_rms.item() < 0.01
