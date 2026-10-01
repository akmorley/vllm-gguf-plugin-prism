from gguf import GGUFReader

r = GGUFReader("../models/Ternary-Bonsai-2-27B-PQ2_0.gguf")

for t in r.tensors:
    if t.name in {
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
    }:
        print(
            t.name,
            "shape=", t.shape,
            "type=", t.tensor_type,
        )
