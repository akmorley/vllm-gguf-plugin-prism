from vllm import LLM, SamplingParams

GGUF = "../models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
TOKENIZER = "../models/qwen38-tokenizer/"

PROMPT_IDS = [
    9419,
    11,
    821,
    803,
    369,
    11,
    821,
    803,
    369,
    11,
    279,
    1865,
    28393,
    220,
]


def main():
    llm = LLM(
        model=GGUF,
        tokenizer=TOKENIZER,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        seed=0,
        limit_mm_per_prompt={
            "image": 0,
            "video": 0,
            "audio": 0,
        },
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20,
        ignore_eos=True,
        seed=0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )


#    LLAMA_TOP_IDS = [
#        17,
#        16,
#        21,
#        18,
#        248046,
#        20,
#        15,
#        24,
#        19,
#        23,
#    ]
#    
#    sampling = SamplingParams(
#        temperature=0.0,
#        max_tokens=1,
#        logprobs=10,
#        logprob_token_ids=LLAMA_TOP_IDS,
#        ignore_eos=True,
#        seed=0,
#    )
#
#
#    print( sampling )
#
    outputs = llm.generate(
        [
            {
                "prompt_token_ids": PROMPT_IDS,
            }
        ],
        sampling,
    )

    completion = outputs[0].outputs[0]

    print()
    print("PROMPT IDS:")
    print(PROMPT_IDS)

    print()
    print("GENERATED:")
    print(completion.token_ids)

    print()
    print("TEXT:")
    print(repr(completion.text))

    print()
    print("TOP 20:")

    rows = []

    for token_id, item in completion.logprobs[0].items():
        rows.append(
            (
                token_id,
                item.logprob,
                getattr(
                    item,
                    "decoded_token",
                    None,
                ),
            )
        )

    rows.sort(
        key=lambda row: row[1],
        reverse=True,
    )

    for token_id, logprob, text in rows:
        print(
            f"{token_id:8d} "
            f"{logprob:12.6f} "
            f"{text!r}"
        )


if __name__ == "__main__":
    main()
