"""Run packed PhaseKV inference with OPT weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from models.OPT.opt_attention import install_opt_phasekv_attention
from PhaseKV import model_runner


MODEL_ID = "facebook/opt-6.7b"


def dense_prefill(model, input_ids, need_attentions):
    return model(input_ids=input_ids, use_cache=True, output_attentions=need_attentions)


def packed_step(model, cache, token):
    output = model(
        input_ids=token.reshape(1, 1), past_key_values=cache,
        use_cache=True, output_attentions=False,
        position_ids=torch.tensor([[cache.true_seq_length]], device=token.device),
    )
    cache.finish_decode_step()
    return output.logits[:, -1, :]


def main(argv=None):
    parser = argparse.ArgumentParser(description="OPT PhaseKV 单轮推理（CUDA、batch=1）")
    model_runner.add_arguments(parser, MODEL_ID)
    args = parser.parse_args(argv)
    model_runner.validate_arguments(parser, args)
    try:
        gpu = model_runner.check_cuda()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GPU: {gpu}")
    if args.check_cuda:
        return
    model, tokenizer = model_runner.load_model(args, "opt")
    install_opt_phasekv_attention(model)
    input_ids = model_runner.encode_prompt(tokenizer, args.prompt, use_chat_template=False)
    answer, cache = model_runner.generate(
        model, tokenizer, input_ids, args,
        dense_prefill=dense_prefill, packed_step=packed_step,
    )
    print(answer)
    print(f"[PhaseKV 物理 KV 字节: {cache.physical_nbytes():,}]")


if __name__ == "__main__":
    main()
