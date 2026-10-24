"""Preflight and launch native vLLM serving for DeepSeek-V4-Flash.

Run this with a separate, current vLLM environment. This is a native serving
entry point; it does not claim PhaseKV attention compatibility.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


VARIANTS = {
    "preview": ("deepseek-ai/DeepSeek-V4-Flash", 170, "0.20.0"),
    "0731": ("deepseek-ai/DeepSeek-V4-Flash-0731", 200, "0.25.0"),
}


def build_command(args) -> list[str]:
    model = args.model or VARIANTS[args.variant][0]
    return [
        "vllm", "serve", model,
        "--host", args.host, "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--kv-cache-dtype", "fp8",
        "--block-size", "256",
        "--trust-remote-code",
        "--tokenizer-mode", "deepseek_v4",
        "--served-model-name", args.served_model_name,
    ]


def check_hardware(args) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA；请在租用 GPU 的实例内运行")
    count = torch.cuda.device_count()
    if args.tensor_parallel_size > count:
        raise RuntimeError(
            f"请求 {args.tensor_parallel_size} 张 GPU，当前仅可见 {count} 张"
        )
    capacities = [torch.cuda.get_device_properties(i).total_memory
                  for i in range(args.tensor_parallel_size)]
    usable_bytes = sum(capacities) * args.gpu_memory_utilization
    required_gb = VARIANTS[args.variant][1]
    if usable_bytes < required_gb * 1_000_000_000:
        raise RuntimeError(
            f"{args.variant} 权重最低显存参考值约 {required_gb}GB；"
            f"当前 {args.tensor_parallel_size} 张卡可用预算约 "
            f"{usable_bytes / 1_000_000_000:.1f}GB（按 --gpu-memory-utilization 折算）。"
            "请增加 GPU，或选择更小的模型。"
        )
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(args.tensor_parallel_size))
    return f"GPU: {names}; 估计可用预算 {usable_bytes / 1_000_000_000:.1f}GB"


def check_vllm(args) -> str:
    if shutil.which("vllm") is None:
        raise RuntimeError("未找到 vllm 命令；请使用独立的、支持 DeepSeek V4 的 vLLM 环境")
    try:
        installed = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("找到了 vllm 命令，但当前 Python 环境没有 vLLM 包") from exc
    from packaging.version import Version

    required = VARIANTS[args.variant][2]
    if Version(installed) < Version(required):
        raise RuntimeError(f"{args.variant} 至少需要 vLLM {required}；当前为 {installed}")
    return installed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek V4 Flash 原生 vLLM 部署及显存预检")
    parser.add_argument("--variant", choices=VARIANTS, default="preview")
    parser.add_argument("--model", help="覆盖模型 ID 或指定已下载的本地权重目录")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="deepseek-v4-flash")
    parser.add_argument("--check", action="store_true", help="只检查 GPU 和 vLLM，不启动服务")
    parser.add_argument("--dry-run", action="store_true", help="只打印部署命令，不检查 GPU")
    return parser


def main(argv=None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.tensor_parallel_size <= 0 or args.max_model_len <= 0 or not (0 < args.gpu_memory_utilization <= 1):
        parser.error("GPU 数、上下文长度须为正，显存利用率须位于 (0, 1]")
    if not 1 <= args.port <= 65535:
        parser.error("端口号必须位于 1–65535")
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    try:
        print(check_hardware(args), flush=True)
        print(f"vLLM: {check_vllm(args)}", flush=True)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.check:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
