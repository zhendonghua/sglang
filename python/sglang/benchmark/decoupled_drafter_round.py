"""Microbenchmark for the decoupled drafter's enumeration round.

Drives ``EnumDraftEngine.draft_round`` directly against a real draft model --
no verifier process, no transport: each benchmark round applies a synthetic
commit delta (as the verifier would) and runs one enumeration round, which is
exactly the drafter's steady-state loop body.

Example (drafter defaults from the e2e setup)::

    python -m sglang.benchmark.decoupled_drafter_round \
        --model-path Qwen/Qwen3-0.6B --num-steps 5 --fanout 4 \
        --batch-size 1 --prompt-len 1024 --rounds 60 --profile
"""

import argparse
import logging
import random
import time

import torch

from sglang.benchmark.one_batch import load_model
from sglang.srt.environ import envs
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.decoupled_draft_engine import EnumDraftEngine
from sglang.srt.speculative.decoupled_spec_io import DraftReqKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--warmup-rounds", type=int, default=8)
    parser.add_argument(
        "--accept-len",
        type=int,
        default=2,
        help="Synthetic per-round accept length; commit delta = accept-len + 1.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Forwarded to ServerArgs when set (else the server default).",
    )
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Per-phase breakdown via SGLANG_DEBUG_DECOUPLED_DRAFT_PROFILE "
        "(adds phase-boundary device syncs; wall time will be inflated).",
    )
    return parser.parse_args()


def _run_rounds(
    *,
    engine: EnumDraftEngine,
    keys: list[DraftReqKey],
    rng: random.Random,
    vocab_size: int,
    delta_len: int,
    rounds: int,
) -> list[float]:
    round_ms = []
    for _ in range(rounds):
        for key in keys:
            delta = [rng.randrange(vocab_size) for _ in range(delta_len)]
            engine.apply_commit(key, delta)
        start = time.monotonic()
        packed = engine.draft_round(keys)
        torch.cuda.synchronize()
        round_ms.append(1000.0 * (time.monotonic() - start))
        assert packed is not None and packed["units_device"].shape[0] == len(keys)
    return round_ms


def main() -> None:
    args = _parse_args()
    if args.profile:
        envs.SGLANG_DEBUG_DECOUPLED_DRAFT_PROFILE.set(True)

    # The drafter's engine sizes its decode batches at bs and bs*(K+1)*F rows;
    # make sure both have a captured decode graph.
    branch_rows = args.batch_size * (args.num_steps + 1) * args.fanout
    server_args_kwargs = dict(
        model_path=args.model_path,
        mem_fraction_static=args.mem_fraction_static,
        cuda_graph_bs_decode=sorted({args.batch_size, branch_rows}),
        disable_cuda_graph=args.disable_cuda_graph,
    )
    if args.attention_backend is not None:
        server_args_kwargs["attention_backend"] = args.attention_backend
    server_args = ServerArgs(**server_args_kwargs)
    port_args = PortArgs.init_new(server_args)
    bench_runner, _tokenizer = load_model(server_args, port_args, 0, 0)
    model_runner = bench_runner.torch_runner
    vocab_size = model_runner.model_config.vocab_size

    engine = EnumDraftEngine(
        model_runner=model_runner,
        num_steps=args.num_steps,
        fanout=args.fanout,
    )
    rng = random.Random(42)
    keys = []
    for i in range(args.batch_size):
        key = DraftReqKey(src_verifier_rank=0, request_id=f"bench-{i}")
        engine.open(
            key,
            req_pool_idx=i,
            prompt_tokens=[rng.randrange(vocab_size) for _ in range(args.prompt_len)],
            committed_outputs=[],
        )
        keys.append(key)
    delta_len = args.accept_len + 1

    _run_rounds(
        engine=engine,
        keys=keys,
        rng=rng,
        vocab_size=vocab_size,
        delta_len=delta_len,
        rounds=args.warmup_rounds,
    )
    engine.profiler.round_ct = 0
    engine.profiler.phase_ms = {}
    round_ms = _run_rounds(
        engine=engine,
        keys=keys,
        rng=rng,
        vocab_size=vocab_size,
        delta_len=delta_len,
        rounds=args.rounds,
    )

    round_ms.sort()
    n = len(round_ms)
    logger.info(
        "drafter round (bs=%d K=%d F=%d prompt=%d delta=%d rounds=%d): "
        "p50=%.2fms mean=%.2fms p90=%.2fms max=%.2fms",
        args.batch_size,
        args.num_steps,
        args.fanout,
        args.prompt_len,
        delta_len,
        n,
        round_ms[n // 2],
        sum(round_ms) / n,
        round_ms[(n * 9) // 10],
        round_ms[-1],
    )
    if args.profile:
        logger.info("phase breakdown: %s", engine.profiler.summary())


if __name__ == "__main__":
    main()
