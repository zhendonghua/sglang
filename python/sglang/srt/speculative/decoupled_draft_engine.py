"""Enumeration draft engine: the decoupled drafter's core compute.

Runs the draft model directly through ``ScheduleBatch`` + ``ModelRunner``
(the bench_one_batch harness pattern) -- no user requests, no admission, no
radix cache. Per request it keeps only the committed prefix's KV (token slots
it owns); every round is a re-extension from that prefix:

1. **advance** (extend): compute KV for the newly committed tokens; the last
   position's logits are enumeration node 0.
2. **backbone** (extend + K-1 decodes on scratch rows): greedily draft
   c_1..c_K; the logits at node ``a`` (after c_1..c_a) provide the top-F
   bonus guesses g_{a, 0..F-1} for accept case ``a`` (c_{a+1} == g_{a,0}).
3. **branches** (one batch of (K+1) x F scratch rows, extend + K-1 decodes):
   chain(a, f) = K tokens drafted after prefix + c_1..c_a + g_{a,f}. Nested
   prefixes are shared by slot id (page_size == 1, read-only), never copied.
4. **pack**: unit(a, f) = [g_{a,f}, chain_1..chain_K]; the block's stamp is
   the total committed length the tree grew from.

All scratch state (rows + KV slots written for backbone / branch tokens) is
freed at the end of the round: a wrong branch is never selected, and the next
commit re-extends from the committed prefix (keep-winning-branch KV is a
listed future optimization).
"""

from __future__ import annotations

import logging
import time
from array import array
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_spec_io import DraftReqKey
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class _ScratchTreeCache(SimpleNamespace):
    """Allocation-only tree cache stub (no prefix caching), bench-harness style."""

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, params: EvictParams):
        pass


class _RoundProfiler:
    """Per-phase host-time accumulator for the enumeration round.

    Syncs the device at every mark so each phase's wall time is attributed
    exactly -- which also serializes host and GPU work. Numbers are for
    relative breakdown, not absolute round latency (debug only).
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.round_ct = 0
        self.phase_ms: dict[str, float] = {}
        self._t_last = 0.0

    def start_round(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.round_ct += 1
        self._t_last = time.monotonic()

    def mark(self, phase: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.monotonic()
        self.phase_ms[phase] = self.phase_ms.get(phase, 0.0) + 1000.0 * (
            now - self._t_last
        )
        self._t_last = now

    def summary(self) -> str:
        if self.round_ct == 0:
            return "no profiled rounds"
        parts = [
            f"{phase}={ms / self.round_ct:.2f}"
            for phase, ms in sorted(
                self.phase_ms.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        total = sum(self.phase_ms.values()) / self.round_ct
        return f"rounds={self.round_ct} total_ms={total:.2f} | " + " ".join(parts)


class _DraftReqState:
    def __init__(self, *, req_pool_idx: int) -> None:
        # The seat on the OWNING VERIFIER (echoed into every block row); this
        # engine's own scratch rows are unrelated and transient.
        self.req_pool_idx = req_pool_idx
        self.committed_tokens: list[int] = []
        # Slot ids of the committed prefix's KV in the drafter's pool.
        self.committed_slots = torch.empty((0,), dtype=torch.int64)


class EnumDraftEngine:
    """Per-request committed KV + one enumeration tree per commit round."""

    def __init__(
        self,
        *,
        model_runner: ModelRunner,
        num_steps: int,
        fanout: int,
    ) -> None:
        self.model_runner = model_runner
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        self.unit_width = self.num_steps + 1
        self.device = model_runner.device
        self._tree_cache = _ScratchTreeCache(
            page_size=model_runner.server_args.page_size,
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
        # Greedy, never finishing on its own; lifecycle is DraftSync/DraftClose.
        self._sampling_params = SamplingParams(temperature=0, max_new_tokens=1 << 30)
        self._states: dict[DraftReqKey, _DraftReqState] = {}
        self.profiler = _RoundProfiler(
            enabled=envs.SGLANG_DEBUG_DECOUPLED_DRAFT_PROFILE.get()
        )

    # ------------------------------------------------------------------ #
    # Lifecycle (control plane)
    # ------------------------------------------------------------------ #

    def open(
        self,
        key: DraftReqKey,
        *,
        req_pool_idx: int,
        prompt_tokens: list[int],
        committed_outputs: list[int],
    ) -> None:
        # Re-open (retraction re-sync) drops the old prefix KV entirely.
        self.close(key)
        state = _DraftReqState(req_pool_idx=req_pool_idx)
        state.committed_tokens = list(prompt_tokens) + list(committed_outputs)
        self._states[key] = state

    def apply_commit(self, key: DraftReqKey, committed_tokens: list[int]) -> None:
        state = self._states.get(key)
        if state is None:
            return
        state.committed_tokens.extend(int(t) for t in committed_tokens)

    def close(self, key: DraftReqKey) -> None:
        state = self._states.pop(key, None)
        if state is not None and state.committed_slots.numel() > 0:
            self.model_runner.token_to_kv_pool_allocator.free(
                state.committed_slots.to(self.device)
            )

    def has(self, key: DraftReqKey) -> bool:
        return key in self._states

    # ------------------------------------------------------------------ #
    # One enumeration round
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def draft_round(self, keys: list[DraftReqKey]) -> Optional[dict]:
        """Draft one enumeration tree per key; returns the packed parallel
        arrays {pool_indices, base_committed_lens, tokens} or None if no key
        is live. Frees every scratch row / slot before returning.
        """
        keys = [key for key in keys if key in self._states]
        if not keys:
            return None
        states = [self._states[key] for key in keys]
        scratch_batches: list[ScheduleBatch] = []
        scratch_slots: list[torch.Tensor] = []
        self.profiler.start_round()
        try:
            return self._draft_round_inner(states, scratch_batches, scratch_slots)
        finally:
            self._free_scratch(scratch_batches, scratch_slots)
            self.profiler.mark("free")

    def _draft_round_inner(
        self,
        states: list[_DraftReqState],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
    ) -> dict:
        num_steps, fanout = self.num_steps, self.fanout
        num_cases = num_steps + 1
        bs = len(states)

        # -- Phase 1: advance the committed prefix; last logits = node 0 ----
        node_logits: list[torch.Tensor] = []
        advance_batch, advance_slots = self._extend_batch(
            token_lists=[state.committed_tokens for state in states],
            prefix_slots=[state.committed_slots for state in states],
            tag="advance",
        )
        scratch_batches.append(advance_batch)
        logits = self._forward(advance_batch, tag="advance")
        node_logits.append(logits)
        # Newly written KV joins the committed prefix (kept across rounds).
        offset = 0
        for state in states:
            new_len = len(state.committed_tokens) - state.committed_slots.numel()
            state.committed_slots = torch.cat(
                [
                    state.committed_slots,
                    advance_slots[offset : offset + new_len].to("cpu"),
                ]
            )
            offset += new_len
        self.profiler.mark("commit_slots")

        # -- Phase 2: backbone c_1..c_K + per-node top-F guesses ------------
        # guesses[a]: [bs, F] int64; backbone_tokens[j]: [bs] (c_{j+1}).
        guesses = [torch.topk(node_logits[0], fanout, dim=-1).indices]
        backbone_tokens: list[torch.Tensor] = [guesses[0][:, 0]]
        backbone_slot_steps: list[torch.Tensor] = []
        if num_steps >= 1:
            backbone_batch, first_slots = self._extend_batch(
                token_lists=[
                    state.committed_tokens + [int(backbone_tokens[0][i])]
                    for i, state in enumerate(states)
                ],
                prefix_slots=[state.committed_slots for state in states],
                tag="backbone",
            )
            scratch_batches.append(backbone_batch)
            scratch_slots.append(first_slots)
            backbone_slot_steps.append(first_slots)
            logits = self._forward(backbone_batch, tag="backbone")
            node_logits.append(logits)
            guesses.append(torch.topk(logits, fanout, dim=-1).indices)
            for _ in range(num_steps - 1):
                next_tokens = guesses[-1][:, 0]
                backbone_tokens.append(next_tokens)
                logits, step_slots = self._decode_step(
                    backbone_batch, next_tokens, tag="backbone"
                )
                scratch_slots.append(step_slots)
                backbone_slot_steps.append(step_slots)
                node_logits.append(logits)
                guesses.append(torch.topk(logits, fanout, dim=-1).indices)

        # -- Phase 3: branch chains for every (case, guess) -----------------
        # Row order: (req 0: (a0,f0), (a0,f1) ... (aK,fF-1)), (req 1: ...).
        guesses_stack = torch.stack(guesses, dim=1)  # [bs, K+1, F]
        guesses_cpu = guesses_stack.tolist()
        backbone_cpu = [
            [int(backbone_tokens[j][i]) for j in range(num_steps)] for i in range(bs)
        ]
        branch_token_lists: list[list[int]] = []
        branch_prefix_slots: list[torch.Tensor] = []
        for i, state in enumerate(states):
            backbone_slots_i = torch.tensor(
                [int(step[i]) for step in backbone_slot_steps], dtype=torch.int64
            )
            for case in range(num_cases):
                for f in range(fanout):
                    branch_token_lists.append(
                        state.committed_tokens
                        + backbone_cpu[i][:case]
                        + [guesses_cpu[i][case][f]]
                    )
                    branch_prefix_slots.append(
                        torch.cat([state.committed_slots, backbone_slots_i[:case]])
                    )
        self.profiler.mark("branch_lists")
        branch_batch, branch_first_slots = self._extend_batch(
            token_lists=branch_token_lists,
            prefix_slots=branch_prefix_slots,
            tag="branch",
        )
        scratch_batches.append(branch_batch)
        scratch_slots.append(branch_first_slots)
        logits = self._forward(branch_batch, tag="branch")
        chain_steps: list[torch.Tensor] = [logits.argmax(dim=-1)]
        for _ in range(num_steps - 1):
            logits, step_slots = self._decode_step(
                branch_batch, chain_steps[-1], tag="branch"
            )
            scratch_slots.append(step_slots)
            chain_steps.append(logits.argmax(dim=-1))

        # -- Phase 4: pack units [guess, chain_1..chain_K] ------------------
        # chains: [bs * num_cases * F, K] -> [bs, K+1, F, K]; stays on device
        # so the CUDA IPC data plane can push it D2D (the ZMQ path D2Hs it).
        chains = torch.stack(chain_steps, dim=1).view(bs, num_cases, fanout, num_steps)
        guesses_col = guesses_stack.unsqueeze(-1)  # [bs, K+1, F, 1]
        units_device = torch.cat([guesses_col, chains], dim=-1)  # [bs, K+1, F, K+1]
        self.profiler.mark("pack")
        return {
            "pool_indices": [state.req_pool_idx for state in states],
            "base_committed_lens": [len(state.committed_tokens) for state in states],
            "units_device": units_device,
        }

    # ------------------------------------------------------------------ #
    # Batch plumbing (bench_one_batch harness pattern)
    # ------------------------------------------------------------------ #

    def _extend_batch(
        self,
        *,
        token_lists: list[list[int]],
        prefix_slots: list[torch.Tensor],
        tag: str,
    ) -> tuple[ScheduleBatch, torch.Tensor]:
        """Extend each row's tokens beyond its (slot-shared) prefix.

        Returns (batch, newly_allocated_slots_flat).
        """
        reqs = []
        for i, tokens in enumerate(token_lists):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=array("q", tokens),
                sampling_params=self._sampling_params,
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.prefix_indices = prefix_slots[i].to(self.device)
            req.set_extend_range(prefix_slots[i].numel(), len(tokens))
            reqs.append(req)
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=self._tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        self.profiler.mark(f"{tag}_build")
        return batch, batch.out_cache_loc

    def _forward(self, batch: ScheduleBatch, *, tag: str) -> torch.Tensor:
        forward_batch = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        self.profiler.mark(f"{tag}_fb")
        logits_output = self.model_runner.forward(forward_batch).logits_output
        self.profiler.mark(f"{tag}_fwd")
        return logits_output.next_token_logits

    def _decode_step(
        self, batch: ScheduleBatch, input_tokens: torch.Tensor, *, tag: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch.input_ids = input_tokens.to(torch.int64)
        batch.prepare_for_decode()
        self.profiler.mark(f"{tag}_step_prep")
        forward_batch = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        self.profiler.mark(f"{tag}_step_fb")
        logits_output = self.model_runner.forward(forward_batch).logits_output
        self.profiler.mark(f"{tag}_step_fwd")
        return logits_output.next_token_logits, batch.out_cache_loc

    def _free_scratch(
        self,
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
    ) -> None:
        for slots in scratch_slots:
            if slots is not None and slots.numel() > 0:
                self.model_runner.token_to_kv_pool_allocator.free(slots)
        for batch in scratch_batches:
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    self.model_runner.req_to_token_pool.free(req)
