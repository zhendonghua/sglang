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
    def __init__(self, *, req_pool_idx: int, device: torch.device) -> None:
        # The seat on the OWNING VERIFIER (echoed into every block row); this
        # engine's own scratch rows are unrelated and transient.
        self.req_pool_idx = req_pool_idx
        self.committed_tokens: list[int] = []
        # Slot ids of the committed prefix's KV in the drafter's pool
        # (device-resident: the round must never sync on slot bookkeeping).
        self.committed_slots = torch.empty((0,), dtype=torch.int64, device=device)
        # Last round's block, kept for the glue fast path: the winning unit's
        # chain IS the next round's backbone (greedy re-draft is a no-op).
        self.last_units_dev: Optional[torch.Tensor] = None  # [K+1, F, K+1]
        self.last_units_host: Optional[torch.Tensor] = None  # pinned mirror
        self.last_backbone_host: Optional[list[int]] = None  # c_1..c_K
        self.mirror_event: Optional[torch.cuda.Event] = None

    def pending_delta(self) -> list[int]:
        """Committed tokens whose KV has not been advanced yet."""
        return self.committed_tokens[self.committed_slots.numel() :]


class _RoundCarriers:
    """Retained fast-path batches for one fixed key set.

    ``glue_batch``: bs*K one-token extend rows; row (i, g) re-materializes
    backbone token c_{g+1} on top of committed + c_1..c_g. Prefixes are
    slot-shared ACROSS ROWS OF THE SAME FORWARD: per layer, the batched KV
    write precedes the attention read, and c_g's KV depends only on its own
    row, so row g+1 reads row g's fresh KV exactly as a sequential chain
    would.

    ``branch_batch``: bs*(K+1)*F persistent decode rows; each round only
    seq_lens and the pool-row tail entries move, then the K chain steps run
    as plain decode (cuda-graph replays).

    Pool-row content is maintained incrementally: rows carry the committed
    prefix up to ``synced_lens[i]``; the region past it is per-round scratch
    mapping (delta slots, then backbone slots) and is rewritten every round.
    """

    def __init__(
        self,
        *,
        keys: tuple[DraftReqKey, ...],
        glue_batch: ScheduleBatch,
        branch_batch: ScheduleBatch,
        synced_lens: list[int],
        num_steps: int,
        fanout: int,
        device: torch.device,
    ) -> None:
        self.keys = keys
        self.glue_batch = glue_batch
        self.branch_batch = branch_batch
        self.synced_lens = synced_lens
        num_cases = num_steps + 1
        rows_per_seat = num_cases * fanout
        bs = len(keys)
        glue_rows = glue_batch.req_pool_indices.view(bs, num_steps)
        branch_rows = branch_batch.req_pool_indices.view(bs, rows_per_seat)
        # All carrier rows sharing the committed prefix (delta broadcast).
        self.all_rows = torch.cat([glue_rows, branch_rows], dim=1)  # [bs, K+R]
        # Combined scatter template: glue triangle (row g needs c_1..c_g's
        # slots at [L:L+g] INCLUSIVE -- fa3 extend reads the current token's
        # own K/V through the page table too) + branch case prefixes (row
        # (c, f) needs c_1..c_c's slots at [L:L+c); its own entry is written
        # by alloc_for_decode). Entry j's value is backbone slot j.
        tri_g = [g for g in range(num_steps) for j in range(g + 1)]
        tri_j = [j for g in range(num_steps) for j in range(g + 1)]
        br_r = [
            c * fanout + f
            for c in range(num_cases)
            for f in range(fanout)
            for j in range(c)
        ]
        br_j = [j for c in range(num_cases) for f in range(fanout) for j in range(c)]
        tri_g_dev = torch.tensor(tri_g, dtype=torch.int64, device=device)
        br_r_dev = torch.tensor(br_r, dtype=torch.int64, device=device)
        self.comb_rows = torch.cat(
            [glue_rows[:, tri_g_dev], branch_rows[:, br_r_dev]], dim=1
        )  # [bs, T]
        self.comb_j = torch.tensor(tri_j + br_j, dtype=torch.int64, device=device)
        self.case_of_row = [c for c in range(num_cases) for _ in range(fanout)]


class EnumDraftEngine:
    """Per-request committed KV + one enumeration tree per commit round."""

    def __init__(
        self,
        *,
        model_runner: ModelRunner,
        num_steps: int,
        fanout: int,
        enable_glue_fast_path: bool = True,
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
        self._carriers: Optional[_RoundCarriers] = None
        self._enable_glue_fast_path = bool(enable_glue_fast_path)
        self.hit_ct = 0
        self.miss_ct = 0
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
        state = _DraftReqState(req_pool_idx=req_pool_idx, device=self.device)
        state.committed_tokens = list(prompt_tokens) + list(committed_outputs)
        self._states[key] = state

    def apply_commit(self, key: DraftReqKey, committed_tokens: list[int]) -> None:
        state = self._states.get(key)
        if state is None:
            return
        state.committed_tokens.extend(int(t) for t in committed_tokens)

    def close(self, key: DraftReqKey) -> None:
        # The carriers are keyed on the exact key set; any membership change
        # invalidates them.
        self._evict_carriers()
        state = self._states.pop(key, None)
        if state is not None and state.committed_slots.numel() > 0:
            self.model_runner.token_to_kv_pool_allocator.free(state.committed_slots)

    def has(self, key: DraftReqKey) -> bool:
        return key in self._states

    # ------------------------------------------------------------------ #
    # One enumeration round
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def draft_round(self, keys: list[DraftReqKey]) -> Optional[dict]:
        """Draft one enumeration tree per key; returns the packed parallel
        arrays {pool_indices, base_committed_lens, tokens} or None if no key
        is live. Frees every scratch slot before returning.

        Two forms of the same tree:

        - **glue fast path** (every seat's commit matched a unit of its last
          block): the winning unit's chain IS the new backbone (greedy
          re-draft is deterministic), so one K-row extend re-materializes
          its KV and yields all node logits; the branch phase runs as plain
          decode replays on the retained carrier batch.
        - **slow path** (first round / any miss / key-set change): the
          original build-everything round; it also (re)builds the carriers
          and host mirrors that arm the fast path.
        """
        keys = [key for key in keys if key in self._states]
        if not keys:
            return None
        states = [self._states[key] for key in keys]
        scratch_batches: list[ScheduleBatch] = []
        scratch_slots: list[torch.Tensor] = []
        self.profiler.start_round()
        try:
            selections = self._match_fast_path(keys, states)
            if selections is not None:
                self.hit_ct += 1
                return self._fast_round(
                    states, selections, scratch_batches, scratch_slots
                )
            self.miss_ct += 1
            return self._slow_round(keys, states, scratch_batches, scratch_slots)
        finally:
            self._free_scratch(scratch_batches, scratch_slots)
            self.profiler.mark("free")

    def _match_fast_path(
        self, keys: list[DraftReqKey], states: list[_DraftReqState]
    ) -> Optional[list[tuple[int, int]]]:
        """Match every seat's pending delta against its last block; returns
        the winning (accept_case, fanout_index) per seat, or None if any seat
        misses (which routes the whole round to the slow path)."""
        if not self._enable_glue_fast_path:
            return None
        if self._carriers is None or self._carriers.keys != tuple(keys):
            return None
        selections: list[tuple[int, int]] = []
        for state in states:
            if state.last_units_host is None or state.last_backbone_host is None:
                return None
            delta = state.pending_delta()
            case = len(delta) - 1
            if case < 0 or case > self.num_steps:
                return None
            if delta[:case] != state.last_backbone_host[:case]:
                return None
            if state.mirror_event is not None:
                state.mirror_event.synchronize()
            guesses_row = state.last_units_host[case, :, 0].tolist()
            bonus = delta[case]
            if bonus not in guesses_row:
                return None
            selections.append((case, guesses_row.index(bonus)))
        return selections

    def _fast_round(
        self,
        states: list[_DraftReqState],
        selections: list[tuple[int, int]],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
    ) -> dict:
        num_steps, fanout = self.num_steps, self.fanout
        bs = len(states)
        carriers = self._carriers
        allocator = self.model_runner.token_to_kv_pool_allocator
        pool = self.model_runner.req_to_token_pool

        # -- Advance the committed prefix (node 0 logits), as on the slow
        # path; its freshly written KV slots feed the carrier row updates.
        base_lens = [state.committed_slots.numel() for state in states]
        advance_batch, advance_slots = self._extend_batch(
            token_lists=[state.committed_tokens for state in states],
            prefix_slots=[state.committed_slots for state in states],
            tag="advance",
        )
        scratch_batches.append(advance_batch)
        node0_logits = self._forward(advance_batch, tag="advance")
        # Graph-runner logits live in a static output buffer that the NEXT
        # forward overwrites -- consume them (topk) before the glue forward,
        # exactly like the slow path consumes each step's logits immediately.
        node0_guesses = torch.topk(node0_logits, fanout, dim=-1).indices  # [bs, F]
        self._absorb_advance_slots(states, advance_slots)

        # -- Carrier pool rows: broadcast the committed delta, then scatter
        # this round's backbone slots into the glue triangle + branch cases.
        backbone_slots = allocator.alloc(bs * num_steps)
        if backbone_slots is None:
            raise RuntimeError("drafter KV pool exhausted (glue backbone)")
        scratch_slots.append(backbone_slots)
        backbone_slots = backbone_slots.view(bs, num_steps)
        chains: list[torch.Tensor] = []
        new_backbones: list[list[int]] = []
        for i, state in enumerate(states):
            new_len = state.committed_slots.numel()
            synced = min(carriers.synced_lens[i], base_lens[i])
            pool.req_to_token[carriers.all_rows[i], synced:new_len] = (
                state.committed_slots[synced:new_len].to(torch.int32)
            )
            carriers.synced_lens[i] = new_len
            slots_i32 = backbone_slots[i].to(torch.int32)
            pool.req_to_token[carriers.comb_rows[i], carriers.comb_j + new_len] = (
                slots_i32[carriers.comb_j]
            )
            case, f = selections[i]
            chains.append(state.last_units_dev[case, f, 1:])
            # The old host mirror was synced during matching; snapshot the new
            # backbone before _pack_and_mirror overwrites it.
            new_backbones.append(state.last_units_host[case, f, 1:].tolist())
        self.profiler.mark("carrier_sync")

        # -- Glue extend: all K backbone tokens in one forward = node 1..K
        # logits; their KV lands in this round's backbone slots.
        glue_logits = self._glue_forward(
            states=states, chains=chains, backbone_slots=backbone_slots
        )
        glue_guesses = torch.topk(
            glue_logits.view(bs, num_steps, -1), fanout, dim=-1
        ).indices  # [bs, K, F]
        guesses_stack = torch.cat([node0_guesses.unsqueeze(1), glue_guesses], dim=1)

        # -- Branch chains: K decode replays on the retained carrier batch.
        chain_steps = self._branch_decode_chain(
            states=states, guesses_stack=guesses_stack, scratch_slots=scratch_slots
        )
        return self._pack_and_mirror(
            states=states,
            guesses_stack=guesses_stack,
            chain_steps=chain_steps,
            new_backbones=new_backbones,
        )

    def _glue_forward(
        self,
        *,
        states: list[_DraftReqState],
        chains: list[torch.Tensor],
        backbone_slots: torch.Tensor,
    ) -> torch.Tensor:
        num_steps = self.num_steps
        bs = len(states)
        glue = self._carriers.glue_batch
        lens = [state.committed_slots.numel() for state in states]
        seq_host = [lens[i] + g + 1 for i in range(bs) for g in range(num_steps)]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        glue.input_ids = torch.cat(chains) if bs > 1 else chains[0]
        glue.out_cache_loc = backbone_slots.view(-1)
        glue.seq_lens = seq_cpu.to(self.device, non_blocking=True)
        glue.seq_lens_cpu = seq_cpu
        glue.seq_lens_sum = sum(seq_host)
        glue.orig_seq_lens = glue.seq_lens.to(torch.int32)
        glue.prefix_lens = [s - 1 for s in seq_host]
        self.profiler.mark("glue_mut")
        return self._forward(glue, tag="glue")

    def _branch_decode_chain(
        self,
        *,
        states: list[_DraftReqState],
        guesses_stack: torch.Tensor,
        scratch_slots: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        num_steps = self.num_steps
        branch = self._carriers.branch_batch
        case_of_row = self._carriers.case_of_row
        seq_host = [
            state.committed_slots.numel() + case
            for state in states
            for case in case_of_row
        ]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        branch.seq_lens = seq_cpu.to(self.device, non_blocking=True)
        branch.seq_lens_cpu = seq_cpu
        branch.seq_lens_sum = None
        branch.orig_seq_lens = branch.seq_lens.to(torch.int32)
        self.profiler.mark("branch_mut")
        logits, step_slots = self._decode_step(
            branch, guesses_stack.reshape(-1), tag="branch"
        )
        scratch_slots.append(step_slots)
        chain_steps: list[torch.Tensor] = [logits.argmax(dim=-1)]
        for _ in range(num_steps - 1):
            logits, step_slots = self._decode_step(
                branch, chain_steps[-1], tag="branch"
            )
            scratch_slots.append(step_slots)
            chain_steps.append(logits.argmax(dim=-1))
        return chain_steps

    def _absorb_advance_slots(
        self, states: list[_DraftReqState], advance_slots: torch.Tensor
    ) -> None:
        """Newly written KV joins the committed prefix (kept across rounds)."""
        offset = 0
        for state in states:
            new_len = len(state.committed_tokens) - state.committed_slots.numel()
            state.committed_slots = torch.cat(
                [state.committed_slots, advance_slots[offset : offset + new_len]]
            )
            offset += new_len
        self.profiler.mark("commit_slots")

    def _slow_round(
        self,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
    ) -> dict:
        num_steps, fanout = self.num_steps, self.fanout
        num_cases = num_steps + 1
        bs = len(states)
        # A slow round never uses the carriers and rebuilds them at the end;
        # evicting up front halves the round's peak pool-row usage.
        self._evict_carriers()

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
        self._absorb_advance_slots(states, advance_slots)

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
        backbone_slots_dev = (
            torch.stack(backbone_slot_steps, dim=1)
            if backbone_slot_steps
            else torch.empty((bs, 0), dtype=torch.int64, device=self.device)
        )  # [bs, K]
        branch_token_lists: list[list[int]] = []
        branch_prefix_slots: list[torch.Tensor] = []
        for i, state in enumerate(states):
            for case in range(num_cases):
                for f in range(fanout):
                    branch_token_lists.append(
                        state.committed_tokens
                        + backbone_cpu[i][:case]
                        + [guesses_cpu[i][case][f]]
                    )
                    branch_prefix_slots.append(
                        torch.cat([state.committed_slots, backbone_slots_dev[i, :case]])
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

        packed = self._pack_and_mirror(
            states=states,
            guesses_stack=guesses_stack,
            chain_steps=chain_steps,
            new_backbones=backbone_cpu,
        )
        self._rebuild_carriers(
            keys=keys,
            states=states,
            branch_batch=branch_batch,
            scratch_batches=scratch_batches,
            scratch_slots=scratch_slots,
            backbone_cpu=backbone_cpu,
            backbone_slots_dev=backbone_slots_dev,
        )
        return packed

    # ------------------------------------------------------------------ #
    # Packing, mirrors, carriers
    # ------------------------------------------------------------------ #

    def _pack_and_mirror(
        self,
        *,
        states: list[_DraftReqState],
        guesses_stack: torch.Tensor,
        chain_steps: list[torch.Tensor],
        new_backbones: list[list[int]],
    ) -> dict:
        """Pack units [guess, chain_1..chain_K] and arm the fast path.

        chains: [bs * (K+1) * F, K] -> [bs, K+1, F, K]; stays on device so
        the CUDA IPC data plane can push it D2D (the ZMQ path D2Hs it). Each
        seat keeps the block on device (next round's glue input) plus an
        async pinned host mirror (next round's hit test).
        """
        num_cases = self.num_steps + 1
        bs = len(states)
        chains = torch.stack(chain_steps, dim=1).view(
            bs, num_cases, self.fanout, self.num_steps
        )
        guesses_col = guesses_stack.unsqueeze(-1)  # [bs, K+1, F, 1]
        units_device = torch.cat([guesses_col, chains], dim=-1)  # [bs, K+1, F, K+1]
        self.profiler.mark("pack")
        mirror_event = torch.cuda.Event()
        for i, state in enumerate(states):
            state.last_units_dev = units_device[i]
            if state.last_units_host is None:
                state.last_units_host = torch.empty(
                    units_device[i].shape, dtype=units_device.dtype, pin_memory=True
                )
            state.last_units_host.copy_(units_device[i], non_blocking=True)
            state.last_backbone_host = list(new_backbones[i])
            state.mirror_event = mirror_event
        mirror_event.record()
        self.profiler.mark("mirror")
        return {
            "pool_indices": [state.req_pool_idx for state in states],
            "base_committed_lens": [len(state.committed_tokens) for state in states],
            "units_device": units_device,
        }

    def _rebuild_carriers(
        self,
        *,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        branch_batch: ScheduleBatch,
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
        backbone_cpu: list[list[int]],
        backbone_slots_dev: torch.Tensor,
    ) -> None:
        """Retain this slow round's branch batch + a freshly built glue batch
        as the fast path's carriers (their pool rows persist; KV slots stay
        per-round scratch)."""
        if not self._enable_glue_fast_path:
            return
        self._evict_carriers()
        # The branch batch's pool rows survive the round; its KV slots are
        # already tracked in scratch_slots and freed as usual.
        scratch_batches.remove(branch_batch)
        glue_batch, glue_slots = self._extend_batch(
            token_lists=[
                state.committed_tokens + backbone_cpu[i][: g + 1]
                for i, state in enumerate(states)
                for g in range(self.num_steps)
            ],
            prefix_slots=[
                torch.cat([state.committed_slots, backbone_slots_dev[i, :g]])
                for i, state in enumerate(states)
                for g in range(self.num_steps)
            ],
            tag="glue_build",
        )
        # Build-time extend slots are placeholders (no forward ran); the fast
        # path re-points out_cache_loc at each round's backbone slots.
        scratch_slots.append(glue_slots)
        self._carriers = _RoundCarriers(
            keys=tuple(keys),
            glue_batch=glue_batch,
            branch_batch=branch_batch,
            synced_lens=[state.committed_slots.numel() for state in states],
            num_steps=self.num_steps,
            fanout=self.fanout,
            device=self.device,
        )
        self.profiler.mark("carrier_build")

    def _evict_carriers(self) -> None:
        carriers, self._carriers = self._carriers, None
        if carriers is None:
            return
        for batch in (carriers.glue_batch, carriers.branch_batch):
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    self.model_runner.req_to_token_pool.free(req)

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
