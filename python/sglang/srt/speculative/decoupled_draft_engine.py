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

A miss round (the commit fell outside the last block) collapses to case 0:
the verifier's select missed the same block the same way and falls back, so
the next commit can only be a single bonus -- only the F case-0 chains are
drafted and the dead cells are poisoned (see ``_case0_round``).
"""

from __future__ import annotations

import logging
import time
from array import array
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import msgspec
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
        # Top-1 prerun bet: number of speculatively committed tokens (0 = no
        # active bet) + the pre-bet mirror snapshot for rollback.
        self.prerun_len = 0
        self.prerun_snapshot: Optional[tuple] = None

    def pending_delta(self) -> list[int]:
        """Committed tokens whose KV has not been advanced yet."""
        return self.committed_tokens[self.committed_slots.numel() :]


class _CascadeMetadata(msgspec.Struct):
    """Shared-prefix cascade inputs for one branch-decode chain (fa3 decode
    consumes it via ForwardBatch.decoupled_cascade). Page tables/lens are
    int32 per the fa3 convention; tail state advances in place per step."""

    prefix_page_table: torch.Tensor  # [seats, max_prefix_len]
    prefix_lens: torch.Tensor  # [seats]
    tail_page_table: torch.Tensor  # [rows, 2K+2]
    tail_lens: torch.Tensor  # [rows]


class _SeatCarrier:
    """Retained fast-path pool rows + scatter template for ONE seat.

    ``glue_rows``: K one-token extend rows; row g re-materializes backbone
    token c_{g+1} on top of committed + c_1..c_g. Prefixes are slot-shared
    ACROSS ROWS OF THE SAME FORWARD: per layer, the batched KV write precedes
    the attention read, and c_g's KV depends only on its own row, so row g+1
    reads row g's fresh KV exactly as a sequential chain would.

    ``branch_rows``: (K+1)*F persistent decode rows; each round only seq_lens
    and the pool-row tail entries move, then the K chain steps run as plain
    decode (cuda-graph replays).

    Pool-row content is maintained incrementally: rows carry the committed
    prefix up to ``synced_len``; the region past it is per-round scratch
    mapping (delta slots, then backbone slots) and is rewritten every round.
    Seats are independent so any hit subset of a batch can run the fast path
    (per-seat mixing); rows live until the seat closes.
    """

    def __init__(
        self,
        *,
        glue_rows: torch.Tensor,  # [K] device
        branch_rows: torch.Tensor,  # [(K+1)*F] device
        glue_reqs: list,
        branch_reqs: list,
        synced_len: int,
        tri_g: torch.Tensor,
        br_r: torch.Tensor,
    ) -> None:
        self.glue_rows = glue_rows
        self.branch_rows = branch_rows
        # Req stubs owning the pool rows (freed via ReqToTokenPool.free(req)).
        self.glue_reqs = glue_reqs
        self.branch_reqs = branch_reqs
        self.synced_len = synced_len
        # All carrier rows sharing the committed prefix (delta broadcast).
        self.all_rows = torch.cat([glue_rows, branch_rows])
        # Combined scatter rows (values/cols come from the engine templates).
        self.comb_rows = torch.cat([glue_rows[tri_g], branch_rows[br_r]])


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
        self._seat_carriers: dict[DraftReqKey, _SeatCarrier] = {}
        self._enable_glue_fast_path = bool(enable_glue_fast_path)
        self.prerun_hit_ct = 0
        self.prerun_miss_ct = 0
        # Shared-prefix cascade for branch decodes (fa3): only past this
        # prefix length does dedup beat the L2-served re-reads. 0 = off.
        self._cascade_min_prefix_len = (
            envs.SGLANG_DECOUPLED_CASCADE_MIN_PREFIX_LEN.get()
        )
        # Static scatter templates shared by every seat carrier: glue triangle
        # (row g needs c_1..c_g's slots at [L:L+g] INCLUSIVE -- fa3 extend
        # reads the current token's own K/V through the page table too) +
        # branch case prefixes (row (c, f) needs c_1..c_c's slots at [L:L+c);
        # its own entry is written by alloc_for_decode). Entry j's value is
        # backbone slot j.
        num_cases = self.num_steps + 1
        tri_g = [g for g in range(self.num_steps) for j in range(g + 1)]
        tri_j = [j for g in range(self.num_steps) for j in range(g + 1)]
        br_r = [
            c * self.fanout + f
            for c in range(num_cases)
            for f in range(self.fanout)
            for j in range(c)
        ]
        br_j = [
            j for c in range(num_cases) for f in range(self.fanout) for j in range(c)
        ]
        self._tri_g = torch.tensor(tri_g, dtype=torch.int64, device=self.device)
        self._br_r = torch.tensor(br_r, dtype=torch.int64, device=self.device)
        self._br_j = torch.tensor(br_j, dtype=torch.int64, device=self.device)
        self._comb_j = torch.tensor(tri_j + br_j, dtype=torch.int64, device=self.device)
        self._case_of_row = [c for c in range(num_cases) for _ in range(self.fanout)]
        # Reusable batch shells for assembled fast subrounds (retained from
        # slow rounds; per-round fields are fully rebound before each use).
        self._glue_template: Optional[ScheduleBatch] = None
        self._branch_template: Optional[ScheduleBatch] = None
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

    def apply_commit(self, key: DraftReqKey, committed_tokens: list[int]) -> bool:
        """Apply a real commit; returns True when an active top-1 prerun bet
        matched it exactly (the seat's next block is already on the verifier,
        so the caller skips drafting it this round)."""
        state = self._states.get(key)
        if state is None:
            return False
        delta = [int(t) for t in committed_tokens]
        if state.prerun_len > 0:
            bet = state.committed_tokens[-state.prerun_len :]
            if delta == bet:
                state.prerun_len = 0
                state.prerun_snapshot = None
                self.prerun_hit_ct += 1
                return True
            self._rollback_prerun(state)
            self.prerun_miss_ct += 1
            if self.prerun_miss_ct <= 3:
                # Alignment probe: a systematic construction/compare bug shows
                # up in the first few misses (bet vs real delta side by side).
                logger.info(
                    "prerun miss #%d: bet=%s delta=%s",
                    self.prerun_miss_ct,
                    bet[:8],
                    delta[:8],
                )
        state.committed_tokens.extend(delta)
        return False

    def _rollback_prerun(self, state: _DraftReqState) -> None:
        """Undo a wrong bet: drop the speculative tokens, free their KV, and
        restore the pre-bet mirrors (the real delta must be matched against
        the pre-bet block, not the bet one)."""
        base_len = len(state.committed_tokens) - state.prerun_len
        if state.committed_slots.numel() > base_len:
            self.model_runner.token_to_kv_pool_allocator.free(
                state.committed_slots[base_len:]
            )
            state.committed_slots = state.committed_slots[:base_len]
        state.committed_tokens = state.committed_tokens[:base_len]
        units_dev, units_host_clone, backbone_host, mirror_event = state.prerun_snapshot
        state.last_units_dev = units_dev
        if state.last_units_host is not None and units_host_clone is not None:
            # Restore INTO the pinned buffer so the mirror keeps its identity.
            state.last_units_host.copy_(units_host_clone)
        state.last_backbone_host = backbone_host
        state.mirror_event = mirror_event
        state.prerun_len = 0
        state.prerun_snapshot = None

    @torch.no_grad()
    def speculative_prerun(self, keys: list[DraftReqKey]) -> Optional[dict]:
        """Bet each seat's most likely next commit (full accept + its own top
        bonus guess g_{K,0}), pre-run that round now, and return the packed
        block to ship speculatively. By construction the bet delta hits the
        glue fast path. A wrong bet is rolled back by apply_commit and only
        cost idle drafter time."""
        ready: list[tuple[DraftReqKey, _DraftReqState]] = []
        for key in keys:
            state = self._states.get(key)
            if (
                state is None
                or state.prerun_len > 0
                or state.last_units_host is None
                # Empty after a case-0 miss round: that block carries no
                # backbone, so there is no full-accept outcome to bet.
                or not state.last_backbone_host
                or key not in self._seat_carriers
            ):
                continue
            ready.append((key, state))
        if not ready:
            return None
        for _, state in ready:
            if state.mirror_event is not None:
                state.mirror_event.synchronize()
            bet_delta = list(state.last_backbone_host) + [
                int(state.last_units_host[self.num_steps, 0, 0])
            ]
            state.prerun_snapshot = (
                state.last_units_dev,
                state.last_units_host.clone(),
                state.last_backbone_host,
                state.mirror_event,
            )
            state.committed_tokens.extend(bet_delta)
            state.prerun_len = len(bet_delta)
        try:
            return self.draft_round([key for key, _ in ready])
        except Exception:
            for _, state in ready:
                if state.prerun_len > 0:
                    self._rollback_prerun(state)
            raise

    def close(self, key: DraftReqKey) -> None:
        self._evict_seat(key)
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

        Three forms of the same tree, mixed PER SEAT within one call:

        - **glue fast path** (this seat's commit matched a unit of its last
          block): the winning unit's chain IS the new backbone (greedy
          re-draft is deterministic), so one K-row extend re-materializes
          its KV and yields all node logits; the branch phase runs as plain
          decode replays assembled from the seat's retained carrier rows.
        - **case-0 miss round** (commit fell outside the last block, carrier
          exists): a drafter miss mirrors a verifier select miss, so the
          next commit is necessarily a single fallback bonus -- only the
          case-0 rows of this block can ever be read. Enumerate just those
          F chains on the carrier rows and poison the dead cells.
        - **bootstrap** (no carrier yet): the original build-everything
          round for the fresh seats only; it also builds their seat
          carriers and host mirrors.
        """
        keys = [key for key in keys if key in self._states]
        if not keys:
            return None
        scratch_batches: list[ScheduleBatch] = []
        scratch_slots: list[torch.Tensor] = []
        self.profiler.start_round()
        try:
            hit_keys: list[DraftReqKey] = []
            hit_states: list[_DraftReqState] = []
            selections: list[tuple[int, int]] = []
            case0_keys: list[DraftReqKey] = []
            case0_states: list[_DraftReqState] = []
            slow_keys: list[DraftReqKey] = []
            slow_states: list[_DraftReqState] = []
            for key in keys:
                state = self._states[key]
                selection = self._match_seat(key, state)
                if selection is not None:
                    hit_keys.append(key)
                    hit_states.append(state)
                    selections.append(selection)
                elif key in self._seat_carriers:
                    case0_keys.append(key)
                    case0_states.append(state)
                else:
                    slow_keys.append(key)
                    slow_states.append(state)
            parts: list[dict] = []
            if hit_states:
                self.hit_ct += 1
                parts.append(
                    self._fast_round(
                        hit_keys,
                        hit_states,
                        selections,
                        scratch_batches,
                        scratch_slots,
                    )
                )
            if case0_states or slow_states:
                self.miss_ct += 1
            if case0_states:
                parts.append(
                    self._case0_round(
                        case0_keys, case0_states, scratch_batches, scratch_slots
                    )
                )
            if slow_states:
                parts.append(
                    self._slow_round(
                        slow_keys, slow_states, scratch_batches, scratch_slots
                    )
                )
            if len(parts) == 1:
                return parts[0]
            return {
                "pool_indices": [
                    pool_idx for part in parts for pool_idx in part["pool_indices"]
                ],
                "base_committed_lens": [
                    base_len
                    for part in parts
                    for base_len in part["base_committed_lens"]
                ],
                "units_device": torch.cat([part["units_device"] for part in parts]),
            }
        finally:
            self._free_scratch(scratch_batches, scratch_slots)
            self.profiler.mark("free")

    def _match_seat(
        self, key: DraftReqKey, state: _DraftReqState
    ) -> Optional[tuple[int, int]]:
        """Match one seat's pending delta against its last block; returns the
        winning (accept_case, fanout_index) or None (seat misses)."""
        if not self._enable_glue_fast_path:
            return None
        if key not in self._seat_carriers:
            return None
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
        return (case, guesses_row.index(bonus))

    def _fast_round(
        self,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        selections: list[tuple[int, int]],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
    ) -> dict:
        num_steps, fanout = self.num_steps, self.fanout
        bs = len(states)
        carriers = [self._seat_carriers[key] for key in keys]
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
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            new_len = state.committed_slots.numel()
            synced = min(carrier.synced_len, base_lens[i])
            pool.req_to_token[carrier.all_rows, synced:new_len] = state.committed_slots[
                synced:new_len
            ].to(torch.int32)
            carrier.synced_len = new_len
            slots_i32 = backbone_slots[i].to(torch.int32)
            pool.req_to_token[carrier.comb_rows, self._comb_j + new_len] = slots_i32[
                self._comb_j
            ]
            case, f = selections[i]
            chains.append(state.last_units_dev[case, f, 1:])
            # The old host mirror was synced during matching; snapshot the new
            # backbone before _pack_and_mirror overwrites it.
            new_backbones.append(state.last_units_host[case, f, 1:].tolist())
        self.profiler.mark("carrier_sync")

        # -- Glue extend: all K backbone tokens in one forward = node 1..K
        # logits; their KV lands in this round's backbone slots.
        glue_logits = self._glue_forward(
            carriers=carriers,
            states=states,
            chains=chains,
            backbone_slots=backbone_slots,
        )
        glue_guesses = torch.topk(
            glue_logits.view(bs, num_steps, -1), fanout, dim=-1
        ).indices  # [bs, K, F]
        guesses_stack = torch.cat([node0_guesses.unsqueeze(1), glue_guesses], dim=1)

        # -- Branch chains: K decode replays on the assembled carrier rows.
        chain_steps = self._branch_decode_chain(
            carriers=carriers,
            states=states,
            guesses_stack=guesses_stack,
            backbone_slots=backbone_slots,
            scratch_slots=scratch_slots,
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
        carriers: list[_SeatCarrier],
        states: list[_DraftReqState],
        chains: list[torch.Tensor],
        backbone_slots: torch.Tensor,
    ) -> torch.Tensor:
        num_steps = self.num_steps
        bs = len(states)
        glue = self._glue_template
        # Assemble the subset's rows onto the shared shell (rebind-only).
        glue.reqs = [req for carrier in carriers for req in carrier.glue_reqs]
        glue.req_pool_indices = (
            torch.cat([carrier.glue_rows for carrier in carriers])
            if bs > 1
            else carriers[0].glue_rows
        )
        glue.extend_lens = [1] * (bs * num_steps)
        glue.extend_logprob_start_lens = [0] * (bs * num_steps)
        glue.extend_num_tokens = bs * num_steps
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
        carriers: list[_SeatCarrier],
        states: list[_DraftReqState],
        guesses_stack: torch.Tensor,
        backbone_slots: torch.Tensor,
        scratch_slots: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        num_steps = self.num_steps
        branch = self._branch_template
        branch.reqs = [req for carrier in carriers for req in carrier.branch_reqs]
        branch.req_pool_indices = (
            torch.cat([carrier.branch_rows for carrier in carriers])
            if len(carriers) > 1
            else carriers[0].branch_rows
        )
        seq_host = [
            state.committed_slots.numel() + case
            for state in states
            for case in self._case_of_row
        ]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        branch.seq_lens = seq_cpu.to(self.device, non_blocking=True)
        branch.seq_lens_cpu = seq_cpu
        branch.seq_lens_sum = None
        branch.orig_seq_lens = branch.seq_lens.to(torch.int32)
        cascade = self._build_branch_cascade(
            states=states, backbone_slots=backbone_slots
        )
        self.profiler.mark("branch_mut")
        logits, step_slots = self._decode_step(
            branch, guesses_stack.reshape(-1), tag="branch", cascade=cascade
        )
        scratch_slots.append(step_slots)
        chain_steps: list[torch.Tensor] = [logits.argmax(dim=-1)]
        for _ in range(num_steps - 1):
            logits, step_slots = self._decode_step(
                branch, chain_steps[-1], tag="branch", cascade=cascade
            )
            scratch_slots.append(step_slots)
            chain_steps.append(logits.argmax(dim=-1))
        return chain_steps

    def _build_branch_cascade(
        self,
        *,
        states: list[_DraftReqState],
        backbone_slots: torch.Tensor,
    ) -> Optional[_CascadeMetadata]:
        """Shared-prefix cascade inputs for this round's branch chain, or None
        below the L2 threshold (where per-row re-reads are effectively free
        and the two-call split only adds overhead)."""
        min_prefix = self._cascade_min_prefix_len
        lens = [state.committed_slots.numel() for state in states]
        if min_prefix <= 0 or min(lens) < min_prefix:
            return None
        seats = len(states)
        rows_per_seat = (self.num_steps + 1) * self.fanout
        prefix_page_table = torch.zeros(
            (seats, max(lens)), dtype=torch.int32, device=self.device
        )
        for i, state in enumerate(states):
            prefix_page_table[i, : lens[i]] = state.committed_slots.to(torch.int32)
        tail_page_table = torch.zeros(
            (seats * rows_per_seat, 2 * self.num_steps + 2),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(seats):
            block = tail_page_table[i * rows_per_seat : (i + 1) * rows_per_seat]
            block[self._br_r, self._br_j] = backbone_slots[i].to(torch.int32)[
                self._br_j
            ]
        tail_lens = torch.tensor(
            [case for _ in states for case in self._case_of_row],
            dtype=torch.int32,
            device=self.device,
        )
        return _CascadeMetadata(
            prefix_page_table=prefix_page_table,
            prefix_lens=torch.tensor(lens, dtype=torch.int32, device=self.device),
            tail_page_table=tail_page_table,
            tail_lens=tail_lens,
        )

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

    def _case0_round(
        self,
        keys: list[DraftReqKey],
        states: list[_DraftReqState],
        scratch_batches: list[ScheduleBatch],
        scratch_slots: list[torch.Tensor],
    ) -> dict:
        """Miss round collapsed to case 0 (the dead-cell theorem).

        The drafter and the verifier judge the same (block, delta) pair, so a
        drafter miss means the verifier's select missed too and fell back to a
        plain decode: the NEXT commit is a single case-0 bonus, and every
        case >= 1 cell of this block is dead. (If the fallback's junk drafts
        happen to be target-agreeing the next delta can still be longer; it
        then simply misses again -- one wasted round, never a wrong token.)

        So: advance the delta, take the top-F node-0 guesses, and run ONE
        F-row decode chain on the carrier's case-0 rows (which carry no
        backbone prefix -- their sequences start at the committed length).
        No backbone, no glue, no carrier rebuild; dead guess cells are
        poisoned with -1 (matches nothing on either side) and dead chains
        with 0 (never read behind a poisoned guess).
        """
        num_steps, fanout = self.num_steps, self.fanout
        bs = len(states)
        carriers = [self._seat_carriers[key] for key in keys]
        pool = self.model_runner.req_to_token_pool

        # -- Advance the committed prefix; last logits = node 0 --------------
        base_lens = [state.committed_slots.numel() for state in states]
        advance_batch, advance_slots = self._extend_batch(
            token_lists=[state.committed_tokens for state in states],
            prefix_slots=[state.committed_slots for state in states],
            tag="advance",
        )
        scratch_batches.append(advance_batch)
        node0_logits = self._forward(advance_batch, tag="advance")
        # Consume the graph runner's static logits buffer before the next
        # forward overwrites it.
        node0_guesses = torch.topk(node0_logits, fanout, dim=-1).indices  # [bs, F]
        self._absorb_advance_slots(states, advance_slots)

        # -- Carrier rows only need the committed delta (no backbone) --------
        for i, (state, carrier) in enumerate(zip(states, carriers)):
            new_len = state.committed_slots.numel()
            synced = min(carrier.synced_len, base_lens[i])
            pool.req_to_token[carrier.all_rows, synced:new_len] = state.committed_slots[
                synced:new_len
            ].to(torch.int32)
            carrier.synced_len = new_len
        self.profiler.mark("case0_sync")

        # -- Case-0 chains: per seat, the first F carrier rows ---------------
        branch = self._branch_template
        branch.reqs = [
            req for carrier in carriers for req in carrier.branch_reqs[:fanout]
        ]
        branch.req_pool_indices = (
            torch.cat([carrier.branch_rows[:fanout] for carrier in carriers])
            if bs > 1
            else carriers[0].branch_rows[:fanout]
        )
        seq_host = [
            state.committed_slots.numel() for state in states for _ in range(fanout)
        ]
        seq_cpu = torch.tensor(seq_host, dtype=torch.int64)
        branch.seq_lens = seq_cpu.to(self.device, non_blocking=True)
        branch.seq_lens_cpu = seq_cpu
        branch.seq_lens_sum = None
        branch.orig_seq_lens = branch.seq_lens.to(torch.int32)
        self.profiler.mark("case0_mut")
        logits, step_slots = self._decode_step(
            branch, node0_guesses.reshape(-1), tag="case0"
        )
        scratch_slots.append(step_slots)
        chain_steps: list[torch.Tensor] = [logits.argmax(dim=-1)]
        for _ in range(num_steps - 1):
            logits, step_slots = self._decode_step(branch, chain_steps[-1], tag="case0")
            scratch_slots.append(step_slots)
            chain_steps.append(logits.argmax(dim=-1))

        # -- Expand to full-size units; poison the dead cells ----------------
        guesses_stack = torch.full(
            (bs, num_steps + 1, fanout),
            -1,
            dtype=node0_guesses.dtype,
            device=self.device,
        )
        guesses_stack[:, 0] = node0_guesses
        full_steps: list[torch.Tensor] = []
        for step in chain_steps:
            full_step = torch.zeros(
                (bs, num_steps + 1, fanout), dtype=step.dtype, device=self.device
            )
            full_step[:, 0] = step.view(bs, fanout)
            full_steps.append(full_step.view(-1))
        # No backbone this round: only a case-0 match can hit next round, and
        # a case-0 hit reads its chain from the units mirror, not the backbone.
        return self._pack_and_mirror(
            states=states,
            guesses_stack=guesses_stack,
            chain_steps=full_steps,
            new_backbones=[[] for _ in states],
        )

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
        # A slow subround rebuilds its seats' carriers at the end; evicting
        # them up front bounds the subround's peak pool-row usage.
        for key in keys:
            self._evict_seat(key)

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
        self._build_seat_carriers(
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

    def _build_seat_carriers(
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
        """Donate this slow subround's branch rows + a freshly built glue
        batch to per-seat carriers (pool rows persist until the seat closes;
        KV slots stay per-round scratch). The built batches double as the
        assembly shells for later fast subrounds."""
        if not self._enable_glue_fast_path:
            return
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
        bs = len(keys)
        num_steps = self.num_steps
        rows_per_seat = (num_steps + 1) * self.fanout
        glue_rows_all = glue_batch.req_pool_indices.view(bs, num_steps)
        branch_rows_all = branch_batch.req_pool_indices.view(bs, rows_per_seat)
        for i, (key, state) in enumerate(zip(keys, states)):
            self._seat_carriers[key] = _SeatCarrier(
                glue_rows=glue_rows_all[i],
                branch_rows=branch_rows_all[i],
                glue_reqs=glue_batch.reqs[i * num_steps : (i + 1) * num_steps],
                branch_reqs=branch_batch.reqs[
                    i * rows_per_seat : (i + 1) * rows_per_seat
                ],
                synced_len=state.committed_slots.numel(),
                tri_g=self._tri_g,
                br_r=self._br_r,
            )
        self._glue_template = glue_batch
        self._branch_template = branch_batch
        self.profiler.mark("carrier_build")

    def _evict_seat(self, key: DraftReqKey) -> None:
        carrier = self._seat_carriers.pop(key, None)
        if carrier is None:
            return
        for req in carrier.glue_reqs + carrier.branch_reqs:
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
        self,
        batch: ScheduleBatch,
        input_tokens: torch.Tensor,
        *,
        tag: str,
        cascade: Optional[_CascadeMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch.input_ids = input_tokens.to(torch.int64)
        batch.prepare_for_decode()
        if cascade is not None:
            # Append this step's KV slot to each row's private tail, then
            # advance the tail lengths so the kernel covers the new token.
            rows = cascade.tail_lens.shape[0]
            cascade.tail_page_table[
                torch.arange(rows, device=self.device), cascade.tail_lens.long()
            ] = batch.out_cache_loc.to(torch.int32)
            cascade.tail_lens.add_(1)
        self.profiler.mark(f"{tag}_step_prep")
        forward_batch = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        forward_batch.decoupled_cascade = cascade
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
