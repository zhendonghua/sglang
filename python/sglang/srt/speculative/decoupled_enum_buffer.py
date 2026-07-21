"""Verifier-side GPU landing buffer for decoupled enumeration spec.

The drafter ships one DraftEnumerationBufferBatch per round (one round ahead),
pre-enumerating every (accept_case, bonus_guess) chain the verifier could select
next; this is where those blocks land on the verifier's GPU for the verify
forward to consume.

Pool-indexed by req_pool_idx like FutureMap's relays (managers/overlap_utils.py),
so the forward gathers it with the same batch.req_pool_indices as every other
per-request tensor; leading dim = req_to_token.shape[0] (max_running + 1; seat 0
is the harmless cuda-graph padding row). Routing is carried by the block itself:
each row echoes the req_pool_idx the verifier announced in DraftSync, so landing
is one GPU scatter with no host rid lookup. Each seat carries a
base_committed_len stamp; the verify forward (phase 4a) compares it against the
request's live committed length for fresh-vs-fallback, and a never-written /
reset seat holds a sentinel that always falls back. A late block whose request
already left lands harmlessly: its seat is either free (nobody gathers it) or
reused, where reset_slot() at re-assignment plus the stamp mismatch keep it on
the fallback path.

Phase 1b is SYNC (buf_count == 1: land scatters on the current stream, the caller
synchronizes before gather). The async overlap form (pinned staging + a private
copy stream + a double-buffer swap fence) lands in phase 6.3; the API is shaped so
6.3 only flips buf_count and moves the H2D onto the copy stream.
"""

from __future__ import annotations

import numpy as np
import torch

from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch

# Stamp for a seat with no valid block (never written, or reset on realloc); a
# negative committed length never matches a real one, so the seat falls back.
_STAMP_EMPTY = -1


class DecoupledEnumBuffer:
    """GPU landing buffer for enumeration blocks, indexed by req_pool_idx."""

    def __init__(
        self,
        *,
        device: str,
        req_to_token_pool,
        num_steps: int,  # K = draft chain length per case
        fanout: int,  # F = bonus-token guesses per accept case
        verifier_rank: int,
        enable_overlap: bool,
    ) -> None:
        if enable_overlap:
            raise NotImplementedError(
                "overlap landing needs the async copy-stream + swap fence "
                "(phase 6.3); run with enable_overlap=False for now"
            )
        self.device = device
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        # From DecoupledSpecIpcConfig.rank in phase 5b; land() rejects a block
        # routed to a different verifier.
        self.verifier_rank = int(verifier_rank)
        # (K+1) accept cases * F bonus guesses * (K+1)-wide [guess, chain] units
        # (unit element 0 = the guessed bonus itself = the GPU match key; a hit
        # unit is the verify row), flat per row.
        self.unit_width = self.num_steps + 1
        self.row_width = self.unit_width * self.fanout * self.unit_width
        # req_to_token.shape[0] == max_running + 1, so seat 0 stays the harmless
        # cuda-graph padding row; never size to bare max_running.
        self.seats = int(req_to_token_pool.req_to_token.shape[0])
        # Each seat holds the TWO newest stamped blocks (generations). The block
        # serving round r was enumerated two commits back (stamp |P_{r-2}|),
        # while the commit of round r-1 has already pushed the next block
        # (stamp |P_{r-1}|): with one generation the newer push would clobber
        # the block round r selects from, turning every round into a staleness
        # fallback under sync pacing. gather() returns both; the select matches
        # stamps against the expected base and picks the right generation.
        self.gen_count = 2
        # Double-buffer (mamba idiom, memory_pool.py:849) is the phase 6.3 form;
        # buf_count is always 1 here (enable_overlap is rejected above).
        self.buf_count = 2 if enable_overlap else 1
        self._write_slot = 0

        # int64 matches the forward's token-id convention (input_ids,
        # FutureMap.output_tokens_buf, EagleDraftInput.draft_token are all int64);
        # int32 is numerically enough but would force an up-cast. Not
        # req_to_token's int32, which is a KV-slot index pool, not vocab ids.
        self.enum_tokens = [
            torch.zeros(
                (self.seats, self.gen_count, self.row_width),
                dtype=torch.int64,
                device=device,
            )
            for _ in range(self.buf_count)
        ]
        # Per-(seat, generation) freshness stamp; starts at the sentinel so an
        # unwritten generation falls back. int64 to match the committed length
        # it is compared against.
        self.enum_base_committed_lens = [
            torch.full(
                (self.seats, self.gen_count),
                _STAMP_EMPTY,
                dtype=torch.int64,
                device=device,
            )
            for _ in range(self.buf_count)
        ]
        # Which generation was written last per seat; the next land overwrites
        # the OTHER one, so the previous block survives exactly one more push.
        self.enum_last_gen = [
            torch.zeros((self.seats,), dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]

    @property
    def _read_slot(self) -> int:
        # Single buffer: daemon and forward share one slot (SYNC). Double buffer:
        # the forward reads the slot the daemon is not writing.
        return self._write_slot if self.buf_count == 1 else 1 - self._write_slot

    def land(self, block: DraftEnumerationBufferBatch) -> None:
        """Scatter a block's rows + stamps into the seats its pool_indices name.

        Called by the recv daemon. Routing rides in the block (the drafter
        echoes each request's DraftSync req_pool_idx), so this is validation +
        one GPU scatter. Phase 1b scatters on the current stream (SYNC); phase
        6.3 moves it onto a private copy stream with pinned staging.

        NOTE: the raises below run on the recv daemon thread, whose loop dies on
        an uncaught exception. TODO(phase 5c): quarantine.
        """
        if int(block.dst_verifier_rank) != self.verifier_rank:
            # A misrouted / M:N block: seats are only meaningful within the
            # owning verifier, so a foreign block's pool_indices would land in
            # unrelated local seats.
            raise RuntimeError(
                "enumeration block routed to the wrong verifier: "
                f"dst_verifier_rank={block.dst_verifier_rank} "
                f"verifier_rank={self.verifier_rank} "
                f"src_drafter_rank={block.src_drafter_rank} "
                f"batch_size={block.batch_size}"
            )
        if int(block.num_steps) != self.num_steps or int(block.fanout) != self.fanout:
            raise RuntimeError(
                "enumeration block dims differ from the buffer's config "
                "(mismatched K/F shape-errors the scatter, or silently mis-lays "
                "out the flat [accept_case][guess][step] layout if the products "
                f"coincide): block=({block.num_steps}, {block.fanout}) "
                f"buffer=({self.num_steps}, {self.fanout})"
            )
        block.validate()  # parallel-array shape, unique pool_indices >= 1
        if not block.pool_indices:
            return
        max_pool_idx = max(int(pool_idx) for pool_idx in block.pool_indices)
        if max_pool_idx >= self.seats:
            # validate() cannot know this buffer's seat count; a peer echoing a
            # seat we never announced must not corrupt an unrelated seat.
            raise RuntimeError(
                "enumeration block pool_idx exceeds the seat table: "
                f"pool_idx={max_pool_idx} seats={self.seats} "
                f"src_drafter_rank={block.src_drafter_rank}"
            )

        # Reshape the block's flat token tuple once (C-order, so rows_host[i] ==
        # block.row_tokens(i)); blocking H2D from pageable host (the copy stream
        # + pinned staging arrive in phase 6.3).
        pool_indices = torch.tensor(
            block.pool_indices, dtype=torch.int64, device=self.device
        )
        rows = torch.from_numpy(
            np.asarray(block.tokens, dtype=np.int64).reshape(
                block.batch_size, block.row_stride
            )
        ).to(device=self.device)
        base_committed_lens = torch.tensor(
            block.base_committed_lens, dtype=torch.int64, device=self.device
        )

        slot = self._write_slot
        # Write the generation NOT written last, then mark it newest: the
        # previous block survives exactly one more push (see gen_count above).
        write_gens = 1 - self.enum_last_gen[slot][pool_indices]
        self.enum_tokens[slot][pool_indices, write_gens] = rows
        self.enum_base_committed_lens[slot][
            pool_indices, write_gens
        ] = base_committed_lens
        self.enum_last_gen[slot][pool_indices] = write_gens

    def gather(
        self, req_pool_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this batch's rows + freshness stamps from the read-side slot:
        (rows [B, gen_count, (K+1)*F*(K+1)], base_committed_lens [B, gen_count]).
        The verify forward (phase 4a) matches base_committed_lens against the
        live expected base to pick the serving generation (fresh-vs-fallback),
        then selects the winning [guess, chain] unit by (accept_case,
        bonus_guess).
        """
        slot = self._read_slot
        rows = self.enum_tokens[slot][req_pool_indices]
        base_committed_lens = self.enum_base_committed_lens[slot][req_pool_indices]
        return rows, base_committed_lens

    def reset_slot(self, pool_idx: int) -> None:
        # Invalidate a seat's stamps when it is (re)assigned, so the reused seat
        # falls back until its new occupant's own block lands. Called by the
        # scheduler at prefill alloc / retraction re-admit.
        for slot in range(self.buf_count):
            self.enum_base_committed_lens[slot][pool_idx, :] = _STAMP_EMPTY

    def swap(self) -> None:
        # Advance the write/read double-buffer at a round boundary; no-op under
        # buf_count == 1 (phase 6.3 pairs this with the copy-stream event fence).
        if self.buf_count > 1:
            self._write_slot = 1 - self._write_slot
