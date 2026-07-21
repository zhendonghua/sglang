"""Drafter-side manager + dedicated event loop for decoupled enumeration spec.

The decoupled drafter serves no user requests: its whole job is answering the
verifier's control plane with enumeration blocks, exactly one round ahead.
Instead of threading mirror requests through the normal scheduler machinery,
the drafter runs this manager's ``run_loop`` as its event loop:

    drain ready controls -> close / open / apply commits -> one enumeration
    round for every touched request -> push blocks -> idle-wait.

Pacing is inherent: one block per DraftSync / VerifyCommit, no backpressure
machinery. The draft model is driven directly by ``EnumDraftEngine``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sglang.srt.speculative.decoupled_draft_engine import EnumDraftEngine
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftEnumerationBufferBatch,
    DraftReqKey,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    build_transport,
)
from sglang.srt.speculative.drafter_ipc_thread import DrafterIpcThread

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.0005

# CUDA IPC slot capacity in block rows; bounds the verifier batch size a
# single push can carry (the verifier's default running cap is far below it).
IPC_POOL_MAX_ROWS = 256


class DecoupledDraftManager:
    """Drafter engine driver: controls in, enumeration blocks out."""

    def __init__(
        self,
        *,
        ipc_config: DecoupledSpecIpcConfig,
        model_runner: ModelRunner,
        num_steps: int,
        fanout: int,
        data_transport: str = "zmq",
    ) -> None:
        import zmq

        self.ipc_config = ipc_config
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        self.engine = EnumDraftEngine(
            model_runner=model_runner,
            num_steps=self.num_steps,
            fanout=self.fanout,
        )
        transport = build_transport(
            kind=DecoupledSpecTransportKind.ZMQ,
            bind_endpoint=ipc_config.bind_endpoint,
            connect_endpoints=ipc_config.connect_endpoints,
            context=zmq.Context(2),
        )
        self.ipc_thread = DrafterIpcThread(
            transport=transport, drafter_rank=ipc_config.rank
        )
        self.ipc_thread.start()
        self._round_ct = 0
        self._round_time_s = 0.0

        self.ipc_block_pool = None
        if data_transport == "cuda_ipc":
            from sglang.srt.speculative.cuda_ipc_enum_transport import (
                CudaIpcEnumBlockPool,
            )

            unit_width = self.num_steps + 1
            self.ipc_block_pool = CudaIpcEnumBlockPool(
                device=model_runner.device,
                # Both sides derive the shm rendezvous name from the drafter's
                # bind endpoint.
                endpoint=ipc_config.bind_endpoint,
                max_rows=IPC_POOL_MAX_ROWS,
                row_width=2 + unit_width * self.fanout * unit_width,
            )

    def run_loop(self) -> None:
        """The drafter scheduler's event loop (never returns)."""
        logger.info(
            "Decoupled drafter loop started (rank=%d, K=%d, F=%d)",
            self.ipc_config.rank,
            self.num_steps,
            self.fanout,
        )
        while True:
            ready = self.ipc_thread.collect_ready_draft_controls(
                lambda inbox: inbox.extract_ready_controls_locked(
                    # The enumeration drafter re-extends from the committed
                    # prefix, so every commit is consumable in full.
                    lambda segment: len(segment.committed_tokens)
                )
            )
            if ready.is_empty():
                time.sleep(_IDLE_WAIT_S)
                continue
            try:
                self._apply_controls_and_draft(ready)
            except Exception:
                # A bad round must not kill the drafter for every request; the
                # affected verifier rounds simply fall back. TODO(5c-class):
                # quarantine the offending request instead of best-effort.
                logger.exception("decoupled drafter round failed; controls dropped")

    def _apply_controls_and_draft(self, ready) -> None:
        for draft_key in ready.close_keys:
            self.engine.close(draft_key)
        touched: dict[DraftReqKey, None] = {}
        for sync in ready.sync_messages:
            self.engine.open(
                sync.draft_key,
                req_pool_idx=int(sync.req_pool_idx),
                prompt_tokens=list(sync.prompt_token_ids),
                committed_outputs=list(sync.committed_outputs),
            )
            touched[sync.draft_key] = None
        for segment in ready.ready_commit_segments:
            if not self.engine.has(segment.draft_key):
                continue
            self.engine.apply_commit(segment.draft_key, list(segment.committed_tokens))
            touched[segment.draft_key] = None
        if not touched:
            return
        # One block per owning verifier (1:1 today: a single peer).
        by_verifier: dict[int, list[DraftReqKey]] = {}
        for draft_key in touched:
            by_verifier.setdefault(draft_key.src_verifier_rank, []).append(draft_key)
        for verifier_rank, draft_keys in by_verifier.items():
            round_start = time.monotonic()
            packed = self.engine.draft_round(draft_keys)
            self._round_ct += 1
            self._round_time_s += time.monotonic() - round_start
            if self._round_ct % 200 == 0:
                logger.info(
                    "decoupled drafter rounds: ct=%d avg_ms=%.1f last_bs=%d",
                    self._round_ct,
                    1000.0 * self._round_time_s / self._round_ct,
                    len(draft_keys),
                )
            if packed is None:
                continue
            if self.ipc_block_pool is not None:
                # CUDA IPC data plane: D2D into the shared pool; the shm flag
                # bump after the device sync is the arrival signal.
                self.ipc_block_pool.push(
                    pool_indices=packed["pool_indices"],
                    base_committed_lens=packed["base_committed_lens"],
                    units=packed["units_device"],
                )
                continue
            self.ipc_thread.submit_draft_results(
                DraftEnumerationBufferBatch(
                    src_drafter_rank=self.ipc_config.rank,
                    dst_verifier_rank=verifier_rank,
                    num_steps=self.num_steps,
                    fanout=self.fanout,
                    pool_indices=packed["pool_indices"],
                    base_committed_lens=packed["base_committed_lens"],
                    tokens=tuple(packed["units_device"].to("cpu").reshape(-1).tolist()),
                )
            )

    def close(self) -> None:
        self.ipc_thread.close()
