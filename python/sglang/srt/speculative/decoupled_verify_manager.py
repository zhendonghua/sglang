"""Verifier-side scheduler collaborator for decoupled enumeration spec.

The scheduler delegates here once per batch result; everything else (wire,
landing, GPU select) lives in the IPC thread and the verify worker. The
manager owns:

- **Control plane bookkeeping** (host): an unseen rid gets a ``DraftSync``
  announcing the prompt, committed outputs, and its seat (req_pool_idx); each
  round's newly committed slice becomes a ``VerifyCommit``; a finished request
  sends ``DraftClose``. A seat change (retraction re-admit) re-syncs the full
  committed prefix -- the drafter-carried pool_idx protocol's only re-sync
  obligation.
- **Sync-mode pacing** (the C6 host latch, phase 5b form): after forwarding a
  round's commits, wait -- bounded -- until the next enumeration block of
  every still-running request has landed, so the next verify round selects
  instead of falling back. A timeout is never an error: the round degrades to
  the unified fallback. Phase 6.3 replaces this wait with launch-gating.
- **Hit / fallback accounting** from the worker's ``last_select_hits``.

Expected arrival stamps are pure host math: the drafter stamps a block with
its total committed length (prompt + committed outputs) at enumeration time,
which the manager mirrors from its own commit bookkeeping.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_scripted_drafter import ScriptedFakeDrafter
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftEnumerationBufferBatch,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    FakeTransportMesh,
    build_transport,
)
from sglang.srt.speculative.verifier_ipc_thread import VerifierIpcThread

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.speculative.verify_worker import VerifyWorker

logger = logging.getLogger(__name__)

# Sync-mode bound on waiting for the next enumeration block. Generous vs a
# real drafter round (a few ms); on expiry the round simply falls back.
_SYNC_ARRIVAL_TIMEOUT_S = 0.2

_LOOPBACK_VERIFIER_ENDPOINT = "loopback://decoupled-spec-verifier"
_LOOPBACK_DRAFTER_ENDPOINT = "loopback://decoupled-spec-drafter"


class EnumArrivalBoard:
    """Host mirror of landed stamps per seat (daemon writes, scheduler waits).

    The GPU buffer holds the authoritative stamps; this mirror exists only so
    the sync-mode gate can wait without a device sync.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._stamps: dict[int, int] = {}

    def record(self, block: DraftEnumerationBufferBatch) -> None:
        with self._cond:
            for pool_idx, stamp in zip(block.pool_indices, block.base_committed_lens):
                self._stamps[int(pool_idx)] = int(stamp)
            self._cond.notify_all()

    def wait_for(self, expected: dict[int, int], timeout_s: float) -> bool:
        """Wait until every seat's landed stamp equals its expected base.

        Returns False on timeout (the verify round then falls back for the
        seats that never arrived).
        """

        def _arrived() -> bool:
            return all(
                self._stamps.get(pool_idx) == stamp
                for pool_idx, stamp in expected.items()
            )

        deadline = time.monotonic() + timeout_s
        with self._cond:
            while not _arrived():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True


class _ReqState:
    def __init__(self, *, pool_idx: int, prompt_len: int, committed_len: int) -> None:
        self.pool_idx = pool_idx
        self.prompt_len = prompt_len
        self.committed_len = committed_len  # committed OUTPUT tokens

    @property
    def total_committed_len(self) -> int:
        return self.prompt_len + self.committed_len


class DecoupledVerifyManager:
    """Scheduler collaborator: control plane + sync pacing + accounting."""

    def __init__(
        self,
        *,
        ipc_config: DecoupledSpecIpcConfig,
        verify_worker: VerifyWorker,
    ) -> None:
        self.ipc_config = ipc_config
        self.verify_worker = verify_worker
        self.arrival_board = EnumArrivalBoard()
        # M:N drafter-assignment policy is out of scope; everything routes to
        # drafter rank 0 (the 1:1 topology).
        self.dst_drafter_rank = 0

        self._rid_states: dict[str, _ReqState] = {}
        self.enum_round_ct = 0
        self.enum_hit_ct = 0
        self.sync_wait_timeout_ct = 0

        self.scripted_drafter: Optional[ScriptedFakeDrafter] = None
        loopback_mode = envs.SGLANG_TEST_DECOUPLED_LOOPBACK.get()
        if loopback_mode is not None:
            transport = self._build_loopback(loopback_mode)
        else:
            import zmq

            transport = build_transport(
                kind=DecoupledSpecTransportKind.ZMQ,
                bind_endpoint=ipc_config.bind_endpoint,
                connect_endpoints=ipc_config.connect_endpoints,
                context=zmq.Context(2),
            )
        self.ipc_thread = VerifierIpcThread(
            transport=transport,
            enum_buffer=verify_worker.enum_buffer,
            on_land=self.arrival_board.record,
        )
        self.ipc_thread.start()

    def _build_loopback(self, mode: str):
        """Single-process loopback: fake mesh + an in-process scripted drafter."""
        mesh = FakeTransportMesh()
        verifier_transport = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=_LOOPBACK_VERIFIER_ENDPOINT,
            connect_endpoints=[_LOOPBACK_DRAFTER_ENDPOINT],
            mesh=mesh,
        )
        drafter_transport = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=_LOOPBACK_DRAFTER_ENDPOINT,
            connect_endpoints=[_LOOPBACK_VERIFIER_ENDPOINT],
            mesh=mesh,
        )
        self.scripted_drafter = ScriptedFakeDrafter(
            transport=drafter_transport,
            verifier_rank=self.ipc_config.rank,
            drafter_rank=self.dst_drafter_rank,
            num_steps=self.verify_worker.speculative_num_steps,
            fanout=self.verify_worker.speculative_fanout,
            mode=mode,
        )
        self.scripted_drafter.start()
        logger.info(
            "Decoupled-spec loopback: scripted fake drafter (mode=%s) started", mode
        )
        return verifier_transport

    def close(self) -> None:
        if self.scripted_drafter is not None:
            self.scripted_drafter.close()
        self.ipc_thread.close()

    def on_batch_result(self, batch: ScheduleBatch) -> None:
        """Forward this round's lifecycle controls, then pace the next round.

        Runs on the scheduler thread after the batch-result processor appended
        the round's committed tokens to req.output_ids.
        """
        if not batch.reqs:
            return
        control_batch = DraftControlBatch(dst_drafter_rank=self.dst_drafter_rank)
        expected: dict[int, int] = {}
        for req in batch.reqs:
            self._collect_req_controls(req, control_batch, expected)
        if (
            control_batch.sync_messages
            or control_batch.verify_commit_messages
            or control_batch.close_messages
        ):
            self.ipc_thread.submit_control_batch(control_batch)
        self._account_select_hits(batch)
        if expected:
            arrived = self.arrival_board.wait_for(expected, _SYNC_ARRIVAL_TIMEOUT_S)
            if not arrived:
                self.sync_wait_timeout_ct += 1

    def _collect_req_controls(
        self,
        req: Req,
        control_batch: DraftControlBatch,
        expected: dict[int, int],
    ) -> None:
        state = self._rid_states.get(req.rid)
        if req.finished():
            if state is not None:
                control_batch.close_messages.append(
                    DraftClose(
                        request_id=req.rid,
                        src_verifier_rank=self.ipc_config.rank,
                        dst_drafter_rank=self.dst_drafter_rank,
                        reason="finished",
                    )
                )
                self._rid_states.pop(req.rid, None)
            return

        if state is None or state.pool_idx != req.req_pool_idx:
            # New request, or a retraction re-admit moved its seat: (re-)open
            # with the full committed prefix and poison the seat's stamp so the
            # previous occupant's landed block cannot look fresh.
            self.verify_worker.enum_buffer.reset_slot(req.req_pool_idx)
            state = _ReqState(
                pool_idx=req.req_pool_idx,
                prompt_len=len(req.origin_input_ids),
                committed_len=len(req.output_ids),
            )
            self._rid_states[req.rid] = state
            control_batch.sync_messages.append(
                DraftSync(
                    request_id=req.rid,
                    src_verifier_rank=self.ipc_config.rank,
                    dst_drafter_rank=self.dst_drafter_rank,
                    req_pool_idx=req.req_pool_idx,
                    prompt_token_ids=list(req.origin_input_ids),
                    committed_outputs=list(req.output_ids),
                )
            )
        else:
            committed_len = len(req.output_ids)
            if committed_len > state.committed_len:
                control_batch.verify_commit_messages.append(
                    VerifyCommit(
                        request_id=req.rid,
                        src_verifier_rank=self.ipc_config.rank,
                        dst_drafter_rank=self.dst_drafter_rank,
                        pre_verify_committed_len=state.committed_len,
                        committed_tokens=list(req.output_ids[state.committed_len :]),
                    )
                )
                state.committed_len = committed_len
        expected[state.pool_idx] = state.total_committed_len

    def _account_select_hits(self, batch: ScheduleBatch) -> None:
        hits = self.verify_worker.last_select_hits
        if hits is None or not batch.forward_mode.is_decode_or_idle():
            return
        self.verify_worker.last_select_hits = None
        # Sync mode: the result was already D2H-synced by copy_to_cpu, so this
        # read does not add a stall. 6.3 moves accounting off the host path.
        hit_list = hits.tolist()
        self.enum_round_ct += len(hit_list)
        self.enum_hit_ct += sum(hit_list)
        if self.enum_round_ct and self.enum_round_ct % 200 < len(hit_list):
            logger.info(
                "decoupled enum select: hit_ct=%d round_ct=%d hit_rate=%.3f "
                "sync_wait_timeout_ct=%d",
                self.enum_hit_ct,
                self.enum_round_ct,
                self.enum_hit_ct / self.enum_round_ct,
                self.sync_wait_timeout_ct,
            )
