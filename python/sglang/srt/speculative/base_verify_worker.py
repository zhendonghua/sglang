"""Verify-only spec worker contract for decoupled speculative decoding.

A decoupled verifier runs only the verify half of speculative decoding: it
holds the target worker and verifies externally produced draft chains; there
is no draft model in the process. Like the ngram worker (the in-tree
verify-only precedent), everything model-related -- memory pools, attention
backends, cuda graphs -- belongs to the target ``TpModelWorker``;
``BaseSpecWorker``'s draft-aware hooks degrade to no-ops through the ``None``
``draft_worker``, and ``war_fastpath_runner`` / ``spec_v2_attn_backends``
keep their target-only defaults (the correct routing for a worker whose last
shared-pool read is the target verify).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult


class BaseVerifyWorker(BaseSpecWorker):
    """Verify-only contract: target worker + externally provided chains.

    Subclasses implement ``forward_batch_generation`` (prefill = plain target
    extend; decode = build a verify input from external chain tokens and run
    the shared verify path). They must not load or drive a draft model.
    """

    @property
    def draft_worker(self) -> Optional[object]:
        # No draft model in a verify-only worker; BaseSpecWorker's draft-aware
        # init hooks (alloc_memory_pool / init_attention_backends /
        # init_cuda_graphs) all no-op on None.
        return None

    @abstractmethod
    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult: ...
