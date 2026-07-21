"""Two-GPU, two-process tests for the decoupled-spec CUDA IPC enum data plane.

CUDA IPC handles cannot be reopened inside the exporting process, so the
producer pool runs in a spawned child on cuda:0 and the consumer maps it in
THIS process onto cuda:1 through the handle + peer access (the NVLink path) --
the same topology as the real drafter/verifier split. What is guarded:

- rendezvous + handle redirect: mapped rows read back exactly what the
  producer wrote, across devices and processes;
- the ring protocol: unacked slots refuse reuse (cross-process WAR guard),
  acked slots recycle.
"""

import multiprocessing as mp
import unittest

import torch

from sglang.srt.speculative.cuda_ipc_enum_transport import CudaIpcEnumBlockReader
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

ENDPOINT = "ipc:///tmp/decoupled-cuda-ipc-test"
ROW_WIDTH = 2 + 12  # [pool_idx, stamp] + an arbitrary unit payload
NUM_SLOTS = 2


def _producer_main(cmd_queue: mp.Queue, ack_queue: mp.Queue) -> None:
    import torch

    from sglang.srt.speculative.cuda_ipc_enum_transport import CudaIpcEnumBlockPool

    pool = CudaIpcEnumBlockPool(
        device="cuda:0",
        endpoint=ENDPOINT,
        max_rows=8,
        row_width=ROW_WIDTH,
        num_slots=NUM_SLOTS,
    )
    ack_queue.put("ready")
    while True:
        command = cmd_queue.get()
        if command is None:
            break
        pool_indices, stamps, base = command
        units = (
            torch.arange(
                len(pool_indices) * (ROW_WIDTH - 2),
                dtype=torch.int64,
                device="cuda:0",
            )
            + base
        ).view(len(pool_indices), -1)
        ack_queue.put(
            pool.push(
                pool_indices=pool_indices, base_committed_lens=stamps, units=units
            )
        )
    pool.close()
    ack_queue.put("closed")


@unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")
class TestCudaIpcEnumTransport(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ctx = mp.get_context("spawn")
        cls.cmd_queue = ctx.Queue()
        cls.ack_queue = ctx.Queue()
        cls.producer = ctx.Process(
            target=_producer_main, args=(cls.cmd_queue, cls.ack_queue), daemon=True
        )
        cls.producer.start()
        assert cls.ack_queue.get(timeout=120) == "ready"
        cls.reader = CudaIpcEnumBlockReader(
            device="cuda:1", endpoint=ENDPOINT, attach_timeout_s=30.0
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "reader"):
            cls.reader.close()
        if hasattr(cls, "cmd_queue"):
            cls.cmd_queue.put(None)
        if hasattr(cls, "producer") and cls.producer.is_alive():
            cls.producer.join(timeout=30)
            if cls.producer.is_alive():
                cls.producer.terminate()

    def _push(self, pool_indices, stamps, base) -> bool:
        self.cmd_queue.put((pool_indices, stamps, base))
        return self.ack_queue.get(timeout=60)

    def test_transport_protocol(self):
        # 1. Cross-process, cross-device roundtrip: mapped rows equal what the
        #    producer wrote, with [pool_idx, stamp] leading each row.
        self.assertTrue(self._push([5, 6, 7], [11, 12, 13], base=1000))
        polled = self.reader.poll()
        self.assertIsNotNone(polled)
        slot, rows = polled
        self.assertEqual(rows.device.index, 1)
        self.assertEqual(rows[:, 0].tolist(), [5, 6, 7])
        self.assertEqual(rows[:, 1].tolist(), [11, 12, 13])
        expected_units = torch.arange(3 * (ROW_WIDTH - 2), dtype=torch.int64) + 1000
        self.assertEqual(
            rows[:, 2:].cpu().reshape(-1).tolist(), expected_units.tolist()
        )

        # 2. Ring WAR guard: with one slot still unacked, filling the other and
        #    pushing a third block must be refused, never overwritten.
        self.assertTrue(self._push([1], [21], base=2000))
        self.assertFalse(self._push([2], [22], base=3000))

        # 3. Acking recycles: after consuming both outstanding slots the ring
        #    accepts pushes again, and the new block round-trips.
        self.reader.ack(slot)
        polled = self.reader.poll()
        self.assertIsNotNone(polled)
        slot2, rows2 = polled
        self.assertEqual(rows2[:, 0].tolist(), [1])
        self.reader.ack(slot2)
        self.assertTrue(self._push([2], [22], base=3000))
        slot3, rows3 = self.reader.poll()
        self.assertEqual(rows3[:, 1].tolist(), [22])
        self.reader.ack(slot3)


if __name__ == "__main__":
    unittest.main()
