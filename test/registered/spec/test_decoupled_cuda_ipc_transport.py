"""Two-GPU tests for the decoupled-spec CUDA IPC enumeration data plane.

Exercises the real mechanism in one process across two devices: the producer
pool lives on cuda:0, the consumer maps it onto cuda:1 through the CUDA IPC
handle + peer access (the NVLink path), and the shm flag protocol carries
arrival (ready_seq) and reuse (consumed_seq). What is guarded:

- rendezvous + handle redirect: the mapped rows read back exactly what the
  producer wrote, across devices;
- the ring protocol: unacked slots refuse reuse (cross-process WAR guard),
  acked slots recycle;
- a stale shm segment from a killed run is replaced, not fatal.
"""

import unittest

import torch

from sglang.srt.speculative.cuda_ipc_enum_transport import (
    CudaIpcEnumBlockPool,
    CudaIpcEnumBlockReader,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")

ENDPOINT = "ipc:///tmp/decoupled-cuda-ipc-test"
ROW_WIDTH = 2 + 12  # [pool_idx, stamp] + a K=1,F=... arbitrary unit payload


def _units(batch_size: int, base: int) -> torch.Tensor:
    return (
        torch.arange(batch_size * (ROW_WIDTH - 2), dtype=torch.int64, device="cuda:0")
        + base
    ).view(batch_size, ROW_WIDTH - 2)


@unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")
class TestCudaIpcEnumTransport(CustomTestCase):
    def setUp(self):
        self.pool = CudaIpcEnumBlockPool(
            device="cuda:0",
            endpoint=ENDPOINT,
            max_rows=8,
            row_width=ROW_WIDTH,
            num_slots=2,
        )
        self.reader = CudaIpcEnumBlockReader(
            device="cuda:1", endpoint=ENDPOINT, attach_timeout_s=10.0
        )

    def tearDown(self):
        self.reader.close()
        self.pool.close()

    def test_cross_device_roundtrip(self):
        units = _units(3, base=1000)
        self.assertTrue(
            self.pool.push(
                pool_indices=[5, 6, 7],
                base_committed_lens=[11, 12, 13],
                units=units,
            )
        )
        polled = self.reader.poll()
        self.assertIsNotNone(polled)
        slot, rows = polled
        self.assertEqual(rows.device.index, 1)
        self.assertEqual(rows[:, 0].tolist(), [5, 6, 7])
        self.assertEqual(rows[:, 1].tolist(), [11, 12, 13])
        self.assertTrue(torch.equal(rows[:, 2:].cpu(), units.cpu()))
        self.reader.ack(slot)
        self.assertIsNone(self.reader.poll())

    def test_ring_refuses_unacked_slot_reuse(self):
        # Fill both slots without acking; the third push must be dropped (the
        # cross-process WAR guard: never overwrite an unconsumed block).
        for i in range(2):
            self.assertTrue(
                self.pool.push(
                    pool_indices=[1 + i],
                    base_committed_lens=[i],
                    units=_units(1, base=i * 100),
                )
            )
        self.assertFalse(
            self.pool.push(
                pool_indices=[3], base_committed_lens=[9], units=_units(1, base=900)
            )
        )
        # Consume one slot; the ring accepts a push again.
        slot, _rows = self.reader.poll()
        self.reader.ack(slot)
        self.assertTrue(
            self.pool.push(
                pool_indices=[3], base_committed_lens=[9], units=_units(1, base=900)
            )
        )

    def test_stale_segment_is_replaced(self):
        # A second producer on the same endpoint (as after a crashed run)
        # replaces the stale segment instead of failing forever.
        replacement = CudaIpcEnumBlockPool(
            device="cuda:0",
            endpoint=ENDPOINT,
            max_rows=4,
            row_width=ROW_WIDTH,
            num_slots=2,
        )
        try:
            fresh_reader = CudaIpcEnumBlockReader(
                device="cuda:1", endpoint=ENDPOINT, attach_timeout_s=10.0
            )
            self.assertEqual(fresh_reader.max_rows, 4)
            fresh_reader.close()
        finally:
            replacement.close()


if __name__ == "__main__":
    unittest.main()
