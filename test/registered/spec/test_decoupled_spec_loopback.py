"""Single-process loopback correctness lock for decoupled enumeration spec.

Launches the decoupled VERIFIER engine with an in-process scripted fake
drafter (fake transport; SGLANG_TEST_DECOUPLED_LOOPBACK) and asserts the
enumeration-verify contract: the committed output is token-for-token identical
to the non-speculative baseline **for any drafter behavior**, because a wrong,
stale, or absent enumeration block only ever degrades a round to the unified
bonus-seeded fallback (verify accepts only target-agreeing tokens). Both
deterministic degradation paths are pinned:

- ``garbage``: fresh stamps, guesses that (almost) never match -> guess-miss
  fallback path.
- ``stale``: stamps always one behind -> staleness fallback path, judged
  before any guess is consulted.

This is the correctness lock every later phase (real drafter, cross-process
transports, overlap flags) must keep green.
"""

import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 48

PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5,",
    "The quick brown fox jumps over",
    "Water is composed of hydrogen and",
]

DECOUPLED_ARGS = [
    "--speculative-algorithm",
    "STANDALONE",
    "--speculative-draft-model-path",
    MODEL,
    "--speculative-num-steps",
    "3",
    "--decoupled-spec-role",
    "verifier",
    "--decoupled-spec-rank",
    "0",
    # Present to satisfy the role validator; the loopback fake mesh never
    # opens them.
    "--decoupled-spec-bind-endpoint",
    "ipc:///tmp/decoupled-loopback-verifier",
    "--decoupled-spec-connect-endpoints",
    '["ipc:///tmp/decoupled-loopback-drafter"]',
]


def _generate_output_ids(base_url: str) -> list[list[int]]:
    response = requests.post(
        base_url + "/generate",
        json={
            "text": PROMPTS,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return [entry["output_ids"] for entry in response.json()]


class TestDecoupledSpecLoopback(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Non-speculative baseline outputs, collected once.
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        try:
            cls.baseline_output_ids = _generate_output_ids(DEFAULT_URL_FOR_TEST)
        finally:
            kill_process_tree(process.pid)

    def _run_loopback(self, mode: str) -> list[list[int]]:
        with envs.SGLANG_TEST_DECOUPLED_LOOPBACK.override(mode):
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=DECOUPLED_ARGS,
            )
        try:
            return _generate_output_ids(DEFAULT_URL_FOR_TEST)
        finally:
            kill_process_tree(process.pid)

    def test_garbage_drafter_matches_baseline(self):
        # Guess-miss fallback every round: committed output must still equal
        # the non-speculative baseline token for token.
        output_ids = self._run_loopback("garbage")
        self.assertEqual(output_ids, self.baseline_output_ids)

    def test_stale_drafter_matches_baseline(self):
        # Staleness fallback every round (judged before the guess match).
        output_ids = self._run_loopback("stale")
        self.assertEqual(output_ids, self.baseline_output_ids)


if __name__ == "__main__":
    unittest.main(verbosity=3)
