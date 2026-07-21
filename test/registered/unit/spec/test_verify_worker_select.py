"""CPU unit tests for the decoupled verifier's GPU enum-select math.

``select_enum_units`` is the heart of enumeration verify: given the
seat-gathered rows (num_cases x fanout x [guess, chain] units) and freshness
stamps, it must pick exactly the unit matching reality -- (previous round's
accept case, real bonus token) -- and degrade every miss (stale stamp, bonus
outside the F guesses, never-landed seat) to the bonus-seeded fallback row.
Pure tensor math, so CPU tensors drive the identical code path. A wrong pick
here silently verifies the wrong chain (costing acceptance, never correctness
-- verify only accepts target-agreeing tokens), which no other test observes
directly.
"""

import unittest

import torch

from sglang.srt.speculative.verify_worker import select_enum_units
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Small dims keep the expected layout hand-checkable: K=2 steps -> 3 accept
# cases, F=2 guesses, unit_width=3 ([guess, c1, c2]).
NUM_CASES = 3
FANOUT = 2
UNIT_WIDTH = 3
ROW_WIDTH = NUM_CASES * FANOUT * UNIT_WIDTH


def _row(units):
    """units: {(case, guess_idx): [guess, c1, c2]} -> flat [ROW_WIDTH] row.

    Unfilled units get a poison guess (-7) that never matches a real bonus.
    """
    flat = torch.full((NUM_CASES, FANOUT, UNIT_WIDTH), -7, dtype=torch.int64)
    for (case, f), tokens in units.items():
        flat[case, f] = torch.tensor(tokens, dtype=torch.int64)
    return flat.reshape(ROW_WIDTH)


def _select(rows, stamps, bonus, cases, bases, rows_prev=None, stamps_prev=None):
    """Drive the two-generation select; the second generation defaults to an
    unwritten (sentinel-stamped) block."""
    newest = torch.stack(rows)
    previous = (
        torch.stack(rows_prev) if rows_prev is not None else torch.zeros_like(newest)
    )
    stamps_newest = torch.tensor(stamps, dtype=torch.int64)
    stamps_previous = (
        torch.tensor(stamps_prev, dtype=torch.int64)
        if stamps_prev is not None
        else torch.full_like(stamps_newest, -1)
    )
    return select_enum_units(
        torch.stack([newest, previous], dim=1),
        torch.stack([stamps_newest, stamps_previous], dim=1),
        bonus_tokens=torch.tensor(bonus, dtype=torch.int64),
        prev_accept_lens=torch.tensor(cases, dtype=torch.int64),
        base_committed_lens=torch.tensor(bases, dtype=torch.int64),
        num_cases=NUM_CASES,
        fanout=FANOUT,
        unit_width=UNIT_WIDTH,
    )


class TestSelectEnumUnits(CustomTestCase):
    def test_hit_selects_matching_case_and_guess_unit(self):
        # The unit at (accept case, matching guess) is returned verbatim: it IS
        # the verify row [root=bonus, chain...]. Guess 0 of case 1 misses (55),
        # guess 1 hits (77) -- the match must scan the guess axis, not take f=0.
        row = _row({(1, 0): [55, 10, 11], (1, 1): [77, 20, 21]})
        selected, hits = _select(
            [row], stamps=[100], bonus=[77], cases=[1], bases=[100]
        )
        self.assertEqual(hits.tolist(), [True])
        self.assertEqual(selected[0].tolist(), [77, 20, 21])

    def test_stale_stamp_falls_back(self):
        # Same content, wrong base stamp (the block was drafted from an older
        # committed prefix): must fall back even though the guess matches.
        row = _row({(1, 1): [77, 20, 21]})
        selected, hits = _select([row], stamps=[99], bonus=[77], cases=[1], bases=[100])
        self.assertEqual(hits.tolist(), [False])
        self.assertEqual(selected[0].tolist(), [77, 77, 77])

    def test_guess_miss_falls_back_with_real_bonus_root(self):
        # Fresh stamp but the real bonus is outside the F guesses: fallback row
        # must be rooted at the REAL bonus (a plain 1-token decode), never at a
        # guessed token.
        row = _row({(0, 0): [55, 10, 11], (0, 1): [66, 20, 21]})
        selected, hits = _select([row], stamps=[42], bonus=[77], cases=[0], bases=[42])
        self.assertEqual(hits.tolist(), [False])
        self.assertEqual(selected[0].tolist(), [77, 77, 77])

    def test_wrong_case_lookup_misses(self):
        # The guess only lives under case 2; reality accepted 0 drafts, so case
        # 0 is consulted and the bonus is not among ITS guesses -> fallback.
        # Guards against flattening the (case, guess) axes into one search.
        row = _row({(2, 0): [77, 20, 21]})
        selected, hits = _select([row], stamps=[42], bonus=[77], cases=[0], bases=[42])
        self.assertEqual(hits.tolist(), [False])
        self.assertEqual(selected[0].tolist(), [77, 77, 77])

    def test_first_matching_guess_wins(self):
        # Duplicate guesses in one case: selection must be deterministic
        # (first match), or the verified chain differs run to run.
        row = _row({(0, 0): [77, 10, 11], (0, 1): [77, 20, 21]})
        selected, hits = _select([row], stamps=[42], bonus=[77], cases=[0], bases=[42])
        self.assertEqual(hits.tolist(), [True])
        self.assertEqual(selected[0].tolist(), [77, 10, 11])

    def test_out_of_range_case_clamps_and_falls_back(self):
        # A protocol bug producing case > K must not crash the gather with a
        # device-side OOB; it clamps and (the clamped case not matching) falls
        # back like any miss.
        row = _row({(2, 0): [77, 20, 21]})
        selected, hits = _select([row], stamps=[42], bonus=[77], cases=[9], bases=[42])
        self.assertEqual(hits.tolist(), [True])  # clamped to case 2, which matches
        self.assertEqual(selected[0].tolist(), [77, 20, 21])

    def test_previous_generation_serves_when_newer_block_landed(self):
        # Regression (first cross-process e2e, 0-hit): the block serving THIS
        # round was enumerated two commits back, but the last commit already
        # pushed a newer block into the seat. The select must match the
        # expected base against BOTH stamped generations and pick the older
        # one, or sync pacing degrades every round to a staleness fallback.
        serving = _row({(0, 0): [77, 20, 21]})
        newer = _row({(0, 0): [55, 30, 31]})
        selected, hits = _select(
            [newer],
            stamps=[50],  # the newer push (|P_{r-1}|)
            bonus=[77],
            cases=[0],
            bases=[42],  # this round expects the older base (|P_{r-2}|)
            rows_prev=[serving],
            stamps_prev=[42],
        )
        self.assertEqual(hits.tolist(), [True])
        self.assertEqual(selected[0].tolist(), [77, 20, 21])

    def test_mixed_batch_rows_are_independent(self):
        # One hit + one stale row in the same batch: per-row judgment, no
        # cross-row leakage of stamps or guesses.
        hit_row = _row({(0, 0): [11, 1, 2]})
        stale_row = _row({(0, 0): [22, 3, 4]})
        selected, hits = _select(
            [hit_row, stale_row],
            stamps=[10, 999],
            bonus=[11, 22],
            cases=[0, 0],
            bases=[10, 20],
        )
        self.assertEqual(hits.tolist(), [True, False])
        self.assertEqual(selected[0].tolist(), [11, 1, 2])
        self.assertEqual(selected[1].tolist(), [22, 22, 22])


if __name__ == "__main__":
    unittest.main()
