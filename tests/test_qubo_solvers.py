import itertools
import unittest

import numpy as np

from qubo.solvers import (
    POSTPROCESSORS,
    hmm_forward_causal,
    hmm_viterbi,
    smooth_moving_average,
    smooth_moving_average_causal,
    solve_chain_qubo_causal,
    solve_chain_qubo_exact,
)


class ExactQuboSolverTests(unittest.TestCase):
    def test_empty_sequence(self):
        self.assertEqual(len(solve_chain_qubo_exact(np.array([]))), 0)

    def test_dynamic_program_matches_brute_force(self):
        scores = np.array([0.8, 0.2, 0.9])
        lmbda = 0.25
        threshold = 0.5
        actual = solve_chain_qubo_exact(scores, lmbda, threshold)

        def energy(values):
            values = np.asarray(values)
            unary = -np.sum((scores - threshold) * values)
            smooth = lmbda * np.sum(np.diff(values) ** 2)
            return unary + smooth

        candidates = list(itertools.product([0, 1], repeat=len(scores)))
        expected = np.asarray(min(candidates, key=energy))
        np.testing.assert_array_equal(actual, expected)


class CausalQuboSolverTests(unittest.TestCase):
    def test_empty_sequence(self):
        self.assertEqual(len(solve_chain_qubo_causal(np.array([]))), 0)

    def test_each_label_is_last_label_of_prefix_optimum(self):
        rng = np.random.default_rng(0)
        scores = rng.random(12)
        lmbda, threshold = 0.3, 0.5
        actual = solve_chain_qubo_causal(scores, lmbda, threshold)
        expected = [
            solve_chain_qubo_exact(scores[: index + 1], lmbda, threshold)[-1]
            for index in range(len(scores))
        ]
        np.testing.assert_array_equal(actual, expected)

    def test_future_scores_do_not_change_past_labels(self):
        rng = np.random.default_rng(1)
        scores = rng.random(20)
        changed = scores.copy()
        changed[10:] = 1.0 - changed[10:]
        np.testing.assert_array_equal(
            solve_chain_qubo_causal(scores, 0.4, 0.5)[:10],
            solve_chain_qubo_causal(changed, 0.4, 0.5)[:10],
        )


class MovingAverageTests(unittest.TestCase):
    def test_causal_matches_trailing_mean(self):
        scores = np.random.default_rng(2).random(15)
        expected = [scores[max(0, t - 3): t + 1].mean() > 0.5 for t in range(15)]
        np.testing.assert_array_equal(smooth_moving_average_causal(scores, 4, 0.5), expected)

    def test_centered_matches_centered_mean(self):
        scores = np.random.default_rng(3).random(15)
        expected = [scores[max(0, t - 2): t + 3].mean() > 0.5 for t in range(15)]
        np.testing.assert_array_equal(smooth_moving_average(scores, 5, 0.5), expected)

    def test_window_one_is_plain_threshold(self):
        scores = np.array([0.1, 0.6, 0.4, 0.9])
        np.testing.assert_array_equal(smooth_moving_average(scores, 1, 0.5), scores > 0.5)
        np.testing.assert_array_equal(
            smooth_moving_average_causal(scores, 1, 0.5), scores > 0.5
        )


class HmmTests(unittest.TestCase):
    def test_viterbi_matches_brute_force(self):
        scores = np.array([0.9, 0.3, 0.8, 0.05, 0.7])
        lmbda, threshold = 1.5, 0.4
        llr = np.log(scores / (1 - scores)) - np.log(threshold / (1 - threshold))

        def energy(values):
            values = np.asarray(values)
            return -np.sum(llr * values) + lmbda * np.sum(np.diff(values) ** 2)

        candidates = list(itertools.product([0, 1], repeat=len(scores)))
        expected = np.asarray(min(candidates, key=energy))
        np.testing.assert_array_equal(hmm_viterbi(scores, lmbda, threshold), expected)

    def test_forward_without_memory_is_plain_threshold(self):
        scores = np.array([0.1, 0.6, 0.35, 0.9])
        np.testing.assert_array_equal(hmm_forward_causal(scores, 0.0, 0.4), scores > 0.4)

    def test_forward_large_switch_cost_suppresses_isolated_spike(self):
        scores = np.full(30, 0.1)
        scores[15] = 0.9
        self.assertEqual(hmm_forward_causal(scores, 8.0, 0.5).sum(), 0)


class CausalPostprocessorTests(unittest.TestCase):
    def test_future_scores_do_not_change_past_labels(self):
        rng = np.random.default_rng(4)
        scores = rng.random(40)
        changed = scores.copy()
        changed[20:] = 1.0 - changed[20:]
        for solver, lmbda in [(smooth_moving_average_causal, 7), (hmm_forward_causal, 2.0)]:
            np.testing.assert_array_equal(
                solver(scores, lmbda, 0.4)[:20], solver(changed, lmbda, 0.4)[:20]
            )

    def test_all_postprocessors_handle_empty_input(self):
        for name, solver in POSTPROCESSORS.items():
            if name == "solve_qubo_seizure":
                continue
            self.assertEqual(len(solver(np.array([]), 1.0, 0.5)), 0, name)


if __name__ == "__main__":
    unittest.main()
