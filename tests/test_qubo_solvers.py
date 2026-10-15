import itertools
import unittest

import numpy as np

from qubo.solvers import solve_chain_qubo_causal, solve_chain_qubo_exact


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


if __name__ == "__main__":
    unittest.main()
