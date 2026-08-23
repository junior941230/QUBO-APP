import itertools
import unittest

import numpy as np

from qubo.solvers import solve_chain_qubo_exact


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


if __name__ == "__main__":
    unittest.main()
