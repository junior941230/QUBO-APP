import itertools
import unittest

import numpy as np

from qubo.cross_channel import qubo_energy, solve_cross_channel_qubo
from qubo.solvers import solve_chain_qubo_exact


class CrossChannelQuboTests(unittest.TestCase):
    def test_min_cut_matches_brute_force(self):
        rng = np.random.default_rng(0)
        neighbors = ((0, 1),)
        for trial in range(20):
            scores, channel_scores = rng.random(3), rng.random((2, 3))
            params = dict(
                lam_t=rng.uniform(0, 0.5), theta=rng.uniform(0.2, 0.8),
                lam_s=rng.uniform(0, 0.5), mu=rng.uniform(0, 1.0),
                theta_c=rng.uniform(0.2, 0.8),
            )
            z, x = solve_cross_channel_qubo(
                scores, channel_scores, neighbors=neighbors, return_channels=True, **params
            )
            best = min(
                qubo_energy(bits[:3], np.reshape(bits[3:], (2, 3)), scores, channel_scores,
                            neighbors=neighbors, **params)
                for bits in itertools.product([0, 1], repeat=9)
            )
            found = qubo_energy(z, x, scores, channel_scores, neighbors=neighbors, **params)
            self.assertAlmostEqual(found, best, places=3, msg=f"trial {trial}")

    def test_without_coupling_matches_chain_qubo(self):
        rng = np.random.default_rng(1)
        scores = rng.random(200)
        channel_scores = rng.random((18, 200))
        z = solve_cross_channel_qubo(scores, channel_scores, 0.6, 0.4, 0.3, 0.0, 0.5)
        np.testing.assert_array_equal(z, solve_chain_qubo_exact(scores, 0.6, 0.4))

    def test_active_channels_can_turn_global_state_on(self):
        scores = np.full(50, 0.35)
        channel_scores = np.full((18, 50), 0.1)
        channel_scores[:4, 20:30] = 0.9
        uncoupled = solve_cross_channel_qubo(scores, channel_scores, 0.3, 0.4, 0.2, 0.0, 0.5)
        coupled = solve_cross_channel_qubo(scores, channel_scores, 0.3, 0.4, 0.2, 0.5, 0.5)
        self.assertEqual(uncoupled.sum(), 0)
        self.assertTrue(coupled[20:30].all())
        self.assertEqual(coupled[:15].sum() + coupled[35:].sum(), 0)


if __name__ == "__main__":
    unittest.main()
