from collections import defaultdict
import inspect

import numpy as np

from config import RANDOM_SEED

try:
    from neal import SimulatedAnnealingSampler
except ImportError:
    SimulatedAnnealingSampler = None


def solve_qubo_seizure(all_scores, lmbda=0.5, threshold=0.5, seed=RANDOM_SEED):
    if SimulatedAnnealingSampler is None:
        raise ImportError("solve_qubo_seizure requires dwave-neal")
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    qubo = defaultdict(float)
    for index, score in enumerate(scores):
        qubo[(index, index)] = -(score - threshold)
    for index in range(len(scores) - 1):
        qubo[(index, index)] += lmbda
        qubo[(index + 1, index + 1)] += lmbda
        qubo[(index, index + 1)] = -2 * lmbda
    sample = SimulatedAnnealingSampler().sample_qubo(
        qubo, num_reads=20, seed=int(seed)
    ).first.sample
    return np.asarray([sample[index] for index in range(len(scores))], dtype=int)


def solve_chain_qubo_exact(all_scores, lmbda=0.5, threshold=0.5, seed=RANDOM_SEED):
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    unary = -(scores - threshold)
    dp = np.full((len(scores), 2), float("inf"))
    path = np.zeros((len(scores), 2), dtype=int)
    dp[0] = (0.0, unary[0])
    for index in range(1, len(scores)):
        for current in range(2):
            costs = [
                dp[index - 1, previous]
                + unary[index] * current
                + lmbda * (previous - current) ** 2
                for previous in range(2)
            ]
            path[index, current] = int(np.argmin(costs))
            dp[index, current] = min(costs)
    result = np.zeros(len(scores), dtype=int)
    result[-1] = int(np.argmin(dp[-1]))
    for index in range(len(scores) - 2, -1, -1):
        result[index] = path[index + 1, result[index + 1]]
    return result


def get_qubo_solver(name):
    if name == "solve_qubo_seizure":
        return solve_qubo_seizure
    if name == "solve_chain_qubo_exact":
        return solve_chain_qubo_exact
    raise ValueError(f"Unknown solver: {name}")


def safe_solver_call(solver, scores, lmbda, threshold, seed=RANDOM_SEED):
    """Run a solver with a reproducible seed when its interface supports one."""
    scores = np.asarray(scores)
    kwargs = {"lmbda": float(lmbda), "threshold": float(threshold)}
    try:
        parameters = inspect.signature(solver).parameters.values()
        accepts_seed = any(
            param.name == "seed" or param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters
        )
    except (TypeError, ValueError):
        accepts_seed = True
    if accepts_seed:
        kwargs["seed"] = int(seed)

    out = solver(scores, **kwargs)
    out = np.asarray(out)
    if out.ndim != 1 or out.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Solver output shape {out.shape} does not match scores shape {scores.shape}"
        )
    return (out > 0).astype(int)
