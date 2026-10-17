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


def _chain_viterbi(unary, lmbda):
    """Minimise sum(unary * x) + lmbda * sum((x[t] - x[t-1]) ** 2) over binary x."""
    dp = np.full((len(unary), 2), float("inf"))
    path = np.zeros((len(unary), 2), dtype=int)
    dp[0] = (0.0, unary[0])
    for index in range(1, len(unary)):
        for current in range(2):
            costs = [
                dp[index - 1, previous]
                + unary[index] * current
                + lmbda * (previous - current) ** 2
                for previous in range(2)
            ]
            path[index, current] = int(np.argmin(costs))
            dp[index, current] = min(costs)
    result = np.zeros(len(unary), dtype=int)
    result[-1] = int(np.argmin(dp[-1]))
    for index in range(len(unary) - 2, -1, -1):
        result[index] = path[index + 1, result[index + 1]]
    return result


def solve_chain_qubo_exact(all_scores, lmbda=0.5, threshold=0.5, seed=RANDOM_SEED):
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    return _chain_viterbi(-(scores - threshold), lmbda)


def solve_chain_qubo_causal(all_scores, lmbda=0.5, threshold=0.5, seed=RANDOM_SEED):
    """Online version of the chain QUBO: label t uses only scores[0..t].

    Runs the same forward dynamic program as solve_chain_qubo_exact but takes
    the minimum-energy state of each prefix instead of back-tracking from the
    end of the record, so no future epoch influences an earlier decision.
    """
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    unary = -(scores - threshold)
    result = np.zeros(len(scores), dtype=int)
    cost_off, cost_on = 0.0, unary[0]
    result[0] = int(cost_on < cost_off)
    for index in range(1, len(scores)):
        cost_off, cost_on = (
            min(cost_off, cost_on + lmbda),
            min(cost_off + lmbda, cost_on) + unary[index],
        )
        result[index] = int(cost_on < cost_off)
    return result


# Classical post-processing baselines with the same (lmbda, threshold) interface,
# so they are tuned on the same inner-validation grid as the QUBO solvers.

_PROB_EPS = 1e-6


def _logit(values):
    values = np.clip(np.asarray(values, dtype=float), _PROB_EPS, 1 - _PROB_EPS)
    return np.log(values) - np.log1p(-values)


def smooth_moving_average(all_scores, lmbda=5, threshold=0.5, seed=RANDOM_SEED):
    """Centered moving average of the scores over lmbda epochs, then threshold."""
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    window = max(1, int(round(lmbda)))
    before = (window - 1) // 2
    padded = np.concatenate([[0.0], np.cumsum(scores)])
    index = np.arange(len(scores))
    start = np.clip(index - before, 0, len(scores))
    end = np.clip(index - before + window, 0, len(scores))
    return ((padded[end] - padded[start]) / (end - start) > threshold).astype(int)


def smooth_moving_average_causal(all_scores, lmbda=5, threshold=0.5, seed=RANDOM_SEED):
    """Trailing moving average over the current and previous lmbda - 1 epochs."""
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    window = max(1, int(round(lmbda)))
    padded = np.concatenate([[0.0], np.cumsum(scores)])
    end = np.arange(1, len(scores) + 1)
    start = np.maximum(0, end - window)
    return ((padded[end] - padded[start]) / (end - start) > threshold).astype(int)


def hmm_viterbi(all_scores, lmbda=3.0, threshold=0.5, seed=RANDOM_SEED):
    """Two-state HMM Viterbi path treating scores as calibrated posteriors.

    The emission term is the log-likelihood ratio logit(score) - logit(threshold)
    and lmbda is the switching cost in nats (-log of the switch probability,
    up to a constant). Same chain as solve_chain_qubo_exact, log-odds unary.
    """
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    return _chain_viterbi(-(_logit(scores) - _logit(threshold)), lmbda)


def hmm_forward_causal(all_scores, lmbda=3.0, threshold=0.5, seed=RANDOM_SEED):
    """Online two-state HMM: label t is 1 when P(seizure | scores[0..t]) > 0.5.

    Forward filtering with switch probability 1 / (1 + exp(lmbda)) and the same
    log-likelihood-ratio emission as hmm_viterbi; no future epoch is used.
    """
    del seed
    scores = np.asarray(all_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    switch = 1.0 / (1.0 + np.exp(float(lmbda)))
    emission = _logit(scores) - _logit(threshold)
    result = np.zeros(len(scores), dtype=int)
    log_odds = 0.0
    for index, llr in enumerate(emission):
        if index > 0:
            p_on = 1.0 / (1.0 + np.exp(-log_odds))
            p_on = p_on * (1 - switch) + (1 - p_on) * switch
            log_odds = float(_logit(p_on))
        log_odds += llr
        result[index] = int(log_odds > 0)
    return result


POSTPROCESSORS = {
    "solve_qubo_seizure": solve_qubo_seizure,
    "solve_chain_qubo_exact": solve_chain_qubo_exact,
    "solve_chain_qubo_causal": solve_chain_qubo_causal,
    "smooth_moving_average": smooth_moving_average,
    "smooth_moving_average_causal": smooth_moving_average_causal,
    "hmm_viterbi": hmm_viterbi,
    "hmm_forward_causal": hmm_forward_causal,
}


def get_qubo_solver(name):
    try:
        return POSTPROCESSORS[name]
    except KeyError:
        raise ValueError(f"Unknown solver: {name}") from None


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
