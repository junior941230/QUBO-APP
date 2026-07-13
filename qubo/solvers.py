import inspect

import numpy as np
from config import RANDOM_SEED
from pipeline import solve_qubo_seizure, solve_chain_qubo_exact
def get_qubo_solver(name):
    if name == "solve_qubo_seizure":
        return solve_qubo_seizure
    if name == "solve_chain_qubo_exact":
        return solve_chain_qubo_exact
    raise ValueError(f"Unknown solver: {name}")


def safe_solver_call(solver, scores, lmbda, threshold, seed=RANDOM_SEED):
    """Run a solver with a reproducible seed when its interface supports one."""
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
