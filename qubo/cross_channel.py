"""Cross-channel QUBO: a global seizure chain coupled to per-channel chains.

Variables are z[t] (global seizure state) and x[c, t] (channel c involved at
epoch t). The energy is

    sum_t      -(s[t] - theta) z[t]
  + sum_{c,t}  -(p[c, t] - theta_c) x[c, t]
  + lam_t      sum_t [z[t] != z[t-1]] + lam_t sum_{c,t} [x[c,t] != x[c,t-1]]
  + lam_s      sum_{(c,c') adjacent, t} [x[c,t] != x[c',t]]
  + mu         sum_{c,t} x[c,t] (1 - z[t])

Every pairwise term is attractive, so the QUBO is submodular and an s-t
minimum cut gives the exact optimum. With mu = 0 the channels decouple and z
is exactly the 1-D chain QUBO solved by solve_chain_qubo_exact.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from core.channels import CANONICAL_CHB_CHANNELS

N_CHANNELS = len(CANONICAL_CHB_CHANNELS)
# Integer capacity resolution for scipy's max-flow; energies are O(1).
_SCALE = 10_000


def _montage_neighbors():
    """Neighbouring bipolar channels: consecutive links in each chain plus the
    same position in the adjacent chain on the same hemisphere."""
    chains = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13], [14, 15, 16, 17]]
    pairs = [(a, b) for chain in chains for a, b in zip(chain, chain[1:])]
    pairs += [(a, b) for a, b in zip(chains[0], chains[1])]
    pairs += [(a, b) for a, b in zip(chains[3], chains[4])]
    return tuple(pairs)


MONTAGE_NEIGHBORS = _montage_neighbors()


def qubo_energy(z, x, scores, channel_scores, lam_t, theta, lam_s, mu, theta_c,
                neighbors=MONTAGE_NEIGHBORS):
    """Energy of a labelling; used by tests to check the min-cut optimum."""
    z, x = np.asarray(z), np.asarray(x)
    energy = -np.sum((scores - theta) * z) - np.sum((channel_scores - theta_c) * x)
    energy += lam_t * (np.abs(np.diff(z)).sum() + np.abs(np.diff(x, axis=1)).sum())
    energy += lam_s * sum(np.abs(x[a] - x[b]).sum() for a, b in neighbors)
    energy += mu * np.sum(x * (1 - z))
    return float(energy)


def solve_cross_channel_qubo(scores, channel_scores, lam_t, theta, lam_s, mu, theta_c,
                             neighbors=MONTAGE_NEIGHBORS, return_channels=False):
    """Exact minimiser of the cross-channel QUBO via s-t minimum cut.

    scores has shape (T,), channel_scores (C, T). Returns z (and x when
    return_channels is True). Node i is labelled 1 when it stays on the
    source side of the cut.
    """
    scores = np.asarray(scores, dtype=float)
    channel_scores = np.asarray(channel_scores, dtype=float)
    n_epochs = len(scores)
    if n_epochs == 0:
        empty = np.array([], dtype=int)
        return (empty, np.zeros((len(channel_scores), 0), dtype=int)) if return_channels else empty
    n_channels = channel_scores.shape[0]
    n_vars = (n_channels + 1) * n_epochs
    source, sink = n_vars, n_vars + 1
    z_ids = np.arange(n_epochs)
    x_ids = n_epochs + np.arange(n_channels * n_epochs).reshape(n_channels, n_epochs)

    rows, cols, caps = [], [], []

    def add(tail, head, capacity):
        capacity = np.broadcast_to(np.asarray(capacity, dtype=float), np.shape(tail))
        keep = capacity > 0
        rows.append(np.asarray(tail)[keep])
        cols.append(np.broadcast_to(head, np.shape(tail))[keep])
        caps.append(capacity[keep])

    # Unary u * label: cost u when label is 1 (i -> sink), -u when 0 (source -> i).
    unary = np.concatenate([-(scores - theta), -(channel_scores - theta_c).ravel()])
    ids = np.arange(n_vars)
    add(ids, sink, unary)
    add(np.full(n_vars, source), ids, -unary)

    # Disagreement penalties w [a != b]: arcs in both directions.
    def add_pair(a, b, weight):
        if weight > 0:
            add(a, b, weight)
            add(b, a, weight)

    add_pair(z_ids[:-1], z_ids[1:], lam_t)
    add_pair(x_ids[:, :-1].ravel(), x_ids[:, 1:].ravel(), lam_t)
    for a, b in neighbors:
        add_pair(x_ids[a], x_ids[b], lam_s)
    # mu x (1 - z): paid when x is 1 (source side) and z is 0 (sink side).
    if mu > 0:
        add(x_ids.ravel(), np.tile(z_ids, n_channels), mu)

    tails, heads = np.concatenate(rows), np.concatenate(cols)
    capacity = np.rint(np.concatenate(caps) * _SCALE).astype(np.int32)
    graph = csr_matrix((capacity, (tails, heads)), shape=(n_vars + 2, n_vars + 2))
    graph.sum_duplicates()
    flow = maximum_flow(graph, source, sink, method="dinic").flow
    residual = (graph - flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    labels = np.zeros(n_vars + 2, dtype=int)
    labels[reachable] = 1
    z = labels[z_ids]
    if return_channels:
        return z, labels[x_ids]
    return z
