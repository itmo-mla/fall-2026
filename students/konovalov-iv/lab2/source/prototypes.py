from math import lgamma

import numpy as np

from knn import pairwise_distances


def log_comb(n, r):
    return lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)


def ccv_weights(L, k, eps=1e-12):
    ell = L - k
    weights = []
    for m in range(1, k + 1):
        if L - 1 - m < ell - 1:
            break
        r = np.exp(log_comb(L - 1 - m, ell - 1) - log_comb(L - 1, ell))
        if r < eps:
            break
        weights.append(r)
    return np.array(weights)


class CCVSelector:
    def __init__(self, X, y, control_fraction=0.3):
        self.y = y
        self.L = len(y)
        self.R = ccv_weights(self.L, max(1, round(control_fraction * self.L)))
        D = pairwise_distances(X, X)
        np.fill_diagonal(D, np.inf)
        self.D = D
        self.order = np.argsort(D, axis=1, kind="stable")[:, :-1]
        self.neighbor_errors = y[self.order] != y[:, None]

    def contributions(self, mask):
        in_omega = mask[self.order]
        rank = np.cumsum(in_omega, axis=1)
        valid = in_omega & (rank <= len(self.R))
        r = np.where(valid, self.R[np.clip(rank - 1, 0, len(self.R) - 1)], 0.0)
        return (r * self.neighbor_errors).sum(axis=1)

    def ccv(self, mask):
        return self.contributions(mask).mean()

    def compactness_profile(self, mask, m_max):
        in_omega = mask[self.order]
        profile = []
        for m in range(1, m_max + 1):
            mth = np.argmax(np.cumsum(in_omega, axis=1) == m, axis=1)
            profile.append(self.neighbor_errors[np.arange(self.L), mth].mean())
        return np.array(profile)

    def _can_remove(self, mask, i):
        return np.sum(mask & (self.y == self.y[i])) > 1

    def greedy_remove(self, tol=0.005):
        mask = np.ones(self.L, dtype=bool)
        removed, path = [], [self.ccv(mask)]
        while True:
            candidates = [i for i in np.flatnonzero(mask) if self._can_remove(mask, i)]
            if not candidates:
                break
            scores = []
            for i in candidates:
                mask[i] = False
                scores.append(self.ccv(mask))
                mask[i] = True
            best = candidates[int(np.argmin(scores))]
            mask[best] = False
            removed.append(best)
            path.append(min(scores))

        path = np.array(path)
        n_removed = int(np.max(np.flatnonzero(path <= path.min() + tol)))
        omega = np.ones(self.L, dtype=bool)
        omega[removed[:n_removed]] = False
        return omega, path, n_removed

    def greedy_add(self, X):
        mask = np.zeros(self.L, dtype=bool)
        for c in np.unique(self.y):
            idx = np.flatnonzero(self.y == c)
            center = X[idx].mean(axis=0)
            mask[idx[np.argmin(((X[idx] - center) ** 2).sum(axis=1))]] = True
        path = [self.ccv(mask)]
        while not mask.all():
            candidates = np.flatnonzero(~mask)
            scores = []
            for i in candidates:
                mask[i] = True
                scores.append(self.ccv(mask))
                mask[i] = False
            if min(scores) >= path[-1]:
                break
            mask[candidates[int(np.argmin(scores))]] = True
            path.append(min(scores))
        return mask, np.array(path)
