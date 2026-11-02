"""Numerical checks for "The logarithmic factor in the tuned constant-step rate of random reshuffling".

Instance: n1 components f(x) = (L/2) x^2 - s x and n2 = n - n1 components f(x) = s (n1/n2) x, n1 <= n/2.
F(x) = (mu/2) x^2 with mu = L n1/n, x* = 0, G = max_i |f_i'(0)| = s.
One epoch of random reshuffling with step g maps X to A X + B, A = (1 - gL)^n1, B random through the order.
The epoch moments E B, E B^2 are computed exactly by a dynamic programme over (step, number of stiff
components used); epochs are independent, so E X_K and E X_K^2 follow by recursion.

Run: python rr_check.py            (all checks, a few minutes)
"""
import math
import numpy as np


def epoch_moments(gs, n, n1, L=1.0, s=1.0):
    """Exact E[B], E[B^2] of one epoch for each step in gs (A = (1-gL)^n1 is deterministic)."""
    gs = np.asarray(gs, dtype=float)[:, None]
    n2 = n - n1
    a = 1.0 - gs * L
    cS, cF = gs * s, -gs * s * n1 / n2
    J = n1 + 1
    j = np.arange(J)
    P = np.zeros((1, J)); P[0, 0] = 1.0
    B = np.zeros((gs.shape[0], J)); B2 = np.zeros_like(B)

    def shift(X):
        Y = np.zeros_like(X); Y[:, 1:] = X[:, :-1]; return Y

    for k in range(n):
        ps = np.clip((n1 - j) / (n - k), 0.0, 1.0)[None, :]
        wS, wF = P * ps, P * (1.0 - ps)
        nP = shift(wS) + wF
        nB = shift(wS * (a * B + cS)) + wF * (B + cF)
        nB2 = shift(wS * (a * a * B2 + 2 * a * cS * B + cS * cS)) + wF * (B2 + 2 * cF * B + cF * cF)
        d = np.where(nP > 0, nP, 1.0)
        P, B, B2 = nP, nB / d, nB2 / d
    return B[:, n1], B2[:, n1]


def last_iterate_error(gs, n, n1, K, X0, L=1.0, s=1.0):
    """Exact E F(x_K) - F* for each step in gs."""
    mu = L * n1 / n
    EB, EB2 = epoch_moments(gs, n, n1, L, s)
    A = (1.0 - np.asarray(gs) * L) ** n1
    m1 = np.full_like(EB, X0); m2 = m1 * m1
    for _ in range(K):
        m1, m2 = A * m1 + EB, A * A * m2 + 2 * A * EB * m1 + EB2
    return 0.5 * mu * m2


def phi(r, n1):
    k = np.arange(n1)
    return 2.0 * np.sum((k + 1) * r ** k) / ((n1 + 1) * np.sum(r ** k))


def check_phi():
    worst_hi, worst_lo = 0.0, np.inf
    for n1 in (1, 2, 3, 5, 10, 50, 200):
        for r in np.linspace(0.0, 1.0, 2001):
            p = phi(r, n1)
            worst_hi = max(worst_hi, p)
            worst_lo = min(worst_lo, p - max(r ** (n1 - 1), 2.0 / (n1 + 1)))
    print(f"[phi] max phi = {worst_hi:.12f} (Lemma 2.1: <= 1); min of phi - max(r^(n1-1), 2/(n1+1)) = {worst_lo:.3e} (>= 0)")


def check_mean_formula():
    for n, n1, g in ((10, 3, 0.07), (40, 1, 0.01), (100, 10, 0.002)):
        EB, _ = epoch_moments([g], n, n1)
        r = 1 - g
        m_dp = EB[0] / (1 - r ** n1)
        print(f"[mean] n {n} n1 {n1} g {g}: stationary mean, DP {m_dp:.12e}, (g s/2) phi(r) {g / 2 * phi(r, n1):.12e}")


def check_variance():
    for n, n1 in ((40, 1), (40, 3), (100, 10), (100, 50), (1000, 100)):
        n2 = n - n1
        for g in (1e-5, 1e-4, 1e-3):
            if g * n1 > 0.5:
                continue
            EB, EB2 = epoch_moments([g], n, n1)
            vb = EB2[0] - EB[0] ** 2
            first = g ** 4 * n1 ** 3 * (n + 1) / (12 * n2)
            bound = g ** 4 * (math.sqrt(n1 ** 3 * (n + 1) / (12 * n2)) + g * n1 ** 3) ** 2
            print(f"[var] n {n} n1 {n1} g {g:.0e}: Var B {vb:.4e}, first order {first:.4e} (ratio {vb / first:.4f}), "
                  f"Lemma 2.2 bound {bound:.4e}, holds {vb <= bound}")


def check_rates():
    """Minimum over a grid of steps in (0, 1/L] of the exact error, against Theorem 1.1(a) and (b)."""
    for n, n1, K in ((40, 1, 1000), (40, 1, 10000), (100, 10, 1000), (100, 10, 10000), (100, 50, 10000), (1000, 100, 10000)):
        mu, T, D = n1 / n, n * K, 1.0
        Lam = math.log(mu * T * D)
        lower = (Lam - math.log(Lam)) ** 2 / (512 * mu * T * T)
        unit = Lam ** 2 / (8 * mu * T * T)
        gs = np.unique(np.concatenate([np.exp(np.linspace(math.log(1e-3 * Lam / (mu * T)), 0.0, 2500)),
                                       np.linspace(1e-4, 1.0, 1000)]))
        gs = gs[gs <= 1.0]
        e_plus = last_iterate_error(gs, n, n1, K, +D)
        e_minus = last_iterate_error(gs, n, n1, K, -D)
        i = int(e_plus.argmin())
        g_t = Lam / (mu * T)
        e_t = last_iterate_error([g_t], n, n1, K, +D)[0]
        upper = ((1 + Lam / 2) ** 2 + Lam ** 3 * n1 * (0.5 + Lam * math.sqrt(n1) / K) ** 2 / K) / (2 * mu * T * T)
        print(f"[rate] n {n} n1 {n1} K {K}: X0=+D min {e_plus[i]:.4e} = {e_plus[i] / unit:.3f} x unit "
              f"= {e_plus[i] / lower:.0f} x lower bound; argmin g mu T/Lam {gs[i] * mu * T / Lam:.3f}; "
              f"at g = Lam/(mu T): {e_t:.4e} <= upper {upper:.4e} {e_t <= upper}; "
              f"X0=-D min {e_minus.min() / unit:.4f} x unit")


def check_monte_carlo():
    rng = np.random.default_rng(362)
    for n, n1, K, g, X0 in ((10, 3, 5, 0.05, 1.0), (12, 1, 8, 0.2, -0.5)):
        n2 = n - n1
        h = np.array([1.0] * n1 + [0.0] * n2)
        c = np.array([1.0] * n1 + [-n1 / n2] * n2)
        R = 400000
        X = np.full(R, X0)
        for _ in range(K):
            order = np.argsort(rng.random((R, n)), axis=1)
            for t in range(n):
                X = X - g * (h[order[:, t]] * X - c[order[:, t]])
        mc = 0.5 * (n1 / n) * X ** 2
        ex = last_iterate_error([g], n, n1, K, X0)[0]
        print(f"[mc] n {n} n1 {n1} K {K} g {g} X0 {X0}: exact {ex:.6e}, simulated {mc.mean():.6e} +- {mc.std() / math.sqrt(R):.1e}")


if __name__ == "__main__":
    check_phi()
    check_mean_formula()
    check_variance()
    check_monte_carlo()
    check_rates()
