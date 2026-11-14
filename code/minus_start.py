# Start x0 = -D: the step at which E X_K = 0 and the exact minimum of E F(x_K) - F* next to it (L = s = D = 1).
# One-epoch moments by a dynamic programme over (stiff steps seen, flat steps seen), counted from the end of the epoch.
import math
import numpy as np
from scipy.optimize import minimize_scalar

def epoch_moments(n, n1, g):
    """E B and E B^2 for one epoch by a backward DP over (stiff seen, flat seen); B = sum c_t (r^q_t - 1)."""
    n2 = n - n1; lr = math.log1p(-g) if g < 1 else -np.inf
    em1 = [math.expm1(q*lr) if g < 1 else (0.0 if q == 0 else -1.0) for q in range(n1+1)]  # r^q - 1
    cs, cf = g, -g*n1/n2
    P = np.zeros((n1+1, n2+1)); M1 = np.zeros_like(P); M2 = np.zeros_like(P); P[0, 0] = 1.0
    for tot in range(n):            # states with j+i = tot
        for j in range(max(0, tot-n2), min(n1, tot)+1):
            i = tot - j; p = P[j, i]
            if p == 0: continue
            left = n - tot; ps = (n1-j)/left; pf = (n2-i)/left
            for (dj, di, pr, c) in ((1, 0, ps, cs), (0, 1, pf, cf)):
                if pr == 0: continue
                d = c*em1[j]                       # q_t = number of stiff steps after t = j
                P[j+dj, i+di] += p*pr
                M1[j+dj, i+di] += pr*(M1[j, i] + p*d)
                M2[j+dj, i+di] += pr*(M2[j, i] + 2*d*M1[j, i] + p*d*d)
    return M1[n1, n2], M2[n1, n2]

def phi(r, n1):
    k = np.arange(n1); w = r**k
    return 2/(n1+1)*np.sum((k+1)*w)/np.sum(w)

def err(n, n1, K, g, x0=1.0):
    EB, EB2 = epoch_moments(n, n1, g); VB = EB2 - EB**2
    A = (1-g)**n1; AK = A**K
    m = EB/(-math.expm1(n1*math.log1p(-g))) if g < 1 else EB
    EX = AK*x0 + (1-AK)*m
    VX = VB*(-math.expm1(2*K*n1*math.log1p(-g)))/(-math.expm1(2*n1*math.log1p(-g))) if g < 1 else VB
    mu = n1/n
    return mu/2*(EX**2 + VX), EX, VX, VB, m

from scipy.optimize import brentq

for (n, n1, K) in [(40, 1, 1000), (40, 1, 10000), (100, 10, 1000), (100, 10, 10000),
                   (100, 50, 1000), (100, 50, 10000), (1000, 100, 1000), (1000, 100, 10000)]:
    mu = n1 / n; T = n * K; Lam = math.log(mu * T); unit = Lam ** 2 / (8 * mu * T ** 2)

    def EX(g):
        A = (1 - g) ** n1; AK = A ** K
        return -AK + (1 - AK) * g / 2 * phi(1 - g, n1)

    gs = np.geomspace(1e-3 * Lam / (mu * T), 1, 4000); v = [EX(g) for g in gs]
    roots = [brentq(EX, gs[i], gs[i + 1], xtol=1e-16) for i in range(len(gs) - 1) if v[i] * v[i + 1] < 0]
    best = (np.inf, None)
    for g0 in roots:
        f = lambda lg: err(n, n1, K, math.exp(lg), x0=-1.0)[0]
        r = minimize_scalar(f, bounds=(math.log(g0) - 0.05, math.log(g0) + 0.05), method="bounded",
                            options={"xatol": 1e-13})
        best = min(best, (r.fun, math.exp(r.x)))
    at_root = err(n, n1, K, roots[0], x0=-1.0)[0] / unit if roots else float("nan")
    print("(%d,%d,%d) roots %s  min/unit %.4e at g*mu*T/Lam %.4f  (error at root %.4e)"
          % (n, n1, K, ["%.6g" % (g * mu * T / Lam) for g in roots], best[0] / unit,
             best[1] * mu * T / Lam if best[1] else float("nan"), at_root), flush=True)
