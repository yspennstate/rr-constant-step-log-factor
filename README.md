# The logarithmic factor in the tuned constant-step rate of random reshuffling

Yitzchak Shmalo

For strongly convex finite sums, the known last-iterate upper bounds for random reshuffling with a constant step
chosen from the number of epochs carry a logarithmic factor, and the known lower bounds carry none. This paper shows
that for constant steps the factor cannot be removed.

The instance is one-dimensional: n1 components (L/2)x^2 - sx and n - n1 components s(n1/n2)x, so that
F(x) = (mu/2)x^2 with mu = L n1/n, G = s at the minimiser and kappa = n/n1. With T = nK steps, initial distance D
and Lambda = ln(mu T D/G):

- every constant step gamma <= 1/L leaves E F(x_K) - F* >= G^2 (Lambda - ln Lambda)^2 / (512 mu T^2) once
  K >= n1(Lambda - ln Lambda)/2;
- the step gamma = Lambda/(mu T) gives at most G^2 Lambda^2 (1 + o(1)) / (8 mu T^2) once K is large compared
  with n1 Lambda.

So the tuned constant-step error is Theta(G^2 ln^2(mu T D/G) / (mu T^2)). With n1 = 1 the lower bound has the n and K
dependence of the known constant-step lower bound sigma*^2/(mu n K^2), multiplied by ln^2 K. The proof computes one
epoch exactly: the stationary mean of the epoch-end iterate is gamma G phi/2 with phi a weighted mean between
(1 - gamma L)^(n1 - 1) and 1, and the epoch variance reduces to the variance of a Mann-Whitney statistic.

## Contents

- `paper/rr_log_factor.tex`, `paper/rr_log_factor.pdf` - the paper.
- `code/rr_check.py` - the numerical checks of Section 5: an exact dynamic programme for the first two moments of one
  epoch, checks of Lemmas 2.1 and 2.2, a Monte Carlo comparison, and the tuned rate on six instances against both
  bounds of Theorem 1.1. Requires Python 3 and NumPy; runs in a few minutes.
- `code/check_output.txt` - the output of `python rr_check.py`.
- `code/minus_start.py` - for the start x0 = -D, the step at which the mean iterate vanishes and the exact minimum of
  the error next to it, on eight instances (the six above and two more with K = 1000). Requires NumPy and SciPy.
- `code/minus_start_output.txt` - its output.

## License

The paper is released under CC BY 4.0 and the code under the MIT License.
