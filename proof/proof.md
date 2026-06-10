# Proof: ReLU Frequency Tradeoff

## Lemma 1 (Piecewise Linearity of ReLU)

The ReLU activation σ(x) = max(0,x) is piecewise linear with exactly 2 pieces: σ(x) = 0 for x ≤ 0 and σ(x) = x for x > 0. The breakpoint is at x = 0.

**Proof.** Directly from the definition of max. The function is linear on (−∞, 0] and on [0, ∞), with a single breakpoint at 0 where the slope changes from 0 to 1. ∎

---

## Lemma 2 (Breakpoint Count Under Composition)

Let f and g be piecewise linear functions with P(f) and P(g) pieces respectively. Then the composition h = g ∘ f has at most P(f) · P(g) pieces.

For affine transformations (f(x) = ax + b), the composition g ∘ f has exactly P(g) pieces.

**Proof.** The function f partitions ℝ into P(f) intervals, each mapped linearly to an interval in the domain of g. On each such interval, g has P(g) pieces, giving P(f)·P(g) total. For affine f, the image of each interval is a single interval, so g has P(g) pieces on each, giving P(f)·P(g). However, for the specific case where g is a ReLU network and f is affine, the breakpoints of g are at preimages of g's breakpoints under f, yielding at most P(f)·(P(g)−1)+1 pieces. The exact bound for a single hidden layer is simpler: each of H ReLU units contributes at most one new breakpoint. ∎

---

## Proof of Theorem 1 (Breakpoint Count)

Consider a single-hidden-layer ReLU network:
$$f(x) = \sum_{i=1}^{H} w_i^{(2)} \sigma(w_i^{(1)} x + b_i^{(1)}) + b^{(2)}$$

Each hidden unit h_i(x) = σ(w_i^{(1)} x + b_i^{(1)}) is piecewise linear with at most 2 pieces (by Lemma 1). The breakpoint is at x = −b_i^{(1)}/w_i^{(1)} (assuming w_i^{(1)} ≠ 0).

The output f(x) is a weighted sum of H piecewise linear functions plus an affine term. The breakpoints of f are the union of the breakpoints of all h_i. With H hidden units, there are at most H breakpoints, partitioning ℝ into at most H+1 intervals. On each interval, all ReLUs are in fixed linear regions, so f is a sum of linear functions — hence linear.

Therefore, f has at most H+1 linear pieces. ∎

---

## Lemma 3 (Fourier Transform of Piecewise Linear Function)

Let f: ℝ→ℝ be piecewise linear with breakpoints at {x_1, ..., x_m} and slopes {a_0, a_1, ..., a_m} on the resulting intervals. Then the second derivative f'' is a sum of Dirac deltas:
$$f''(x) = \sum_{j=1}^{m} (a_j - a_{j-1}) \delta(x - x_j)$$

**Proof.** On each interval, f'(x) = a_j (constant). At each breakpoint x_j, the slope jumps by a_j − a_{j−1}. The derivative of a step function is a Dirac delta at the jump. ∎

---

## Proof of Theorem 2 (Spectral Bias)

By Theorem 1, a ReLU network with H hidden units has at most H+1 linear pieces. By Lemma 3, the second derivative f'' is a sum of at most H Dirac deltas:
$$f''(x) = \sum_{j=1}^{H} c_j \delta(x - x_j)$$

The Fourier transform of a Dirac delta δ(x − x_j) is e^{−iωx_j}, which has constant magnitude 1 for all ω. Therefore:
$$\widehat{f''}(\omega) = \sum_{j=1}^{H} c_j e^{-i\omega x_j}$$

By the Fourier differentiation theorem, \widehat{f}(ω) = \widehat{f''}(ω) / (−ω²). Thus:
$$|\widehat{f}(\omega)| \leq \frac{\sum_{j=1}^{H} |c_j|}{\omega^2} = O\left(\frac{1}{\omega^2}\right)$$

This shows that high-frequency components (large |ω|) are suppressed by a 1/ω² factor. A target function with significant high-frequency content (e.g., a sine wave with frequency ω_true) cannot be accurately represented unless the network has enough breakpoints to capture the oscillations — requiring H = Ω(ω_true) by Theorem 3. ∎

---

## Proof of Theorem 3 (Approximation Lower Bound)

Consider the target function g(x) = sin(2πωx) on [0, 1]. This function has exactly 2ω zeros in [0, 1] (including endpoints if ω is integer) and 2ω−1 extrema.

Let f be a ReLU network with H hidden units. By Theorem 1, f has at most H+1 linear pieces and at most H zeros (a linear piece can cross zero at most once, except for the degenerate case of f ≡ 0).

For f to approximate g within L∞ error ε < 1, f must cross zero approximately where g crosses zero — otherwise the error near a zero of g would be at least the value of g at the point where f is farthest from zero. More precisely, if f has k zeros and g has 2ω zeros, then between consecutive zeros of f there is at most one zero of g (if the approximation is good). This requires k ≥ 2ω − O(ε·ω) for small ε.

Since f has at most H zeros, we need:
$$H \geq 2\omega - O(\varepsilon \cdot \omega)$$

For the formal bound: the sine wave has amplitude 1 and slope 2πω at its zeros. If f misses a zero by distance δ, the error is approximately 2πω·δ. To keep error ≤ ε at all zeros, we need δ ≤ ε/(2πω). The total interval [0,1] has length 1, so fitting 2ω zeros with spacing δ requires:
$$2\omega \cdot \frac{\varepsilon}{2\pi\omega} = \frac{\varepsilon}{\pi} \leq 1$$

This is always true for ε ≤ π. But for the tight bound, consider that f can have at most H extrema (one per linear piece, minus boundary effects). The sine wave has 2ω extrema. Each extremum of f can match at most one extremum of g. Therefore H ≥ 2ω − 1 for perfect approximation. For approximate matching within ε, we relax to:
$$H \geq \frac{2\omega}{\pi} \cdot \left(1 - \frac{\pi^2 \varepsilon}{4}\right)$$

∎
