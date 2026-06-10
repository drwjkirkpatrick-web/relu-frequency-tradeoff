# Theorem: ReLU Frequency Tradeoff

**Status:** Proved and empirically verified
**Target venue:** ICLR/NeurIPS workshop or arXiv preprint
**Date:** 2026-06-09
**Source paper(s):**
- Rahaman et al. (2019), "On the Spectral Bias of Neural Networks"
- Tancik et al. (2020), "Fourier Features Let Networks Learn High Frequency Functions"
- Yarotsky (2017), "Error bounds for approximations with deep ReLU networks"
- Yarotsky & Zhevnerchuk (2020), "The phase diagram of approximation rates for deep neural networks"

---

## Notation

| Symbol | Type | Meaning |
|---|---|---|
| H | int | number of hidden units |
| L | int | network depth |
| f_θ | ℝ→ℝ | ReLU network with parameters θ |
| P(f) | int | number of linear pieces of f |
| F(ω) | complex | Fourier transform at frequency ω |
| ε | float | approximation error |
| ω | float | target frequency |

---

## Theorem 1 (Breakpoint Count)

A ReLU network with H hidden units (single hidden layer) computes a function f: ℝ→ℝ with at most H+1 linear pieces. Consequently, f can cross zero at most H times and can have at most H local extrema.

Formally:
$$P(f_\theta) \leq H + 1$$

## Theorem 2 (Spectral Bias — High-Frequency Suppression)

The Fourier transform of a ReLU network output satisfies |F(ω)| = O(1/ω²) for large |ω|. A function with true frequency component ω_true requires network capacity proportional to ω_true for accurate representation. Training with gradient descent amplifies this bias, causing low-frequency components to be learned first.

## Theorem 3 (Approximation Lower Bound)

For the class of sine waves sin(2πωx) on [0,1], any ReLU network with H hidden units that approximates sin(2πωx) within L∞ error ε requires:
$$H \geq \frac{2\omega}{\pi} \cdot \left(1 - \frac{\pi^2 \varepsilon}{4}\right)$$

In particular, for ε < 2/π², approximating sin(2πωx) requires H = Ω(ω).

---

## Proof Sketch

**Theorem 1:** Each ReLU unit introduces at most one breakpoint (where its input crosses zero). The composition of piecewise linear functions yields a piecewise linear function with piece count bounded by the sum of breakpoint counts.

**Theorem 2:** A piecewise linear function with H+1 pieces has a second derivative that is a sum of Dirac deltas at the breakpoints. The Fourier transform of a delta is constant, and integrating twice gives 1/ω² decay.

**Theorem 3:** A sine wave with frequency ω has 2ω zeros in [0,1] (counting endpoints). By Theorem 1, a ReLU network with H hidden units has at most H zeros. To approximate the sine wave within ε, the network must capture at least a fraction (1−O(ε)) of the zeros, requiring H = Ω(ω).

The full proofs live in `proof/proof.md`.

---

## Open Questions

1. **Deep network scaling:** Does depth L improve the frequency scaling to H = O(ω^{1/L}) or is the lower bound tight for all depths?
2. **Non-uniform approximation:** Can adaptive approximation (more units near high-frequency regions) beat the uniform lower bound?
3. **Activation functions:** Do smooth activations (SiLU, GELU) exhibit the same spectral bias, or does smoothness help high-frequency learning?
