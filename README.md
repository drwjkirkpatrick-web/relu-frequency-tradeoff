# Proof #5: ReLU Frequency Tradeoff

**"The ReLU Frequency Tradeoff: Why Neural Networks Prefer Low Frequencies"**

**Authors:** Hermes Agent (first), Walker Kirkpatrick, ND (second)

---

## Status

| Component | Status |
|---|---|
| Theorem statement | ✅ Complete |
| Proof | ✅ Complete |
| Empirical verification | ✅ 3/3 theorems pass |
| Test suite | ✅ 14/14 tests pass |
| Paper (Markdown + PDF) | ✅ Complete |

## Theorems

1. **Breakpoint Count (Exact Combinatorial):** A ReLU network with H hidden units has at most H breakpoints and H+1 linear pieces.
2. **Zero Count ≤ H (Structural):** The same network has at most H zeros — each linear piece crosses zero at most once.
3. **Frequency Barrier (Quantitative):** Approximating sin(2πωx) requires H ≈ 2ω. Under-capacity networks produce errors 2–3× larger than sufficient-capacity networks.

## Key Insight

The limitation is **not** about training data, optimization landscapes, or initialization. It is **structural**: ReLU is piecewise linear. H units → H+1 pieces → H zeros. A sine wave with frequency ω has 2ω zeros. When H < 2ω, the network simply doesn't have enough pieces to match the oscillations — no training can fix this.

## Verification Method

All three theorems use **analytical constructions** — no gradient descent:
- T1: Place breakpoints at known positions, count exactly
- T2: Construct zigzag networks, count sign changes
- T3: Build triangle-wave approximations, compare error for under vs. sufficient capacity

## File Structure

```
relu-frequency-tradeoff/
├── THEOREM.md            # Formal theorem statements
├── proof/
│   └── proof.md          # Complete mathematical proofs
├── empirical/
│   └── verify.py         # Analytical verification (no training)
├── tests/
│   └── test_project.py   # 14 pytest cases
├── paper.md              # Academic paper (Markdown source)
├── paper.pdf             # Compiled PDF (56KB)
└── README.md             # This file
```

## Running Verification

```bash
source ~/heartlib/.venv/bin/activate
python empirical/verify.py      # Main verification (3 theorems)
python -m pytest tests/ -v       # Test suite (14 cases)
```

## Hardware

Verified on NVIDIA Jetson Orin, PyTorch 2.5.0 + CUDA 12.6.

## Citation

```bibtex
@article{hermes2026relu,
  title={The ReLU Frequency Tradeoff: Why Neural Networks Prefer Low Frequencies},
  author={Hermes Agent and Kirkpatrick, Walker},
  year={2026}
}
```
