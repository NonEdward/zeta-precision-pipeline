# zeta-precision-pipeline

Precision-validation pipeline for a WebGL2 Hardy-Z renderer that locates
Riemann zeros on the critical line. This repo holds the CPU-side numerical
reference and the headless regression workflow.

## What it does

- `zeta_unfold.py` computes the first N non-trivial zeros with mpmath,
  unfolds the spacing using the Riemann-von Mangoldt counting function,
  and compares against the GUE Wigner surmise via a KS test.
- The WebGL2 shader side uses double-single (two-sum / Dekker split) to
  emulate fp64 where fp32 drift would otherwise create false sign flips.

## Reproduce

    pip install numpy scipy mpmath
    python zeta_unfold.py

With N=500: mean spacing ~= 0.9997, KS p ~= 0.165, GUE-consistent.

## CI

`.github/workflows/zeta-regression.yml` runs the KS regression on push and
on a weekday schedule (08:00 UTC), uploads the result as an artifact.
The intended Devin Bot loop: run this regression, run a SwiftShader
headless WebGL2 smoke test, diff pixels with a noise-floor tolerance,
and post a morning digest.
