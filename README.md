# Reduced-Order Kalman Filtering with Nuisance Variables

Simulations of the paper

> A. Cellier-Devaux, D. Astolfi, V. Andrieu, U. Serres,
> *Reduced-Order Kalman Filtering with Nuisance Variables*, submitted to the European Control Conference (ECC) 2027.
> Extended version: HAL preprint (link to be added).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vincent69410/reduced-kalman-nuisance/blob/main/simulations.ipynb)

Click the badge above to run the simulations in your browser (Google account required, nothing to install).
Then use *Runtime → Run all*. The computation takes one to two minutes.

## Content

| File | Description |
|---|---|
| `sim2d.py` | All the computations: exact error covariances (Proposition 1), tuning of the reduced filter (Remark 3), Monte-Carlo check, figures. |
| `simulations.ipynb` | Colab notebook which runs `sim2d.py` and displays Fig. 1, Fig. 2 and Table I of the paper. |
| `requirements.txt` | Python dependencies (NumPy, SciPy, Matplotlib). |

## What is computed

* **Example 1** (scalar example, Fig. 1): a velocity with linear damping and an unknown acceleration modeled as a random walk.
  The plain truncation (Gamma = 0) has an error which grows without bound, whatever the noise inflation;
  the reduced filter built on Gamma = -1/tau is consistent.
* **Section V** (Fig. 2, Table I): navigation of a turning vehicle in the plane, with an unknown slowly varying force
  and a biased position measurement. Four filters are compared: the full-order Kalman filter (8 states),
  the plain truncation, the truncation with inflated noise, and the reduced filter (4 states) built on the
  time-varying solution Gamma(t) of the nonsymmetric Riccati equation, with the tuning given by the consistency condition
  (Proposition 2).

## Running locally

```bash
pip install -r requirements.txt
python sim2d.py
```

This produces `fig_scalar.pdf/png`, `fig_nav2d.pdf/png`, `results1d.json` and `results2d.json` in the current folder.
The parameters are defined at the top of `sim2d.py` (navigation example) and at the end of the file (scalar example).

## License

MIT License, see `LICENSE`.
