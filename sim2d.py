"""Simulations of the paper "Reduced-Order Kalman Filtering with Nuisance Variables" (ECC 2027 submission).

Section V: navigation of a turning vehicle in the plane (anisotropic linear damping).

  x = (v, p) in R^4,  z = (d, b_g) in R^4
  dv = -D(t) v - d + w,   dp = v,   dd = n_d,   db_g = -b_g/tau_g + n_g,   y = p + b_g + r
  D(t) = R_psi diag(1/tau_l, 1/tau_t) R_psi^T,  psi: 0 on [0,100), U-turn on [100,120], pi afterwards.

Exact error covariances (Proposition 1) for: the full-order KF, the plain truncation (Gamma=0, Q=Qx),
the truncation Gamma=0 with inflated (Q,R), and the reduced filter with Gamma(t) and the certified constant tuning.
Monte-Carlo check of Proposition 1 on the last two.  Everything in continuous time, Euler with step dt
(matrices frozen on each step).  Example 1 (scalar example, Fig. 1) is treated at the end of the file.

Outputs: fig_nav2d.{pdf,png}, fig_scalar.{pdf,png}, results2d.json, results1d.json.
Runtime: about one minute (300 Monte-Carlo runs).
"""
import numpy as np, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"full": "#444444", "naive": "#d62728", "infl": "#ff7f0e", "good": "#1f77b4"}

tau_l, tau_t, tau_g = 5.0, 1.0, 300.0
qw, qd, qg, R = 1e-2, 1e-4, 2 * 1.0**2 / tau_g, 1.0     # (m/s^2)^2/Hz, (m/s^3)^2/Hz, m^2/Hz, m^2 s
T, dt = 400.0, 0.01
N = int(round(T / dt)); t = np.arange(N + 1) * dt
t1, t2 = 100.0, 120.0
I2, Z2 = np.eye(2), np.zeros((2, 2))

def psi(s): return 0.0 if s < t1 else (np.pi * (s - t1) / (t2 - t1) if s < t2 else np.pi)
def Rot(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def Dm(s): Rp = Rot(psi(s)); return Rp @ np.diag([1 / tau_l, 1 / tau_t]) @ Rp.T

Axz = np.block([[-I2, Z2], [Z2, Z2]]); Az = np.block([[Z2, Z2], [Z2, -I2 / tau_g]])
Cx = np.hstack([Z2, I2]); Cz = np.hstack([Z2, I2])
Qx = np.block([[qw * I2, Z2], [Z2, Z2]]); Qz = np.block([[qd * I2, Z2], [Z2, qg * I2]])
Rm = R * I2
def Ax(s): return np.block([[-Dm(s), Z2], [I2, Z2]])
D0 = Dm(0.0)

# ---- Gamma(t) by the linear flow: Gamma_dv = -D0 X^{-1}, Xdot = -D(t) X + D0 ------------
Xv = I2.copy(); Gdv = np.empty((N + 1, 2, 2)); Gdv[0] = -D0
for k in range(N):
    Xv = Xv + dt * (-Dm(k * dt) @ Xv + D0)
    Gdv[k + 1] = -D0 @ np.linalg.inv(Xv)
# check the DRE residual
res = max(np.abs((Gdv[k + 1] - Gdv[k]) / dt - (Gdv[k] @ Dm(k * dt) + Gdv[k] @ Gdv[k])).max() for k in range(0, N, 50))
dev = max(np.abs(Gdv[k] + Dm(k * dt)).max() for k in range(N + 1))
alpha = -max(np.max(np.linalg.eigvalsh((Gdv[k] + Gdv[k].T) / 2)) for k in range(N + 1))
print(f"DRE residual {res:.1e}; max |Gamma_dv + D| = {dev:.3f}; uniform decay rate of A_zeta >= {alpha:.3f}")

def Gam(k, on): return np.block([[Gdv[k], Z2], [Z2, Z2]]) if on else np.zeros((4, 4))

# ---- covariance propagation (Prop. 1) --------------------------------------------------
P0 = np.diag([1.0, 1.0, 1.0, 1.0]); Pd0 = (0.05)**2 * I2; Pg0 = tau_g / 2 * qg * I2
def run(gamma_on, Q, Rt, mc=0, seed=0):
    S = P0.copy()
    G = Gam(0, gamma_on)
    Pzz0 = np.block([[Pd0, Z2], [Z2, Pg0]])
    Pi = np.block([[P0 + np.zeros((4, 4)), -P0 @ G.T], [-G @ P0, Pzz0 + G @ P0 @ G.T]])   # (e, zeta), xhat0 = 0
    out = np.zeros((N + 1, 4)); cross = np.zeros((N + 1, 4, 4))
    rng = np.random.default_rng(seed)
    if mc:
        x = rng.multivariate_normal(np.zeros(4), P0, mc)       # true x, xhat = 0
        d = rng.multivariate_normal(np.zeros(2), Pd0, mc); bg = rng.multivariate_normal(np.zeros(2), Pg0, mc)
        xh = np.zeros((mc, 4)); msq = np.zeros(N + 1)
    for k in range(N + 1):
        s = k * dt; G = Gam(k, gamma_on)
        Ag = Ax(s) + Axz @ G; Cg = Cx + Cz @ G; Azeta = Az - G @ Axz
        K = S @ Cg.T @ np.linalg.inv(Rt)
        A = np.block([[Ag - K @ Cg, Axz - K @ Cz], [np.zeros((4, 4)), Azeta]])
        Gb = np.block([[np.eye(4), np.zeros((4, 4)), -K], [-G, np.eye(4), np.zeros((4, 2))]])
        W = Gb @ np.block([[Qx, np.zeros((4, 6))], [np.zeros((4, 4)), Qz, np.zeros((4, 2))], [np.zeros((2, 8)), Rm]]) @ Gb.T
        out[k] = [np.sqrt(np.trace(Pi[:2, :2])), np.sqrt(np.trace(S[:2, :2])), np.sqrt(np.trace(Pi[2:4, 2:4])), np.sqrt(np.trace(S[2:4, 2:4]))]
        Mk = Axz - K @ Cz; X = Mk @ Pi[4:, :4]; cross[k] = X + X.T
        if mc:
            msq[k] = np.mean(np.sum((x[:, :2] - xh[:, :2])**2, axis=1))
            y = x[:, 2:4] + bg + rng.normal(0, np.sqrt(R / dt), (mc, 2))
            xh = xh + dt * (xh @ Ag.T + (y - xh @ Cg.T) @ K.T)
            x = x + dt * (x @ Ax(s).T + np.hstack([-d, np.zeros((mc, 2))]) + np.hstack([rng.normal(0, np.sqrt(qw / dt), (mc, 2)), np.zeros((mc, 2))]))
            d = d + rng.normal(0, np.sqrt(qd * dt), (mc, 2)); bg = bg + dt * (-bg / tau_g) + rng.normal(0, np.sqrt(qg * dt), (mc, 2))
        if k < N:
            Pi = Pi + dt * (A @ Pi + Pi @ A.T + W)
            S = S + dt * (Ag @ S + S @ Ag.T + Q - S @ Cg.T @ np.linalg.inv(Rt) @ Cg @ S)
    run.cross = cross
    return (out, np.sqrt(msq)) if mc else out

def run_full():
    P = np.block([[P0, np.zeros((4, 4))], [np.zeros((4, 4)), np.block([[Pd0, Z2], [Z2, Pg0]])]]); out = np.zeros((N + 1, 4))
    H = np.hstack([Cx, Cz]); Om = np.block([[Qx, np.zeros((4, 4))], [np.zeros((4, 4)), Qz]])
    for k in range(N + 1):
        F = np.block([[Ax(k * dt), Axz], [np.zeros((4, 4)), Az]])
        out[k] = [np.sqrt(np.trace(P[:2, :2]))] * 2 + [np.sqrt(np.trace(P[2:4, 2:4]))] * 2
        if k < N: P = P + dt * (F @ P + P @ F.T + Om - P @ H.T @ np.linalg.inv(Rm) @ H @ P)
    return out

# ---- tuning of the reduced filter Gamma(t): constant (Q, R) from the certificate in steady state ----
pz_d = np.array([(qw / tau_l**2 + qd) * tau_l / 2, (qw / tau_t**2 + qd) * tau_t / 2])   # stationary Pi_zeta_d, body frame
pz_g = tau_g / 2 * qg
infl_v = pz_d.max()                                   # heading-independent bound of A_xz Pi A_xz^T
print(f"stationary residual std: force {np.sqrt(pz_d)} m/s^2 (body axes), position bias {np.sqrt(pz_g):.2f} m")
Ag_ss = np.block([[Z2, Z2], [I2, Z2]]); Cg_ss = Cx
def tuned(g1, g2):
    Rt = R + pz_g / g2
    Q = Qx.copy()
    for _ in range(60):                               # fixed point: Q = Qx + (g1+g2) S_inf(Q) + infl/g1
        S = solve_continuous_are(Ag_ss.T, Cg_ss.T, Q, Rt * I2)
        Q = Qx + (g1 + g2) * S + np.block([[infl_v / g1 * I2, Z2], [Z2, Z2]])
    return Q, Rt * I2
def true_ss(Q, Rt):
    S = solve_continuous_are(Ag_ss.T, Cg_ss.T, Q, Rt); K = S @ Cg_ss.T @ np.linalg.inv(Rt)
    G = np.block([[-D0, Z2], [Z2, Z2]]); Azeta = Az - G @ Axz
    A = np.block([[Ag_ss - K @ Cg_ss, Axz - K @ Cz], [np.zeros((4, 4)), Azeta]])
    Gb = np.block([[np.eye(4), np.zeros((4, 4)), -K], [-G, np.eye(4), np.zeros((4, 2))]])
    W = Gb @ np.block([[Qx, np.zeros((4, 6))], [np.zeros((4, 4)), Qz, np.zeros((4, 2))], [np.zeros((2, 8)), Rm]]) @ Gb.T
    from scipy.linalg import solve_continuous_lyapunov
    Pi = solve_continuous_lyapunov(A, -W); return np.trace(Pi[:2, :2])
grid = []
for g1 in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    for g2 in [0.005, 0.02, 0.05, 0.1, 0.3, 1.0]:
        Q, Rt = tuned(g1, g2); c = true_ss(Q, Rt)
        S = solve_continuous_are(Ag_ss.T, Cg_ss.T, Q, Rt)
        grid.append((c, np.trace(S), g1, g2, Q, Rt))
cmin = min(g[0] for g in grid)
# rule: smallest announced variance among the tunings whose true variance is within 20 % of the minimum
_, _, g1, g2, Qg, Rg = min((g for g in grid if g[0] <= 1.2 * cmin), key=lambda g: g[1])
print(f"certified tuning: gamma1={g1}, gamma2={g2}, Q_v={Qg[0,0]:.4g}, Q_p={Qg[2,2]:.4g}, R={Rg[0,0]:.3g}  (qw={qw}, R={R})")

# inflated Gamma=0: same (Q, R) as the certified filter, for comparison
filt = {}
filt["full"] = run_full()
filt["naive"] = run(False, Qx, Rm)
filt["infl"] = run(False, Qg, Rg)
filt["good"], mc_good = run(True, Qg, Rg, mc=300, seed=1)
_, mc_infl = run(False, Qg, Rg, mc=300, seed=2)

# ---- numbers for the table --------------------------------------------------------------
seg = {"cruise": (t > 60) & (t < 100), "turn": (t >= 100) & (t <= 140), "end": (t > 380)}
rows = {}
for k, o in filt.items():
    rows[k] = {s: (o[m, 0].max(), o[m, 1].max(), o[m, 2].max(), o[m, 3].max()) for s, m in seg.items()}
    cons_v = np.all(o[t > 1, 1] >= o[t > 1, 0] * (1 - 1e-6)); cons_p = np.all(o[t > 1, 3] >= o[t > 1, 2] * (1 - 1e-6))
    rows[k]["consistent"] = bool(cons_v and cons_p)
    print(f"{k:6s}", {s: tuple(round(x, 2) for x in v) for s, v in rows[k].items() if s != 'consistent'}, "consistent:", rows[k]["consistent"])
print("MC check (max rel. dev. for t>50): good", np.abs(mc_good[5000:] / filt['good'][5000:, 0] - 1).max().round(3),
      " infl", np.abs(mc_infl[5000:] / filt['infl'][5000:, 0] - 1).max().round(3))
json.dump({"params": dict(tau_l=tau_l, tau_t=tau_t, tau_g=tau_g, qw=qw, qd=qd, qg=qg, R=R, g1=g1, g2=g2, Qv=Qg[0, 0], Qp=Qg[2, 2], Rt=Rg[0, 0],
                          dev=dev, alpha=alpha, pz_d=pz_d.tolist(), pz_g=pz_g),
           "rows": rows}, open("results2d.json", "w"), indent=1, default=float)

# ---- figure ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.0))
lab = {"full": "full KF (8 states)", "naive": r"$\Gamma=0$, $\mathcal{Q}=Q_x$", "infl": r"$\Gamma=0$, inflated $\mathcal{Q}$", "good": r"$\Gamma(t)$, certified $\mathcal{Q}$"}
for ax, (i, j, yl, ym) in zip(axs, [(0, 1, "velocity error std (m/s)", 1.6), (2, 3, "position error std (m)", 7)]):
    for k in ["full", "naive", "infl", "good"]:
        ax.plot(t, filt[k][:, i], color=C[k], lw=1.2 if k != "full" else 0.9, label=lab[k])
        if k != "full": ax.plot(t, filt[k][:, j], color=C[k], lw=0.8, ls="--")
    sub = slice(0, N + 1, 1000)
    if i == 0:
        ax.plot(t[sub], mc_good[sub], "o", color=C["good"], ms=1.8)
        ax.plot(t[sub], mc_infl[sub], "o", color=C["infl"], ms=1.8)
    ax.axvspan(t1, t2, color="gray", alpha=0.15)
    ax.set_xlabel("time (s)"); ax.set_ylabel(yl); ax.set_ylim(0, ym); ax.set_xlim(0, T)
axs[0].text((t1 + t2) / 2, 1.45, "U-turn", ha="center", fontsize=5.5, color="gray")
axs[0].legend(fontsize=5, loc="upper left", handlelength=1.5, borderpad=0.3, labelspacing=0.25)
fig.subplots_adjust(wspace=0.35)
fig.savefig("fig_nav2d.pdf"); fig.savefig("fig_nav2d.png", dpi=200)

# ---- scalar example figure (Example 1), same style, tau = 5 s --------------------------
tau, qw1, qb1, R1 = 5.0, 1e-2, 1e-4, 0.05
Ts, dts = 600.0, 0.01; Ns = int(Ts / dts); ts = np.arange(Ns + 1) * dts
def scalar(G, Q, Rt):
    Ag, Cg, Azeta = -1 / tau - G, 1.0, -G
    Om_nu = G * qw1 * G + qb1
    S, Pi = 1.0, np.array([[1.0, -G], [-G, 0.01 + G * G]]); outv = np.zeros((Ns + 1, 2))
    for k in range(Ns + 1):
        K = S * Cg / Rt
        A = np.array([[Ag - K * Cg, -1.0], [0.0, Azeta]]); Gb = np.array([[1, 0, -K], [-G, 1, 0]])
        W = Gb @ np.diag([qw1, qb1, R1]) @ Gb.T
        outv[k] = [np.sqrt(Pi[0, 0]), np.sqrt(S)]
        Pi = Pi + dts * (A @ Pi + Pi @ A.T + W); S = S + dts * (2 * Ag * S + Q - S * S * Cg * Cg / Rt)
    return outv
def scalar_full():
    F = np.array([[-1 / tau, -1.0], [0.0, 0.0]]); H = np.array([[1.0, 0.0]]); P = np.diag([1.0, 0.01]); o = np.zeros(Ns + 1)
    for k in range(Ns + 1):
        o[k] = np.sqrt(P[0, 0]); P = P + dts * (F @ P + P @ F.T + np.diag([qw1, qb1]) - P @ H.T @ H @ P / R1)
    return o
pzz1 = tau / 2 * (qw1 / tau**2 + qb1)
def tuned1(g):
    Q = qw1
    for _ in range(100): S = np.sqrt(Q * R1); Q = qw1 + g * S + pzz1 / g
    return Q
# exact steady-state true variance for Gamma=-1/tau, gain K: Lyapunov 2x2 closed form via numeric
from scipy.linalg import solve_continuous_lyapunov
def true1(Q):
    K = np.sqrt(Q * R1) / R1; A = np.array([[-K, -1.0], [0.0, -1 / tau]]); Gb = np.array([[1, 0, -K], [1 / tau, 1, 0]])
    W = Gb @ np.diag([qw1, qb1, R1]) @ Gb.T; return solve_continuous_lyapunov(A, -W)[0, 0]
gbest = min(np.geomspace(0.01, 3, 60), key=lambda g: true1(tuned1(g))); Q1 = tuned1(gbest)
print(f"scalar example: gamma={gbest:.3f}, Q={Q1:.4g}, true steady std={np.sqrt(true1(Q1)):.3f}, announced={np.sqrt(np.sqrt(Q1*R1)):.3f}")
s_full = scalar_full(); s_naive = scalar(0.0, qw1, R1); s_good = scalar(-1 / tau, Q1, R1)
fig, ax = plt.subplots(figsize=(1.95, 1.1))
ax.plot(ts, s_full, color=C["full"], lw=0.9, label="full KF")
ax.plot(ts, s_naive[:, 0], color=C["naive"], lw=1.5, label=r"$\Gamma=0$: true"); ax.plot(ts, s_naive[:, 1], color=C["naive"], lw=1.0, ls="--", label=r"$\Gamma=0$: announced")
ax.plot(ts, s_good[:, 0], color=C["good"], lw=1.5, label=r"$\Gamma=-1/\tau$: true"); ax.plot(ts, s_good[:, 1], color=C["good"], lw=1.0, ls="--", label=r"$\Gamma=-1/\tau$: announced")
ax.set_xlabel("time (s)"); ax.set_ylabel("velocity error std (m/s)"); ax.set_ylim(0, 0.75); ax.set_xlim(0, Ts); ax.legend(fontsize=4.8, loc="upper left", ncol=1, handlelength=1.5, borderpad=0.3, labelspacing=0.25)
fig.savefig("fig_scalar.pdf"); fig.savefig("fig_scalar.png", dpi=200)
json.dump(dict(tau=tau, qw=qw1, qb=qb1, R=R1, gamma=gbest, Q=Q1, true_ss=float(np.sqrt(true1(Q1))), ann_ss=float(np.sqrt(np.sqrt(Q1 * R1))),
               naive_true_600=float(s_naive[-1, 0]), naive_ann_600=float(s_naive[-1, 1]), full_ss=float(s_full[-1])), open("results1d.json", "w"), indent=1)
print("done")
