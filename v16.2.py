# -*- coding: utf-8 -*-
""" 
Fair comparison under the same number of quadrature nodes:

  1) Optimal MVDR (oracle)
  2) PGLQ-Full [12]    : piecewise Gauss-Legendre full-matrix reconstruction
  3) GRDR-CMR [25]     : Gaussian random dimensionality reduction + CMR
  4) Toeplitz-SSBF [26]: Toeplitz / structured-subspace beamforming
  5) Proposed SCC-BF   : segmented Clenshaw-Curtis beamforming

All reconstruction-type methods use the SAME total number of quadrature nodes:
    B = 2 * M_each_shared * Q_each_shared

Newly added in this version:
  A) SINR-Runtime trade-off figure
  B) Computational-cost vs performance comparison table
     (using FLOPs proxy / complexity proxy, not exact hardware FLOPs)

Dependencies:
  pip install numpy scipy matplotlib
"""

import os
import csv
import time
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.integrate import newton_cotes

# ============================================================
# Color palette
# ============================================================
COLORS = {
    "optimal":  "#2F2F2F",   # deep gray
    "pglq":     "#4C78A8",   # academic blue
    "grdr":     "#59A14F",   # green
    "toep":     "#B07AA1",   # purple
    "proposed": "#D62728",   # academic red
    "dense":    "#8C8C8C",   # optional dense baseline
    "grid":     "#C9C9C9",
    "spine":    "#333333",
    "ref":      "#666666",
}

METHOD_META = {
    "optimal": {
        "label": "Optimal",
        "color": COLORS["optimal"],
        "marker": "D",
        "linestyle": "-.",
    },
    "dense": {
        "label": "Dense IR",
        "color": COLORS["dense"],
        "marker": "X",
        "linestyle": "--",
    },
    "pglq": {
        "label": "PGLQ-Full [12]",
        "color": COLORS["pglq"],
        "marker": "s",
        "linestyle": "--",
    },
    "grdr": {
        "label": "GRDR-CMR [25]",
        "color": COLORS["grdr"],
        "marker": "o",
        "linestyle": ":",
    },
    "toep": {
        "label": "Toeplitz-SSBF [26]",
        "color": COLORS["toep"],
        "marker": "P",
        "linestyle": (0, (5, 2)),
    },
    "prop": {
        "label": "Proposed SCC-BF",
        "color": COLORS["proposed"],
        "marker": "^",
        "linestyle": "-",
    },
}

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 15,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "axes.linewidth": 1.0,
    "lines.linewidth": 2.25,
    "lines.markersize": 5.5,

    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "figure.dpi": 160,
})

# ============================================================
# Helpers
# ============================================================
def db(x, eps=1e-12):
    return 10.0 * np.log10(np.maximum(np.real(x), eps))


def steering_u(u, N):
    """ULA steering vector in u = sin(theta), d = lambda/2"""
    n = np.arange(N)
    return np.exp(1j * np.pi * n * u)


def steering_deg(theta_deg, N):
    u = np.sin(np.deg2rad(theta_deg))
    return steering_u(u, N)

def auto_diag_loading(R, alpha=1e-2):
    """
    Scale-aware diagonal loading.
    """
    N = R.shape[0]
    return alpha * np.real(np.trace(R)) / max(N, 1)


def auto_toep_L(N, K=None):
    """
    Separate rank choice for Toeplitz-SSBF.
    Usually Toeplitz needs a bit larger L than SCC surrogate.
    """
    if K is None:
        return max(8, min(24, N // 8 + 4))
    return max(8, min(24, min(N, K) // 8 + 4))


def project_psd(R, floor_ratio=1e-3):
    """
    Hermitian PSD projection by eigenvalue flooring.
    """
    Rh = 0.5 * (R + R.conj().T)
    vals, vecs = np.linalg.eigh(Rh)
    floor = floor_ratio * np.real(np.trace(Rh)) / Rh.shape[0]
    vals = np.maximum(np.real(vals), floor)
    return vecs @ np.diag(vals) @ vecs.conj().T


def toeplitz_psd_project(R, floor_ratio=1e-3, iters=2):
    """
    Practical projection to 'approximately Toeplitz + PSD':
    alternate Toeplitz averaging and PSD projection.
    """
    Rt, _ = toeplitz_project_from_matrix(R)

    for _ in range(iters):
        Rt = project_psd(Rt, floor_ratio=floor_ratio)
        Rt, _ = toeplitz_project_from_matrix(Rt)

    Rt = 0.5 * (Rt + Rt.conj().T)
    return Rt


def simulate_snapshots(
    N=32,
    K=64,
    theta_s_deg=0.0,
    theta_j_deg=(-42.0, 56.0),
    snr_db=5.0,
    inr_db=35.0,
    noise_var=1.0,
    seed=0
):
    rng = np.random.default_rng(seed)

    a_s = steering_deg(theta_s_deg, N)
    A_j = np.column_stack([steering_deg(t, N) for t in theta_j_deg])

    sigma_s2 = noise_var * 10 ** (snr_db / 10.0)
    sigma_j2 = noise_var * 10 ** (inr_db / 10.0)

    s = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2.0)
    s *= np.sqrt(sigma_s2)

    J = (rng.standard_normal((len(theta_j_deg), K)) +
         1j * rng.standard_normal((len(theta_j_deg), K))) / np.sqrt(2.0)
    J *= np.sqrt(sigma_j2)

    Nn = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))) / np.sqrt(2.0)
    Nn *= np.sqrt(noise_var)

    X = np.outer(a_s, s) + A_j @ J + Nn
    Rhat = (X @ X.conj().T) / K

    Rin = noise_var * np.eye(N, dtype=np.complex128)
    for t in theta_j_deg:
        a = steering_deg(t, N)
        Rin += sigma_j2 * np.outer(a, a.conj())

    return X, Rhat, a_s, Rin, sigma_s2


def output_sinr(w, a_s, Rin, sigma_s2):
    num = sigma_s2 * np.abs(w.conj().T @ a_s) ** 2
    den = np.real(w.conj().T @ Rin @ w)
    return float(np.real(num / np.maximum(den, 1e-12)))


def make_sidelobe_u_intervals(theta0_deg=0.0, main_half_width_deg=4.0):
    u0_left = np.sin(np.deg2rad(theta0_deg - main_half_width_deg))
    u0_right = np.sin(np.deg2rad(theta0_deg + main_half_width_deg))
    return [(-1.0, u0_left), (u0_right, 1.0)]


# ============================================================
# Quadrature helpers
# ============================================================
def gauss_legendre_on_interval(a, b, Q=3):
    z, w = np.polynomial.legendre.leggauss(Q)
    x = 0.5 * (a + b) + 0.5 * (b - a) * z
    alpha = 0.5 * (b - a) * w
    return x, alpha


def clenshaw_curtis_on_interval(a, b, Q=17):
    if Q < 2:
        raise ValueError("Q must be >= 2 for Clenshaw-Curtis quadrature.")

    k = np.arange(Q)
    x = np.cos(np.pi * k / (Q - 1))
    x = x[::-1]

    V = np.vander(x, N=Q, increasing=True)
    rhs = np.zeros(Q)
    for m in range(Q):
        rhs[m] = 2.0 / (m + 1) if (m % 2 == 0) else 0.0

    w_std = np.linalg.solve(V.T, rhs)

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    u = c + h * x
    alpha = h * w_std

    return u, alpha


def partition_intervals(intervals, M_each=12, Q=17, rule="cc"):
    u_nodes = []
    alpha_nodes = []

    for (ua, ub) in intervals:
        bounds = np.linspace(ua, ub, M_each + 1)
        for i in range(M_each):
            a = bounds[i]
            b = bounds[i + 1]

            if rule.lower() == "gl":
                x, w = gauss_legendre_on_interval(a, b, Q=Q)
            elif rule.lower() == "cc":
                x, w = clenshaw_curtis_on_interval(a, b, Q=Q)
            else:
                raise ValueError(f"Unknown quadrature rule: {rule}")

            u_nodes.append(x)
            alpha_nodes.append(w)

    return np.concatenate(u_nodes), np.concatenate(alpha_nodes)


# ============================================================
# Method 0: Oracle / Optimal MVDR
# ============================================================
def beamformer_optimal_mvdr(Rin, a0, reg=1e-8):
    x = sla.solve(Rin + reg * np.eye(Rin.shape[0]), a0, assume_a='her')
    w = x / (a0.conj().T @ x)
    return w


# ============================================================
# Dense integral reconstruction (optional, not plotted in main figures)
# ============================================================
def beamformer_dense_ir(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    P_each=300,
    reg=1e-6
):
    N = Rhat.shape[0]
    Rinv = sla.inv(Rhat + reg * np.eye(N))

    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)
    Rrec = np.zeros((N, N), dtype=np.complex128)

    for (ua, ub) in intervals:
        eps = 1e-5
        uu = np.linspace(ua + eps, ub - eps, P_each)
        du = (ub - ua - 2 * eps) / max(P_each - 1, 1)

        for u in uu:
            a = steering_u(u, N)
            q = np.real(a.conj().T @ Rinv @ a)
            q = max(q, 1e-8)
            jac = 1.0 / np.sqrt(max(1.0 - u * u, 1e-8))
            coeff = du * jac / q
            Rrec += coeff * np.outer(a, a.conj())

    a0 = steering_deg(theta0_deg, N)
    x = sla.solve(Rrec + reg * np.eye(N), a0, assume_a='her')
    w = x / (a0.conj().T @ x)
    return w, Rrec


# ============================================================
# Literature method: PGLQ-Full [12]
# ============================================================
def beamformer_pglq_full(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    M_each=12,
    Q=17,
    reg=1e-6
):
    N = Rhat.shape[0]
    Rinv = sla.inv(Rhat + reg * np.eye(N))

    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)
    u_nodes, alpha_nodes = partition_intervals(intervals, M_each=M_each, Q=Q, rule="gl")

    Rrec = np.zeros((N, N), dtype=np.complex128)

    for u, alpha in zip(u_nodes, alpha_nodes):
        a = steering_u(u, N)
        q = np.real(a.conj().T @ Rinv @ a)
        q = max(q, 1e-8)
        jac = 1.0 / np.sqrt(max(1.0 - u * u, 1e-8))
        coeff = alpha * jac / q
        Rrec += coeff * np.outer(a, a.conj())

    a0 = steering_deg(theta0_deg, N)
    x = sla.solve(Rrec + reg * np.eye(N), a0, assume_a='her')
    w = x / (a0.conj().T @ x)
    return w, Rrec


# ============================================================
# Added baseline 1: GRDR-CMR [25]
# ============================================================
def beamformer_grdr_cmr(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    M_each=12,
    Q=17,
    rank_ratio=0.5,
    reg=1e-6,
    seed=1234,
):
    """
    Representative implementation of GRDR-CMR [25]:
      1) Gaussian random dimensionality reduction
      2) Reduced-dimension inverse evaluation
      3) Full-matrix covariance reconstruction via piecewise GL quadrature
    """
    N = Rhat.shape[0]
    r = max(4, min(N, int(np.ceil(rank_ratio * N))))

    rng = np.random.default_rng(seed)
    G = (rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))) / np.sqrt(2.0 * r)

    Rr = G.conj().T @ Rhat @ G
    Rr_inv = sla.inv(Rr + reg * np.eye(r))

    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)
    u_nodes, alpha_nodes = partition_intervals(intervals, M_each=M_each, Q=Q, rule="gl")

    Rrec = np.zeros((N, N), dtype=np.complex128)

    for u, alpha in zip(u_nodes, alpha_nodes):
        a = steering_u(u, N)
        ar = G.conj().T @ a

        q = np.real(ar.conj().T @ Rr_inv @ ar)
        q = max(q, 1e-8)

        jac = 1.0 / np.sqrt(max(1.0 - u * u, 1e-8))
        coeff = alpha * jac / q
        Rrec += coeff * np.outer(a, a.conj())

    a0 = steering_deg(theta0_deg, N)
    x = sla.solve(Rrec + reg * np.eye(N), a0, assume_a='her')
    w = x / (a0.conj().T @ x)
    return w, Rrec


# ============================================================
# Added baseline 2: Toeplitz-SSBF [26]
# ============================================================
def toeplitz_project_from_matrix(R):
    """
    Project a matrix onto the Hermitian Toeplitz set
    by averaging along diagonals.
    """
    N = R.shape[0]
    rcol = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        if k == 0:
            diag_vals = np.diag(R, k=0)
        else:
            diag_vals_pos = np.diag(R, k=k)
            diag_vals_neg = np.diag(R, k=-k)
            diag_vals = np.concatenate([diag_vals_pos, np.conj(diag_vals_neg)])
        rcol[k] = np.mean(diag_vals)

    rcol[0] = np.real(rcol[0])
    return sla.toeplitz(rcol, np.conj(rcol)), rcol


# ============================================================
# Proposed SCC-BF helpers
# ============================================================
def lowrank_inverse_surrogate(Rhat, L=3):
    N = Rhat.shape[0]
    L = max(1, min(L, N - 1))

    vals, vecs = eigsh(Rhat, k=L, which='LM')
    idx = np.argsort(vals)[::-1]
    vals = np.real(vals[idx])
    vecs = vecs[:, idx]

    lam_bar = (np.real(np.trace(Rhat)) - np.sum(vals)) / max(N - L, 1)
    lam_bar = max(lam_bar, 1e-8)

    d = (vals - lam_bar) / (lam_bar * np.maximum(vals, 1e-10))
    d = np.real(d)

    return vecs, vals, lam_bar, d


def beamformer_toeplitz_ssbf(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    M_each=12,
    Q=17,
    L=3,
    reg=1e-6,
):
    """
    Representative implementation of Toeplitz-SSBF [26]:
      1) Toeplitz projection of sample covariance
      2) Low-rank structured inverse surrogate
      3) Toeplitz covariance reconstruction via piecewise GL quadrature
    """
    N = Rhat.shape[0]

    Rt, _ = toeplitz_project_from_matrix(Rhat)
    U, vals, lam_bar, d = lowrank_inverse_surrogate(Rt, L=L)

    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)
    u_nodes, alpha_nodes = partition_intervals(intervals, M_each=M_each, Q=Q, rule="gl")

    lags = np.arange(N)
    rcol = np.zeros(N, dtype=np.complex128)

    for u, alpha in zip(u_nodes, alpha_nodes):
        a = steering_u(u, N)
        proj = U.conj().T @ a

        q = (N / lam_bar) - np.sum(d * np.abs(proj) ** 2)
        q = max(np.real(q), 1e-8)

        jac = 1.0 / np.sqrt(max(1.0 - u * u, 1e-8))
        coeff = alpha * jac / q

        v = np.exp(1j * np.pi * lags * u)
        rcol += coeff * v

    rcol[0] = np.real(rcol[0]) + 5.0 * reg

    a0 = steering_deg(theta0_deg, N)
    x = sla.solve_toeplitz((rcol, np.conj(rcol)), a0)
    w = x / (a0.conj().T @ x)

    return w, rcol


# ============================================================
# Proposed SCC-BF
# ============================================================
def beamformer_scc_bf(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    M_each=12,
    Q=17,
    L=3,
    reg=1e-6
):
    N = Rhat.shape[0]

    U, vals, lam_bar, d = lowrank_inverse_surrogate(Rhat, L=L)

    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)
    u_nodes, alpha_nodes = partition_intervals(intervals, M_each=M_each, Q=Q, rule="cc")

    lags = np.arange(N)
    rcol = np.zeros(N, dtype=np.complex128)

    for u, alpha in zip(u_nodes, alpha_nodes):
        a = steering_u(u, N)

        proj = U.conj().T @ a
        q = (N / lam_bar) - np.sum(d * np.abs(proj) ** 2)
        q = max(np.real(q), 1e-8)

        jac = 1.0 / np.sqrt(max(1.0 - u * u, 1e-8))
        coeff = alpha * jac / q

        v = np.exp(1j * np.pi * lags * u)
        rcol += coeff * v

    rcol[0] = np.real(rcol[0]) + 5.0 * reg

    a0 = steering_deg(theta0_deg, N)
    x = sla.solve_toeplitz((rcol, np.conj(rcol)), a0)
    w = x / (a0.conj().T @ x)

    return w, rcol

def beamformer_scc_bf_no_partition(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    Q=17,
    L=3,
    reg=1e-6
):
    """
    SCC-BF without interval partitioning:
    each sidelobe side is treated as one whole interval.
    total nodes = 2 * Q
    """
    return beamformer_scc_bf(
        Rhat,
        theta0_deg=theta0_deg,
        main_half_width_deg=main_half_width_deg,
        M_each=1,
        Q=Q,
        L=L,
        reg=reg
    )
# ============================================================
# Unified runner
# ============================================================
def auto_prop_L(N):
    return max(4, min(20, N // 6))


def run_methods_same_nodes(
    N=32,
    K=64,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=40.0,
    main_half_width=4.0,
    seed=1,
    dense_P_each=300,
    M_each_shared=12,
    Q_each_shared=17,
    prop_L=None,
    grdr_rank_ratio=0.5,
):
    if prop_L is None:
        prop_L = auto_prop_L(N)

    _, Rhat, a_s, Rin, sigma_s2 = simulate_snapshots(
        N=N,
        K=K,
        theta_s_deg=theta_s,
        theta_j_deg=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        seed=seed
    )

    t0 = time.perf_counter()
    w_opt = beamformer_optimal_mvdr(Rin, a_s)
    t_opt = time.perf_counter() - t0
    sinr_opt = output_sinr(w_opt, a_s, Rin, sigma_s2)

    t0 = time.perf_counter()
    w_dense, _ = beamformer_dense_ir(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        P_each=dense_P_each
    )
    t_dense = time.perf_counter() - t0
    sinr_dense = output_sinr(w_dense, a_s, Rin, sigma_s2)

    t0 = time.perf_counter()
    w_pglq, _ = beamformer_pglq_full(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_each_shared,
        Q=Q_each_shared,
    )
    t_pglq = time.perf_counter() - t0
    sinr_pglq = output_sinr(w_pglq, a_s, Rin, sigma_s2)

    t0 = time.perf_counter()
    w_grdr, _ = beamformer_grdr_cmr(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_each_shared,
        Q=Q_each_shared,
        rank_ratio=grdr_rank_ratio,
        seed=seed + 777,
    )
    t_grdr = time.perf_counter() - t0
    sinr_grdr = output_sinr(w_grdr, a_s, Rin, sigma_s2)

    t0 = time.perf_counter()
    toep_L = max(8, min(24, N // 8 + 4))
    w_toep, _ = beamformer_toeplitz_ssbf(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_each_shared,
        Q=Q_each_shared,
        L=toep_L,
    )
    t_toep = time.perf_counter() - t0
    sinr_toep = output_sinr(w_toep, a_s, Rin, sigma_s2)

    t0 = time.perf_counter()
    w_prop, _ = beamformer_scc_bf(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_each_shared,
        Q=Q_each_shared,
        L=prop_L,
    )
    t_prop = time.perf_counter() - t0
    sinr_prop = output_sinr(w_prop, a_s, Rin, sigma_s2)

    total_nodes = 2 * M_each_shared * Q_each_shared

    return {
        "w_opt": w_opt,
        "w_dense": w_dense,
        "w_pglq": w_pglq,
        "w_grdr": w_grdr,
        "w_toep": w_toep,
        "w_prop": w_prop,
        "a_s": a_s,
        "Rin": Rin,
        "sigma_s2": sigma_s2,
        "sinr_opt": sinr_opt,
        "sinr_dense": sinr_dense,
        "sinr_pglq": sinr_pglq,
        "sinr_grdr": sinr_grdr,
        "sinr_toep": sinr_toep,
        "sinr_prop": sinr_prop,
        "t_opt": t_opt,
        "t_dense": t_dense,
        "t_pglq": t_pglq,
        "t_grdr": t_grdr,
        "t_toep": t_toep,
        "t_prop": t_prop,
        "prop_L": prop_L,
        "grdr_rank_ratio": grdr_rank_ratio,
        "M_each_shared": M_each_shared,
        "Q_each_shared": Q_each_shared,
        "total_nodes": total_nodes,
        "dense_P_each": dense_P_each,
    }

def run_self_param_case(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=43.0,
    main_half_width=4.0,
    seed=1,
    Q_each=3,
    M_opt_like=12,
    M_ncq=8,
    prop_L=None,
):
    """
    只比较自己的参数设置：
      1) Δ_interval ≈ Δ_opt     -> 用 M_opt_like 表示
      2) Using NCQ, Δ_interval  -> 用 M_ncq 表示
      3) Without interval partitioning -> M_each = 1
      4) Optimal
    """
    if prop_L is None:
        prop_L = auto_prop_L(N)

    _, Rhat, a_s, Rin, sigma_s2 = simulate_snapshots(
        N=N,
        K=K,
        theta_s_deg=theta_s,
        theta_j_deg=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        seed=seed
    )

    w_opt = beamformer_optimal_mvdr(Rin, a_s)
    sinr_opt = output_sinr(w_opt, a_s, Rin, sigma_s2)

    w_optlike, _ = beamformer_scc_bf(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_opt_like,
        Q=Q_each,
        L=prop_L,
    )
    sinr_optlike = output_sinr(w_optlike, a_s, Rin, sigma_s2)

    w_ncq, _ = beamformer_scc_bf(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_ncq,
        Q=Q_each,
        L=prop_L,
    )
    sinr_ncq = output_sinr(w_ncq, a_s, Rin, sigma_s2)

    w_no_part, _ = beamformer_scc_bf_no_partition(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        Q=Q_each,
        L=prop_L,
    )
    sinr_no_part = output_sinr(w_no_part, a_s, Rin, sigma_s2)

    return {
        "sinr_opt": sinr_opt,
        "sinr_optlike": sinr_optlike,
        "sinr_ncq": sinr_ncq,
        "sinr_no_part": sinr_no_part,
    }

def run_scc_single_case(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=43.0,
    main_half_width=4.0,
    seed=1,
    M_each=12,
    Q_each=17,
    prop_L=None,
):
    """
    只运行 Proposed SCC-BF 和一个 no-partition 基线
    """
    if prop_L is None:
        prop_L = auto_prop_L(N)

    _, Rhat, a_s, Rin, sigma_s2 = simulate_snapshots(
        N=N,
        K=K,
        theta_s_deg=theta_s,
        theta_j_deg=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        seed=seed
    )

    # SCC-BF with partition
    w_prop, _ = beamformer_scc_bf(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=M_each,
        Q=Q_each,
        L=prop_L,
    )
    sinr_prop = output_sinr(w_prop, a_s, Rin, sigma_s2)

    # no partition baseline: M_each = 1
    w_no_part, _ = beamformer_scc_bf(
        Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        M_each=1,
        Q=Q_each,
        L=prop_L,
    )
    sinr_no_part = output_sinr(w_no_part, a_s, Rin, sigma_s2)

    return {
        "sinr_prop": sinr_prop,
        "sinr_no_part": sinr_no_part,
    }
# ============================================================
# Quadrature-rule ablation for SCC-BF:
#   GL vs NC vs CC under the SAME SCC-BF framework
# ============================================================

RULE_META = {
    "gl": {
        "label": "Gauss-Legendre",
        "color": "#4C78A8",
        "marker": "s",
        "linestyle": "--",
    },
    "nc": {
        "label": "Newton-Cotes",
        "color": "#59A14F",
        "marker": "o",
        "linestyle": ":",
    },
    "cc": {
        "label": "Clenshaw-Curtis",
        "color": "#D62728",
        "marker": "^",
        "linestyle": "-",
    },
}


def rule_label(rule):
    return RULE_META[rule]["label"]


def safe_sidelobe_u_intervals(theta0_deg=0.0, main_half_width_deg=4.0, endpoint_eps=1e-6):
    """
    For fair comparison among GL / NC / CC:
    trim only the global endpoints ±1 slightly, because
    CC / NC are closed rules and may sample the endpoints directly.
    """
    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)
    safe = []
    for ua, ub in intervals:
        ua2 = ua + endpoint_eps if np.isclose(ua, -1.0) else ua
        ub2 = ub - endpoint_eps if np.isclose(ub, 1.0) else ub
        if ub2 <= ua2:
            raise ValueError("Invalid interval after endpoint trimming. Reduce endpoint_eps.")
        safe.append((ua2, ub2))
    return safe


def newton_cotes_on_interval(a, b, Q=5):
    """
    Closed Newton-Cotes on [a,b] with Q equally spaced nodes.
    Q nodes => degree index n = Q-1 for scipy.integrate.newton_cotes
    """
    if Q < 2:
        raise ValueError("Q must be >= 2 for Newton-Cotes quadrature.")

    x = np.linspace(a, b, Q)
    w_std, _ = newton_cotes(Q - 1, equal=1)   # weights for equally spaced samples
    alpha = ((b - a) / (Q - 1)) * np.asarray(w_std, dtype=np.float64)
    return x, alpha


def quadrature_on_interval(a, b, Q=17, rule="cc"):
    rule = rule.lower()
    if rule == "gl":
        return gauss_legendre_on_interval(a, b, Q=Q)
    elif rule == "cc":
        return clenshaw_curtis_on_interval(a, b, Q=Q)
    elif rule in ("nc", "newton-cotes", "newton_cotes"):
        return newton_cotes_on_interval(a, b, Q=Q)
    else:
        raise ValueError(f"Unknown quadrature rule: {rule}")


def partition_intervals_by_rule(intervals, M_each=12, Q=17, rule="cc"):
    u_nodes = []
    alpha_nodes = []

    for (ua, ub) in intervals:
        bounds = np.linspace(ua, ub, M_each + 1)
        for i in range(M_each):
            a = bounds[i]
            b = bounds[i + 1]
            x, w = quadrature_on_interval(a, b, Q=Q, rule=rule)
            u_nodes.append(x)
            alpha_nodes.append(w)

    return np.concatenate(u_nodes), np.concatenate(alpha_nodes)


def beamformer_scc_bf_rule(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    M_each=12,
    Q=17,
    L=3,
    reg=1e-6,
    rule="cc",
    endpoint_eps=1e-6,
):
    """
    SAME SCC-BF framework, only the quadrature rule changes:
        rule = 'gl' / 'nc' / 'cc'
    """
    N = Rhat.shape[0]

    U, vals, lam_bar, d = lowrank_inverse_surrogate(Rhat, L=L)
    intervals = safe_sidelobe_u_intervals(
        theta0_deg=theta0_deg,
        main_half_width_deg=main_half_width_deg,
        endpoint_eps=endpoint_eps,
    )
    u_nodes, alpha_nodes = partition_intervals_by_rule(
        intervals, M_each=M_each, Q=Q, rule=rule
    )

    lags = np.arange(N)
    rcol = np.zeros(N, dtype=np.complex128)

    for u, alpha in zip(u_nodes, alpha_nodes):
        a = steering_u(u, N)

        proj = U.conj().T @ a
        q = (N / lam_bar) - np.sum(d * np.abs(proj) ** 2)
        q = max(np.real(q), 1e-8)

        jac = 1.0 / np.sqrt(max(1.0 - u * u, 1e-8))
        coeff = alpha * jac / q

        rcol += coeff * np.exp(1j * np.pi * lags * u)

    rcol[0] = np.real(rcol[0]) + 5.0 * reg

    a0 = steering_deg(theta0_deg, N)
    x = sla.solve_toeplitz((rcol, np.conj(rcol)), a0)
    w = x / (a0.conj().T @ x)

    return w, rcol


def run_scc_rules_same_budget(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=43.0,
    main_half_width=4.0,
    seed=1,
    M_each=12,
    Q_each=17,
    prop_L=None,
    endpoint_eps=1e-6,
):
    """
    SAME data / SAME SCC-BF framework / SAME budget B = 2*M_each*Q_each
    only quadrature rule changes.
    """
    if prop_L is None:
        prop_L = auto_prop_L(N)

    _, Rhat, a_s, Rin, sigma_s2 = simulate_snapshots(
        N=N,
        K=K,
        theta_s_deg=theta_s,
        theta_j_deg=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        seed=seed
    )

    out = {
        "B": 2 * M_each * Q_each,
        "M_each": M_each,
        "Q_each": Q_each,
    }

    w_opt = beamformer_optimal_mvdr(Rin, a_s)
    out["w_opt"] = w_opt
    out["sinr_opt"] = output_sinr(w_opt, a_s, Rin, sigma_s2)

    for rule in ("gl", "nc", "cc"):
        w, rcol = beamformer_scc_bf_rule(
            Rhat,
            theta0_deg=theta_s,
            main_half_width_deg=main_half_width,
            M_each=M_each,
            Q=Q_each,
            L=prop_L,
            reg=1e-6,
            rule=rule,
            endpoint_eps=endpoint_eps,
        )
        out[f"w_{rule}"] = w
        out[f"rcol_{rule}"] = rcol
        out[f"sinr_{rule}"] = output_sinr(w, a_s, Rin, sigma_s2)

    return out


# ============================================================
# Figure 1: nodes on 1D axis
# ============================================================
def plot_rule_nodes_1d_topjournal_bottom_titles(
    Q_list=(3, 5, 7, 9, 11, 13, 17, 21),
    out_dir="./outputs_pub_quad_rules",
    name="fig1_rule_nodes_1d_topjournal_bottom_titles",
):
    """
    4x2 对称节点分布图
    - 每个子图标题放在下方
    - 每个子图都单独有 x / y 轴标注
    """
    if len(Q_list) != 8:
        raise ValueError("For a symmetric 4x2 layout, Q_list must contain exactly 8 entries.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
    })

    rule_order = ("gl", "nc", "cc")
    y_map = {"gl": 2.0, "nc": 1.0, "cc": 0.0}
    y_ticklabels = ["GL", "NC", "CC"]

    rule_meta = {
        "gl": {
            "label": "Gauss-Legendre",
            "color": "#4C78A8",
            "marker": "s",
        },
        "nc": {
            "label": "Newton-Cotes",
            "color": "#59A14F",
            "marker": "o",
        },
        "cc": {
            "label": "Clenshaw-Curtis",
            "color": "#D62728",
            "marker": "^",
        },
    }

    panel_labels = [f"({chr(97+i)})" for i in range(8)]

    fig, axes = plt.subplots(
        4, 2,
        figsize=(12.8, 11.2),
        sharex=False,
        sharey=False,
        constrained_layout=False
    )
    axes = axes.ravel()

    for idx, (ax, Q) in enumerate(zip(axes, Q_list)):
        ax.set_facecolor("white")

        # 三条水平参考线
        for rule in rule_order:
            ax.hlines(
                y=y_map[rule],
                xmin=-1.0,
                xmax=1.0,
                colors=rule_meta[rule]["color"],
                linestyles=":",
                linewidth=0.8,
                alpha=0.40,
                zorder=1
            )

        # 节点
        for rule in rule_order:
            x, _ = quadrature_on_interval(-1.0, 1.0, Q=Q, rule=rule)
            y = np.full_like(x, y_map[rule], dtype=float)

            ax.scatter(
                x, y,
                s=42,
                marker=rule_meta[rule]["marker"],
                facecolors="white",
                edgecolors=rule_meta[rule]["color"],
                linewidths=1.0,
                zorder=3,
                label=rule_meta[rule]["label"] if idx == 0 else None
            )

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.55, 2.55)

        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(y_ticklabels, fontsize=12)

        # 每个子图单独的横纵坐标标题
        ax.set_xlabel(r"Nodes on $[-1,1]$", fontsize=16, labelpad=2)
        ax.set_ylabel("Quadrature rule", fontsize=16, labelpad=2)

        ax.grid(
            True, axis="x",
            color="#D9D9D9",
            linestyle="--",
            linewidth=0.55,
            alpha=0.85,
            zorder=0
        )

        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            length=4.2,
            width=0.85,
            labelsize=16
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#333333")

        # -------- 子图标题下置 --------
        ax.text(
            0.5, -0.15,
            rf"{panel_labels[idx]}  $Q={Q}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16
        )

    # 总图例
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.35,
        handletextpad=0.6,
        columnspacing=1.8,
        fontsize=16
    )
    leg.get_frame().set_edgecolor("#B5B5B5")
    leg.get_frame().set_linewidth(0.9)
    leg.get_frame().set_facecolor("white")

    # 注意：标题放下面后，必须拉大 hspace
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.05,
        top=0.92,
        wspace=0.28,
        hspace=0.46
    )

    save_figure(fig, out_dir, name)
    plt.show()


# ============================================================
# Figure 2: actual-integrand quadrature error
# ============================================================
def reference_integral_piecewise_safe(func, intervals, num_per_interval=40001):
    val = 0.0 + 0.0j
    for (ua, ub) in intervals:
        u = np.linspace(ua, ub, num_per_interval)
        val += np.trapz(func(u), u)
    return val


def quadrature_integral_piecewise_rule(func, intervals, M_each=12, Q=17, rule="cc"):
    val = 0.0 + 0.0j

    for (ua, ub) in intervals:
        bounds = np.linspace(ua, ub, M_each + 1)
        for i in range(M_each):
            a = bounds[i]
            b = bounds[i + 1]
            x, alpha = quadrature_on_interval(a, b, Q=Q, rule=rule)
            val += np.sum(alpha * func(x))

    return val


def plot_quadrature_error_three_rules_sci_local_legend(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    ell_list=(0, 10, 30, 60),
    M_each=12,
    Q_list=(3, 5, 7, 9, 13, 17, 21, 25, 33),
    reg=1e-6,
    endpoint_eps=1e-6,
    out_dir="./outputs_pub_quad_rules",
    name="fig2_quadrature_error_three_rules_sci_local_legend",
):
    """
    SCI 风格版本：
    1) 每个子图标题放在下方
    2) 每个子图都单独标注 x / y 轴
    3) 每个子图左上角单独 legend
    """

    N = Rhat.shape[0]
    Rinv = sla.inv(Rhat + reg * np.eye(N))
    intervals = safe_sidelobe_u_intervals(
        theta0_deg=theta0_deg,
        main_half_width_deg=main_half_width_deg,
        endpoint_eps=endpoint_eps,
    )

    def make_integrand(ell):
        def func(u):
            u = np.asarray(u)
            out = np.zeros_like(u, dtype=np.complex128)

            for idx, uu in enumerate(u):
                a = steering_u(float(uu), N)
                q = np.real(a.conj().T @ Rinv @ a)
                q = max(q, 1e-10)
                jac = 1.0 / np.sqrt(max(1.0 - float(uu) * float(uu), 1e-10))
                out[idx] = (jac / q) * np.exp(1j * np.pi * ell * float(uu))
            return out

        return func

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    total_nodes = 2 * M_each * np.array(Q_list)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.6, 8.8),
        sharex=False,
        sharey=False
    )
    axes = axes.ravel()

    for idx, (ax, ell) in enumerate(zip(axes, ell_list)):
        func = make_integrand(ell)

        ref_val = reference_integral_piecewise_safe(
            func,
            intervals,
            num_per_interval=40001
        )

        for rule in ("gl", "nc", "cc"):
            errs = []
            for Q in Q_list:
                val = quadrature_integral_piecewise_rule(
                    func, intervals, M_each=M_each, Q=Q, rule=rule
                )
                errs.append(np.abs(val - ref_val))

            errs = np.maximum(np.asarray(errs), 1e-16)
            meta = RULE_META[rule]

            ax.plot(
                total_nodes, errs,
                color=meta["color"],
                linestyle=meta["linestyle"],
                marker=meta["marker"],
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=2.0,
                label=meta["label"],
            )

        # 每个子图各自 legend
        leg = ax.legend(
            loc="upper left",
            fontsize=10,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.25,
            handlelength=2.2,
            handletextpad=0.5,
            borderaxespad=0.6
        )
        leg.get_frame().set_edgecolor("#A8A8A8")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_facecolor("white")

        # 坐标轴
        ax.set_yscale("log")
        ax.set_xlabel("Total number of quadrature nodes", fontsize=14, labelpad=6)
        ax.set_ylabel("Absolute integration error", fontsize=14, labelpad=6)

        ax.grid(True, which="major", color="#D6D6D6", linestyle='-', linewidth=0.45, alpha=0.55)
        ax.grid(True, which="minor", color="#E8E8E8", linestyle=':', linewidth=0.35, alpha=0.45)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='major', length=5.5, width=0.9, pad=4, labelsize=12)
        ax.tick_params(direction='in', which='minor', length=3.0, width=0.75)
        ax.tick_params(axis='y', which='both', labelleft=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#333333")

        # 子图标题放下方
        ax.text(
            0.5, -0.30,
            rf"{panel_labels[idx]}  $\ell = {ell}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15
        )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.97,
        bottom=0.10,
        wspace=0.20,
        hspace=0.48
    )

    save_figure(fig, out_dir, name)
    plt.show()
# ============================================================
# Figure 3: same SCC-BF framework, only quadrature rule changes
# ============================================================
def plot_snr_vs_sinr_three_rules(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_list=np.arange(-10, 31, 5),
    inr_db=43.0,
    main_half_width=4.0,
    mc_runs=20,
    M_each=12,
    Q_each=17,
    endpoint_eps=1e-6,
    out_dir="./outputs_pub_quad_rules",
    name="fig3_snr_vs_sinr_three_rules",
):
    y = {"opt": [], "gl": [], "nc": [], "cc": []}

    for snr_db in snr_list:
        vals = {"opt": [], "gl": [], "nc": [], "cc": []}

        for mc in range(mc_runs):
            res = run_scc_rules_same_budget(
                N=N,
                K=K,
                theta_s=theta_s,
                theta_j=theta_j,
                snr_db=snr_db,
                inr_db=inr_db,
                main_half_width=main_half_width,
                seed=12000 + mc,
                M_each=M_each,
                Q_each=Q_each,
                prop_L=auto_prop_L(N),
                endpoint_eps=endpoint_eps,
            )
            vals["opt"].append(db(res["sinr_opt"]))
            vals["gl"].append(db(res["sinr_gl"]))
            vals["nc"].append(db(res["sinr_nc"]))
            vals["cc"].append(db(res["sinr_cc"]))

        for k in y:
            y[k].append(np.mean(vals[k]))

        print(f"[SNR vs SINR / quadrature-rule ablation] SNR = {snr_db:>5.1f} dB done")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    ax.plot(
        snr_list, y["opt"],
        color=COLORS["optimal"],
        linestyle="-.",
        marker="D",
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=2.0,
        label="Optimal",
    )

    for rule in ("gl", "nc", "cc"):
        meta = RULE_META[rule]
        ax.plot(
            snr_list, y[rule],
            color=meta["color"],
            linestyle=meta["linestyle"],
            marker=meta["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=2.1,
            label=f"SCC-BF-{rule.upper()}",
        )

    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Output SINR (dB)")
    ax.set_title(rf"Same SCC-BF framework, same quadrature budget: $B={2*M_each*Q_each}$")
    setup_axis(ax, add_minor=True)
    add_soft_legend(ax, loc="upper left")

    fig.tight_layout()
    save_figure(fig, out_dir, name)
    plt.show()


# ============================================================
# Figure 4: M-Q heatmaps for GL / NC / CC
# ============================================================
def build_rule_heatmap_data(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=43.0,
    main_half_width=4.0,
    mc_runs=10,
    M_list=(1, 2, 3, 4, 6, 8, 12, 16),
    Q_list=(3, 5, 7, 9, 13, 17),
    endpoint_eps=1e-6,
):
    data = {rule: np.zeros((len(M_list), len(Q_list))) for rule in ("gl", "nc", "cc")}

    for i, M_each in enumerate(M_list):
        for j, Q_each in enumerate(Q_list):
            vals = {rule: [] for rule in ("gl", "nc", "cc")}

            for mc in range(mc_runs):
                res = run_scc_rules_same_budget(
                    N=N,
                    K=K,
                    theta_s=theta_s,
                    theta_j=theta_j,
                    snr_db=snr_db,
                    inr_db=inr_db,
                    main_half_width=main_half_width,
                    seed=22000 + 1000 * i + 100 * j + mc,
                    M_each=M_each,
                    Q_each=Q_each,
                    prop_L=auto_prop_L(N),
                    endpoint_eps=endpoint_eps,
                )

                vals["gl"].append(db(res["sinr_gl"]))
                vals["nc"].append(db(res["sinr_nc"]))
                vals["cc"].append(db(res["sinr_cc"]))

            for rule in ("gl", "nc", "cc"):
                data[rule][i, j] = np.mean(vals[rule])

            print(f"[Heatmap] M = {M_each:>2d}, Q = {Q_each:>2d} done")

    return data


def plot_mq_heatmaps_three_rules(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=43.0,
    main_half_width=4.0,
    mc_runs=10,
    M_list=(1, 2, 3, 4, 6, 8, 12, 16),
    Q_list=(3, 5, 7, 9, 13, 17),
    endpoint_eps=1e-6,
    out_dir="./outputs_pub_quad_rules",
    name="fig4_mq_heatmaps_three_rules",
):
    data = build_rule_heatmap_data(
        N=N,
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=mc_runs,
        M_list=M_list,
        Q_list=Q_list,
        endpoint_eps=endpoint_eps,
    )

    vmin = min(np.min(data[r]) for r in ("gl", "nc", "cc"))
    vmax = max(np.max(data[r]) for r in ("gl", "nc", "cc"))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=False, sharey=False)

    for ax, rule in zip(axes, ("gl", "nc", "cc")):
        im = ax.imshow(
            data[rule],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )

        ax.set_title(rule_label(rule))
        ax.set_xticks(np.arange(len(Q_list)))
        ax.set_xticklabels(Q_list)
        ax.set_yticks(np.arange(len(M_list)))
        ax.set_yticklabels(M_list)
        ax.set_xlabel("Q per segment")
        ax.tick_params(direction="in", top=True, right=True)

        for i in range(len(M_list)):
            for j in range(len(Q_list)):
                ax.text(
                    j, i, f"{data[rule][i, j]:.1f}",
                    ha="center", va="center",
                    fontsize=8, color="white"
                )

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color(COLORS["spine"])

    axes[0].set_ylabel("M (segments per sidelobe side)")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Mean output SINR (dB)")

    fig.suptitle("Sensitivity to $(M, Q)$ under the identical SCC-BF framework", y=1.02, fontsize=14)
    fig.tight_layout()
    save_figure(fig, out_dir, name)
    plt.show()


# ============================================================
# One-click runner for the 4 figures
# ============================================================
def main_quadrature_rule_ablation():
    out_dir = "./outputs_pub_quad_rules"

    N = 32
    K = 128
    theta_s = 0.0
    theta_j = (-46.0, 32.0)
    snr_db = 10.0
    inr_db = 43.0
    main_half_width = 4.0

    M_each = 12
    Q_each = 17

    # ---------------- Figure 1 ----------------
    plot_rule_nodes_1d_topjournal_bottom_titles(
        Q_list=(3, 5, 7, 9, 11, 13, 17, 21),
        out_dir=out_dir,
        name="fig1_rule_nodes_1d_topjournal_bottom_titles",
    )

    # A fixed Rhat for Figure 2
    Rhat = simulate_snapshots(
        N=N,
        K=K,
        theta_s_deg=theta_s,
        theta_j_deg=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        seed=1
    )[1]

    # ---------------- Figure 2 ----------------
    plot_quadrature_error_three_rules_sci_local_legend(
        Rhat=Rhat,
        theta0_deg=theta_s,
        main_half_width_deg=main_half_width,
        ell_list=(0, 10, 30, 60),
        M_each=M_each,
        Q_list=(3, 5, 7, 9, 13, 17, 21, 25, 33),
        reg=1e-6,
        endpoint_eps=1e-6,
        out_dir=out_dir,
        name="fig2_quadrature_error_three_rules_sci_local_legend",
    )

    # ---------------- Figure 3 ----------------
    plot_snr_vs_sinr_three_rules(
        N=N,
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_list=np.arange(-10, 31, 5),
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=20,
        M_each=M_each,
        Q_each=Q_each,
        endpoint_eps=1e-6,
        out_dir=out_dir,
        name="fig3_snr_vs_sinr_three_rules",
    )

    # ---------------- Figure 4 ----------------
    plot_mq_heatmaps_three_rules(
        N=N,
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=10,
        M_list=(1, 2, 3, 4, 6, 8, 12, 16),
        Q_list=(3, 5, 7, 9, 13, 17),
        endpoint_eps=1e-6,
        out_dir=out_dir,
        name="fig4_mq_heatmaps_three_rules",
    )

# ============================================================
# Beampattern
# ============================================================
def beampattern_db(w, theta_grid_deg, N):
    vals = []
    for th in theta_grid_deg:
        a = steering_deg(th, N)
        vals.append(np.abs(w.conj().T @ a))
    vals = np.array(vals)
    vals = vals / np.max(np.maximum(vals, 1e-12))
    return db(vals ** 2)


# ============================================================
# Plot helpers
# ============================================================
def setup_axis(ax, add_minor=True):
    ax.grid(True, which="major", color=COLORS["grid"], linestyle='-', linewidth=0.55, alpha=0.18)
    if add_minor:
        ax.grid(True, which="minor", color=COLORS["grid"], linestyle=':', linewidth=0.42, alpha=0.10)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(direction='in', which='major', length=5.5, width=0.9, pad=4)
    ax.tick_params(direction='in', which='minor', length=3.0, width=0.75)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(COLORS["spine"])


def setup_inset_axis(axins):
    axins.grid(True, which="major", color=COLORS["grid"], linestyle='-', linewidth=0.45, alpha=0.16)
    axins.tick_params(direction='in', which='major', length=3.5, width=0.75, labelsize=10, pad=2)
    axins.xaxis.set_minor_locator(AutoMinorLocator())
    axins.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in axins.spines.values():
        spine.set_linewidth(0.85)
        spine.set_color(COLORS["spine"])


def save_figure(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    png_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {png_path}")


def add_soft_legend(ax, loc='upper left'):
    leg = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        borderpad=0.35,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.5,
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#A8A8A8')
    leg.get_frame().set_linewidth(0.8)
    return leg


# ============================================================
# Figure 1: Beampattern
# ============================================================
def plot_beampattern_comparison(
    w_opt, w_pglq, w_grdr, w_toep, w_prop,
    N=32,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    out_dir="./outputs_pub",
):
    theta_grid = np.linspace(-90, 90, 1441)
    bp_pglq = beampattern_db(w_pglq, theta_grid, N)
    bp_grdr = beampattern_db(w_grdr, theta_grid, N)
    bp_toep = beampattern_db(w_toep, theta_grid, N)
    bp_prop = beampattern_db(w_prop, theta_grid, N)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    ax.plot(theta_grid, bp_pglq,
            label='PGLQ-Full [12]',
            color=COLORS["pglq"], linestyle='--', linewidth=2.0, zorder=3)
    ax.plot(theta_grid, bp_grdr,
            label='GRDR-CMR [25]',
            color=COLORS["grdr"], linestyle=':', linewidth=2.0, zorder=3)
    ax.plot(theta_grid, bp_toep,
            label='Toeplitz-SSBF [26]',
            color=COLORS["toep"], linestyle=(0, (5, 2)), linewidth=2.0, zorder=3)
    ax.plot(theta_grid, bp_prop,
            label='Proposed SCC-BF',
            color=COLORS["proposed"], linestyle='-', linewidth=2.4, zorder=4)

    ax.axvline(theta_s, linestyle='-.', linewidth=0.9, color=COLORS["ref"], alpha=0.55, zorder=1)
    for tj in theta_j:
        ax.axvline(tj, linestyle=':', linewidth=0.9, color=COLORS["ref"], alpha=0.55, zorder=1)

    ax.set_xlim(-90, 90)
    ax.set_ylim(-80, 3)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Beampattern (dB)')
    setup_axis(ax, add_minor=True)
    add_soft_legend(ax)

    x1, x2 = 6.0, 15.0
    y1, y2 = -40.0, -20.0
    axins = inset_axes(ax, width="30%", height="28%", loc='upper right', borderpad=1.1)
    axins.plot(theta_grid, bp_pglq, color=COLORS["pglq"], linestyle='--', linewidth=1.35)
    axins.plot(theta_grid, bp_grdr, color=COLORS["grdr"], linestyle=':', linewidth=1.35)
    axins.plot(theta_grid, bp_toep, color=COLORS["toep"], linestyle=(0, (5, 2)), linewidth=1.35)
    axins.plot(theta_grid, bp_prop, color=COLORS["proposed"], linestyle='-', linewidth=1.5)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    setup_inset_axis(axins)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.75)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_beampattern_comparison_same_nodes")
    plt.show()

def plot_self_param_group_2x3(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_list=np.arange(-10, 31, 5),
    inr_db=43.0,
    main_half_width=4.0,
    mc_runs=10,
    out_dir="./outputs_pub_same_nodes_gl_vs_cc",
):
    """
    2x3组图：只比较自己方法在不同参数设置下的表现

    列：Q = 3, 5
    行：三种 Δz_k 场景（这里用三组 M 配置模拟）

    每个子图四条曲线：
      1) Q=?, Δ_interval = Δ_opt
      2) Using NCQ, Δ_interval = xxx
      3) Without interval partitioning
      4) Optimal
    """

    # 三组场景：你可以把它理解成三种 Δz_k 条件
    # 每组给一个 “近似 Δopt 的 M” 和一个 “NCQ 对比 M”
    scenarios = [
        {"dzk_label": "0.03", "M_opt_like": 12, "M_ncq": 8},
        {"dzk_label": "0.05", "M_opt_like": 8,  "M_ncq": 5},
        {"dzk_label": "0.07", "M_opt_like": 5,  "M_ncq": 3},
    ]
    Q_cases = [3, 5]

    fig, axes = plt.subplots(3, 2, figsize=(10.8, 13.0), sharex=False, sharey=False)
    axes = np.array(axes)

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    pidx = 0

    for i, sc in enumerate(scenarios):
        for j, Q_each in enumerate(Q_cases):
            ax = axes[i, j]

            y_opt = []
            y_optlike = []
            y_ncq = []
            y_no_part = []

            for snr_db in snr_list:
                opt_mc = []
                optlike_mc = []
                ncq_mc = []
                no_part_mc = []

                for mc in range(mc_runs):
                    res = run_self_param_case(
                        N=N,
                        K=K,
                        theta_s=theta_s,
                        theta_j=theta_j,
                        snr_db=snr_db,
                        inr_db=inr_db,
                        main_half_width=main_half_width,
                        seed=20000 + 1000*i + 100*j + mc,
                        Q_each=Q_each,
                        M_opt_like=sc["M_opt_like"],
                        M_ncq=sc["M_ncq"],
                        prop_L=auto_prop_L(N),
                    )
                    opt_mc.append(db(res["sinr_opt"]))
                    optlike_mc.append(db(res["sinr_optlike"]))
                    ncq_mc.append(db(res["sinr_ncq"]))
                    no_part_mc.append(db(res["sinr_no_part"]))

                y_opt.append(np.mean(opt_mc))
                y_optlike.append(np.mean(optlike_mc))
                y_ncq.append(np.mean(ncq_mc))
                y_no_part.append(np.mean(no_part_mc))

            y_opt = np.array(y_opt)
            y_optlike = np.array(y_optlike)
            y_ncq = np.array(y_ncq)
            y_no_part = np.array(y_no_part)

            # 曲线风格尽量靠近你给的图
            ax.plot(
                snr_list, y_optlike,
                color='blue', linestyle='-',
                marker='s', markerfacecolor='none', markeredgewidth=1.0,
                linewidth=1.4,
                label=fr'Q={Q_each}, $\Delta_{{\mathrm{{interval}}}} \approx \Delta_{{\mathrm{{opt}}}}$'
            )

            ax.plot(
                snr_list, y_ncq,
                color='magenta', linestyle='-',
                marker='^', markerfacecolor='none', markeredgewidth=1.0,
                linewidth=1.4,
                label=fr'Using NCQ, $\Delta_{{\mathrm{{interval}}}}$ (M={sc["M_ncq"]})'
            )

            ax.plot(
                snr_list, y_no_part,
                color='red', linestyle='-',
                marker='o', markerfacecolor='none', markeredgewidth=1.0,
                linewidth=1.2,
                label='Without interval partitioning'
            )

            ax.plot(
                snr_list, y_opt,
                color='black', linestyle='-',
                linewidth=1.4,
                label='Optimal'
            )

            # 轴风格改得更像论文图
            ax.grid(True, which='major', color='#BEBEBE', linestyle='-', linewidth=0.35, alpha=0.9)
            ax.tick_params(direction='in', top=True, right=True)
            ax.set_xlim(-10, 30)
            ax.set_ylim(-20, 60)

            ax.set_xticks(np.arange(-10, 31, 5))
            ax.set_yticks(np.arange(-20, 61, 10))

            if i == 2:
                ax.set_xlabel('SNR(dB)')
            if j == 0:
                ax.set_ylabel('SINR(dB)')

            # 只在每个子图放自己的小图例，更像参考图
            leg = ax.legend(
                loc='upper left',
                fontsize=8,
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                borderpad=0.3,
                handlelength=2.4,
                handletextpad=0.4,
                borderaxespad=0.8,
            )
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#555555')
            leg.get_frame().set_linewidth(0.8)

            # inset
            axins = inset_axes(ax, width="33%", height="28%", loc='lower right', borderpad=1.3)
            axins.plot(snr_list, y_optlike, color='blue', linestyle='-', linewidth=1.0)
            axins.plot(snr_list, y_ncq, color='magenta', linestyle='-', linewidth=1.0)
            axins.plot(snr_list, y_no_part, color='red', linestyle='-', linewidth=0.9)
            axins.plot(snr_list, y_opt, color='black', linestyle='-', linewidth=1.0)

            # 放大中高SNR的微小差异
            x1, x2 = 4.5, 5.5
            idx = (snr_list >= 5) & (snr_list <= 5)
            local_vals = np.concatenate([
                y_optlike[idx], y_ncq[idx], y_no_part[idx], y_opt[idx]
            ])
            yc = np.mean(local_vals)
            y1, y2 = yc - 0.2, yc + 0.2

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.tick_params(direction='in', top=True, right=True, labelsize=7, length=2)
            axins.grid(True, which='major', color='#D0D0D0', linestyle='-', linewidth=0.3)
            for spine in axins.spines.values():
                spine.set_linewidth(0.8)

            # 主图中的虚线框
            rect_x1, rect_x2 = 4.0, 6.0
            rect_y1, rect_y2 = y1, y2
            rect = plt.Rectangle(
                (rect_x1, rect_y1),
                rect_x2 - rect_x1,
                rect_y2 - rect_y1,
                fill=False,
                linestyle=(0, (5, 3)),
                linewidth=0.9,
                edgecolor='gray'
            )
            ax.add_patch(rect)

            # 连线箭头
            ax.annotate(
                '',
                xy=(11.8, y1 - 0.5),
                xytext=(rect_x2, rect_y1),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color='black')
            )

            # 子图下方标题
            ax.text(
                0.5, -0.22,
                f'{panel_labels[pidx]}  Q = {Q_each}, ' + r'$\Delta_{z_k}$' + f' = {sc["dzk_label"]}',
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=12
            )

            print(f"[Self-param subplot] scenario={sc['dzk_label']}, Q={Q_each} done")
            pidx += 1

    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    save_figure(fig, out_dir, "fig_self_param_group_2x3")
    plt.show()
# ============================================================
# Figure 2: SNR vs Output SINR
# ============================================================
def sweep_snr_plot(
    N=32,
    K=64,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_list=np.arange(-10, 31, 5),
    inr_db=40.0,
    main_half_width=4.0,
    mc_runs=20,
    out_dir="./outputs_pub",
    M_each_shared=24,
    Q_each_shared=5,
):
    sinr_opt_list = []
    sinr_pglq_list = []
    sinr_grdr_list = []
    sinr_toep_list = []
    sinr_prop_list = []

    for snr_db in snr_list:
        opt_mc = []
        pglq_mc = []
        grdr_mc = []
        toep_mc = []
        prop_mc = []

        for mc in range(mc_runs):
            res = run_methods_same_nodes(
                N=N,
                K=K,
                theta_s=theta_s,
                theta_j=theta_j,
                snr_db=snr_db,
                inr_db=inr_db,
                main_half_width=main_half_width,
                seed=1000 + mc,
                dense_P_each=300,
                M_each_shared=M_each_shared,
                Q_each_shared=Q_each_shared,
                prop_L=auto_prop_L(N),
            )
            opt_mc.append(db(res["sinr_opt"]))
            pglq_mc.append(db(res["sinr_pglq"]))
            grdr_mc.append(db(res["sinr_grdr"]))
            toep_mc.append(db(res["sinr_toep"]))
            prop_mc.append(db(res["sinr_prop"]))

        sinr_opt_list.append(np.mean(opt_mc))
        sinr_pglq_list.append(np.mean(pglq_mc))
        sinr_grdr_list.append(np.mean(grdr_mc))
        sinr_toep_list.append(np.mean(toep_mc))
        sinr_prop_list.append(np.mean(prop_mc))
        print(f"[SNR sweep] input SNR = {snr_db:>5.1f} dB done")

    sinr_opt_list = np.array(sinr_opt_list)
    sinr_pglq_list = np.array(sinr_pglq_list)
    sinr_grdr_list = np.array(sinr_grdr_list)
    sinr_toep_list = np.array(sinr_toep_list)
    sinr_prop_list = np.array(sinr_prop_list)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    ax.plot(snr_list, sinr_opt_list,
            marker='D', markevery=1, markerfacecolor='white', markeredgewidth=0.9,
            color=COLORS["optimal"], linestyle='-.', linewidth=2.0, label='Optimal')
    ax.plot(snr_list, sinr_pglq_list,
            marker='s', markevery=1, markerfacecolor='white', markeredgewidth=0.9,
            color=COLORS["pglq"], linestyle='--', linewidth=2.0, label='PGLQ-Full [12]')
    ax.plot(snr_list, sinr_grdr_list,
            marker='o', markevery=1, markerfacecolor='white', markeredgewidth=0.9,
            color=COLORS["grdr"], linestyle=':', linewidth=2.0, label='GRDR-CMR [25]')
    ax.plot(snr_list, sinr_toep_list,
            marker='P', markevery=1, markerfacecolor='white', markeredgewidth=0.9,
            color=COLORS["toep"], linestyle=(0, (5, 2)), linewidth=2.0, label='Toeplitz-SSBF [26]')
    ax.plot(snr_list, sinr_prop_list,
            marker='^', markevery=1, markerfacecolor='white', markeredgewidth=0.9,
            color=COLORS["proposed"], linestyle='-', linewidth=2.25, label='Proposed SCC-BF')

    ax.set_xlabel('Input SNR (dB)')
    ax.set_ylabel('Output SINR (dB)')
    setup_axis(ax, add_minor=True)
    add_soft_legend(ax)

    x1 = 5
    x2 = 8
    y1 = 23
    y2 = 28

    axins = inset_axes(ax, width="28%", height="25%", loc='center right', borderpad=1.0)
    axins.plot(snr_list, sinr_opt_list, color=COLORS["optimal"], linestyle='-.', linewidth=1.25)
    axins.plot(snr_list, sinr_pglq_list, color=COLORS["pglq"], linestyle='--', linewidth=1.25)
    axins.plot(snr_list, sinr_grdr_list, color=COLORS["grdr"], linestyle=':', linewidth=1.25)
    axins.plot(snr_list, sinr_toep_list, color=COLORS["toep"], linestyle=(0, (5, 2)), linewidth=1.25)
    axins.plot(snr_list, sinr_prop_list, color=COLORS["proposed"], linestyle='-', linewidth=1.4)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    setup_inset_axis(axins)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.75)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_snr_vs_output_sinr_same_nodes")
    plt.show()


# ============================================================
# Figure 3: N vs Output SINR
# ============================================================
def sweep_array_size_plot(
    N_list=(8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 100, 150, 200, 250, 300),
    K=64,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=40.0,
    main_half_width=4.0,
    mc_runs=20,
    out_dir="./outputs_pub",
    M_each_shared=12,
    Q_each_shared=17,
):
    sinr_opt_list = []
    sinr_pglq_list = []
    sinr_grdr_list = []
    sinr_toep_list = []
    sinr_prop_list = []

    N_list = np.array(N_list)

    for N in N_list:
        opt_mc = []
        pglq_mc = []
        grdr_mc = []
        toep_mc = []
        prop_mc = []

        prop_L = auto_prop_L(N)

        for mc in range(mc_runs):
            res = run_methods_same_nodes(
                N=N,
                K=K,
                theta_s=theta_s,
                theta_j=theta_j,
                snr_db=snr_db,
                inr_db=inr_db,
                main_half_width=main_half_width,
                seed=3000 + mc,
                dense_P_each=300,
                M_each_shared=M_each_shared,
                Q_each_shared=Q_each_shared,
                prop_L=prop_L,
            )
            opt_mc.append(db(res["sinr_opt"]))
            pglq_mc.append(db(res["sinr_pglq"]))
            grdr_mc.append(db(res["sinr_grdr"]))
            toep_mc.append(db(res["sinr_toep"]))
            prop_mc.append(db(res["sinr_prop"]))

        sinr_opt_list.append(np.mean(opt_mc))
        sinr_pglq_list.append(np.mean(pglq_mc))
        sinr_grdr_list.append(np.mean(grdr_mc))
        sinr_toep_list.append(np.mean(toep_mc))
        sinr_prop_list.append(np.mean(prop_mc))
        print(f"[N sweep / SINR] N = {N:>3d}, L = {prop_L:>2d} done")

    sinr_opt_list = np.array(sinr_opt_list)
    sinr_pglq_list = np.array(sinr_pglq_list)
    sinr_grdr_list = np.array(sinr_grdr_list)
    sinr_toep_list = np.array(sinr_toep_list)
    sinr_prop_list = np.array(sinr_prop_list)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.plot(N_list, sinr_opt_list,
            marker='D', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["optimal"], linestyle='-.', linewidth=2.0, label='Optimal')
    ax.plot(N_list, sinr_pglq_list,
            marker='s', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["pglq"], linestyle='--', linewidth=2.0, label='PGLQ-Full [12]')
    ax.plot(N_list, sinr_grdr_list,
            marker='o', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["grdr"], linestyle=':', linewidth=2.0, label='GRDR-CMR [25]')
    ax.plot(N_list, sinr_toep_list,
            marker='P', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["toep"], linestyle=(0, (5, 2)), linewidth=2.0, label='Toeplitz-SSBF [26]')
    ax.plot(N_list, sinr_prop_list,
            marker='^', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["proposed"], linestyle='-', linewidth=2.25, label='Proposed SCC-BF')

    ax.set_xlabel('Number of Array Elements')
    ax.set_ylabel('Output SINR (dB)')
    setup_axis(ax, add_minor=True)
    add_soft_legend(ax)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_array_size_vs_output_sinr_same_nodes")
    plt.show()


# ============================================================
# Figure 4: N vs Runtime
# ============================================================
def sweep_array_size_runtime_plot(
    N_list=(8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 100, 150, 200, 250, 300),
    K=64,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_db=10.0,
    inr_db=40.0,
    main_half_width=4.0,
    mc_runs=10,
    out_dir="./outputs_pub",
    use_logy=False,
    M_each_shared=12,
    Q_each_shared=17,
):
    N_list = np.array(N_list)
    t_pglq_list = []
    t_grdr_list = []
    t_toep_list = []
    t_prop_list = []

    for N in N_list:
        pglq_mc = []
        grdr_mc = []
        toep_mc = []
        prop_mc = []

        prop_L = auto_prop_L(N)

        for mc in range(mc_runs):
            res = run_methods_same_nodes(
                N=N,
                K=K,
                theta_s=theta_s,
                theta_j=theta_j,
                snr_db=snr_db,
                inr_db=inr_db,
                main_half_width=main_half_width,
                seed=5000 + mc,
                dense_P_each=300,
                M_each_shared=M_each_shared,
                Q_each_shared=Q_each_shared,
                prop_L=prop_L,
            )
            pglq_mc.append(res["t_pglq"])
            grdr_mc.append(res["t_grdr"])
            toep_mc.append(res["t_toep"])
            prop_mc.append(res["t_prop"])

        t_pglq_list.append(np.mean(pglq_mc))
        t_grdr_list.append(np.mean(grdr_mc))
        t_toep_list.append(np.mean(toep_mc))
        t_prop_list.append(np.mean(prop_mc))
        print(f"[N sweep / Runtime] N = {N:>3d}, L = {prop_L:>2d} done")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.plot(N_list, t_pglq_list,
            marker='s', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["pglq"], linestyle='--', linewidth=2.0, label='PGLQ-Full [12]')
    ax.plot(N_list, t_grdr_list,
            marker='o', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["grdr"], linestyle=':', linewidth=2.0, label='GRDR-CMR [25]')
    ax.plot(N_list, t_toep_list,
            marker='P', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["toep"], linestyle=(0, (5, 2)), linewidth=2.0, label='Toeplitz-SSBF [26]')
    ax.plot(N_list, t_prop_list,
            marker='^', markevery=1, markerfacecolor='white', markeredgewidth=0.85,
            color=COLORS["proposed"], linestyle='-', linewidth=2.25, label='Proposed SCC-BF')

    ax.set_xlabel('Number of Array Elements')
    ax.set_ylabel('Runtime (s)')

    if use_logy:
        ax.set_yscale('log')

    setup_axis(ax, add_minor=True)
    add_soft_legend(ax)

    fig.tight_layout()
    save_name = "fig_array_size_vs_runtime_same_nodes_log" if use_logy else "fig_array_size_vs_runtime_same_nodes"
    save_figure(fig, out_dir, save_name)
    plt.show()


# ============================================================
# Figure 5: Quadrature error vs nodes using the actual integrand
# ============================================================
def reference_integral_piecewise(func, intervals, num_per_interval=20001):
    """
    High-accuracy reference integral over piecewise intervals using
    dense composite trapezoidal rule.
    """
    val = 0.0 + 0.0j
    for (ua, ub) in intervals:
        eps = 1e-8
        u = np.linspace(ua + eps, ub - eps, num_per_interval)
        fu = func(u)
        val += np.trapz(fu, u)
    return val


def quadrature_integral_piecewise(func, intervals, M_each=12, Q=17, rule="gl"):
    """
    Piecewise quadrature over the same sidelobe intervals used in beamforming.
    """
    val = 0.0 + 0.0j

    for (ua, ub) in intervals:
        bounds = np.linspace(ua, ub, M_each + 1)
        for i in range(M_each):
            a = bounds[i]
            b = bounds[i + 1]

            if rule.lower() == "gl":
                x, alpha = gauss_legendre_on_interval(a, b, Q=Q)
            elif rule.lower() == "cc":
                x, alpha = clenshaw_curtis_on_interval(a, b, Q=Q)
            else:
                raise ValueError(f"Unknown quadrature rule: {rule}")

            val += np.sum(alpha * func(x))

    return val


def plot_quadrature_error_vs_nodes_actual_integrand(
    Rhat,
    theta0_deg=0.0,
    main_half_width_deg=4.0,
    ell_list=(0, 10, 30, 60),
    M_each=12,
    Q_list=(3, 5, 7, 9, 13, 17, 21, 25, 33),
    reg=1e-6,
    out_dir="./outputs_pub",
):
    """
    Compare GL and CC quadrature errors versus node count using the actual
    lag-domain integrand appearing in the reconstruction:
        f_l(u) = [jac(u)/q(u)] * exp(j*pi*l*u)

    where
        jac(u) = 1/sqrt(1-u^2),
        q(u)   = a^H(u) R^{-1} a(u).
    """
    N = Rhat.shape[0]
    Rinv = sla.inv(Rhat + reg * np.eye(N))
    intervals = make_sidelobe_u_intervals(theta0_deg, main_half_width_deg)

    def make_integrand(ell):
        def func(u):
            u = np.asarray(u)
            out = np.zeros_like(u, dtype=np.complex128)

            for idx, uu in enumerate(u):
                a = steering_u(float(uu), N)
                q = np.real(a.conj().T @ Rinv @ a)
                q = max(q, 1e-10)
                jac = 1.0 / np.sqrt(max(1.0 - float(uu) * float(uu), 1e-10))
                out[idx] = (jac / q) * np.exp(1j * np.pi * ell * float(uu))
            return out

        return func

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    markers_gl = ['s', 'o', 'D', '<']
    markers_cc = ['^', 'v', '>', 'p']

    total_nodes = 2 * M_each * np.array(Q_list)

    for i, ell in enumerate(ell_list):
        func = make_integrand(ell)

        ref_val = reference_integral_piecewise(
            func,
            intervals,
            num_per_interval=20001
        )

        err_gl = []
        err_cc = []

        for Q in Q_list:
            val_gl = quadrature_integral_piecewise(
                func, intervals, M_each=M_each, Q=Q, rule="gl"
            )
            val_cc = quadrature_integral_piecewise(
                func, intervals, M_each=M_each, Q=Q, rule="cc"
            )

            err_gl.append(np.abs(val_gl - ref_val))
            err_cc.append(np.abs(val_cc - ref_val))

        err_gl = np.maximum(np.array(err_gl), 1e-16)
        err_cc = np.maximum(np.array(err_cc), 1e-16)

        ax.plot(
            total_nodes, err_gl,
            linestyle='--',
            linewidth=2.0,
            marker=markers_gl[i % len(markers_gl)],
            markerfacecolor='white',
            markeredgewidth=0.9,
            color=COLORS["pglq"],
            label=fr'GL, $\ell={ell}$'
        )

        ax.plot(
            total_nodes, err_cc,
            linestyle='-',
            linewidth=2.2,
            marker=markers_cc[i % len(markers_cc)],
            markerfacecolor='white',
            markeredgewidth=0.9,
            color=COLORS["proposed"],
            label=fr'CC, $\ell={ell}$'
        )

    ax.set_yscale('log')
    ax.set_xlabel('Total Number of Quadrature Nodes')
    ax.set_ylabel('Absolute Integration Error')
    setup_axis(ax, add_minor=True)
    add_soft_legend(ax)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_quadrature_error_vs_nodes_actual_integrand")
    plt.show()


# ============================================================
# Added Figure 6: SINR-Runtime trade-off
# ============================================================
def summarize_tradeoff_same_scene(
    N=32,
    K=64,
    theta_s=0.0,
    theta_j=(-42.0, 56.0),
    snr_db=10.0,
    inr_db=40.0,
    main_half_width=4.0,
    mc_runs=20,
    dense_P_each=300,
    M_each_shared=12,
    Q_each_shared=17,
    prop_L=None,
    grdr_rank_ratio=0.5,
):
    """
    Average SINR and runtime over multiple MC runs under one fixed scenario.
    This summary is used for the SINR-Runtime trade-off figure and the table.
    """
    if prop_L is None:
        prop_L = auto_prop_L(N)

    stats = {
        "optimal": {"sinr_db": [], "runtime": []},
        "dense": {"sinr_db": [], "runtime": []},
        "pglq": {"sinr_db": [], "runtime": []},
        "grdr": {"sinr_db": [], "runtime": []},
        "toep": {"sinr_db": [], "runtime": []},
        "prop": {"sinr_db": [], "runtime": []},
    }

    for mc in range(mc_runs):
        res = run_methods_same_nodes(
            N=N,
            K=K,
            theta_s=theta_s,
            theta_j=theta_j,
            snr_db=snr_db,
            inr_db=inr_db,
            main_half_width=main_half_width,
            seed=7000 + mc,
            dense_P_each=dense_P_each,
            M_each_shared=M_each_shared,
            Q_each_shared=Q_each_shared,
            prop_L=prop_L,
            grdr_rank_ratio=grdr_rank_ratio,
        )

        stats["optimal"]["sinr_db"].append(db(res["sinr_opt"]))
        stats["optimal"]["runtime"].append(res["t_opt"])

        stats["dense"]["sinr_db"].append(db(res["sinr_dense"]))
        stats["dense"]["runtime"].append(res["t_dense"])

        stats["pglq"]["sinr_db"].append(db(res["sinr_pglq"]))
        stats["pglq"]["runtime"].append(res["t_pglq"])

        stats["grdr"]["sinr_db"].append(db(res["sinr_grdr"]))
        stats["grdr"]["runtime"].append(res["t_grdr"])

        stats["toep"]["sinr_db"].append(db(res["sinr_toep"]))
        stats["toep"]["runtime"].append(res["t_toep"])

        stats["prop"]["sinr_db"].append(db(res["sinr_prop"]))
        stats["prop"]["runtime"].append(res["t_prop"])

        print(f"[Trade-off summary] MC {mc + 1:>2d}/{mc_runs} done")

    summary = {
        "meta": {
            "N": N,
            "K": K,
            "theta_s": theta_s,
            "theta_j": theta_j,
            "snr_db": snr_db,
            "inr_db": inr_db,
            "main_half_width": main_half_width,
            "mc_runs": mc_runs,
            "dense_P_each": dense_P_each,
            "M_each_shared": M_each_shared,
            "Q_each_shared": Q_each_shared,
            "B": 2 * M_each_shared * Q_each_shared,
            "prop_L": prop_L,
            "grdr_rank_ratio": grdr_rank_ratio,
        },
        "methods": {}
    }

    for key, val in stats.items():
        summary["methods"][key] = {
            "mean_sinr_db": float(np.mean(val["sinr_db"])),
            "std_sinr_db": float(np.std(val["sinr_db"])),
            "mean_runtime": float(np.mean(val["runtime"])),
            "std_runtime": float(np.std(val["runtime"])),
        }

    return summary


def plot_sinr_runtime_tradeoff(
    summary,
    out_dir="./outputs_pub",
    include_dense=False,
    use_logx=True,
):
    methods_order = ["optimal", "pglq", "grdr", "toep", "prop"]
    if include_dense:
        methods_order.insert(1, "dense")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    for key in methods_order:
        item = summary["methods"][key]
        meta = METHOD_META[key]

        ax.scatter(
            item["mean_runtime"],
            item["mean_sinr_db"],
            s=95,
            marker=meta["marker"],
            facecolors='white',
            edgecolors=meta["color"],
            linewidths=1.3,
            label=meta["label"],
            zorder=4,
        )

        ax.annotate(
            meta["label"],
            (item["mean_runtime"], item["mean_sinr_db"]),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=11,
            color=meta["color"],
        )

    if use_logx:
        ax.set_xscale('log')

    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Output SINR (dB)')
    setup_axis(ax, add_minor=False)
    add_soft_legend(ax, loc='lower right')

    fig.tight_layout()
    save_name = "fig_sinr_runtime_tradeoff_same_scene"
    save_figure(fig, out_dir, save_name)
    plt.show()
def plot_group_snr_vs_sinr_by_Q_with_M_sweeps(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_list=np.arange(-10, 31, 5),
    inr_db=43.0,
    main_half_width=4.0,
    mc_runs=10,
    Q_list=(3, 5, 7, 9, 13, 17),
    M_list=(2, 3, 4, 6, 8, 12, 16),
    out_dir="./outputs_pub_same_nodes_gl_vs_cc",
):
    """
    6 子图：
      - 每个子图固定一个 Q
      - 子图中画多个 M
      - 额外加一条 without partitioning (M=1)
      - 每个子图都单独加 x / y 轴标注
      - 子图标题放在下方
      - 只对 Q = 9 和 Q = 17 添加 inset 放大图
    """
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 14.2), sharex=False, sharey=False)
    axes = axes.ravel()

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    markers = ['s', '^', 'o', 'D', 'v', 'P', 'X']

    for idx, Q_each in enumerate(Q_list):
        ax = axes[idx]

        # 缓存曲线，方便主图和 inset 复用
        curve_cache = {}

        # -----------------------------
        # 多条 M 曲线
        # -----------------------------
        for m_idx, M_each in enumerate(M_list):
            y_vals = []

            for snr_db in snr_list:
                vals_mc = []
                for mc in range(mc_runs):
                    res = run_scc_single_case(
                        N=N,
                        K=K,
                        theta_s=theta_s,
                        theta_j=theta_j,
                        snr_db=snr_db,
                        inr_db=inr_db,
                        main_half_width=main_half_width,
                        seed=30000 + idx * 1000 + m_idx * 100 + mc,
                        M_each=M_each,
                        Q_each=Q_each,
                        prop_L=auto_prop_L(N),
                    )
                    vals_mc.append(db(res["sinr_prop"]))
                y_vals.append(np.mean(vals_mc))

            y_vals = np.array(y_vals)
            curve_cache[f"M={M_each}"] = y_vals

            ax.plot(
                snr_list, y_vals,
                marker=markers[m_idx % len(markers)],
                markerfacecolor='none',
                markeredgewidth=0.9,
                linewidth=1.5,
                label=f'M={M_each}'
            )

        # -----------------------------
        # without partitioning (M=1)
        # -----------------------------
        y_no_part = []

        for snr_db in snr_list:
            vals_mc = []
            for mc in range(mc_runs):
                res = run_scc_single_case(
                    N=N,
                    K=K,
                    theta_s=theta_s,
                    theta_j=theta_j,
                    snr_db=snr_db,
                    inr_db=inr_db,
                    main_half_width=main_half_width,
                    seed=40000 + idx * 1000 + mc,
                    M_each=1,
                    Q_each=Q_each,
                    prop_L=auto_prop_L(N),
                )
                vals_mc.append(db(res["sinr_no_part"]))
            y_no_part.append(np.mean(vals_mc))

        y_no_part = np.array(y_no_part)
        curve_cache["Without partitioning"] = y_no_part

        ax.plot(
            snr_list, y_no_part,
            color='red',
            linestyle='-',
            marker='o',
            markerfacecolor='none',
            markeredgewidth=0.9,
            linewidth=1.4,
            label='Without partitioning'
        )

        # -----------------------------
        # 每个子图都单独加横纵轴
        # -----------------------------
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Output SINR (dB)', fontsize=12)

        ax.set_xlim(min(snr_list), max(snr_list))
        ax.set_ylim(-20, 60)
        ax.set_xticks(np.arange(-10, 31, 5))
        ax.set_yticks(np.arange(-20, 61, 10))
        ax.grid(True, which='major', color='#C8C8C8', linestyle='-', linewidth=0.45, alpha=0.9)
        ax.tick_params(direction='in', top=True, right=True)

        # -----------------------------
        # 子图标题放到下方
        # -----------------------------
        ax.text(
            0.5, -0.15,
            f'{panel_labels[idx]} Q = {Q_each}',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=12
        )

        leg = ax.legend(
            loc='upper left',
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.3,
            handlelength=2.2,
            handletextpad=0.4,
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#666666')
        leg.get_frame().set_linewidth(0.8)

        # -----------------------------
        # 只对 Q = 9 和 Q = 17 加放大镜
        # -----------------------------
        if Q_each in (9, 17):
            axins = inset_axes(ax, width="34%", height="28%", loc='lower right', borderpad=1.0)

            # 先画所有 M 曲线
            for m_idx, M_each in enumerate(M_list):
                axins.plot(
                    snr_list,
                    curve_cache[f"M={M_each}"],
                    marker=markers[m_idx % len(markers)],
                    markerfacecolor='none',
                    markeredgewidth=0.7,
                    linewidth=1.0
                )

            # 再画 no-partition
            axins.plot(
                snr_list,
                curve_cache["Without partitioning"],
                color='red',
                linestyle='-',
                marker='o',
                markerfacecolor='none',
                markeredgewidth=0.7,
                linewidth=1.0
            )

            # 放大高 SNR 区域
            x1, x2 = 18, 20
            mask = (snr_list >= x1) & (snr_list <= x2)

            local_vals = []
            for M_each in M_list:
                local_vals.append(curve_cache[f"M={M_each}"][mask])
            local_vals.append(curve_cache["Without partitioning"][mask])

            local_vals = np.concatenate(local_vals)
            y1 = 22
            y2 = 35

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.grid(True, which='major', color='#D0D0D0', linestyle='-', linewidth=0.3)
            axins.tick_params(direction='in', top=True, right=True, labelsize=7, length=2)

            for spine in axins.spines.values():
                spine.set_linewidth(0.8)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.75)

        print(f"[Group by Q] Q={Q_each} done")

    fig.tight_layout(h_pad=3.8, w_pad=2.2)
    save_figure(fig, out_dir, "fig_group_snr_vs_sinr_by_Q_with_M_sweeps")
    plt.show()

    def plot_group_snr_vs_sinr_by_M_with_Q_sweeps(
            N=32,
            K=128,
            theta_s=0.0,
            theta_j=(-46.0, 32.0),
            snr_list=np.arange(-10, 31, 5),
            inr_db=43.0,
            main_half_width=4.0,
            mc_runs=10,
            M_fixed_list=(2, 3, 4, 6, 8, 12),
            Q_list=(3, 5, 7, 9, 13, 17),
            out_dir="./outputs_pub_same_nodes_gl_vs_cc",
    ):
        """
        6子图：
          - 每个子图固定一个 M
          - 子图中画多个 Q
          - 只对 M = 6 和 M = 12 添加 inset 放大图
          - 每个子图都单独加 x / y 轴标注
          - 子图标题放在下方
        """
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.8), sharex=False, sharey=False)
        axes = axes.ravel()

        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        markers = ['s', '^', 'o', 'D', 'v', 'P']

        for idx, M_each in enumerate(M_fixed_list):
            ax = axes[idx]

            # 缓存每条曲线，后面主图和 inset 都复用
            curve_cache = {}

            for q_idx, Q_each in enumerate(Q_list):
                y_vals = []

                for snr_db in snr_list:
                    vals_mc = []
                    for mc in range(mc_runs):
                        res = run_scc_single_case(
                            N=N,
                            K=K,
                            theta_s=theta_s,
                            theta_j=theta_j,
                            snr_db=snr_db,
                            inr_db=inr_db,
                            main_half_width=main_half_width,
                            seed=50000 + idx * 1000 + q_idx * 100 + mc,
                            M_each=M_each,
                            Q_each=Q_each,
                            prop_L=auto_prop_L(N),
                        )
                        vals_mc.append(db(res["sinr_prop"]))
                    y_vals.append(np.mean(vals_mc))

                y_vals = np.array(y_vals)
                curve_cache[Q_each] = y_vals

                ax.plot(
                    snr_list, y_vals,
                    marker=markers[q_idx % len(markers)],
                    markerfacecolor='none',
                    markeredgewidth=0.9,
                    linewidth=1.5,
                    label=f'Q={Q_each}'
                )

            # -----------------------------
            # 每个子图都单独加横纵轴标注
            # -----------------------------
            ax.set_xlabel('SNR (dB)', fontsize=12)
            ax.set_ylabel('Output SINR (dB)', fontsize=12)

            ax.set_xlim(min(snr_list), max(snr_list))
            ax.set_ylim(-20, 60)
            ax.set_xticks(np.arange(-10, 31, 5))
            ax.set_yticks(np.arange(-20, 61, 10))
            ax.grid(True, which='major', color='#C8C8C8', linestyle='-', linewidth=0.45, alpha=0.9)
            ax.tick_params(direction='in', top=True, right=True)

            # -----------------------------
            # 子图标题放到下方
            # -----------------------------
            ax.text(
                0.5, -0.15,
                f'{panel_labels[idx]} M = {M_each}',
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=12
            )

            leg = ax.legend(
                loc='upper left',
                fontsize=8,
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                borderpad=0.3,
                handlelength=2.2,
                handletextpad=0.4,
            )
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#666666')
            leg.get_frame().set_linewidth(0.8)

            # -----------------------------
            # 只给 M = 6 和 M = 12 加放大镜
            # -----------------------------
            if M_each in (6, 12):
                axins = inset_axes(ax, width="34%", height="28%", loc='lower right', borderpad=1.0)

                for q_idx, Q_each in enumerate(Q_list):
                    axins.plot(
                        snr_list,
                        curve_cache[Q_each],
                        marker=markers[q_idx % len(markers)],
                        markerfacecolor='none',
                        markeredgewidth=0.7,
                        linewidth=1.0
                    )

                # 放大高 SNR 区域
                x1, x2 = 18, 20
                mask = (snr_list >= x1) & (snr_list <= x2)

                local_vals = np.concatenate([curve_cache[Q][mask] for Q in Q_list])
                y1 = np.min(local_vals) - 1.0
                y2 = np.max(local_vals) + 1.0

                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.grid(True, which='major', color='#D0D0D0', linestyle='-', linewidth=0.3)
                axins.tick_params(direction='in', top=True, right=True, labelsize=7, length=2)

                for spine in axins.spines.values():
                    spine.set_linewidth(0.8)

                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.75)

            print(f"[Group by M] M={M_each} done")

        # 注意：标题放下方以后，h_pad 要大一点
        fig.tight_layout(h_pad=3.6, w_pad=2.2)
        save_figure(fig, out_dir, "fig_group_snr_vs_sinr_by_M_with_Q_sweeps")
        plt.show()

# ============================================================
# Added Table: Computational-cost vs performance
# ============================================================
def estimate_complexity_proxy(
    method,
    N,
    B,
    L=4,
    rank_ratio=0.5,
    dense_P_each=300,
):
    """
    Very rough FLOPs proxy / complexity proxy.
    It is intended for relative comparison only, not for exact instruction-level FLOPs.
    """
    r = max(4, min(N, int(np.ceil(rank_ratio * N))))
    B_dense = 2 * dense_P_each

    if method == "optimal":
        # one Hermitian solve
        return (1.0 / 3.0) * (N ** 3)

    if method == "dense":
        # inverse + dense-node reconstruction + full solve
        return 2.0 * (N ** 3) + 2.0 * B_dense * (N ** 2)

    if method == "pglq":
        # inverse + B full-node evaluations + full solve
        return 2.0 * (N ** 3) + 2.0 * B * (N ** 2)

    if method == "grdr":
        # random projection + reduced inverse + full reconstruction + full solve
        return (
            2.0 * (N ** 2) * r +
            N * (r ** 2) +
            (r ** 3) +
            B * ((N ** 2) + (r ** 2)) +
            (N ** 3)
        )

    if method == "toep":
        # Toeplitz projection + low-rank surrogate + B lag updates + Toeplitz solve
        return (
            (N ** 2) +
            2.0 * (N ** 2) * L +
            B * (2.0 * N * L + N) +
            (N ** 2)
        )

    if method == "prop":
        # low-rank surrogate + B lag updates + Toeplitz solve
        return (
            2.0 * (N ** 2) * L +
            B * (2.0 * N * L + N) +
            (N ** 2)
        )

    raise ValueError(f"Unknown method: {method}")


def build_cost_performance_table(summary, include_dense=False):
    meta = summary["meta"]
    N = meta["N"]
    B = meta["B"]
    L = meta["prop_L"]
    rank_ratio = meta["grdr_rank_ratio"]
    dense_P_each = meta["dense_P_each"]

    methods_order = ["optimal", "pglq", "grdr", "toep", "prop"]
    if include_dense:
        methods_order.insert(1, "dense")

    opt_sinr = summary["methods"]["optimal"]["mean_sinr_db"]
    prop_runtime = summary["methods"]["prop"]["mean_runtime"]
    prop_flops = estimate_complexity_proxy(
        "prop", N=N, B=B, L=L, rank_ratio=rank_ratio, dense_P_each=dense_P_each
    )

    rows = []
    for key in methods_order:
        item = summary["methods"][key]
        flops_proxy = estimate_complexity_proxy(
            key, N=N, B=B, L=L, rank_ratio=rank_ratio, dense_P_each=dense_P_each
        )
        rows.append({
            "Method": METHOD_META[key]["label"],
            "Mean Output SINR (dB)": item["mean_sinr_db"],
            "SINR Std (dB)": item["std_sinr_db"],
            "Mean Runtime (s)": item["mean_runtime"],
            "Runtime Std (s)": item["std_runtime"],
            "Runtime Ratio to Proposed": item["mean_runtime"] / max(prop_runtime, 1e-15),
            "FLOPs Proxy": flops_proxy,
            "FLOPs Ratio to Proposed": flops_proxy / max(prop_flops, 1e-15),
            "SINR Gap to Optimal (dB)": opt_sinr - item["mean_sinr_db"],
        })

    return rows

def plot_group_snr_vs_sinr_by_M_with_Q_sweeps(
    N=32,
    K=128,
    theta_s=0.0,
    theta_j=(-46.0, 32.0),
    snr_list=np.arange(-10, 31, 5),
    inr_db=43.0,
    main_half_width=4.0,
    mc_runs=10,
    M_fixed_list=(2, 3, 4, 6, 8, 12),
    Q_list=(3, 5, 7, 9, 13, 17),
    out_dir="./outputs_pub_same_nodes_gl_vs_cc",
):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 14.0), sharex=False, sharey=False)
    axes = axes.ravel()

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    markers = ['s', '^', 'o', 'D', 'v', 'P']

    for idx, M_each in enumerate(M_fixed_list):
        ax = axes[idx]
        curve_cache = {}

        for q_idx, Q_each in enumerate(Q_list):
            y_vals = []

            for snr_db in snr_list:
                vals_mc = []
                for mc in range(mc_runs):
                    res = run_scc_single_case(
                        N=N,
                        K=K,
                        theta_s=theta_s,
                        theta_j=theta_j,
                        snr_db=snr_db,
                        inr_db=inr_db,
                        main_half_width=main_half_width,
                        seed=50000 + idx * 1000 + q_idx * 100 + mc,
                        M_each=M_each,
                        Q_each=Q_each,
                        prop_L=auto_prop_L(N),
                    )
                    vals_mc.append(db(res["sinr_prop"]))
                y_vals.append(np.mean(vals_mc))

            y_vals = np.array(y_vals)
            curve_cache[Q_each] = y_vals

            ax.plot(
                snr_list, y_vals,
                marker=markers[q_idx % len(markers)],
                markerfacecolor='none',
                markeredgewidth=0.9,
                linewidth=1.5,
                label=f'Q={Q_each}'
            )

        # 每个子图都加横纵轴
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Output SINR (dB)', fontsize=12)

        ax.set_xlim(min(snr_list), max(snr_list))
        ax.set_ylim(-20, 60)
        ax.set_xticks(np.arange(-10, 31, 5))
        ax.set_yticks(np.arange(-20, 61, 10))
        ax.grid(True, which='major', color='#C8C8C8', linestyle='-', linewidth=0.45, alpha=0.9)
        ax.tick_params(direction='in', top=True, right=True)

        # 标题放下方
        ax.text(
            0.5, -0.15,
            f'{panel_labels[idx]} M = {M_each}',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=12
        )

        leg = ax.legend(
            loc='upper left',
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.3,
            handlelength=2.2,
            handletextpad=0.4,
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#666666')
        leg.get_frame().set_linewidth(0.8)

        # 只对 M=6 和 M=12 加放大镜
        if M_each in (6, 12):
            axins = inset_axes(ax, width="34%", height="28%", loc='lower right', borderpad=1.0)

            for q_idx, Q_each in enumerate(Q_list):
                axins.plot(
                    snr_list,
                    curve_cache[Q_each],
                    marker=markers[q_idx % len(markers)],
                    markerfacecolor='none',
                    markeredgewidth=0.7,
                    linewidth=1.0
                )

            x1, x2 = 18, 20
            mask = (snr_list >= x1) & (snr_list <= x2)
            local_vals = np.concatenate([curve_cache[Q][mask] for Q in Q_list])
            y1 = 28
            y2 = 35

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.grid(True, which='major', color='#D0D0D0', linestyle='-', linewidth=0.3)
            axins.tick_params(direction='in', top=True, right=True, labelsize=7, length=2)

            for spine in axins.spines.values():
                spine.set_linewidth(0.8)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.75)

        print(f"[Group by M] M={M_each} done")

    fig.tight_layout(h_pad=3.8, w_pad=2.2)
    save_figure(fig, out_dir, "fig_group_snr_vs_sinr_by_M_with_Q_sweeps")
    plt.show()
def print_cost_performance_table(rows):
    headers = [
        "Method",
        "Mean Output SINR (dB)",
        "SINR Std (dB)",
        "Mean Runtime (s)",
        "Runtime Std (s)",
        "Runtime Ratio to Proposed",
        "FLOPs Proxy",
        "FLOPs Ratio to Proposed",
        "SINR Gap to Optimal (dB)",
    ]

    def _fmt(h, v):
        if h == "Method":
            return str(v)
        if "FLOPs Proxy" in h:
            return f"{v:.3e}"
        return f"{v:.6f}"

    widths = {}
    for h in headers:
        widths[h] = max(len(h), max(len(_fmt(h, row[h])) for row in rows))

    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print("\n========== Computational-cost vs Performance Table ==========")
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(_fmt(h, row[h]).ljust(widths[h]) for h in headers))


def save_cost_performance_table(rows, out_dir="./outputs_pub", name="table_cost_vs_performance"):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{name}.csv")
    txt_path = os.path.join(out_dir, f"{name}.txt")

    headers = list(rows[0].keys())

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Computational-cost vs Performance Table\n")
        f.write("NOTE: FLOPs Proxy is a relative complexity proxy, not exact hardware FLOPs.\n\n")
        f.write("\t".join(headers) + "\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row[h]
                if h == "Method":
                    vals.append(str(v))
                elif "FLOPs Proxy" in h:
                    vals.append(f"{v:.6e}")
                else:
                    vals.append(f"{v:.6f}")
            f.write("\t".join(vals) + "\n")

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {txt_path}")


# ============================================================
# Main demo
# ============================================================
def main():
    out_dir = "./outputs_pub_same_nodes_gl_vs_cc"

    N = 32
    K = 500
    theta_s = 0.0
    theta_j = (-42.0, 56.0)
    snr_db = 10.0
    inr_db = 40.0
    main_half_width = 4.0

    M_each_shared = 12
    Q_each_shared = 17
    prop_L = auto_prop_L(N)

    res = run_methods_same_nodes(
        N=N,
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        seed=1,
        dense_P_each=300,
        M_each_shared=M_each_shared,
        Q_each_shared=Q_each_shared,
        prop_L=prop_L,
    )

    print("========== Shared Quadrature Budget ==========")
    print("Literature method 1: PGLQ-Full [12]")
    print("Literature method 2: GRDR-CMR [25]")
    print("Literature method 3: Toeplitz-SSBF [26]")
    print("Proposed method    : Proposed SCC-BF")
    print(f"M_each             : {M_each_shared}")
    print(f"Q_each             : {Q_each_shared}")
    print(f"Total nodes        : {res['total_nodes']}   (same for all reconstruction methods)")
    print(f"Chosen L           : {res['prop_L']}")
    print(f"GRDR rank ratio    : {res['grdr_rank_ratio']:.2f}")

    print("\n========== Runtime ==========")
    print(f"Optimal            : {res['t_opt']:.6f} s")
    print(f"Dense IR           : {res['t_dense']:.6f} s   [not shown in main figures]")
    print(f"PGLQ-Full [12]     : {res['t_pglq']:.6f} s")
    print(f"GRDR-CMR [25]      : {res['t_grdr']:.6f} s")
    print(f"Toeplitz-SSBF [26] : {res['t_toep']:.6f} s")
    print(f"Proposed SCC-BF    : {res['t_prop']:.6f} s")

    print("\n========== Output SINR ==========")
    print(f"Optimal            : {db(res['sinr_opt']):.3f} dB")
    print(f"Dense IR           : {db(res['sinr_dense']):.3f} dB   [not shown in main figures]")
    print(f"PGLQ-Full [12]     : {db(res['sinr_pglq']):.3f} dB")
    print(f"GRDR-CMR [25]      : {db(res['sinr_grdr']):.3f} dB")
    print(f"Toeplitz-SSBF [26] : {db(res['sinr_toep']):.3f} dB")
    print(f"Proposed SCC-BF    : {db(res['sinr_prop']):.3f} dB")
   # plot_group_snr_vs_sinr_by_Q_with_M_sweeps(
   #      N=N,
   #      K=K,
   #      theta_s=theta_s,
   #      theta_j=theta_j,
   #      snr_list=np.arange(-10, 31, 5),
   #      inr_db=inr_db,
   #      main_half_width=main_half_width,
   #      mc_runs=10,
   #      Q_list=(3, 5, 7, 9, 13, 17),
   #      M_list=(2, 3, 4, 6, 8, 12, 16),
   #      out_dir=out_dir,
   #  )
   #  plot_group_snr_vs_sinr_by_M_with_Q_sweeps(
   #      N=N,
   #      K=K,
   #      theta_s=theta_s,
   #      theta_j=theta_j,
   #      snr_list=np.arange(-10, 31, 5),
   #      inr_db=inr_db,
   #      main_half_width=main_half_width,
   #      mc_runs=10,
   #      M_fixed_list=(2, 3, 4, 6, 8, 12),
   #      Q_list=(3, 5, 7, 9, 13, 17),
   #      out_dir=out_dir,
   #  )

    # plot_beampattern_comparison(
    #     res["w_opt"], res["w_pglq"], res["w_grdr"], res["w_toep"], res["w_prop"],
    #     N=N,
    #     theta_s=theta_s,
    #     theta_j=theta_j,
    #     out_dir=out_dir,
    # )
    #
    # plot_quadrature_error_vs_nodes_actual_integrand(
    #     Rhat=simulate_snapshots(
    #         N=N,
    #         K=K,
    #         theta_s_deg=theta_s,
    #         theta_j_deg=theta_j,
    #         snr_db=snr_db,
    #         inr_db=inr_db,
    #         seed=1
    #     )[1],
    #     theta0_deg=theta_s,
    #     main_half_width_deg=main_half_width,
    #     ell_list=(0, 10, 30, 60),
    #     M_each=M_each_shared,
    #     Q_list=(3, 5, 7, 9, 13, 17, 21, 25, 33),
    #     reg=1e-6,
    #     out_dir=out_dir,
    # )
    #
    # sweep_snr_plot(
    #     N=N,
    #     K=K,
    #     theta_s=theta_s,
    #     theta_j=theta_j,
    #     snr_list=np.arange(-10, 31, 5),
    #     inr_db=inr_db,
    #     main_half_width=main_half_width,
    #     mc_runs=20,
    #     out_dir=out_dir,
    #     M_each_shared=M_each_shared,
    #     Q_each_shared=Q_each_shared,
    # )
    #
    sweep_array_size_plot(
        N_list=(8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 100, 150, 200, 250, 300),
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=20,
        out_dir=out_dir,
        M_each_shared=M_each_shared,
        Q_each_shared=Q_each_shared,
    )

    sweep_array_size_runtime_plot(
        N_list=(8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 100, 150, 200, 250, 300),
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=10,
        out_dir=out_dir,
        use_logy=False,
        M_each_shared=M_each_shared,
        Q_each_shared=Q_each_shared,
    )

    sweep_array_size_runtime_plot(
        N_list=(8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 100, 150, 200, 250, 300),
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=10,
        out_dir=out_dir,
        use_logy=True,
        M_each_shared=M_each_shared,
        Q_each_shared=Q_each_shared,
    )

    # --------------------------------------------------------
    # Newly added: SINR-Runtime trade-off + cost-performance table
    # --------------------------------------------------------
    summary = summarize_tradeoff_same_scene(
        N=N,
        K=K,
        theta_s=theta_s,
        theta_j=theta_j,
        snr_db=snr_db,
        inr_db=inr_db,
        main_half_width=main_half_width,
        mc_runs=20,
        dense_P_each=300,
        M_each_shared=M_each_shared,
        Q_each_shared=Q_each_shared,
        prop_L=prop_L,
        grdr_rank_ratio=0.5,
    )

    plot_sinr_runtime_tradeoff(
        summary,
        out_dir=out_dir,
        include_dense=False,
        use_logx=True,
    )

    table_rows = build_cost_performance_table(summary, include_dense=False)
    print_cost_performance_table(table_rows)
    save_cost_performance_table(
        table_rows,
        out_dir=out_dir,
        name="table_cost_vs_performance_same_scene"
    )

if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     main_quadrature_rule_ablation()
