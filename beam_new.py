# -*- coding: utf-8 -*-
"""
2D planar-array extension of the proposed SCC-BF method
-------------------------------------------------------
Only the proposed method is simulated here, without comparison baselines.

This script generates and saves three separate figures:
  1) Top-view beampattern in the (u, v) domain
  2) Several 2D cuts at selected azimuth angles
  3) Angular contour map in the (theta, phi) domain

Dependencies:
  pip install numpy scipy matplotlib
"""

import os
import time
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


# ============================================================
# Global plot style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,

    "axes.linewidth": 1.2,
    "lines.linewidth": 2.4,
    "lines.markersize": 7,

    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "figure.dpi": 160,
})


# ============================================================
# Basic helpers
# ============================================================
def db(x, eps=1e-12):
    return 10.0 * np.log10(np.maximum(np.real(x), eps))


def uv_from_angles(theta_deg, phi_deg):
    """
    Direction-cosine parameterization:
        u = sin(theta) cos(phi)
        v = sin(theta) sin(phi)

    theta: angle in degrees, can be scalar or ndarray
    phi  : angle in degrees, can be scalar or ndarray
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    u = np.sin(th) * np.cos(ph)
    v = np.sin(th) * np.sin(ph)
    return u, v


def steering_ura_uv(u, v, Nx, Ny):
    """
    URA steering vector with half-wavelength spacing.
    Vectorization order: x-index changes fastest, then y-index.
    """
    nx = np.arange(Nx)
    ny = np.arange(Ny)

    ax = np.exp(1j * np.pi * nx * u)
    ay = np.exp(1j * np.pi * ny * v)

    return np.kron(ay, ax)


def steering_ura_angles(theta_deg, phi_deg, Nx, Ny):
    u, v = uv_from_angles(theta_deg, phi_deg)
    return steering_ura_uv(u, v, Nx, Ny)


def steering_ura_uv_batch(u, v, Nx, Ny):
    """
    Batch steering vectors.
    Inputs:
      u, v: arrays of shape (B,)
    Output:
      A: shape (Nx*Ny, B)
    """
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()

    nx = np.arange(Nx)[:, None]
    ny = np.arange(Ny)[:, None]

    Ax = np.exp(1j * np.pi * nx * u[None, :])   # (Nx, B)
    Ay = np.exp(1j * np.pi * ny * v[None, :])   # (Ny, B)

    A = np.einsum('yb,xb->yxb', Ay, Ax).reshape(Ny * Nx, -1)
    return A


# ============================================================
# Snapshot simulation for a planar array
# ============================================================
def simulate_ura_snapshots(
    Nx=12,
    Ny=12,
    K=300,
    desired=(15.0, 25.0),
    interferences=((-30.0, 70.0), (38.0, -120.0)),
    snr_db=5.0,
    inr_db=35.0,
    noise_var=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    N = Nx * Ny

    a_s = steering_ura_angles(desired[0], desired[1], Nx, Ny)
    A_j = np.column_stack([
        steering_ura_angles(th, ph, Nx, Ny)
        for (th, ph) in interferences
    ])

    sigma_s2 = noise_var * 10 ** (snr_db / 10.0)
    sigma_j2 = noise_var * 10 ** (inr_db / 10.0)

    s = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2.0)
    s *= np.sqrt(sigma_s2)

    J = (rng.standard_normal((len(interferences), K)) +
         1j * rng.standard_normal((len(interferences), K))) / np.sqrt(2.0)
    J *= np.sqrt(sigma_j2)

    Nn = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))) / np.sqrt(2.0)
    Nn *= np.sqrt(noise_var)

    X = np.outer(a_s, s) + A_j @ J + Nn
    Rhat = (X @ X.conj().T) / K

    Rin = noise_var * np.eye(N, dtype=np.complex128)
    for (th, ph) in interferences:
        a = steering_ura_angles(th, ph, Nx, Ny)
        Rin += sigma_j2 * np.outer(a, a.conj())

    return X, Rhat, a_s, Rin, sigma_s2


def output_sinr(w, a_s, Rin, sigma_s2):
    num = sigma_s2 * np.abs(w.conj().T @ a_s) ** 2
    den = np.real(w.conj().T @ Rin @ w)
    return float(np.real(num / np.maximum(den, 1e-12)))


# ============================================================
# Clenshaw-Curtis quadrature
# ============================================================
def clenshaw_curtis_on_interval(a, b, Q=11):
    """
    1D Clenshaw-Curtis nodes and weights on [a,b].
    """
    if Q < 2:
        raise ValueError("Q must be >= 2.")

    k = np.arange(Q)
    z = np.cos(np.pi * k / (Q - 1))
    z = z[::-1]  # ascending order

    V = np.vander(z, N=Q, increasing=True)
    rhs = np.zeros(Q)
    for m in range(Q):
        rhs[m] = 2.0 / (m + 1) if (m % 2 == 0) else 0.0

    w_std = np.linalg.solve(V.T, rhs)

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    x = c + h * z
    alpha = h * w_std
    return x, alpha


def segmented_cc_nodes_weights_2d(Mx=8, My=8, Q=11):
    """
    Tensor-product segmented CC nodes over [-1,1] x [-1,1].
    The visible-region mask is applied later.
    """
    u_bounds = np.linspace(-1.0, 1.0, Mx + 1)
    v_bounds = np.linspace(-1.0, 1.0, My + 1)

    nodes = []
    weights = []

    for ix in range(Mx):
        ux, wu = clenshaw_curtis_on_interval(u_bounds[ix], u_bounds[ix + 1], Q=Q)
        for iy in range(My):
            vy, wv = clenshaw_curtis_on_interval(v_bounds[iy], v_bounds[iy + 1], Q=Q)

            UU, VV = np.meshgrid(ux, vy, indexing='xy')
            WW = np.outer(wv, wu)

            nodes.append(np.column_stack([UU.ravel(), VV.ravel()]))
            weights.append(WW.ravel())

    return np.vstack(nodes), np.concatenate(weights)


# ============================================================
# Low-rank inverse surrogate
# ============================================================
def lowrank_inverse_surrogate(Rhat, L=16):
    N = Rhat.shape[0]
    L = max(1, min(L, N - 1))

    Rh = 0.5 * (Rhat + Rhat.conj().T)

    try:
        vals, vecs = eigsh(Rh, k=L, which='LM')
        idx = np.argsort(np.real(vals))[::-1]
        vals = np.real(vals[idx])
        vecs = vecs[:, idx]
    except Exception:
        vals_all, vecs_all = sla.eigh(Rh)
        idx = np.argsort(np.real(vals_all))[::-1][:L]
        vals = np.real(vals_all[idx])
        vecs = vecs_all[:, idx]

    lam_bar = (np.real(np.trace(Rh)) - np.sum(vals)) / max(N - L, 1)
    lam_bar = max(lam_bar, 1e-8)

    d = (vals - lam_bar) / (lam_bar * np.maximum(vals, 1e-10))
    d = np.real(d)

    return vecs, vals, lam_bar, d


# ============================================================
# Structured 2D SCC-BF reconstruction for a URA
# ============================================================
def build_bttb_covariance_from_lags(lag_grid, Nx, Ny):
    """
    Construct full covariance matrix from 2D lag coefficients.
    lag_grid index mapping:
      dx in [-(Nx-1), ..., Nx-1]
      dy in [-(Ny-1), ..., Ny-1]
    """
    N = Nx * Ny
    R = np.zeros((N, N), dtype=np.complex128)

    x_center = Nx - 1
    y_center = Ny - 1

    for iy1 in range(Ny):
        for ix1 in range(Nx):
            i = ix1 + Nx * iy1
            for iy2 in range(Ny):
                for ix2 in range(Nx):
                    j = ix2 + Nx * iy2
                    dx = ix1 - ix2
                    dy = iy1 - iy2
                    R[i, j] = lag_grid[dx + x_center, dy + y_center]

    return 0.5 * (R + R.conj().T)


def reconstruct_covariance_scc_ura(
    Rhat,
    Nx=12,
    Ny=12,
    desired_uv=(0.0, 0.0),
    Mx=8,
    My=8,
    Q=11,
    L=16,
    exclusion_radius=0.12,
    diag_loading=1e-3,
):
    """
    2D SCC-BF covariance reconstruction in the visible (u,v) domain.

    Notes:
    - Integration is performed directly over the visible (u,v) disk.
    - A circular exclusion region around the desired direction is used
      to suppress desired-signal self-cancellation.
    """
    N = Nx * Ny

    U, vals, lam_bar, d = lowrank_inverse_surrogate(Rhat, L=L)

    dx_vals = np.arange(-(Nx - 1), Nx)
    dy_vals = np.arange(-(Ny - 1), Ny)

    lag_grid = np.zeros((2 * Nx - 1, 2 * Ny - 1), dtype=np.complex128)

    nodes, weights = segmented_cc_nodes_weights_2d(Mx=Mx, My=My, Q=Q)

    u0, v0 = desired_uv
    valid_nodes = 0

    for (u, v), alpha in zip(nodes, weights):
        if u * u + v * v > 1.0:
            continue

        if (u - u0) ** 2 + (v - v0) ** 2 <= exclusion_radius ** 2:
            continue

        a = steering_ura_uv(u, v, Nx, Ny)
        proj = U.conj().T @ a

        q = (Nx * Ny / lam_bar) - np.sum(d * np.abs(proj) ** 2)
        q = max(np.real(q), 1e-8)

        coeff = alpha / q

        ex = np.exp(1j * np.pi * dx_vals * u)[:, None]
        ey = np.exp(1j * np.pi * dy_vals * v)[None, :]
        lag_grid += coeff * (ex @ ey)

        valid_nodes += 1

    Rrec = build_bttb_covariance_from_lags(lag_grid, Nx, Ny)

    delta = diag_loading * np.real(np.trace(Rrec)) / max(N, 1)
    Rrec = Rrec + delta * np.eye(N, dtype=np.complex128)

    return Rrec, lag_grid, valid_nodes


def beamformer_scc_bf_ura(
    Rhat,
    Nx=12,
    Ny=12,
    desired=(15.0, 25.0),
    Mx=8,
    My=8,
    Q=11,
    L=16,
    exclusion_radius=0.12,
    diag_loading=1e-3,
):
    u0, v0 = uv_from_angles(desired[0], desired[1])

    Rrec, lag_grid, valid_nodes = reconstruct_covariance_scc_ura(
        Rhat=Rhat,
        Nx=Nx,
        Ny=Ny,
        desired_uv=(u0, v0),
        Mx=Mx,
        My=My,
        Q=Q,
        L=L,
        exclusion_radius=exclusion_radius,
        diag_loading=diag_loading,
    )

    a0 = steering_ura_angles(desired[0], desired[1], Nx, Ny)
    x = sla.solve(Rrec, a0, assume_a='her')
    w = x / (a0.conj().T @ x)

    return w, Rrec, lag_grid, valid_nodes


# ============================================================
# Pattern evaluation
# ============================================================
def pattern_power_uv_batch(w, u, v, Nx, Ny):
    A = steering_ura_uv_batch(u, v, Nx, Ny)
    y = np.sum(np.conj(w)[:, None] * A, axis=0)
    return np.abs(y) ** 2


def top_view_pattern_db(
    w,
    Nx,
    Ny,
    grid_size=281,
    floor_db=-140.0,
):
    u = np.linspace(-1.0, 1.0, grid_size)
    v = np.linspace(-1.0, 1.0, grid_size)
    U, V = np.meshgrid(u, v, indexing='xy')

    mask = (U ** 2 + V ** 2) <= 1.0
    P = np.full_like(U, np.nan, dtype=float)

    pts_u = U[mask]
    pts_v = V[mask]

    batch = 4096
    vals = np.zeros_like(pts_u, dtype=float)

    for start in range(0, len(pts_u), batch):
        end = min(start + batch, len(pts_u))
        vals[start:end] = pattern_power_uv_batch(
            w, pts_u[start:end], pts_v[start:end], Nx, Ny
        )

    vals_db = db(vals / np.max(np.maximum(vals, 1e-12)))
    vals_db = np.maximum(vals_db, floor_db)
    P[mask] = vals_db

    return U, V, P


def cut_pattern_db(
    w,
    Nx,
    Ny,
    phi_deg=25.0,
    theta_grid=np.linspace(-90, 90, 721),
    floor_db=-100.0,
):
    u, v = uv_from_angles(theta_grid, phi_deg)
    P = pattern_power_uv_batch(w, u, v, Nx, Ny)
    P_db = db(P / np.max(np.maximum(P, 1e-12)))
    return theta_grid, np.maximum(P_db, floor_db)


def angular_heatmap_db(
    w,
    Nx,
    Ny,
    theta_grid=np.linspace(-90, 90, 181),
    phi_grid=np.linspace(-180, 180, 361),
    floor_db=-60.0,
):
    """
    Angular response in the (theta, phi) domain.
    Output shape: (len(theta_grid), len(phi_grid))
    """
    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    U, V = uv_from_angles(TH, PH)

    uu = U.ravel()
    vv = V.ravel()

    batch = 4096
    vals = np.zeros_like(uu, dtype=float)

    for start in range(0, len(uu), batch):
        end = min(start + batch, len(uu))
        vals[start:end] = pattern_power_uv_batch(
            w, uu[start:end], vv[start:end], Nx, Ny
        )

    vals_db = db(vals / np.max(np.maximum(vals, 1e-12)))
    vals_db = np.maximum(vals_db, floor_db)

    return theta_grid, phi_grid, vals_db.reshape(len(theta_grid), len(phi_grid))


# ============================================================
# Figure helpers
# ============================================================
def save_figure(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    png_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf_path, dpi=500, bbox_inches='tight')
    fig.savefig(png_path, dpi=350, bbox_inches='tight')
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {png_path}")


def style_axes(ax, equal=False):
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.22)
    ax.tick_params(direction='in', length=6, width=1.0, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    if equal:
        ax.set_aspect('equal')


def style_colorbar(cbar, label):
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=14, width=0.9, length=4)


def add_marker_labels_uv(ax, desired, interferences):
    u0, v0 = uv_from_angles(desired[0], desired[1])
    ax.plot(
        u0, v0, marker='*', color='red', markersize=13,
        markeredgecolor='white', markeredgewidth=0.8, zorder=5
    )
    ax.text(
        u0 + 0.03, v0 + 0.03, 'Desired',
        fontsize=14, color='black',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85)
    )

    for k, (th, ph) in enumerate(interferences, start=1):
        uj, vj = uv_from_angles(th, ph)
        ax.plot(
            uj, vj, marker='o', color='white', markersize=6.5,
            markeredgecolor='black', markeredgewidth=1.0, zorder=5
        )
        ax.text(
            uj + 0.03, vj + 0.03, f'J{k}',
            fontsize=13, color='black',
            bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.85)
        )


def add_marker_labels_angle(ax, desired, interferences):
    ax.plot(
        desired[1], desired[0], marker='*', color='red', markersize=13,
        markeredgecolor='white', markeredgewidth=0.8, zorder=5
    )
    ax.text(
        desired[1] + 7, desired[0] + 5, 'Desired',
        fontsize=14, color='black',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85)
    )

    for k, (th, ph) in enumerate(interferences, start=1):
        ax.plot(
            ph, th, marker='o', color='white', markersize=6.5,
            markeredgecolor='black', markeredgewidth=1.0, zorder=5
        )
        ax.text(
            ph + 7, th + 5, f'J{k}',
            fontsize=13, color='black',
            bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.85)
        )


# ============================================================
# Plot 1: Top view
# ============================================================
def plot_top_view(
    w,
    Nx,
    Ny,
    desired,
    interferences,
    out_dir="./outputs_planar_sccbf_sep",
):
    U, V, Puv = top_view_pattern_db(w, Nx, Ny, grid_size=281, floor_db=-140.0)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.pcolormesh(U, V, Puv, shading='auto', cmap='turbo', vmin=-140, vmax=0)

    # visible-region boundary
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color='white', linewidth=1.2, alpha=0.85)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title('Top View Diagram', pad=10)

    style_axes(ax, equal=True)
    add_marker_labels_uv(ax, desired, interferences)

    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.035)
    style_colorbar(cbar, 'Pattern (dB)')

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_planar_top_view")
    plt.show()


# ============================================================
# Plot 2: 2D cuts
# ============================================================
def plot_2d_cuts(
    w,
    Nx,
    Ny,
    cut_phis,
    out_dir="./outputs_planar_sccbf_sep",
):
    fig, ax = plt.subplots(figsize=(7.4, 6.0))

    styles = [
        ('-',  '#D62728'),  # red
        ('--', '#1F77B4'),  # blue
        ('-.', '#2F2F2F'),  # dark gray
    ]

    for phi, (ls, color) in zip(cut_phis, styles):
        theta_grid, Pcut = cut_pattern_db(w, Nx, Ny, phi_deg=phi, floor_db=-100.0)
        ax.plot(
            theta_grid, Pcut,
            linestyle=ls, color=color, linewidth=2.6,
            label=rf'$\varphi={phi:.0f}^\circ$'
        )

    ax.set_xlim(-90, 90)
    ax.set_ylim(-100, 0)
    ax.set_xlabel(r'$\theta(^\circ)$')
    ax.set_ylabel('Pattern Amplitude (dB)')
    ax.set_title('2D Cut Diagram', pad=10)

    style_axes(ax, equal=False)

    leg = ax.legend(
        loc='upper left',
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        borderpad=0.35,
        handlelength=2.4
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#A0A0A0')
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_planar_cuts")
    plt.show()


# ============================================================
# Plot 3: Contour in (theta, phi)
# ============================================================
def plot_angular_contour(
    w,
    Nx,
    Ny,
    desired,
    interferences,
    out_dir="./outputs_planar_sccbf_sep",
):
    theta_grid, phi_grid, Pang = angular_heatmap_db(
        w,
        Nx,
        Ny,
        theta_grid=np.linspace(-90, 90, 181),
        phi_grid=np.linspace(-180, 180, 361),
        floor_db=-60.0,
    )

    PHI, THETA = np.meshgrid(phi_grid, theta_grid)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    levels_fill = np.linspace(-60, 0, 16)
    levels_line = np.arange(-60, 1, 10)

    cf = ax.contourf(
        PHI, THETA, Pang,
        levels=levels_fill,
        cmap='turbo',
        vmin=-60, vmax=0
    )
    ax.contour(
        PHI, THETA, Pang,
        levels=levels_line,
        colors='white',
        linewidths=0.85,
        alpha=0.75
    )

    ax.set_xlabel(r'$\varphi(^\circ)$')
    ax.set_ylabel(r'$\theta(^\circ)$')
    ax.set_title(r'Angular Contour in $(\theta,\varphi)$', pad=10)

    style_axes(ax, equal=False)
    add_marker_labels_angle(ax, desired, interferences)

    cbar = plt.colorbar(cf, ax=ax, fraction=0.05, pad=0.035)
    style_colorbar(cbar, 'Pattern (dB)')

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_planar_contour")
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    # ---------------- simulation parameters
    Nx = 12
    Ny = 12
    K = 300

    desired = (15.0, 25.0)
    interferences = (
        (-30.0, 70.0),
        (38.0, -120.0),
    )

    snr_db = 5.0
    inr_db = 35.0
    noise_var = 1.0
    seed = 7

    # ---------------- SCC-BF parameters
    Mx = 8
    My = 8
    Q = 11
    L = 16
    exclusion_radius = 0.12
    diag_loading = 1e-3

    out_dir = "./outputs_planar_sccbf_sep"

    # ---------------- simulate
    _, Rhat, a_s, Rin, sigma_s2 = simulate_ura_snapshots(
        Nx=Nx,
        Ny=Ny,
        K=K,
        desired=desired,
        interferences=interferences,
        snr_db=snr_db,
        inr_db=inr_db,
        noise_var=noise_var,
        seed=seed,
    )

    # ---------------- run proposed 2D SCC-BF
    t0 = time.perf_counter()
    w, Rrec, lag_grid, valid_nodes = beamformer_scc_bf_ura(
        Rhat=Rhat,
        Nx=Nx,
        Ny=Ny,
        desired=desired,
        Mx=Mx,
        My=My,
        Q=Q,
        L=L,
        exclusion_radius=exclusion_radius,
        diag_loading=diag_loading,
    )
    elapsed = time.perf_counter() - t0

    sinr_out = output_sinr(w, a_s, Rin, sigma_s2)

    print("========== 2D Planar SCC-BF Extension ==========")
    print(f"Array size                 : {Nx} x {Ny}  (N = {Nx * Ny})")
    print(f"Snapshots K                : {K}")
    print(f"Desired direction          : (theta, phi) = {desired}")
    print(f"Interference directions    : {interferences}")
    print(f"Input SNR / INR            : {snr_db:.1f} dB / {inr_db:.1f} dB")
    print(f"Segmented CC parameters    : Mx={Mx}, My={My}, Q={Q}")
    print(f"Retained dominant rank L   : {L}")
    print(f"Exclusion radius in (u,v)  : {exclusion_radius}")
    print(f"Valid quadrature nodes     : {valid_nodes}")
    print(f"Runtime                    : {elapsed:.4f} s")
    print(f"Output SINR                : {db(sinr_out):.3f} dB")

    # cuts at desired phi and both interference phis
    cut_phis = [desired[1], interferences[0][1], interferences[1][1]]

    # ---------------- save 3 separate figures
    plot_top_view(
        w=w,
        Nx=Nx,
        Ny=Ny,
        desired=desired,
        interferences=interferences,
        out_dir=out_dir,
    )

    plot_2d_cuts(
        w=w,
        Nx=Nx,
        Ny=Ny,
        cut_phis=cut_phis,
        out_dir=out_dir,
    )

    plot_angular_contour(
        w=w,
        Nx=Nx,
        Ny=Ny,
        desired=desired,
        interferences=interferences,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
