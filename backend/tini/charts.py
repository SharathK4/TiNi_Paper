"""Static chart export (matplotlib, Agg backend).

These PNGs are a persisted record of each run — something to attach to notes
or compare between runs — not what the dashboard renders. The dashboard draws
its own interactive charts from the JSON endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import K_B_T_PN_ANGSTROM  # noqa: E402

INK = "#15161d"
INK2 = "#6e7180"
ACCENT = "#b5462a"
GRID = "#e8e9f4"


def _style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c4c7d8")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _save(fig: plt.Figure, path: Path, dpi: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tmp.replace(path)
    return str(path)


def chart_traces_overlay(t: np.ndarray, traces: np.ndarray, ids: list[str], out: Path, dpi: int, n: int = 8) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k in range(min(n, traces.shape[0])):
        ax.plot(t, traces[k, :, 2], lw=0.6, alpha=0.7, label=ids[k])
    ax.set_xlabel("time t [ps]")
    ax.set_ylabel("force per atom f [pN]")
    ax.set_title(f"Individual force traces (first {min(n, traces.shape[0])})", color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(fontsize=7, ncol=4, frameon=False)
    return _save(fig, out, dpi)


def chart_mean_trace(t: np.ndarray, avg: dict[str, np.ndarray], fit: dict[str, Any], out: Path, dpi: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(t, avg["mean_f"] - avg["std_f"], avg["mean_f"] + avg["std_f"], color=ACCENT, alpha=0.12, lw=0, label="± 1 s.d.")
    ax.plot(t, avg["mean_f"], color=ACCENT, lw=1.4, label="⟨f(t)⟩")
    ax.plot(fit["t_fit"], fit["f_fit"], color=INK, lw=1.0, ls="--", label=f"ḟ = {fit['slope']:.4f} pN/ps (t ≤ {fit['window_ps']:g} ps)")
    ax.set_xlabel("time t [ps]")
    ax.set_ylabel("force per atom [pN]")
    ax.set_title("Ensemble-averaged force trace", color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, out, dpi)


def chart_force_vs_x(avg: dict[str, np.ndarray], out: Path, dpi: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(avg["mean_x"], avg["mean_f"], color=ACCENT, lw=1.4)
    ax.set_xlabel("mean box length ⟨L_x⟩ [Å]")
    ax.set_ylabel("⟨f⟩ [pN]")
    ax.set_title("Mean force against mean box length", color=INK, fontsize=10, loc="left")
    _style(ax)
    return _save(fig, out, dpi)


def chart_potential(curves: dict[str, Any], out: Path, dpi: int) -> str:
    xi = curves["xi_A"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(xi, curves["U0_pN_A"], color=INK, lw=1.4, label="U₀(ξ)")
    cmap = plt.get_cmap("copper")
    for i, tl in enumerate(curves["tilts"]):
        a1.plot(xi, tl["U"], lw=1.0, color=cmap(0.15 + 0.7 * i / max(1, len(curves["tilts"]) - 1)), label=f"f = {tl['f_pN']:.1f} pN")
    a1.set_xlabel("ξ [Å]")
    a1.set_ylabel("U [pN·Å]")
    a1.set_title("Cubic landscape and its tilt under force", color=INK, fontsize=10, loc="left")
    _style(a1)
    a1.legend(fontsize=7, frameon=False)
    a2.plot(xi, curves["F_pN"], color=ACCENT, lw=1.4, label="F(ξ) = −dU₀/dξ (analytic)")
    a2.plot(xi, curves["F_numerical_pN"], color=INK, lw=0.8, ls=":", label="−dU₀/dξ (numerical)")
    a2.axhline(-curves["parameters"]["f_c_pN"], color=INK2, lw=0.8, ls="--")
    a2.set_xlabel("ξ [Å]")
    a2.set_ylabel("F [pN]")
    a2.set_title("Landscape force", color=INK, fontsize=10, loc="left")
    _style(a2)
    a2.legend(fontsize=7, frameon=False)
    return _save(fig, out, dpi)


def chart_events(events: dict[str, Any], out: Path, dpi: int) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    tp = np.array([e["t_peak_ps"] for e in events["events"]])
    fr = np.array([e["f_crit_from_rate_pN"] for e in events["events"]])
    a1.hist(tp, bins=15, color=ACCENT, alpha=0.8)
    a1.set_xlabel("t‡ [ps]")
    a1.set_ylabel("count")
    a1.set_title("Transition time (prototype peak picker)", color=INK, fontsize=10, loc="left")
    _style(a1)
    a2.hist(fr, bins=15, color=INK, alpha=0.8)
    a2.set_xlabel("f‡ = ḟ·t‡ [pN]")
    a2.set_title("Critical force from rate", color=INK, fontsize=10, loc="left")
    _style(a2)
    return _save(fig, out, dpi)
