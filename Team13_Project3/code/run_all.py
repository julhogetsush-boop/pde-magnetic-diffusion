"""Regenerate the numerical evidence and the figures in the report."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.optimize import brentq

from model import Grid, harmonic_grid, integrate, multiplier, period_fit, stepper, transfer, transient_reference


OUT = Path(__file__).resolve().parents[1] / "results"
METHODS = ("FTCS", "CN")
COLORS = {"FTCS": "#E64B35", "CN": "#3C5488", "SD": "#00887A", "reference": "#30333B"}
T = 0.1


def zero(t):
    return 0.0


def write_csv(name, keys, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(rows)


def scientific(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"$\sci{{{mantissa}}}{{{int(exponent)}}}$"


def write_latex_rows(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(" & ".join(map(str, row)) + r" \\" + "\n")
        stream.write("\\bottomrule\n")


def convergence_rows(records, spatial):
    previous = {}
    for row in records:
        boundary, method, n, h, k, _, l2, linf = row[:8]
        key = (boundary, method)
        order = "---" if key not in previous else f"{np.log2(previous[key] / l2):.3f}"
        previous[key] = l2
        base = (boundary, method, n, scientific(k), scientific(l2), scientific(linf), order)
        yield (*base, f"{100*row[8]:.3f}") if spatial else base


def verify():
    for bc in ("DD", "DN"):
        grid = Grid(8, bc)
        dense = grid.A.toarray()
        weighted = grid.weights[:, None] * dense
        assert np.array_equal(weighted, weighted.T)
        for m in range(1, len(grid.x) + 1):
            _, eigenvalue, u = grid.mode(m)
            assert np.allclose(dense @ u, eigenvalue * u, atol=1e-11)
        u = np.linspace(0.1, 0.8, len(grid.x))
        assert np.isclose(u @ weighted @ u, -sum(np.diff(grid.full(u) if bc == "DD" else np.r_[0, u])**2) / grid.h**2)
        k = 0.003
        identity = np.eye(len(u))
        for left, right in ((0.23, 0.71), (0.71, -0.1)):
            explicit = stepper(grid, k, "FTCS")(u, left, right)
            implicit = stepper(grid, k, "CN")(u, left, right)
            assert np.allclose(explicit, (identity + k * dense) @ u + k * grid.q * left)
            assert np.allclose((identity - k * dense / 2) @ implicit,
                               (identity + k * dense / 2) @ u + k * grid.q * (left + right) / 2)
        try:
            stepper(grid, grid.h**2, "FTCS")
        except ValueError:
            pass
        else:
            raise AssertionError("FTCS accepted an unstable step")
    check_grid = Grid(40)
    fitted, _, _ = period_fit(check_grid, 1, 256, "CN")
    expected = harmonic_grid(check_grid, 1, "CN", 2 * np.pi / 256)[19]
    assert abs(fitted - expected) < 1e-8
    print("Checks passed: boundary rows, weighted spectrum, both updates, stability guard and periodic fit.")


def errors(grid, u):
    return grid.norm(u), np.max(abs(u))


def refinements():
    space, time, forced, baseline = [], [], [], []
    for bc in ("DD", "DN"):
        for n in (20, 40, 80, 160):
            grid = Grid(n, bc)
            wave, lam, mode = grid.mode(1)
            sd = mode * np.exp(T * lam)
            exact = mode * np.exp(-T * wave**2)
            edge = (np.arange(n) + 0.5) * grid.h
            grad = wave * np.cos(wave * edge) * np.exp(-T * wave**2)
            for method in METHODS:
                scale = 0.005 if method == "FTCS" else 0.2
                steps = round(T / (scale * grid.h**2))
                u = integrate(grid, mode, zero, T, steps, method)
                nuisance = grid.norm(u - sd) / grid.norm(sd - exact)
                grad_error = np.sqrt(grid.h * np.sum((np.diff(grid.full(u)) / grid.h - grad)**2))
                assert nuisance < 0.1
                space.append((bc, method, n, grid.h, T / steps, T, *errors(grid, u-exact), nuisance, grad_error))
        grid = Grid(40, bc)
        _, lam, mode = grid.mode(1)
        for method in METHODS:
            counts = (400, 800, 1600, 3200) if method == "FTCS" else (5, 10, 20, 40)
            for steps in counts:
                u = integrate(grid, mode, zero, T, steps, method)
                time.append((bc, method, 40, grid.h, T / steps, T, *errors(grid, u-mode*np.exp(T*lam))))
    grid = Grid(128, "DD")
    mode = np.sin(np.pi * grid.x)
    exact = mode * np.exp(-T * np.pi**2)
    for method, steps in (("FTCS", 4096), ("CN", 100)):
        u = integrate(grid, mode, zero, T, steps, method)
        baseline.append((method, 128, grid.h, T / steps, T, u[63], *errors(grid, u-exact)))
    for n in (20, 40, 80):
        grid = Grid(n)
        exact = transient_reference(grid.x, T, 1)
        for method, scale in (("FTCS", 0.4), ("CN", 0.2)):
            steps = round(T / (scale * grid.h**2))
            u = integrate(grid, np.zeros(n), np.sin, T, steps, method)
            forced.append((method, n, grid.h, T / steps, T, *errors(grid, u-exact)))
    cols = ("boundary", "method", "N", "h", "k", "T", "L2h", "Linf")
    write_csv("spatial.csv", (*cols, "time_to_space", "gradient_L2"), space)
    write_csv("temporal.csv", cols, time)
    write_csv("baseline.csv", cols[1:6] + ("centre", "L2h", "Linf"), baseline)
    write_csv("forced.csv", cols[1:], forced)
    write_latex_rows("spatial_rows.tex", convergence_rows(space, spatial=True))
    write_latex_rows("temporal_rows.tex", convergence_rows(time, spatial=False))
    return space, time


def stability():
    rows, histories = [], {}
    for bc in ("DD", "DN"):
        grid = Grid(40, bc)
        for method, r in (("FTCS", 0.49), ("FTCS", 0.5001), ("FTCS", 0.51), ("CN", 2), ("CN", 20)):
            k = r * grid.h**2
            advance = stepper(grid, k, method, allow_growth=(method == "FTCS" and r == 0.51))
            for mode_name, index in (("low", 1), ("high", len(grid.x))):
                _, eig, v = grid.mode(index)
                g = multiplier(method, k * eig)
                u = v.copy()
                amplitude = [1.0]
                energy = [grid.norm(u)**2 / 2]
                for m in range(100):
                    u = advance(u, 0, 0)
                    amplitude.append(np.dot(grid.weights * v, u) / np.dot(grid.weights * v, v))
                    energy.append(grid.norm(u)**2 / 2)
                assert np.allclose(amplitude, g**np.arange(101), rtol=1e-10, atol=1e-12)
                growth = max(np.diff(energy))
                if abs(g) <= 1:
                    assert growth <= 1e-12
                rows.append((bc, method, 40, grid.h, k, r, mode_name, g, amplitude[-1], growth))
                histories[bc, method, r, mode_name] = amplitude
    write_csv("stability.csv", ("boundary", "method", "N", "h", "k", "r", "mode", "G", "a100", "max_energy_increment"), rows)
    return histories


def phase(h):
    return -np.angle(h, deg=True)


def roots(response):
    return (brentq(lambda omega: phase(response(omega)) - 5, 0, 1, xtol=1e-13),
            brentq(lambda omega: abs(response(omega)) - 0.95, 0, 1.1, xtol=1e-13))


def harmonic_study():
    exact = roots(lambda omega: transfer(0.5, omega))
    root_rows, fit_rows = [], []
    for depth in (0.25, 0.5, 0.75, 1.0):
        a, b = roots(lambda omega: transfer(depth, omega))
        root_rows.append(("exact", depth, 0, 0, a, b))
    for n in (20, 40, 80, 160):
        grid = Grid(n)
        for method, resolution in (("SD", 0), ("FTCS", 0.4), ("CN", 2048)):
            def response(omega):
                k = (2 * np.pi / (resolution * omega) if omega else 1 / resolution) if method == "CN" else resolution * grid.h**2
                return harmonic_grid(grid, omega, method, k)[n // 2 - 1]

            root_rows.append((method, 0.5, n, resolution, *roots(response)))
    grid = Grid(80)
    for points in (256, 512, 1024, 4096):
        def response(omega):
            k = 2 * np.pi / (points * omega) if omega else 1 / points
            return harmonic_grid(grid, omega, "CN", k)[39]

        root_rows.append(("CN", 0.5, 80, points, *roots(response)))
    frequencies = (0.1, exact[0], 0.3, exact[1], 1, 2, 5)
    for method, n, values in (("CN", 80, frequencies), ("FTCS", 40, exact)):
        grid = Grid(n)
        for omega in values:
            points = 2048 if method == "CN" else int(np.ceil(2 * np.pi / (omega * 0.4 * grid.h**2)))
            fit, cycles, drift = period_fit(grid, omega, points, method)
            reference = transfer(0.5, omega)
            k = 2 * np.pi / (omega * points)
            harmonic = harmonic_grid(grid, omega, method, k)[n // 2 - 1]
            difference = abs(fit - harmonic)
            assert difference < 1e-8
            fit_rows.append((method, n, grid.h, omega, points, k, cycles, cycles * points * k, drift,
                             abs(reference), abs(fit), phase(reference), phase(fit), difference))
    write_csv("crossings.csv", ("method", "depth", "N", "time_setting", "phase_5deg", "amplitude_5percent"), root_rows)
    write_csv("periodic.csv", ("method", "N", "h", "Omega", "steps_per_period", "k", "cycles", "T", "cycle_L2h", "exact_amplitude", "fit_amplitude", "exact_phase_deg", "fit_phase_deg", "fit_phasor_difference"), fit_rows)
    probes = [row for row in root_rows if row[0] != "exact" and (row[0] != "CN" or row[3] == 2048)]
    settings = {"CN": "$P/2048$", "FTCS": "$0.4h^2$", "SD": "---"}
    write_latex_rows("crossings_rows.tex", ((method, n, settings[method], f"{a:.10f}", f"{b:.10f}",
                     scientific(abs(a-exact[0])), scientific(abs(b-exact[1]))) for method, _, n, _, a, b in probes))
    return exact, root_rows, fit_rows


def cost():
    grid = Grid(80)
    final = 2.0
    reference = transient_reference(grid.x, final, 1, modes=48)
    assert np.array_equal(reference, transient_reference(grid.x, final, 1, modes=96))
    initial = np.zeros(len(grid.x))
    candidates = {"FTCS": [25600, 32000, 40000, 51200], "CN": [4, 8, 16, 32, 64, 128, 256]}
    rows, tried = [], []
    for method, counts in candidates.items():
        for count in counts:
            u = integrate(grid, initial, np.sin, final, count, method)
            tried.append((method, count, final / count, grid.norm(u-reference)))
    for target in (1e-3, 1e-5):
        for method in METHODS:
            passing = [item for item in tried if item[0] == method and item[3] <= target]
            assert passing
            _, count, k, error = passing[0]
            measurements = []
            for _ in range(5):
                start = perf_counter()
                # Include assembly, factorisation, forcing and all time steps.
                trial_grid = Grid(80)
                integrate(trial_grid, initial, np.sin, final, count, method)
                measurements.append(perf_counter() - start)
            rows.append((target, method, 80, grid.h, k, final, count, error,
                         float(np.median(measurements)), min(measurements), max(measurements)))
    write_csv("cost_candidates.csv", ("method", "steps", "k", "L2h"), tried)
    write_csv("cost.csv", ("target", "method", "N", "h", "k", "T", "steps", "L2h", "seconds_median", "seconds_min", "seconds_max"), rows)
    with (OUT / "timing_rows.tex").open("w", encoding="utf-8") as stream:
        for target, method, _, _, _, _, count, error, median, lower, upper in rows:
            mantissa, exponent = f"{error:.3e}".split("e")
            stream.write(
                f"$10^{{{int(np.log10(target))}}}$ & {method} & {count:,} & "
                f"$\\sci{{{mantissa}}}{{{int(exponent)}}}$ & {1000 * median:.3f} & "
                f"{1000 * lower:.3f}--{1000 * upper:.3f} " + r"\\" + "\n"
            )
        stream.write("\\bottomrule\n")
    return rows, tried


def figures(space, time, histories, exact, root_rows, fit_rows, candidates):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#30333B", "text.color": "#30333B",
                         "xtick.color": "#30333B", "ytick.color": "#30333B", "pdf.fonttype": 42,
                         "legend.frameon": False, "lines.linewidth": 1.45})

    def panel(ax, label):
        ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", color="#e6e9ed", linewidth=0.65)
        ax.set_axisbelow(True)

    def spacing_ticks(ax):
        intervals = (160, 80, 40, 20)
        ax.set_xticks([1 / n for n in intervals], [f"1/{n}" for n in intervals])
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45), layout="constrained")
    for method in METHODS:
        data = np.array([row[2:] for row in space if row[:2] == ("DN", method)])
        temporal = np.array([row[2:] for row in time if row[:2] == ("DN", method)])
        explicit = method == "FTCS"
        kw = dict(color=COLORS[method], marker="o" if explicit else "s",
                  markersize=5 if explicit else 3, markerfacecolor="none" if explicit else COLORS[method],
                  linestyle="--" if explicit else "-", zorder=3 if explicit else 2, label=method)
        ax[0].loglog(data[:,1], data[:,4], **kw)
        ax[1].loglog(temporal[:,2], temporal[:,4], **kw)
        ax[2].loglog(data[:,1], data[:,7], **kw)
    for axis, label, xlabel, ylabel in zip(ax, ("a  Space", "b  Time", "c  Current"), ("Spacing h", "Step k", "Spacing h"), ("Weighted field error", "Weighted field error", "Edge gradient error")):
        panel(axis, label)
        axis.set(xlabel=xlabel, ylabel=ylabel)
    spacing_ticks(ax[0])
    spacing_ticks(ax[2])
    ax[1].legend()
    fig.savefig(OUT / "verification.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.55), layout="constrained")
    for method, r, color, style in (("FTCS", 0.49, "#00887A", "--"), ("FTCS", 0.5001, "#7E6148", ":"), ("FTCS", 0.51, COLORS["FTCS"], "-"), ("CN", 20, COLORS["CN"], "-.")):
        for axis, mode in zip(ax, ("high", "low")):
            series = np.asarray(histories["DN", method, r, mode])
            axis.semilogy(np.arange(len(series)), abs(series), linestyle=style, color=color, label=f"{method}, r={r}")
    panel(ax[0], "a  Highest mode")
    panel(ax[1], "b  Lowest mode")
    for axis in ax:
        axis.set(xlabel="Step", ylabel="Absolute modal amplitude")
    ax[0].legend(loc="lower left", fontsize=7.5)
    fig.savefig(OUT / "stability.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained")
    frequencies = np.geomspace(0.08, 5.5, 220)
    reference = transfer(0.5, frequencies)
    fit = np.array([row[2:] for row in fit_rows if row[0] == "CN"])
    for axis, title, ylabel, value, measured, limit, crossing in (
        (ax[0], "a  Phase", "Lag (degrees)", phase(reference), fit[:,10], 5, exact[0]),
        (ax[1], "b  Amplitude", "Amplitude ratio", abs(reference), fit[:,8], .95, exact[1]),
    ):
        panel(axis, title)
        axis.semilogx(frequencies, value, color=COLORS["reference"], label="Analytic")
        axis.semilogx(fit[:,1], measured, "o", markersize=4, markerfacecolor="white", color=COLORS["CN"], label="CN fit")
        axis.axhline(limit, color=COLORS["FTCS"], linestyle="--", linewidth=1, label="Tolerance")
        axis.axvline(crossing, color=COLORS["FTCS"], linestyle=":", linewidth=1)
        axis.set(xlabel="Dimensionless frequency Ω", ylabel=ylabel)
    ax[0].legend(fontsize=8)
    fig.savefig(OUT / "response.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.55), layout="constrained")
    for cols, title, label in ((ax[0], "a  Space", "Spacing h"), (ax[1], "b  Time", "Steps per period")):
        panel(cols, title)
        cols.set(xlabel=label, ylabel="Absolute frequency error")
    semidiscrete = np.array([row[2:] for row in root_rows if row[0] == "SD"])
    cn_time = np.array(sorted([row[2:] for row in root_rows if row[0] == "CN" and row[2] == 80], key=lambda row: row[1]))
    for j, (label, marker, color) in enumerate((("Phase", "o", COLORS["CN"]), ("Amplitude", "s", COLORS["FTCS"]))):
        ax[0].loglog(1 / semidiscrete[:,0], abs(semidiscrete[:,2+j] - exact[j]), color=color, marker=marker, label=label)
        ax[1].loglog(cn_time[:,1], abs(cn_time[:,2+j] - semidiscrete[2,2+j]), color=color, marker=marker, label=label)
    spacing_ticks(ax[0])
    periods = (256, 512, 1024, 2048, 4096)
    ax[1].set_xticks(periods, [str(m) for m in periods])
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[0].legend()
    fig.savefig(OUT / "thresholds.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 2.35), layout="constrained")
    panel(ax, "CN meets both final-field targets with fewer steps")
    for method in METHODS:
        data = np.array([row[1:] for row in candidates if row[0] == method])
        ax.loglog(data[:, 0], data[:, 2], marker="s" if method == "CN" else "o",
                  markersize=4, color=COLORS[method], label=method)
    ax.axhline(1e-3, color="#757575", linewidth=1, linestyle="--", label="Shared targets")
    ax.axhline(1e-5, color="#757575", linewidth=1, linestyle="--")
    ax.set(xlabel="Time steps to final field", ylabel="Weighted final-field error")
    ax.legend()
    fig.savefig(OUT / "cost_accuracy.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run numerical checks only")
    args = parser.parse_args()
    verify()
    if args.check:
        return
    OUT.mkdir(exist_ok=True)
    space, time = refinements()
    histories = stability()
    exact, root_rows, fit_rows = harmonic_study()
    timings, candidates = cost()
    figures(space, time, histories, exact, root_rows, fit_rows, candidates)
    summary = {"phase_crossing": exact[0], "amplitude_crossing": exact[1],
               "maximum_temporal_fraction": max(row[8] for row in space),
               "maximum_cycle_difference": max(row[8] for row in fit_rows),
               "maximum_fit_phasor_difference": max(row[-1] for row in fit_rows),
               "cycles": [min(row[6] for row in fit_rows), max(row[6] for row in fit_rows)],
               "cost": timings}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Phase crossing {exact[0]:.10f}, amplitude crossing {exact[1]:.10f}")


if __name__ == "__main__":
    main()
