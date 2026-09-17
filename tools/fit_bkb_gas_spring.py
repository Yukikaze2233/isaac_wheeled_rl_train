#!/usr/bin/env python3
"""Fit a monotone static BKB force curve; keep image-reading uncertainty explicit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]


def fit_curves(data):
    u = np.asarray(data["chart_x"], dtype=float) / 100.
    if len(u) < 4 or u[0] != 0 or not np.all(np.diff(u) > 0):
        raise ValueError("Need ordered normalized readings starting at zero")
    design = np.column_stack((u, u**2, u**3))
    pressure = data["user_pressure_mpa"]
    if not 9 <= pressure <= 12:
        raise ValueError("This interpolation is scoped to the bracketing 9/12 MPa curves")
    fits = {}
    for p in data["pressure_mpa"]:
        force = np.asarray(data["force_n"][str(p)], dtype=float)
        # Fix the intercept at the printed catalogue label; positive coefficients
        # ensure positive, nondecreasing force and avoid polynomial edge dips.
        higher, _ = nnls(design, force - force[0])
        coefficients = np.r_[force[0], higher]
        predicted = np.polynomial.polynomial.polyval(u, coefficients)
        fits[str(p)] = {"monomial_coefficients_n": coefficients.tolist(),
                       "reading_fit_rmse_n": float(np.sqrt(np.mean((predicted - force)**2))),
                       "reading_fit_max_error_n": float(np.max(np.abs(predicted - force)))}
    weight = (pressure - 9) / 3
    coefficients = ((1 - weight) * np.asarray(fits["9"]["monomial_coefficients_n"])
                    + weight * np.asarray(fits["12"]["monomial_coefficients_n"]))
    reading_interpolation = ((1 - weight) * np.asarray(data["force_n"]["9"])
                             + weight * np.asarray(data["force_n"]["12"]))
    predicted = np.polynomial.polynomial.polyval(u, coefficients)
    rod_area = np.pi * (data["rod_diameter_m"] / 2)**2
    # Compare pressure scaling independently; do not replace the plotted intercept.
    scaled_15 = pressure / 15 * np.polynomial.polynomial.polyval(
        u, fits["15"]["monomial_coefficients_n"])
    return {"schema_version": 1, "model": data["selected_model_assumption"],
            "fit_kind": "nonnegative_cubic_fixed_printed_intercepts_then_pressure_interpolation",
            "pressure_mpa": pressure, "stroke_m": data["stroke_m"],
            "monomial_coefficients_n": coefficients.tolist(),
            "coordinate": "u = compression_m / stroke_m; compression zero at full extension",
            "normalized_fit_domain": [0., float(u[-1])],
            "max_model_compression_m": float(u[-1] * data["stroke_m"]),
            "force_direction": "positive_magnitude_pushing_toward_extension",
            "bracket_weights": {"9_mpa": 1 - weight, "12_mpa": weight},
            "source_curve_fits": fits,
            "interpolated_reading_rmse_n": float(np.sqrt(np.mean((predicted - reading_interpolation)**2))),
            "scaled_15_mpa_crosscheck_max_difference_n": float(np.max(np.abs(predicted - scaled_15))),
            "rod_area_pressure_initial_force_n": float(pressure * 1e6 * rod_area),
            "estimated_reading_uncertainty_n": data["estimated_reading_uncertainty_n"],
            "uncertainty_scope": data["uncertainty_note"],
            "physical_force_model_identified": False,
            "axis_assumption_confirmed": data["axis_assumption_confirmed"],
            "model_assumption_confirmed": data["model_assumption_confirmed"],
            "mounting_pin_distance_reference_identified": False,
            "damping_identified": False, "friction_hysteresis_identified": False,
            "hardware_deployment_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "model/纯底盘_v5/gas_spring/catalogue_points.json")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    fit = fit_curves(data)
    fit["reading_data_sha256"] = hashlib.sha256(raw).hexdigest()
    fit["fitting_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.data.parent
    (output / "fit_10mpa.json").write_text(json.dumps(fit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    coefficients = np.asarray(fit["monomial_coefficients_n"])
    with (output / "force_table_10mpa.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["compression_mm", "stroke_fraction", "force_n", "tangent_stiffness_n_per_mm"])
        for mm in [0, 10, 20, 30, 40, 50, 60, 64, 70, 72]:
            u = mm / (data["stroke_m"] * 1000)
            force = np.polynomial.polynomial.polyval(u, coefficients)
            stiffness = np.polynomial.polynomial.polyval(u, coefficients[1:] * np.arange(1, 4)) / (data["stroke_m"] * 1000)
            writer.writerow([mm, u, round(float(force), 4), round(float(stiffness), 4)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 0.9, 301)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for pressure, color in [(9, "#659354"), (12, "#977b32"), (15, "#cc8e32")]:
        values = fit["source_curve_fits"][str(pressure)]["monomial_coefficients_n"]
        axes[0].scatter(data["chart_x"], data["force_n"][str(pressure)], color=color, s=18)
        axes[0].plot(grid * 100, np.polynomial.polynomial.polyval(grid, values), color=color, label=f"{pressure} MPa: approximate image readings")
    forces = np.polynomial.polynomial.polyval(grid, coefficients)
    axes[0].plot(grid * 100, forces, color="#1761a0", linewidth=2.5, label="10 MPa: interpolated fit")
    axes[0].set(xlabel="Rated stroke (%) [axis assumption]", ylabel="Extension force (N)", title="BKB catalogue curve reconstruction")
    axes[0].legend(fontsize=8)
    mm = grid * data["stroke_m"] * 1000
    axes[1].plot(mm, forces, color="#1761a0", linewidth=2.5)
    axes[1].fill_between(mm, forces - 10, forces + 10, color="#1761a0", alpha=0.15,
                         label="+/-10 N image-reading allowance (not CI)")
    axes[1].set(xlabel="Compression (mm)", ylabel="Extension force (N)", title="10 MPa / assumed 80 mm rated stroke")
    axes[1].axvline(72, color="#9e3333", linestyle="--", label="90% stroke = 72 mm")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "fit_10mpa.png", dpi=160)
    plt.close(fig)
    print(json.dumps(fit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
