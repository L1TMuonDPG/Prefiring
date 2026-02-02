#!/usr/bin/env python3
import ROOT
import argparse
import os
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit

import utils

plt.style.use(hep.style.CMS)
ROOT.gROOT.SetBatch(True)

# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--legend", type=str, default="", help="dataset legend (e.g. 2025F, 2024, Run2024B, ...)")
parser.add_argument("-o", required=True, type=str, help="output dir")
parser.add_argument("-i", required=True, type=str, help="input dir (contains merged_total.root)")
parser.add_argument(
    "--plateau-nbins",
    type=int,
    default=6,
    help="Number of highest-pT bins (with entries) used to estimate plateau (max efficiency). Default: 6",
)
args = parser.parse_args()

output_dir = args.o.rstrip("/") + "/"
input_dir = args.i.rstrip("/") + "/"
os.makedirs(output_dir, exist_ok=True)

in_path = os.path.join(input_dir, "merged_total.root")
if not os.path.isfile(in_path):
    raise FileNotFoundError(f"[ERROR] Missing input file: {in_path}")

in_file = ROOT.TFile(in_path, "READ")
if not in_file or in_file.IsZombie():
    raise RuntimeError(f"[ERROR] Could not open: {in_path}")

# ----------------------------------------------------------------------
# mplhep-friendly labels
# ----------------------------------------------------------------------
TFs = {
    "uGMT":  r"$|\eta| \leq 2.4$",
    "BMTF":  r"$|\eta| \leq 0.83$",
    "OMTF":  r"$0.83 \leq |\eta| \leq 1.24$",
    "EMTF":  r"$1.24 \leq |\eta| \leq 2.4$",
    "BMTF1": r"$0.00 \leq |\eta| \leq 0.20$",
    "BMTF2": r"$0.20 \leq |\eta| \leq 0.40$",
    "BMTF3": r"$0.40 \leq |\eta| \leq 0.55$",
    "BMTF4": r"$0.55 \leq |\eta| \leq 0.83$",
    "BMTF5": r"$0.20 \leq |\eta| \leq 0.30$",
    "BMTF6": r"$0.30 \leq |\eta| \leq 0.55$",
    "OMTF1": r"$0.83 \leq |\eta| \leq 1.00$",
    "OMTF2": r"$1.00 \leq |\eta| \leq 1.24$",
    "EMTF1": r"$1.24 \leq |\eta| \leq 1.40$",
    "EMTF2": r"$1.40 \leq |\eta| \leq 1.60$",
    "EMTF3": r"$1.60 \leq |\eta| \leq 1.80$",
    "EMTF4": r"$1.80 \leq |\eta| \leq 2.10$",
    "EMTF5": r"$2.10 \leq |\eta| \leq 2.25$",
    "EMTF6": r"$2.25 \leq |\eta| \leq 2.40$",
}

vars_title = {
    "eta":  r"$\eta^{\mu,\mathrm{offline}}$",
    "phi":  r"$\phi^{\mu,\mathrm{offline}}$ [rad]",
    "pt":   r"$p_T^{\mu,\mathrm{offline}}$ [GeV]",
    "pt2":  r"$p_T^{\mu,\mathrm{offline}}$ [GeV]",
    # "nPV":  "Number of vertices",
}

FIT_VAR = "pt2"
WP_PREFIX = "L1Mu22_12"

# (only used as text on plot; keep it mpl-friendly)
quality_label = r"L1T Quality $\geq 12$"
pt_l1_label   = r"$p_T^{\mu,\mathrm{L1}} \geq 22~\mathrm{GeV}$"
pt_reco_label = r"$p_T^{\mu,\mathrm{offline}} \geq 26~\mathrm{GeV}$"

# ----------------------------------------------------------------------
# Fit helpers
# ----------------------------------------------------------------------
def plateau_from_max_efficiency(teff: ROOT.TEfficiency, nbins_high: int = 6):
    hP = teff.GetPassedHistogram()
    hT = teff.GetTotalHistogram()
    nB = hT.GetNbinsX()

    chosen = []
    for b in range(nB, 0, -1):
        t = float(hT.GetBinContent(b))
        if t <= 0:
            continue
        p = float(hP.GetBinContent(b))
        chosen.append((b, p, t))
        if len(chosen) >= max(1, nbins_high):
            break

    if not chosen:
        return 0.0, 0.0

    effs = [(p / t) for (_, p, t) in chosen if t > 0]
    A = max(effs) if effs else 0.0

    bmax, pmax, tmax = max(chosen, key=lambda x: (x[1] / x[2]) if x[2] > 0 else -1.0)
    Aerr = math.sqrt(A * (1.0 - A) / tmax) if tmax > 0 else 0.0
    return A, Aerr


def logistic_fixed_A(x, b, c, A):
    return A / (1.0 + np.exp(-(x - b) / c))


def fit_logistic_fixed_A(x, y, yerr_low, yerr_up, A_fixed, xmin=0.0, xmax=60.0):
    sigma = np.maximum(yerr_low, yerr_up)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    m &= (x >= xmin) & (x <= xmax)
    xf, yf, sf = x[m], y[m], sigma[m]

    if xf.size < 5 or A_fixed <= 0.0:
        return None

    def f(x, b, c):
        return logistic_fixed_A(x, b, c, A_fixed)

    p0 = (18.0, 2.5)
    bounds = ([-np.inf, 0.2], [np.inf, 50.0])

    popt, pcov = curve_fit(
        f, xf, yf,
        p0=p0,
        sigma=sf,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )

    b, c = float(popt[0]), float(popt[1])
    yfit = f(xf, *popt)
    chi2 = float(np.sum(((yf - yfit) / sf) ** 2))
    ndf = int(len(yf) - len(popt))

    return {
        "b": b,
        "c": c,
        "chi2": chi2,
        "ndf": ndf,
        "chi2_over_ndf": (chi2 / ndf) if ndf > 0 else None,
        "cov": pcov,
    }


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
fit_results = {}

# ----------------------------------------------------------------------
# Main loop (same plotting format as your example)
# ----------------------------------------------------------------------
for tf in TFs:
    for var in vars_title:
        fig, ax = plt.subplots()

        hist_base = f"{tf}_{WP_PREFIX}_{var}"
        h_passed = in_file.Get(f"{hist_base}_passed")
        h_total = in_file.Get(f"{hist_base}_total")

        if not h_passed or not h_total:
            print(f"[WARN] Could not find {hist_base}_passed/total in file")
            plt.close(fig)
            continue

        h_passed = utils.add_overflow(h_passed)
        h_total = utils.add_overflow(h_total)
        h_eff = ROOT.TEfficiency(h_passed, h_total)

        x, y, yerr_low, yerr_up, xerr = utils.efficiency_to_vector(h_eff)
        valid = (y > 0) & (y <= 1)

        ax.errorbar(
            x[valid], y[valid],
            xerr=xerr[valid],
            yerr=[yerr_low[valid], yerr_up[valid]],
            fmt="o",
            color="#5790fc",
            capsize=3,
            label=TFs[tf],
            markersize=6,
            alpha=0.8,
        )

        ax.set_xlabel(vars_title[var])
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1.2)
        ax.grid(True)

        # CMS label (your utils)
        if args.legend:
            utils.add_cms_label(ax, args.legend, loc=2, text="Internal")
        else:
            hep.cms.label("Internal", data=True, loc=2, com=13.6, ax=ax)

        ax.legend(loc="lower right", fontsize=17, frameon=False)

        # show TF eta range (like your template: not for eta itself)
        if var != "eta":
            ax.text(0.98, 0.93, TFs[tf], transform=ax.transAxes,
                    ha="right", va="top", fontsize=22)

        # Axis scaling and limits (same idea as your template)
        if var == "pt":
            ax.set_xscale("log")
            ax.set_xlim(1, 2000)
        elif var == "pt2":
            ax.set_xlim(0, 60)
        elif var == "phi":
            ax.set_xlim(-3.5, 3.5)
        elif var == "eta":
            ax.set_xlim(-2.5, 2.5)

        # Fit only pt2
        if var == FIT_VAR:
            A, Aerr = plateau_from_max_efficiency(h_eff, nbins_high=args.plateau_nbins)
            fit = fit_logistic_fixed_A(x, y, yerr_low, yerr_up, A_fixed=A, xmin=0.0, xmax=60.0)

            if fit is not None:
                b = fit["b"]
                cpar = fit["c"]
                chi2 = fit["chi2"]
                ndf = fit["ndf"]

                xx = np.linspace(0.0, 60.0, 600)
                yy = logistic_fixed_A(xx, b, cpar, A)
                ax.plot(xx, yy, linewidth=2.2, label=r"Fit (A fixed)")

                fit_results[tf] = {
                    "wp": WP_PREFIX,
                    "var": var,
                    "plateau_method": f"max_eff_highbins(n={args.plateau_nbins})",
                    "A_plateau": float(A),
                    "A_plateau_err": float(Aerr),
                    "b": float(b),
                    "c": float(cpar),
                    "chi2": float(chi2),
                    "ndf": int(ndf),
                    "chi2_over_ndf": float(chi2 / ndf) if ndf > 0 else None,
                }
            else:
                fit_results[tf] = {
                    "wp": WP_PREFIX,
                    "var": var,
                    "plateau_method": f"max_eff_highbins(n={args.plateau_nbins})",
                    "A_plateau": float(A),
                    "A_plateau_err": float(Aerr),
                    "b": None,
                    "c": None,
                    "chi2": None,
                    "ndf": None,
                    "chi2_over_ndf": None,
                }

            ax.text(
                0.25, 0.87,
                rf"{tf}: $A_\mathrm{{plateau}}$ = {A:.4g} $\pm$ {Aerr:.2g}",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.legend(loc="lower right", fontsize=17, frameon=False)

        utils.save_canvas(fig, output_dir, "eff_prefiring", f"{tf}_{var}")
        plt.close(fig)

# ----------------------------------------------------------------------
# Write JSON
# ----------------------------------------------------------------------
out_json = os.path.join(output_dir, "eff_turnon_params_by_TF.json")
with open(out_json, "w") as fjson:
    json.dump(fit_results, fjson, indent=2, sort_keys=True)

in_file.Close()
print(f"[OK] All plots created successfully! Stored in {output_dir}")
print(f"[OK] Wrote efficiency plateau+fit parameters to: {out_json}")



