#!/usr/bin/env python3
import ROOT
import argparse
import os
import sys
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

import utils
from utils import *  # keep your utils as-is

plt.style.use(hep.style.CMS)
ROOT.gROOT.SetBatch(True)

# -------------------- CLI --------------------
ap = argparse.ArgumentParser()
ap.add_argument("--legend", type=str, default="", help="dataset legend (e.g. 2025F, 2024, Run2024B)")
ap.add_argument("-o", required=True, help="output dir")
ap.add_argument("-i", required=True, help="input dir (contains merged_total.root)")
ap.add_argument("--quality", type=int, default=12, help="L1 quality cut used in the ntuples (display only)")
ap.add_argument("--year", type=str, default="2025", help="year label to paint on plots")
ap.add_argument("--fit-json", type=str, default="", help="JSON with stored fit params per TF (expects keys like EMTF5 with fields b,c)")
args = ap.parse_args()

outdir = args.o.rstrip("/") + "/"
indir  = args.i.rstrip("/") + "/"
os.makedirs(outdir, exist_ok=True)

fin_path = os.path.join(indir, "merged_total.root")
if not os.path.isfile(fin_path):
    print(f"[ERROR] File not found: {fin_path}")
    sys.exit(1)

f = ROOT.TFile.Open(fin_path, "READ")
if not f or f.IsZombie():
    print(f"[ERROR] Could not open: {fin_path}")
    sys.exit(1)

# -------------------- load fit params (b,c) from json --------------------
fit_params = {}
if args.fit_json:
    if not os.path.isfile(args.fit_json):
        print(f"[ERROR] --fit-json not found: {args.fit_json}")
        sys.exit(1)
    with open(args.fit_json, "r") as fj:
        fit_params = json.load(fj) or {}

# -------------------- config --------------------
PTCUT = 22

TF_BINS = [
    ("EMTF6", (2.25, 2.40)),
    ("EMTF5", (2.10, 2.25)),
    ("EMTF4", (1.80, 2.10)),
    ("EMTF3", (1.60, 1.80)),
    ("EMTF2", (1.40, 1.60)),
    ("EMTF1", (1.24, 1.40)),
    ("OMTF",  (0.83, 1.24)),
    ("BMTF4", (0.55, 0.83)),
    ("BMTF3", (0.30, 0.55)),
    ("BMTF2", (0.20, 0.30)),
    ("BMTF1", (0.00, 0.20)),
]

# labels
X_TITLE_PT  = r"$p_T^{\mu,\mathrm{offline}}$ [GeV]"
# signed eta for per-TF turn-ons
X_TITLE_ETA = r"$\eta^{\mu,\mathrm{offline}}$"

# -------------------- COLORS (mpl-friendly) --------------------
REGION_COLORS = {
    "BMTF": "#5790fc",  # blue
    "OMTF": "#f89c20",  # orange
    "EMTF": "#e42536",  # red
}
REGION_FILL = {
    "BMTF": "#5790fc",
    "OMTF": "#f89c20",
    "EMTF": "#e42536",
}

def region_of(tfkey: str) -> str:
    if tfkey.startswith("BMTF"):
        return "BMTF"
    if tfkey.startswith("EMTF"):
        return "EMTF"
    if tfkey.startswith("OMTF"):
        return "OMTF"
    return "OTHER"

def color_for_tf(tfkey: str) -> str:
    return REGION_COLORS.get(region_of(tfkey), "black")

# -------------------- strict fetch (NO guessing) --------------------
_missing_cache = set()

def fetch_efficiency_strict(base: str):
    """
    base is WITHOUT _passed/_total.
    Returns (TEfficiency, base) or (None, None).
    No noisy 'missing candidates' prints; only prints once per missing base.
    """
    h_pass = f.Get(base + "_passed")
    h_tot  = f.Get(base + "_total")
    if h_pass and h_tot:
        hp = utils.add_overflow(h_pass)
        ht = utils.add_overflow(h_tot)
        return ROOT.TEfficiency(hp, ht), base

    if base not in _missing_cache:
        print(f"[WARN] Missing hist pair: {base}_passed / {base}_total")
        _missing_cache.add(base)
    return None, None

def base_pt(ptcut, tfkey):   return f"h_prefiring_prob_{ptcut}_{tfkey}"
def base_eta(ptcut, tfkey):  return f"h_prefiring_prob_eta_{ptcut}_{tfkey}"

# -------------------- math helpers --------------------
def integral_prob(eff: ROOT.TEfficiency):
    """sum(passed)/sum(total) (binomial error)."""
    hP = eff.GetPassedHistogram()
    hT = eff.GetTotalHistogram()
    num = den = 0.0
    for b in range(1, hT.GetNbinsX() + 1):
        t = float(hT.GetBinContent(b))
        p = float(hP.GetBinContent(b))
        if t > 0:
            num += p
            den += t
    if den <= 0:
        return 0.0, 0.0
    prob = num / den
    err = math.sqrt(prob * (1.0 - prob) / den)
    return prob, err

def logistic(A, b, c, x):
    return A / (1.0 + np.exp(-(x - b) / c))

def per_bin_payload(eff: ROOT.TEfficiency, xmin=None, xmax=None):
    """
    Works for both pT and eta TEff: dumps per-bin passed/total and errors.
    """
    hP = eff.GetPassedHistogram()
    hT = eff.GetTotalHistogram()
    ax = hT.GetXaxis()
    out = []

    for b in range(1, hT.GetNbinsX() + 1):
        xctr = float(ax.GetBinCenter(b))
        if xmin is not None and xctr < xmin:
            continue
        if xmax is not None and xctr > xmax:
            continue

        tot = float(hT.GetBinContent(b))
        if tot <= 0:
            continue
        pas = float(hP.GetBinContent(b))
        prob = pas / tot
        err_sym = math.sqrt(prob * (1.0 - prob) / tot)

        out.append({
            "bin": int(b),
            "x_low": float(ax.GetBinLowEdge(b)),
            "x_high": float(ax.GetBinUpEdge(b)),
            "x_center": xctr,
            "passed": pas,
            "total": tot,
            "prob": prob,
            "err": err_sym,
            "err_low": float(eff.GetEfficiencyErrorLow(b)),
            "err_up": float(eff.GetEfficiencyErrorUp(b)),
        })
    return out

# -------------------- plotting --------------------
def plot_turnon(eff, tfkey, eta_min, eta_max, outgroup, outtag, xlabel, xlim=None,
                do_fit=False, remove_yerr=False, remove_xerr=False, A_for_fit=None):
    x, y, yerr_low, yerr_up, xerr = utils.efficiency_to_vector(eff)

    valid = np.isfinite(x) & np.isfinite(y) & (y >= 0) & (y <= 1)
    if xlim is not None:
        valid &= (x >= float(xlim[0])) & (x <= float(xlim[1]))
    if not np.any(valid):
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    col = color_for_tf(tfkey)

    yerr = None if remove_yerr else [yerr_low[valid], yerr_up[valid]]
    xxerr = None
    if (xerr is not None) and (not remove_xerr):
        xxerr = xerr[valid]

    ax.errorbar(
        x[valid], y[valid],
        yerr=yerr,
        xerr=xxerr,
        fmt="o",
        markersize=6,
        capsize=2.5 if not remove_yerr else 0.0,
        elinewidth=1.2,
        alpha=0.95,
        color=col,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prefiring Probability")
    ax.grid(True, axis="y", alpha=0.6)

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))

    y_top = max(0.001, 1.15 * float(np.max(y[valid])))
    ax.set_ylim(0.0, y_top)

    reg = region_of(tfkey)
    if reg in REGION_FILL:
        ax.axhspan(0.0, y_top, color=REGION_FILL[reg], alpha=0.05, zorder=0)

    if args.legend:
        utils.add_cms_label(ax, args.legend, loc=2, text="Internal")
    else:
        hep.cms.label("Internal", data=True, loc=2, com=13.6, ax=ax)

    ax.text(
        0.18, 0.84,
        rf"{tfkey}  (${eta_min:.2f} < |\eta| < {eta_max:.2f}$)",
        transform=ax.transAxes, fontsize=14, ha="left", va="top"
    )
    ax.text(0.18, 0.78, rf"L1 $p_T > {PTCUT}$ GeV",
            transform=ax.transAxes, fontsize=13, ha="left", va="top")
    ax.text(0.18, 0.72, rf"L1 quality $\geq {args.quality}$",
            transform=ax.transAxes, fontsize=13, ha="left", va="top")

    if do_fit:
        A_prefire = float(A_for_fit) if A_for_fit is not None else 0.0

        b = cpar = None
        if tfkey in fit_params and isinstance(fit_params[tfkey], dict):
            b = fit_params[tfkey].get("b", None)
            cpar = fit_params[tfkey].get("c", None)

        if (b is not None) and (cpar is not None):
            b = float(b)
            cpar = float(cpar)
            xx0, xx1 = (xlim if xlim is not None else (0.0, 50.0))
            xx = np.linspace(float(xx0), float(xx1), 600)
            yy = logistic(A_prefire, b, cpar, xx)
            ax.plot(xx, yy, linewidth=2.6, alpha=0.95, color="black")
        else:
            ax.text(0.18, 0.66, "No JSON (b,c) for this TF",
                    transform=ax.transAxes, fontsize=12, ha="left", va="top")

    utils.save_canvas(fig, outdir, outgroup, f"{outtag}_{tfkey}_colored_2025")
    plt.close(fig)

def plot_overlay_allTF(xvar: str, xlabel: str, outgroup: str, outname: str,
                       xlim=None, remove_yerr=False, remove_xerr=False):
    """
    Overlay all TF curves on one axes (no 'inclusive' branch needed).
    (kept for pT overlay)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ymax = 0.0
    drew_any = False

    for tfkey, (eta_min, eta_max) in TF_BINS:
        if xvar == "pt":
            eff, _ = fetch_efficiency_strict(base_pt(PTCUT, tfkey))
        elif xvar == "eta":
            eff, _ = fetch_efficiency_strict(base_eta(PTCUT, tfkey))
        else:
            raise ValueError("xvar must be 'pt' or 'eta'")

        if not eff:
            continue

        x, y, yerr_low, yerr_up, xerr = utils.efficiency_to_vector(eff)

        valid = np.isfinite(x) & np.isfinite(y) & (y >= 0) & (y <= 1)
        if xlim is not None:
            valid &= (x >= float(xlim[0])) & (x <= float(xlim[1]))
        if not np.any(valid):
            continue

        col = color_for_tf(tfkey)
        label = rf"{tfkey}  (${eta_min:.2f}<|\eta|<{eta_max:.2f}$)"

        yerr = None if remove_yerr else [yerr_low[valid], yerr_up[valid]]
        xxerr = None
        if (xerr is not None) and (not remove_xerr):
            xxerr = xerr[valid]

        ax.errorbar(
            x[valid], y[valid],
            yerr=yerr,
            xerr=xxerr,
            fmt="o",
            markersize=5,
            capsize=2.0 if not remove_yerr else 0.0,
            elinewidth=1.0,
            alpha=0.95,
            color=col,
            label=label,
        )

        ymax = max(ymax, float(np.max(y[valid])))
        drew_any = True

    if not drew_any:
        plt.close(fig)
        print(f"[WARN] No curves drawn for overlay {xvar}")
        return

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prefiring Probability")
    ax.grid(True, axis="y", alpha=0.6)

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))

    y_top = max(0.001, 1.25 * ymax)
    ax.set_ylim(0.0, y_top)

    if args.legend:
        utils.add_cms_label(ax, args.legend, loc=2, text="Internal")
    else:
        hep.cms.label("Internal", data=True, loc=2, com=13.6, ax=ax)

    ax.text(0.98, 0.94, args.year, transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.text(0.98, 0.88, rf"L1 $p_T > {PTCUT}$ GeV", transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.text(0.98, 0.82, rf"L1 quality $\geq {args.quality}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=16)

    utils.save_canvas(fig, outdir, outgroup, f"{outname}_2025")
    plt.close(fig)

def plot_eta_inclusive_onepoint_per_TF(eta_probs: dict, outgroup: str, outname: str):
    """
    One point per TF for eta:
      y  = sum(passed)/sum(total) from the ETA TEfficiency histogram
      x  = center of TF |eta| range
      xerr = half-width of TF |eta| range
    """
    xs, xerrs, ys, yerrs, cols = [], [], [], [], []

    for tfkey, (eta_min, eta_max) in TF_BINS:
        p, e = eta_probs.get(tfkey, (None, None))
        if p is None:
            continue

        x  = 0.5 * (eta_min + eta_max)
        xe = 0.5 * (eta_max - eta_min)

        xs.append(x)
        xerrs.append(xe)
        ys.append(p)
        yerrs.append(e)
        cols.append(color_for_tf(tfkey))

    if len(xs) == 0:
        print("[WARN] No points to draw for eta inclusive one-point plot")
        return

    xs = np.array(xs, dtype=float)
    xerrs = np.array(xerrs, dtype=float)
    ys = np.array(ys, dtype=float)
    yerrs = np.array(yerrs, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 8))

    for x, xe, y, ye, c in zip(xs, xerrs, ys, yerrs, cols):
        ax.errorbar(
            x, y,
            xerr=xe,
            yerr=ye,
            fmt="o",
            markersize=7,
            capsize=3,
            elinewidth=1.2,
            alpha=0.95,
            color=c,
        )

    ax.set_xlabel(r"$|\eta^{\mu,\mathrm{offline}}|$")
    ax.set_ylabel("Prefiring Probability")
    ax.grid(True, axis="y", alpha=0.6)

    ymax = float(np.max(ys + yerrs))
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, max(1e-5, 1.25 * ymax))

    if args.legend:
        utils.add_cms_label(ax, args.legend, loc=2, text="Internal")
    else:
        hep.cms.label("Internal", data=True, loc=2, com=13.6, ax=ax)

    ax.text(0.98, 0.94, args.year, transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.text(0.98, 0.88, rf"L1 $p_T > {PTCUT}$ GeV", transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.text(0.98, 0.82, rf"L1 quality $\geq {args.quality}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=16)

    utils.save_canvas(fig, outdir, outgroup, f"{outname}_2025")
    plt.close(fig)

# -------------------- compute per-TF prefiring probabilities (from pT efficiencies) --------------------
bin_probs = {}
for tfkey, _ in TF_BINS:
    eff_pt, _ = fetch_efficiency_strict(base_pt(PTCUT, tfkey))
    if not eff_pt:
        bin_probs[tfkey] = (0.0, 0.0)
        continue
    p, e = integral_prob(eff_pt)
    bin_probs[tfkey] = (p, e)

# -------------------- compute per-TF prefiring probabilities (from ETA efficiencies) --------------------
eta_probs = {}
for tfkey, _ in TF_BINS:
    eff_eta, _ = fetch_efficiency_strict(base_eta(PTCUT, tfkey))
    if not eff_eta:
        eta_probs[tfkey] = (None, None)
        continue
    p, e = integral_prob(eff_eta)
    eta_probs[tfkey] = (p, e)

# ============================================================================
# 1) SUMMARY PLOT  (Prefire prob vs eta bins)  (from pT efficiencies)
# ============================================================================
labels = [rf"${a:.2f} < |\eta| < {b:.2f}$" for _, (a, b) in TF_BINS]
n = len(labels)

if n == 0:
    print("[WARN] No TF bins; skipping summary plot.")
else:
    fig, ax = plt.subplots(figsize=(12.5, 8))

    vals = np.array([max(bin_probs[k][0], 1e-12) for k, _ in TF_BINS], dtype=float)
    errs = np.array([bin_probs[k][1] for k, _ in TF_BINS], dtype=float)

    y = np.arange(n)
    ax.invert_yaxis()

    point_colors = [color_for_tf(k) for k, _ in TF_BINS]
    for i in range(n):
        ax.errorbar(
            vals[i], y[i], xerr=errs[i],
            fmt="o", markersize=7, capsize=3,
            color=point_colors[i], alpha=0.95, zorder=3
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e-2)
    ax.set_xlabel("Prefiring Probability")
    ax.grid(True, axis="y", alpha=0.6)

    if args.legend:
        utils.add_cms_label(ax, args.legend, loc=2, text="Internal")
    else:
        hep.cms.label("Internal", data=True, loc=2, com=13.6, ax=ax)

    ax.text(0.98, 0.94, args.year, transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.text(0.98, 0.88, rf"L1 $p_T > {PTCUT}$ GeV", transform=ax.transAxes, ha="right", va="top", fontsize=16)

    utils.save_canvas(fig, outdir, "prefire_probability_vs_eta_bins", "summary_singlepad_colored_2025")
    plt.close(fig)

# ============================================================================
# 2) PER-TF TURN-ON PLOTS + JSON payloads
# ============================================================================
all_payload = {"pt": {}, "eta": {}}

for tfkey, (eta_min, eta_max) in TF_BINS:

    # pT with fit
    eff_pt, basept = fetch_efficiency_strict(base_pt(PTCUT, tfkey))
    if eff_pt:
        A = bin_probs.get(tfkey, (0.0, 0.0))[0]
        plot_turnon(
            eff_pt, tfkey, eta_min, eta_max,
            outgroup="turnon_points_werr_fromjson_pt",
            outtag="pt_fit",
            xlabel=X_TITLE_PT,
            xlim=(0.0, 50.0),
            do_fit=True,
            remove_yerr=False,
            A_for_fit=A,
        )

        payload = {
            "tf": tfkey,
            "eta_min": float(eta_min),
            "eta_max": float(eta_max),
            "ptcut": int(PTCUT),
            "quality": int(args.quality),
            "hist_base": str(basept),
            "xvar": "pt",
            "bins": per_bin_payload(eff_pt, xmin=0.0, xmax=50.0),
        }
        all_payload["pt"][tfkey] = payload
        with open(os.path.join(outdir, f"prefire_perbin_pt_{tfkey}.json"), "w") as fout:
            json.dump(payload, fout, indent=2, sort_keys=False)

    # eta: points only (NO Y ERROR BARS) -- keep signed eta here
    eff_eta, baseeta = fetch_efficiency_strict(base_eta(PTCUT, tfkey))
    if eff_eta:
        plot_turnon(
            eff_eta, tfkey, eta_min, eta_max,
            outgroup="turnon_points_eta_points_only",
            outtag="eta_points",
            xlabel=X_TITLE_ETA,
            xlim=None,
            do_fit=False,
            remove_yerr=True,   # <<< key
        )

        payload = {
            "tf": tfkey,
            "eta_min": float(eta_min),
            "eta_max": float(eta_max),
            "ptcut": int(PTCUT),
            "quality": int(args.quality),
            "hist_base": str(baseeta),
            "xvar": "eta",
            "bins": per_bin_payload(eff_eta, xmin=None, xmax=None),
        }
        all_payload["eta"][tfkey] = payload
        with open(os.path.join(outdir, f"prefire_perbin_eta_{tfkey}.json"), "w") as fout:
            json.dump(payload, fout, indent=2, sort_keys=False)

# ============================================================================
# 3) OVERLAY PLOTS
#    - keep pT overlay as before
#    - replace eta overlay with ONE POINT PER TF (integrated over eta bins)
# ============================================================================
plot_overlay_allTF(
    xvar="pt",
    xlabel=X_TITLE_PT,
    outgroup="overlay_allTF",
    outname="overlay_pt_allTF",
    xlim=(0.0, 50.0),
    remove_yerr=False,
)

plot_eta_inclusive_onepoint_per_TF(
    eta_probs=eta_probs,
    outgroup="inclusive_eta_onepoint",
    outname="inclusive_eta_onepoint_perTF",
)

# Global JSON
out_all_json = os.path.join(outdir, "prefire_perbin_ALL_TFs_allvars.json")
with open(out_all_json, "w") as fout:
    json.dump(all_payload, fout, indent=2, sort_keys=True)

f.Close()
print(f"[OK] Plots saved to {outdir}")
print(f"[OK] Wrote per-bin JSONs (pt/eta) to: {outdir}")








