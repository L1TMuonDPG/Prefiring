#!/usr/bin/env python3
import ROOT, argparse, os, sys
ROOT.gROOT.SetBatch(True)

import utils
from utils import *  # create_canvas, add_overflow, add_cms_label_out, get_dataset_legend, CMS_color_*
latex = ROOT.TLatex()

# -------------------- CLI --------------------
ap = argparse.ArgumentParser()
ap.add_argument("--legend", type=str, default="", help="dataset legend (e.g. Run2024B)")
ap.add_argument("-o", required=True, help="output dir")
ap.add_argument("-i", required=True, help="input dir (contains merged_total.root)")
ap.add_argument("--quality", type=int, default=12, help="L1 quality cut used in the ntuples (display only)")
ap.add_argument("--year", type=str, default="2025", help="year label to paint on plots")
args = ap.parse_args()

outdir = args.o.rstrip("/") + "/"
indir  = args.i.rstrip("/") + "/"
os.makedirs(outdir, exist_ok=True)

fin_path = os.path.join(indir, "merged_total.root")
if not os.path.isfile(fin_path):
    print(f"[ERROR] File not found: {fin_path}")
    sys.exit(1)
f = ROOT.TFile.Open(fin_path, "READ")

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

COLORS = {
    "BMTF1": CMS_color_1, "BMTF2": CMS_color_1, "BMTF3": CMS_color_1, "BMTF4": CMS_color_1,
    "BMTF":  CMS_color_1,
    "OMTF":  CMS_color_2,
    "EMTF1": CMS_color_5, "EMTF2": CMS_color_5, "EMTF3": CMS_color_5,
    "EMTF4": CMS_color_5, "EMTF5": CMS_color_5, "EMTF6": CMS_color_5,
}

# -------------------- style helpers --------------------
def set_base_style():
    g = ROOT.gStyle
    g.SetOptStat(0)
    g.SetTitleFont(42, "XYZ")
    g.SetLabelFont(42, "XYZ")
    g.SetTitleSize(0.045, "XYZ")
    g.SetLabelSize(0.042, "XYZ")
    g.SetTitleOffset(1.10, "X")
    g.SetTitleOffset(1.00, "Y")
    g.SetPadGridX(0)
    g.SetPadGridY(1)
    g.SetEndErrorSize(3)  # chunkier error caps
    g.SetTickLength(0.03, "X")
    g.SetTickLength(0.00, "Y")  # y has category labels; avoid long ticks

set_base_style()

# -------------------- helpers --------------------
def fetch_efficiency_bin(ptcut: int, tfkey: str):
    """Build TEfficiency from stored passed/total histograms."""
    base = f"h_prefiring_prob_{ptcut}_{tfkey}"
    h_pass = f.Get(base + "_passed")
    h_tot  = f.Get(base + "_total")
    if not h_pass or not h_tot:
        return None
    hp = utils.add_overflow(h_pass)
    ht = utils.add_overflow(h_tot)
    eff = ROOT.TEfficiency(hp, ht)
    eff.SetTitle(";p^{#mu,Reco}_{T} [GeV];Prefiring Probability")
    return eff

def integral_prob(eff):
    """sum(passed)/sum(total) (binomial error)."""
    hP = eff.GetPassedHistogram()
    hT = eff.GetTotalHistogram()
    num = den = 0.0
    for b in range(1, hT.GetNbinsX()+1):
        t = hT.GetBinContent(b); p = hP.GetBinContent(b)
        if t > 0:
            num += p; den += t
    if den <= 0:
        return 0.0, 0.0
    p = num/den
    err = (p*(1.0-p)/den)**0.5
    return p, err

def eff_to_point_graph(eff, x_min=None, x_max=None):
    """Convert TEfficiency -> TGraph (points only, no error bars)."""
    hP = eff.GetPassedHistogram()
    hT = eff.GetTotalHistogram()
    nB = hT.GetNbinsX()
    xs, ys = [], []
    for b in range(1, nB + 1):
        t = hT.GetBinContent(b)
        if t <= 0:
            continue
        xc = hT.GetXaxis().GetBinCenter(b)
        if x_min is not None and xc < x_min: continue
        if x_max is not None and xc > x_max: continue
        xs.append(xc)
        ys.append(hP.GetBinContent(b)/t)
    g = ROOT.TGraph(len(xs))
    for i, (x, y) in enumerate(zip(xs, ys)):
        g.SetPoint(i, x, y)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.2)
    g.SetLineWidth(2)
    return g

def graph_ymax(graph, floor=0.001, pad=1.15):
    if graph is None:
        return floor
    n = graph.GetN()
    if n <= 0:
        return floor
    ybuf = graph.GetY()
    ymax = max(float(ybuf[i]) for i in range(n))
    return max(floor, ymax * pad)

def fit_logistic_fixed_A_on_graph(graph, A_fixed, xmin=0.0, xmax=50.0):
    """
    Fit epsilon(pT) = A / (1 + exp(-(pT - b)/c)) with A fixed.
    """
    if graph is None or graph.GetN() < 3 or A_fixed <= 0.0:
        return None
    func = ROOT.TF1("f_logi_inc", "[0]/(1.0 + TMath::Exp(-(x-[1])/[2]))", xmin, xmax)
    func.SetParNames("A","b","c")
    func.SetParameters(A_fixed, 15.0, 3.0)
    func.SetParLimits(0, A_fixed, A_fixed)
    func.SetParLimits(2, 0.3, 50.0)
    graph.Fit(func, "Q")
    func.SetLineColor(ROOT.kRed+1)
    func.SetLineWidth(3)
    func.SetLineStyle(1)
    return func

# -------------------- compute per-η bin probabilities --------------------
bin_probs = {}
for tfkey, _ in TF_BINS:
    eff = fetch_efficiency_bin(PTCUT, tfkey)
    if not eff:
        print(f"[WARN] Missing histos for {tfkey}; setting prob=0")
        bin_probs[tfkey] = (0.0, 0.0)
        continue
    p, e = integral_prob(eff)
    print(tfkey, p, e)
    bin_probs[tfkey] = (p, e)

# -------------------- ONE-PAD SUMMARY PLOT (η bins, no latex box) --------------------
c_sum, Ls, Rs, Ts, Bs = utils.create_canvas("c_prefire_eta_summary_singlepad")
c_sum.SetGridx(False); c_sum.SetGridy(True)
ROOT.gPad.SetLogx(True)
ROOT.gPad.SetLeftMargin(0.26)   # wider to fit long labels
ROOT.gPad.SetRightMargin(0.07)
ROOT.gPad.SetBottomMargin(0.14)
ROOT.gPad.SetTopMargin(0.08)

labels = [f"{a:.2f} < |#eta| < {b:.2f}" for _, (a,b) in TF_BINS]
n = len(labels)

if n == 0:
    print("[WARN] No TF bins; skipping summary plot.")
else:
    x_min, x_max = 1e-6, 1e-2

    # Frame with categorical Y via bin labels (top = EMTF6)
    frame_eta = ROOT.TH2F("frame_eta_bins_sp",
                          ";Prefiring Probability",
                          5, x_min, x_max,
                          n, 0.0, float(n))
    yax = frame_eta.GetYaxis()
    for i, lab in enumerate(labels):               # i=0 -> EMTF6 (top)
        yax.SetBinLabel(n - i, lab)                # reverse so EMTF6 is at the top
    yax.SetLabelSize(0.045)
    yax.SetTitleSize(0.046)
    yax.SetNdivisions(n, False)
    frame_eta.GetXaxis().SetMoreLogLabels(False)
    frame_eta.GetXaxis().SetNoExponent(False)
    frame_eta.GetXaxis().SetTitleOffset(1.05)
    frame_eta.Draw("AXIS")

    # gridlines behind points (horizontal guides)
    hgrid = ROOT.TH2F("hgrid_sp","",1, x_min, x_max, n, 0.0, float(n))
    hgrid.SetLineColor(ROOT.kGray+1)
    hgrid.SetLineStyle(3)
    hgrid.SetLineWidth(1)
    hgrid.Draw("AXIS Y+ SAME")  # redraw Y to keep labels, light guide lines

    # points + errors
    vals = [max(bin_probs[k][0], 1e-9) for k, _ in TF_BINS]
    errs = [bin_probs[k][1]          for k, _ in TF_BINS]

    gr_eta = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        # y at bin centers: bin j spans (j-1, j) => center j-0.5
        j = n - i                 # 1..n from top to bottom
        y = j - 0.5
        x = max(vals[i], x_min*1.01)
        ex = errs[i]
        # keep the lower whisker inside the log frame
        floor  = max(0.20 * x, x_min*1.02)
        exlow  = min(ex, max(0.0, x - floor))
        exhigh = ex
        gr_eta.SetPoint(i, x, y)
        gr_eta.SetPointError(i, exlow, exhigh, 0.0, 0.0)

    gr_eta.SetMarkerStyle(20)
    gr_eta.SetMarkerSize(1.25)
    gr_eta.SetLineWidth(2)
    gr_eta.SetMarkerColor(ROOT.kAzure+2)
    gr_eta.SetLineColor(ROOT.kAzure+2)
    gr_eta.Draw("P SAME")

    # axis (X) on top for readability on log scale
    frame_eta.Draw("AXIS X+ SAME")

    # CMS and meta labels (kept minimal; no extra label box)
    utils.add_cms_label_out(Ls+0.15, Ts-0.02)
    latex.SetTextFont(42); latex.SetTextSize(0.038)
    latex.DrawLatexNDC(0.85, 0.93, args.year)
    latex.DrawLatexNDC(0.62, 0.86, f"L1 p_{{T}} > {PTCUT} GeV")

    c_sum.SaveAs(outdir + "prefire_probability_vs_eta_bins.png")
    c_sum.SaveAs(outdir + "prefire_probability_vs_eta_bins.pdf")

# -------------------- TURN-ON PLOTS per η bin (single pad as before) --------------------
c, L, R, Tm, Bm = utils.create_canvas("c_turnon")
c.SetGridx(False); c.SetGridy(True)
dataset_legend, dataset_x1 = get_dataset_legend(args.legend, R)

for tfkey, (eta_min, eta_max) in TF_BINS:
    eff = fetch_efficiency_bin(PTCUT, tfkey)
    if not eff:
        continue

    gpts = eff_to_point_graph(eff, x_min=0.0, x_max=50.0)
    y_top = graph_ymax(gpts, floor=0.001, pad=1.15)

    frame_name = f"frame_pt_{tfkey}"
    frame_pt = ROOT.TH2F(frame_name, ";p^{#mu,Reco}_{T} [GeV];Efficiency",
                         100, 0.0, 50.0, 100, 0.0, y_top)
    ROOT.gPad.SetLeftMargin(0.12)
    ROOT.gPad.SetRightMargin(0.06)
    ROOT.gPad.SetBottomMargin(0.14)
    ROOT.gPad.SetTopMargin(0.08)
    frame_pt.Draw()

    # draw points
    color = COLORS.get(tfkey, ROOT.kBlack)
    gpts.SetMarkerColor(color)
    gpts.SetLineColor(color)
    gpts.SetMarkerStyle(20)
    gpts.SetMarkerSize(1.15)
    gpts.Draw("P SAME")

    # labels (minimal and non-intrusive)
    add_dataset_legend(dataset_x1, args.legend or "")
    latex.SetTextFont(42); latex.SetTextSize(0.036)
    latex.DrawLatexNDC(0.18, 0.84, f"{tfkey}  ({eta_min:.2f} < |#eta| < {eta_max:.2f})")
    latex.DrawLatexNDC(0.18, 0.78, f"L1 p_{{T}} > {PTCUT} GeV")
    latex.DrawLatexNDC(0.18, 0.72, f"L1 quality #geq {args.quality}")
    utils.add_cms_label_out(L, Tm)
    latex.SetTextFont(42); latex.SetTextSize(0.04)

    # fixed plateau from this bin's prefiring probability
    A_fixed = bin_probs.get(tfkey, (0.0, 0.0))[0]
    f1 = fit_logistic_fixed_A_on_graph(gpts, A_fixed, xmin=0.0, xmax=50.0)
    if f1:
        f1.Draw("SAME")

    c.SaveAs(outdir + f"turnon_{tfkey}_0to50_points_noerr.png")
    c.SaveAs(outdir + f"turnon_{tfkey}_0to50_points_noerr.pdf")

f.Close()
print(f"[OK] Plots saved to", outdir)






