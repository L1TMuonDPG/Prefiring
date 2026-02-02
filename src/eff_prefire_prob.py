#!/usr/bin/env python3
import math
import ROOT
from array import array
import argparse
import sys
import os
import FWCore.PythonUtilities.LumiList as LumiList

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetMarkerStyle(20)
ROOT.gStyle.SetMarkerSize(1)
ROOT.gStyle.SetTitleOffset(1.3)
ROOT.gROOT.ForceStyle()

def CalcDPhi(phi1, phi2):
    dPhi = math.acos(math.cos(phi1 - phi2))
    if math.sin(phi1 - phi2) < 0:
        dPhi *= -1
    return dPhi

def CalcDR(eta1, phi1, eta2, phi2):
    return math.sqrt((CalcDPhi(phi1, phi2))**2 + (eta1 - eta2)**2)

def passedTrig(muon_eta, muon_phi, trg_eta, trg_phi, trg_id, filterBits):
    """Match offline muon to HLT muon (id==13) with filterBits bit-3 set, ΔR≤0.1."""
    for idx, (te, tp) in enumerate(zip(trg_eta, trg_phi)):
        if trg_id[idx] != 13:
            continue
        dphi = abs(math.acos(math.cos(tp - muon_phi)))
        deta = abs(te - muon_eta)
        dr = math.sqrt(deta * deta + dphi * dphi)
        if dr <= 0.1 and ((filterBits[idx] >> 3) & 1) == 1:
            return True
    return False

# -------------------- CLI --------------------
ap = argparse.ArgumentParser()
ap.add_argument("-i", required=True, help="input ROOT file")
ap.add_argument("-o", required=True, help="output dir")
ap.add_argument("--json", required=True, help="golden JSON file")
ap.add_argument("--ptcut", type=int, default=22, help="L1 pt cut (GeV)")
ap.add_argument("--qual", type=int, default=12, help="L1 hwQual minimum (tight=12, medium=8, loose=4)")
ap.add_argument("--print_every", type=int, default=10000, help="print progress every N events")
args = ap.parse_args()

input_file = args.i
output_dir = args.o.rstrip("/")
json_file = LumiList.LumiList(filename=args.json)

os.makedirs(output_dir, exist_ok=True)

# -------------------- L1 TF bins (applied to PROBE |eta|) --------------------
trig_TF = {
    "BMTF1": (0.00, 0.20),
    "BMTF2": (0.20, 0.30),
    "BMTF3": (0.30, 0.55),
    "BMTF4": (0.55, 0.83),
    "OMTF":  (0.83, 1.24),
    "EMTF1": (1.24, 1.40),
    "EMTF2": (1.40, 1.60),
    "EMTF3": (1.60, 1.80),
    "EMTF4": (1.80, 2.10),
    "EMTF5": (2.10, 2.25),
    "EMTF6": (2.25, 2.40),
}

# -------------------- Pt binning for TEfficiency --------------------
scale_pt_edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]
scale_pt = array("d", scale_pt_edges)

# -------------------- Signed eta binning for TEfficiency (Reco eta) --------------------
# Mirror your positive edges into negative, keep strictly increasing order.
pos = [0.0, 0.20, 0.30, 0.55, 0.83, 1.24, 1.40, 1.60, 1.80, 2.10, 2.25, 2.40, 2.50]
neg = [-x for x in reversed(pos)]  # [-2.5, ..., -0.2, -0.0]
# Avoid double 0.0
scale_eta_edges = neg[:-1] + pos
scale_eta = array("d", scale_eta_edges)

# -------------------- IO --------------------
tree = ROOT.TChain("Events")
tree.Add(input_file)
print("Input file:", input_file)

# -------------------- Control histos --------------------
h_l1_pt   = ROOT.TH1F("h_l1_pt",  "", 50, 0, 50)
h_l1_eta  = ROOT.TH1F("h_l1_eta", "", 50, -2.5, 2.5)
h_l1_phi  = ROOT.TH1F("h_l1_phi", "", 80, -4, 4)

h_reco_pt  = ROOT.TH1F("h_reco_pt",  "", 50, 0, 50)
h_reco_eta = ROOT.TH1F("h_reco_eta", "", 50, -2.5, 2.5)
h_reco_phi = ROOT.TH1F("h_reco_phi", "", 80, -4, 4)

h_prefiring_prob = {}
h_prefiring_prob_eta = {}
h_dR_reco_l1 = {}
h_dEta_reco_l1 = {}

for tf in trig_TF:
    key = f"{args.ptcut}_{tf}"

    eff_pt = ROOT.TEfficiency(
        f"h_prefiring_prob_{key}",
        ";Reco p_{T} [GeV];Prefiring probability",
        len(scale_pt_edges) - 1, scale_pt
    )
    eff_pt.GetPassedHistogram().SetName(f"h_prefiring_prob_{key}_passed")
    eff_pt.GetTotalHistogram().SetName(f"h_prefiring_prob_{key}_total")
    h_prefiring_prob[key] = eff_pt

    eff_eta = ROOT.TEfficiency(
        f"h_prefiring_prob_eta_{key}",
        ";Reco #eta;Prefiring probability",
        len(scale_eta_edges) - 1, scale_eta
    )
    eff_eta.GetPassedHistogram().SetName(f"h_prefiring_prob_eta_{key}_passed")
    eff_eta.GetTotalHistogram().SetName(f"h_prefiring_prob_eta_{key}_total")
    h_prefiring_prob_eta[key] = eff_eta

    h_dR_reco_l1[key] = ROOT.TH1F(f"h_dR_reco_l1_{key}", ";#DeltaR(reco,L1);", 100, 0, 1.0)
    h_dEta_reco_l1[key] = ROOT.TH1F(f"h_dEta_reco_l1_{key}", ";#eta^{reco}-#eta^{L1};", 120, -0.6, 0.6)

# -------------------- Event counters (consistent selection) --------------------
h_n_events_sel        = ROOT.TH1F("h_n_events_sel",        "", 1, 0, 1)  # after JSON+HLT
h_n_unprefireable_sel = ROOT.TH1F("h_n_unprefireable_sel", "", 1, 0, 1)  # among selected, passing chosen flag

n_sel = 0
n_unpref = 0

# -------------------- Main loop --------------------
n_entries = tree.GetEntries()
for iEvt in range(n_entries):
    if args.print_every and (iEvt % args.print_every == 0):
        print(f"[{iEvt}/{n_entries}]")

    tree.GetEntry(iEvt)

    # JSON + HLT selection
    if not json_file.contains(int(tree.run), int(tree.luminosityBlock)):
        continue
    if tree.HLT_IsoMu24 != 1:
        continue

    n_sel += 1

    # Choose your unprefireable flag branch (keeping your name as-is)
    use_flag_val = tree.L1_UnprefireableEvent_FirstBxInTrain
    if use_flag_val:
        n_unpref += 1
    else:
        continue

    # Control plots
    for iL1 in range(tree.nL1Mu):
        h_l1_pt.Fill(tree.L1Mu_pt[iL1])
        h_l1_eta.Fill(tree.L1Mu_etaAtVtx[iL1])
        h_l1_phi.Fill(tree.L1Mu_phiAtVtx[iL1])

    for iReco in range(tree.nMuon):
        h_reco_pt.Fill(tree.Muon_pt[iReco])
        h_reco_eta.Fill(tree.Muon_eta[iReco])
        h_reco_phi.Fill(tree.Muon_phi[iReco])

    # ---------- Build TAGS ----------
    iTags = []
    for iTag in range(tree.nMuon):
        eta = tree.Muon_eta[iTag]
        pt  = tree.Muon_pt[iTag]
        if abs(eta) > 2.5:
            continue
        if pt < 30:
            continue
        if tree.Muon_tightId[iTag] != 1:
            continue
        if tree.Muon_isTracker[iTag] != 1:
            continue
        if tree.Muon_highPurity[iTag] != 1:
            continue
        if not passedTrig(tree.Muon_eta[iTag], tree.Muon_phi[iTag],
                          tree.TrigObj_eta, tree.TrigObj_phi,
                          tree.TrigObj_id, tree.TrigObj_filterBits):
            continue
        iTags.append(iTag)

    if not iTags or tree.nMuon < 2:
        continue

    # ---------- Build PROBES (TF by PROBE |η|; denominator = matched L1) ----------
    pt_cut   = args.ptcut
    qual_cut = args.qual

    for iProbe in range(tree.nMuon):
        reco_eta = tree.Muon_eta[iProbe]
        reco_phi = tree.Muon_phi[iProbe]
        reco_pt  = tree.Muon_pt[iProbe]

        if abs(reco_eta) > 2.5:
            continue
        if tree.Muon_tightId[iProbe] != 1:
            continue
        if tree.Muon_tkIsoId[iProbe] == 0:
            continue

        abseta = abs(reco_eta)

        matched_L1s = []
        best_idx, best_dr = -1, 1e9
        for iL1 in range(tree.nL1Mu):
            if tree.L1Mu_hwQual[iL1] < qual_cut:
                continue
            if tree.L1Mu_pt[iL1] <= pt_cut:
                continue
            dR = CalcDR(tree.L1Mu_etaAtVtx[iL1], tree.L1Mu_phiAtVtx[iL1], reco_eta, reco_phi)
            if dR < 0.1:
                matched_L1s.append(iL1)
                if dR < best_dr:
                    best_dr, best_idx = dR, iL1

        if not matched_L1s:
            continue

        any_prefire = any(tree.L1Mu_bx[iL1] < 0 for iL1 in matched_L1s)

        for tf, (emin, emax) in trig_TF.items():
            if not (emin <= abseta < emax):
                continue
            key = f"{pt_cut}_{tf}"

            if best_idx >= 0:
                h_dR_reco_l1[key].Fill(best_dr)
                h_dEta_reco_l1[key].Fill(reco_eta - tree.L1Mu_etaAtVtx[best_idx])

            # ---- pT TEff fill ----
            h_prefiring_prob[key].GetTotalHistogram().Fill(reco_pt)
            if any_prefire:
                h_prefiring_prob[key].GetPassedHistogram().Fill(reco_pt)

            # ---- signed eta TEff fill ----
            h_prefiring_prob_eta[key].GetTotalHistogram().Fill(reco_eta)
            if any_prefire:
                h_prefiring_prob_eta[key].GetPassedHistogram().Fill(reco_eta)

# -------------------- finalize counters --------------------
h_n_events_sel.SetBinContent(1, n_sel)
h_n_unprefireable_sel.SetBinContent(1, n_unpref)

# -------------------- write output --------------------
out_path = f"{output_dir}/{os.path.basename(input_file)}"
fout = ROOT.TFile(out_path, "RECREATE")
fout.cd()

for tf in trig_TF:
    key = f"{args.ptcut}_{tf}"

    eff_pt = h_prefiring_prob[key]
    eff_pt.GetPassedHistogram().Write()
    eff_pt.GetTotalHistogram().Write()
    eff_pt.Write()

    eff_eta = h_prefiring_prob_eta[key]
    eff_eta.GetPassedHistogram().Write()
    eff_eta.GetTotalHistogram().Write()
    eff_eta.Write()

    h_dR_reco_l1[key].Write()
    h_dEta_reco_l1[key].Write()

h_l1_pt.Write()
h_l1_eta.Write()
h_l1_phi.Write()

h_reco_pt.Write()
h_reco_eta.Write()
h_reco_phi.Write()

h_n_events_sel.Write()
h_n_unprefireable_sel.Write()

fout.Write()
fout.Close()

print(f"Wrote: {out_path}")
print(f"Selected events (JSON+HLT): {n_sel}")
print(f"Unprefireable (chosen flag) among selected: {n_unpref}")

