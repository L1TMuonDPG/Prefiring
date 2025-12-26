import ROOT
import argparse
import os
import utils
from utils import *

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--legend', type=str, help='dataset legend')
parser.add_argument('-o', type=str, help='output dir')
parser.add_argument('-i', type=str, help='input dir dir')
args = parser.parse_args()

# Pass arguments
output_dir = args.o
input_dir = args.i
# utils.merge_root_files(input_dir)

in_file = ROOT.TFile(input_dir + "merged_total.root","READ")

WPs = ["L1Mu22_12"]

wp_values = {
    "L1Mu22_12": {"quality": 12, "pt_l1": 22, "pt_reco": 26},
}

TFs = {
    "uGMT": "|#eta| #leq 2.4",
    "BMTF": "|#eta| #leq 0.83",
    "OMTF": "0.83 #leq |#eta| #leq 1.24",
    "EMTF": "1.24 #leq |#eta| #leq 2.4",
    "BMTF1": "0.00 #leq |#eta| #leq 0.20",
    "BMTF2": "0.20 #leq |#eta| #leq 0.40",
    "BMTF3": "0.40 #leq |#eta| #leq 0.55",
    "BMTF4": "0.55 #leq |#eta| #leq 0.83",
    "BMTF5": "0.20 #leq |#eta| #leq 0.30",
    "BMTF6": "0.30 #leq |#eta| #leq 0.55",
    "OMTF1": "0.83 #leq |#eta| #leq 1.00",
    "OMTF2": "1.00 #leq |#eta| #leq 1.24",
    "EMTF1": "1.24 #leq |#eta| #leq 1.40",
    "EMTF2": "1.40 #leq |#eta| #leq 1.60",
    "EMTF3": "1.60 #leq |#eta| #leq 1.80",
    "EMTF4": "1.80 #leq |#eta| #leq 2.10",
    "EMTF5": "2.10 #leq |#eta| #leq 2.25",
    "EMTF6": "2.25 #leq |#eta| #leq 2.40",    
}
vars_title = {
    "eta": "#eta_{Reco}",
    "phi": "#phi_{Reco}",
    "pt": "p^{#mu,offline}_{T} [GeV]",
    "pt2": "p^{#mu,offline}_{T} [GeV]",
}

# Create canvas, receive values for margins
c, L, R, T, B = utils.create_canvas("c")
dataset_legend, dataset_x1 = get_dataset_legend(args.legend, R)

h_passed = {}
h_total = {}
h_eff = {}

## eff vs var
for tf in TFs:
    for var in vars_title:
        key = "L1Mu22_12_" + var
        key2 = tf + "_" + var
        c.SetLogx(0)
        # values = wp_values[wp]
        quality_label = f"L1T Quality #geq 12"
        pt_l1_label = f"p^{{#mu,L1}}_{{T}} #geq 22 GeV"
        pt_reco_label = f"p^{{#mu,Reco}}_{{T}} #geq 26 GeV"

        h_passed[tf] = in_file.Get(f"{tf}_{key}_passed")
        h_passed[tf] = utils.add_overflow(h_passed[tf])
        h_total[tf] = in_file.Get(f"{tf}_{key}_total")
        h_total[tf] = utils.add_overflow(h_total[tf])
        h_eff[tf] = ROOT.TEfficiency(h_passed[tf], h_total[tf])
        draw_hist(h_eff[tf], CMS_color_0, 20, "")
        h_eff[tf].SetTitle(";" + vars_title[var] + ";Efficiency")
        c.Update()
        graph = h_eff[tf].GetPaintedGraph() 
        graph.SetMinimum(0)
        graph.SetMaximum(1.2)
        if var == "pt":
            c.SetLogx(1)
            graph.GetXaxis().SetLimits(1,2000)
            graph.GetXaxis().SetTitleOffset(1.3)
        if var == "pt2":
            graph.GetXaxis().SetLimits(0,60)
            graph.GetXaxis().SetTitleOffset(1.2)
        if var == "nPV":
            graph.GetXaxis().SetLimits(0,70)
        c.Update()

        # Create legend
        leg = ROOT.TLegend(0.61,0.13,0.8,0.38)
        leg.SetFillStyle(0)
        leg.AddEntry(h_eff[tf],TFs[tf],"lep")
        leg.Draw()

        # Latex
        utils.add_dataset_legend(dataset_x1, dataset_legend)
        if var == "phi" or var == "nPV":
            latex.DrawLatexNDC(0.64, 0.53, quality_label)
            latex.DrawLatexNDC(0.64, 0.46, pt_l1_label)
            latex.DrawLatexNDC(0.64, 0.39, pt_reco_label)
        else:
            latex.DrawLatexNDC(0.64, 0.44, quality_label)
            latex.DrawLatexNDC(0.64, 0.39, pt_l1_label)
        utils.add_cms_label_in(L,T)

        c.SaveAs(output_dir + "eff_prefiring_" + key2 + ".png")
        c.SaveAs(output_dir + "eff_prefiring_" + key2 + ".pdf")

# Close input file
in_file.Close()