import math
import os
import ROOT
import numpy as np

# CMS color scheme: https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/

# 2D color palette
ROOT.gStyle.SetPalette(ROOT.kBlueGreenYellow)   

# For two colors use CMS_color_0 and ROOT.kRed

# Six color cycle
CMS_color_0 = ROOT.TColor.GetColor(87, 144, 252)   # Blue        #5790fc
CMS_color_1 = ROOT.TColor.GetColor(248, 156, 32)   # Orange      #f89c20
CMS_color_2 = ROOT.TColor.GetColor(228, 37, 54)    # Red         #e42536
CMS_color_3 = ROOT.TColor.GetColor(150, 74, 139)   # Purple      #964a8b
CMS_color_4 = ROOT.TColor.GetColor(156, 156, 161)  # Gray        #9c9ca1
CMS_color_5 = ROOT.TColor.GetColor(122, 33, 221)   # Deep Purple #7a21dd

# Eight color cycle
CMS_color_6 = ROOT.TColor.GetColor(24, 69, 251)    # Blue        #1845fb
CMS_color_7 = ROOT.TColor.GetColor(255, 94, 2)     # Orange      #ff5e02
CMS_color_8 = ROOT.TColor.GetColor(201, 31, 22)    # Red         #c91f16
CMS_color_9 = ROOT.TColor.GetColor(200, 73, 169)   # Purple      #c849a9
CMS_color_10 = ROOT.TColor.GetColor(173, 173, 125) # Olive green #adad7d
CMS_color_11 = ROOT.TColor.GetColor(134, 200, 221) # Light blue  #86c8dd
CMS_color_12 = ROOT.TColor.GetColor(87, 141, 255)  # Royal blue  #578dff
CMS_color_13 = ROOT.TColor.GetColor(101, 99, 100)  # Gray        #656364

# Ten color cycle
CMS_color_14 = ROOT.TColor.GetColor(63, 144, 218)  # Blue        #3f90da
CMS_color_15 = ROOT.TColor.GetColor(255, 169, 14)  # Orange      #ffa90e
CMS_color_16 = ROOT.TColor.GetColor(189, 31, 1)    # Red         #bd1f01
CMS_color_17 = ROOT.TColor.GetColor(148, 164, 162) # Gray        #94a4a2
CMS_color_18 = ROOT.TColor.GetColor(131, 45, 182)  # Purple      #832db6
CMS_color_19 = ROOT.TColor.GetColor(169, 107, 89)  # Salmon      #a96b59
CMS_color_20 = ROOT.TColor.GetColor(231, 99, 0)    # Orange      #e76300
CMS_color_21 = ROOT.TColor.GetColor(185, 172, 112) # Olive green #b9ac70
CMS_color_22 = ROOT.TColor.GetColor(113, 117, 129) # Slate gray  #717581
CMS_color_23 = ROOT.TColor.GetColor(146, 218, 221) # Cyan        #92dadd

# General settings
ROOT.gROOT.SetBatch(True)            # Set batch mode for ROOT
ROOT.gStyle.SetOptStat(0)            # Disable statistics box
ROOT.gStyle.SetLegendBorderSize(0)   # No border for legend
ROOT.gStyle.SetTitleOffset(1.5,"Z")  # Offset for Z-axis titles
ROOT.gStyle.SetLegendTextSize(0.035) # Legend text size
latex = ROOT.TLatex()                # TLatex

# Legend text for each era
def get_dataset_legend(legend, R=0.1):
    legend_map = {
        #Legend: (Legend text, x coordinate)
        '2024B': ('2024B (0.13 fb^{-1})', 1-(R+0.26)),
        '2024C': ('2024C (7.24 fb^{-1})', 1-(R+0.26)),
        '2024D': ('2024D (7.96 fb^{-1})', 1-(R+0.26)),
        '2024E': ('2024E (11.32 fb^{-1})', 1-(R+0.28)),
        '2024F': ('2024F (27.76 fb^{-1})', 1-(R+0.28)),
        '2024G': ('2024G (37.77 fb^{-1})', 1-(R+0.28)),
        '2024H': ('2024H (5.44 fb^{-1})', 1-(R+0.26)),
        '2024I': ('2024I (11.47 fb^{-1})', 1-(R+0.28)),
        '2024':  ('109 fb^{-1} (13.6 TeV)', 1-(R+0.3)),
        '2023B': ('2023B (0.64 fb^{-1})', 1-(R+0.26)),
        '2023C': ('2023C (18.08 fb^{-1})', 1-(R+0.28)),
        '2023D': ('2023D (9.69 fb^{-1})', 1-(R+0.26)),
        '2023':  ('28.41 fb^{-1} (13.6 TeV)', 1-(R+0.325)),
        
    }
    
    if legend in legend_map:
        return legend_map[legend]
    else:
        return (legend, (1-R-0.1))  # Default case

# Adds overflow in the last bin of the histogram
def add_overflow(hist):
    nbins = hist.GetNbinsX()+1
    e1 = hist.GetBinError(nbins-1)
    e2 = hist.GetBinError(nbins)
    hist.AddBinContent(nbins-1, hist.GetBinContent(nbins))
    hist.SetBinError(nbins-1, math.sqrt(e1*e1 + e2*e2))
    hist.SetBinContent(nbins, 0)
    hist.SetBinError(nbins, 0)
    return hist

# Adds underflow in the first bin of the histogram
def add_underflow(hist):
    e1 = hist.GetBinError(1)
    e0 = hist.GetBinError(0)
    hist.AddBinContent(1, hist.GetBinContent(0))
    hist.SetBinError(1, math.sqrt(e1 * e1 + e0 * e0))
    hist.SetBinContent(0, 0)
    hist.SetBinError(0, 0)
    return hist

# Merges all root files from a directory into "merged_total.root"
def merge_root_files(dir):
    pwd = os.getcwd()
    os.chdir(dir)
    os.system('rm -rf merged_total.root')
    os.system('hadd merged_total.root *.root')
    os.chdir(pwd)

# Draws histograms
def draw_hist(hist, color, marker, draw, line_width=1, marker_size=1.0):
    hist.SetMarkerColor(color)
    hist.SetLineColor(color)
    hist.SetMarkerStyle(marker)
    hist.SetLineWidth(line_width)
    hist.SetMarkerSize(marker_size)
    hist.Draw(draw)

# Draws CMS label in-frame
def add_cms_label_in(L, T):
    latex.SetTextSize(0.045)
    latex.DrawLatexNDC(L+0.02, 1-(T+0.05), "#font[61]{CMS}")
    latex.SetTextSize(0.0346)
    latex.DrawLatexNDC(L+0.02, 1-(T+0.09), "#font[52]{Preliminary}")

# Draws CMS label out-of-frame
def add_cms_label_out(L,T):
    latex.SetTextSize(0.045)
    latex.DrawLatexNDC(L, 1-(T-0.01), "#font[61]{CMS}")
    latex.SetTextSize(0.0346)
    latex.DrawLatexNDC(L+0.095, 1-(T-0.01), "#font[52]{Preliminary}")

# Draws dataset legend
def add_dataset_legend(x,dataset_legend):
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatexNDC(x,0.91,dataset_legend)

# Creates a canvas
def create_canvas(canvas_name, L=0.100, R=0.100, T=0.100, B=0.100):
    canvas = ROOT.TCanvas(canvas_name, canvas_name, 800, 800)
    canvas.SetLeftMargin  (L)
    canvas.SetRightMargin (R)
    canvas.SetTopMargin   (T)
    canvas.SetBottomMargin(B)
    canvas.SetGrid()
    return canvas, L, R, T, B 


#------------------------------------------------------------------------
# Creates a canvas with 1000_800 frame
def create_canvas_wide(canvas_name, L=0.100, R=0.100, T=0.100, B=0.100):
    canvas = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
    canvas.SetLeftMargin  (L)
    canvas.SetRightMargin (R)
    canvas.SetTopMargin   (T)
    canvas.SetBottomMargin(B)
    canvas.SetGrid()
    return canvas, L, R, T, B 

# Draws CMS label in-frame
def add_cms_label_in_wide(L, T):
    latex.SetTextSize(0.045)
    latex.DrawLatexNDC(L+0.02, 1-(T+0.05), "#font[61]{CMS}")
    latex.SetTextSize(0.0346)
    latex.DrawLatexNDC(L+0.02, 1-(T+0.09), "#font[52]{Preliminary}")

# Draws CMS label out-of-frame
def add_cms_label_out_wide(L,T):
    latex.SetTextSize(0.045)
    latex.DrawLatexNDC(L, 1-(T-0.01), "#font[61]{CMS}")
    latex.SetTextSize(0.0346)
    latex.DrawLatexNDC(L+0.075, 1-(T-0.01), "#font[52]{Preliminary}")

# Legend text for each era
def get_dataset_legend_wide(legend, R=0.1):
    legend_map = {
        #Legend: (Legend text, x coordinate)
        '2024B': ('2024B (0.13 fb^{-1})', 1-(R+0.195)),
        '2024C': ('2024C (7.24 fb^{-1})', 1-(R+0.195)),
        '2024D': ('2024D (7.96 fb^{-1})', 1-(R+0.195)),
        '2024E': ('2024E (11.32 fb^{-1})', 1-(R+0.215)),
        '2024F': ('2024F (27.76 fb^{-1})', 1-(R+0.215)),
        '2024G': ('2024G (37.77 fb^{-1})', 1-(R+0.215)),
        '2024H': ('2024H (5.44 fb^{-1})', 1-(R+0.195)),
        '2024I': ('2024I (11.47 fb^{-1})', 1-(R+0.215)),
        '2024':  ('109 fb^{-1} (13.6 TeV)', 1-(R+0.215)),
        '2023B': ('2023B (0.64 fb^{-1})', 1-(R+0.195)),
        '2023C': ('2023C (18.08 fb^{-1})', 1-(R+0.215)),
        '2023D': ('2023D (9.69 fb^{-1})', 1-(R+0.195)),
        '2023':  ('28.41 fb^{-1} (13.6 TeV)', 1-(R+0.24)),
    }
    
    if legend in legend_map:
        return legend_map[legend]
    else:
        return (legend, (1-R-0.1))  # Default case


def efficiency_to_vector(tefficiency):
    if not tefficiency:
        print("Error: TEfficiency object is not valid.")
        return None, None, None
    
    num_bins = tefficiency.GetTotalHistogram().GetNbinsX()

    efficiency_values = []
    error_low_values = []
    error_up_values = []

    for bin in range(1, num_bins + 1):
        efficiency = tefficiency.GetEfficiency(bin)
        error_low = tefficiency.GetEfficiencyErrorLow(bin)
        error_up = tefficiency.GetEfficiencyErrorUp(bin)
        
        efficiency_values.append(efficiency)
        error_low_values.append(error_low)
        error_up_values.append(error_up)

    return efficiency_values, error_low_values, error_up_values

def calculate_ratio_with_error(num_values, num_errors_low, num_errors_up, denom_values, denom_errors_low, denom_errors_up):
    if None in (num_values, num_errors_low, num_errors_up, denom_values, denom_errors_low, denom_errors_up):
        print("Error: One or more input vectors is None.")
        return None, None, None

    num_values = np.array(num_values)
    num_errors_low = np.array(num_errors_low)
    num_errors_up = np.array(num_errors_up)
    denom_values = np.array(denom_values)
    denom_errors_low = np.array(denom_errors_low)
    denom_errors_up = np.array(denom_errors_up)

    zero_mask = (denom_values == 0)
    denom_values[zero_mask] = np.nan

    ratio_values = np.divide(num_values, denom_values, out=np.zeros_like(num_values), where=denom_values!=0)

    denom_values[zero_mask] = 0

    ratio_errors_low = ratio_values*(np.sqrt((num_errors_low / num_values) ** 2 + (denom_errors_low / denom_values) ** 2, where=num_values!=0))
    ratio_errors_up = ratio_values*(np.sqrt((num_errors_up / num_values) ** 2 + (denom_errors_up / denom_values) ** 2, where=num_values!=0))

    ratio_values[zero_mask] = 0
    ratio_errors_low[zero_mask] = 0
    ratio_errors_up[zero_mask] = 0

    return ratio_values, ratio_errors_low, ratio_errors_up


def get_efficiency(file, region, key):
    h_passed = add_overflow(file.Get(f"{region}_{key}_passed"))
    h_total = add_overflow(file.Get(f"{region}_{key}_total"))
    return ROOT.TEfficiency(h_passed, h_total)

def save_canvas(canvas, output_dir, prefix, key):
    canvas.SaveAs(f"{output_dir}/{prefix}_{key}.png")
    canvas.SaveAs(f"{output_dir}/{prefix}_{key}.pdf")

def handle_overlap(h_passed_A, h_total_A, h_passed_B, h_total_B, bin_range):
    for i in range(1, h_total_A.GetNbinsX() + 1):
        eta = h_total_A.GetXaxis().GetBinCenter(i)
        if bin_range[0] < abs(eta) < bin_range[1]:
            h_passed_B.SetBinContent(i, h_passed_B.GetBinContent(i) + h_passed_A.GetBinContent(i))
            h_total_B.SetBinContent(i, h_total_B.GetBinContent(i) + h_total_A.GetBinContent(i))
            h_passed_A.SetBinContent(i, 0)
            h_total_A.SetBinContent(i, 0)