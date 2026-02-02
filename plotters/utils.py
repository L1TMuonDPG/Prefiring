import math
import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Use official CMS style
plt.style.use(hep.style.CMS)

CMS_color_0 = ROOT.TColor.GetColor(87, 144, 252)   # Blue        #5790fc
CMS_color_1 = ROOT.TColor.GetColor(248, 156, 32)   # Orange      #f89c20
CMS_color_2 = ROOT.TColor.GetColor(228, 37, 54)    # Red         #e42536
CMS_color_3 = ROOT.TColor.GetColor(150, 74, 139)   # Purple      #964a8b
CMS_color_4 = ROOT.TColor.GetColor(156, 156, 161)  # Gray        #9c9ca1
CMS_color_5 = ROOT.TColor.GetColor(122, 33, 221)   # Deep Purple #7a21dd

# ----------------------------------------------------------------------
def get_dataset_info(tag):
    dataset_map = {
        "2023B": ("2023B", 0.64),
        "2023C": ("2023C", 18.08),
        "2023D": ("2023D", 9.69),
        "2023":  ("2023", 28.41),
        "2024B": ("2024B", 0.13),
        "2024C": ("2024C", 7.24),
        "2024D": ("2024D", 7.96),
        "2024E": ("2024E", 11.32),
        "2024F": ("2024F", 27.76),
        "2024G": ("2024G", 37.77),
        "2024H": ("2024H", 5.44),
        "2024I": ("2024I", 11.47),
        "2024":  ("2024", 109.0),
        "2025C": ("2025C", 20.78),
        "2025D": ("2025D", 25.29),
        "2025E": ("2025E", 14.00),
        "2025F": ("2025F", 30.35),
        "2025G": ("2025G", 25.23),
        "2025":  ("2025", 115.65),
    }
    return dataset_map.get(tag, (tag, None))  # Return None for unknown luminosity

# ----------------------------------------------------------------------
def add_cms_label(ax, dataset_tag, loc=1, text="Preliminary", com=13.6):
    year, lumi = get_dataset_info(dataset_tag)
    
    # Handle cases where luminosity is unknown
    if lumi is not None:
        hep.cms.label(
            text,
            data=True,
            loc=loc,
            year=year,
            lumi=lumi,
            com=com,
            lumi_format="{0:.2f}",
            ax=ax
        )
    else:
        hep.cms.label(
            text,
            data=True,
            loc=loc,
            year=year,
            com=com,
            ax=ax
        )
# ----------------------------------------------------------------------
# Overflow / underflow helpers
def add_overflow(hist):
    nbins = hist.GetNbinsX()+1
    e1 = hist.GetBinError(nbins-1)
    e2 = hist.GetBinError(nbins)
    hist.AddBinContent(nbins-1, hist.GetBinContent(nbins))
    hist.SetBinError(nbins-1, math.sqrt(e1*e1 + e2*e2))
    hist.SetBinContent(nbins, 0)
    hist.SetBinError(nbins, 0)
    return hist

def add_underflow(hist):
    e1 = hist.GetBinError(1)
    e0 = hist.GetBinError(0)
    hist.AddBinContent(1, hist.GetBinContent(0))
    hist.SetBinError(1, math.sqrt(e1 * e1 + e0 * e0))
    hist.SetBinContent(0, 0)
    hist.SetBinError(0, 0)
    return hist

# ----------------------------------------------------------------------
# Efficiency extraction helpers
def efficiency_to_vector(tefficiency):
    """
    Convert a TEfficiency to numpy arrays including x, y, and errors.
    Now also returns symmetric x-errors = bin_width / 2.
    """
    if not tefficiency:
        print("Error: TEfficiency object is not valid.")
        return None, None, None, None, None

    total_hist = tefficiency.GetTotalHistogram()
    num_bins = total_hist.GetNbinsX()

    x_values = []
    x_err = []
    efficiency_values = []
    error_low_values = []
    error_up_values = []

    for b in range(1, num_bins + 1):
        x_center = total_hist.GetXaxis().GetBinCenter(b)
        bin_width = total_hist.GetXaxis().GetBinWidth(b)
        eff = tefficiency.GetEfficiency(b)
        err_low = tefficiency.GetEfficiencyErrorLow(b)
        err_up = tefficiency.GetEfficiencyErrorUp(b)

        x_values.append(x_center)
        x_err.append(bin_width / 2.0)
        efficiency_values.append(eff)
        error_low_values.append(err_low)
        error_up_values.append(err_up)

    return (
        np.array(x_values),
        np.array(efficiency_values),
        np.array(error_low_values),
        np.array(error_up_values),
        np.array(x_err),
    )


# ----------------------------------------------------------------------
# CMS labels and legends using mplhep
def add_cms_label_in(L, T):
    hep.cms.label("Preliminary", loc=1, data = True, year ='2025F', lumi=27.76, com=13.6, lumi_format='{0:.1f}', ax=ax)

def add_cms_label_out(L, T):
    hep.cms.label("Preliminary", loc=1, data = True, year ='2025F', lumi=27.76, com=13.6, lumi_format='{0:.1f}', ax=ax)

# ----------------------------------------------------------------------
# Histogram utilities
def get_efficiency(file, region, key):
    h_passed = add_overflow(file.Get(f"{region}_{key}_passed"))
    h_total = add_overflow(file.Get(f"{region}_{key}_total"))
    return ROOT.TEfficiency(h_passed, h_total)

# ----------------------------------------------------------------------
# Save matplotlib figure
def save_canvas(fig, output_dir, prefix, key):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{prefix}_{key}.png", dpi=300)
    # fig.savefig(f"{output_dir}/{prefix}_{key}.pdf")
