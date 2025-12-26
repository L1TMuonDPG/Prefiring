#! /usr/bin/env python

## script to make efficiency plots using official NANOAOD
import math
import ROOT
from array import array
import argparse
import sys
import os.path
import FWCore.PythonUtilities.LumiList as LumiList

## Configuration settings
MAX_EVT  = -1          ## Maximum number of events to process
PRT_EVT  = 10000       ## Print every Nth event
MIN_dR_TP      = 0.4   ## Minimum dR for T&P matching (they need to be well separated)
MAX_dR_L1_T    = 0.1   ## Maximum dR for L1T-tag matching
MAX_dR_L1_P    = 0.2   ## Maximum dR for L1T-probe matching
TAG_PT         = 27    ## Minimum offline pT for tag muon
L1_QUAL        = 12    ## Minimum L1 quality for Tag matching
REQ_ZMASS = True       ## Require Z mass window for tag and probe
#######################


## True if passed the trigger (bit indices 3 and 10)
def passedTrig(muon_eta, muon_phi, trg_eta, trg_phi, trg_id, filterBits):

  passed = False

  for idx, (trg_eta, trg_phi) in enumerate(zip(trg_eta,trg_phi)):
    if trg_id[idx] != 13: continue

    dphi = abs(math.acos(math.cos(trg_phi - muon_phi)))
    deta = abs(trg_eta - muon_eta)
    dr = math.sqrt(deta*deta+dphi*dphi)
    if dr <= 0.1: 
      if filterBits[idx]>>3&1 == 1 and filterBits[idx]>>10&1 == 1:
        passed = True

  return passed

def CalcDPhi( phi1, phi2 ):
  dPhi = math.acos( math.cos( phi1 - phi2 ) )
  if math.sin( phi1 - phi2 ) < 0: dPhi *= -1
  return dPhi

def CalcDR( eta1, phi1, eta2, phi2 ):
  return math.sqrt( math.pow(CalcDPhi(phi1, phi2), 2) + math.pow(eta1 - eta2, 2) )

latex = ROOT.TLatex()
latex.SetTextSize(0.04)

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input file')
parser.add_argument('-o', type=str, help='output dir')
parser.add_argument('--json', type=str, help='json file')
args = parser.parse_args()

input_file = args.i
if not input_file:
  sys.exit("No input file given")
output_dir = args.o

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetMarkerStyle(20)
ROOT.gStyle.SetMarkerSize(1)
ROOT.gStyle.SetTitleOffset(1.3)
ROOT.gROOT.ForceStyle()

## L1NTuple branches
tree  = ROOT.TChain('Events')

canvas = ROOT.TCanvas("c","c",750,750)
canvas.SetGridy()
canvas.SetGridx()

## Load input files
print ("Input file: ", input_file)
tree.Add(input_file)

## Trigger settings
trig_WP = {}
trig_WP['L1Mu22_12']  = [12]

trg_pt = {}
trg_pt['L1Mu22_12'] = [22]

trig_TF = {}
trig_TF['uGMT'] = [0.00, 2.40]
trig_TF['BMTF'] = [0.00, 0.83]
trig_TF['OMTF'] = [0.83, 1.24]
trig_TF['EMTF'] = [1.24, 2.40]
trig_TF['BMTF1'] = [0.00, 0.20]
trig_TF['BMTF2'] = [0.20, 0.40]
trig_TF['BMTF3'] = [0.40, 0.55]
trig_TF['BMTF4'] = [0.55, 0.83]
trig_TF['BMTF5'] = [0.20,0.30]
trig_TF['BMTF6'] = [0.30,0.55]
trig_TF['OMTF1'] = [0.83, 1.00]
trig_TF['OMTF2'] = [1.00, 1.24]
trig_TF['EMTF1'] = [1.24, 1.40]
trig_TF['EMTF2'] = [1.40, 1.60]
trig_TF['EMTF3'] = [1.60, 1.80]
trig_TF['EMTF4'] = [1.80, 2.10]
trig_TF['EMTF5'] = [2.10, 2.25]
trig_TF['EMTF6'] = [2.25, 2.40]

## ================ Histograms ======================
scale_pt_temp = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 45, 60, 75, 100, 140, 160, 180, 200, 250, 300, 500, 2000]
scale_pt_temp_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60]
scale_pt_2  = array('d', scale_pt_temp_2)
scale_pt  = array('d', scale_pt_temp)
scale_nPV_temp = [i for i in range(0,72,2)]
scale_nPV  = array('d', scale_nPV_temp)

eta_bins = [48, -2.4, 2.4]
phi_bins = [140, -3.5, 3.5]

h_eff_pt = {}
h_eff_eta = {}
h_eff_phi = {}
h_eff_pt_2 = {}
# h_eff_nPV = {}
h_eff_phi_eta = {}
h_eff_phi_posEta = {}
h_eff_phi_negEta = {}
h_eff_pt_2_posEta = {}
h_eff_pt_2_negEta = {}

h_dr = {}

for TF in trig_TF.keys():
  for WP in trig_WP.keys():
    for pt in trg_pt[WP]:
      key = TF + '_' + WP

      wp_val = trig_WP[WP][0]
      # only allow WP > 12 for BMTF
      if wp_val > 12 and TF != "BMTF":
        continue

      h_eff_pt[key] = ROOT.TEfficiency("h_eff_pt_%s" % key,";Reco p_{T} [GeV];Efficiency", len(scale_pt_temp)-1,  scale_pt)
      h_eff_pt_2[key] = ROOT.TEfficiency("h_eff_pt2_%s" % key,";Reco p_{T} [GeV];Efficiency", len(scale_pt_temp_2)-1,  scale_pt_2)
      h_eff_eta[key] = ROOT.TEfficiency("h_eff_eta_%s" % key,";Reco #eta;Efficiency", eta_bins[0], eta_bins[1], eta_bins[2])
      h_eff_phi[key] = ROOT.TEfficiency("h_eff_phi_%s" % key,";Reco #phi;Efficiency", phi_bins[0], phi_bins[1], phi_bins[2])
      # h_eff_nPV[key] = ROOT.TEfficiency("h_eff_nPV_%s" % key,";nPV;Efficiency", len(scale_nPV_temp)-1,  scale_nPV)
      h_eff_phi_eta[key] = ROOT.TEfficiency("h_eff_phi_eta%s" % key,";#eta;#phi [rad]", eta_bins[0], eta_bins[1], eta_bins[2], phi_bins[0], phi_bins[1], phi_bins[2])

      h_dr[key] = ROOT.TH1F("h_dr_%s" % key,";#DeltaR;",100,0,1)

      h_eff_pt[key].SetDirectory(0)
      h_eff_pt_2[key].SetDirectory(0)
      h_eff_eta[key].SetDirectory(0)
      h_eff_phi[key].SetDirectory(0)
      # h_eff_nPV[key].SetDirectory(0)
      h_eff_phi_eta[key].SetDirectory(0)
## ================================================

## load json file
if args.json is not None:
    json_file = LumiList.LumiList(filename=args.json)
else:
    json_file = None

# Loop over over events in TFile
for iEvt in range(tree.GetEntries()):
  if MAX_EVT > 0 and iEvt > MAX_EVT: break
  if iEvt % PRT_EVT == 0: print ('Event #', iEvt)

  tree.GetEntry(iEvt)

  run = tree.run
  luminosityBlock = tree.luminosityBlock
  if json_file is not None:
    if not json_file.contains(run,luminosityBlock): 
      continue

  # Require HLT muon trigger
  if tree.HLT_IsoMu27 != 1 or tree.HLT_Mu50 != 1: continue

  # good vertex
  if tree.Flag_goodVertices != 1: continue

  # need 2 reco muons for T&P
  if tree.nMuon < 2: continue

  ## Lists of tag and probe RECO muon indices
  iTags, iL1Tags, iProbes = [], [], []

  nPV = tree.PV_npvs

  # if run > 381316: continue ## EMTF issue: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1KnownIssues#Muons

  ##########################################################
  ###  Loop over RECO muons to find all valid tag muons  ###
  ##########################################################
  for iTag in range(tree.nMuon):

    ## Compute tag muon coordinates at 2nd station, require to be valid
    recoEta = tree.Muon_eta[iTag]
    recoPhi = tree.Muon_phi[iTag]

    recoAbsEta = abs(recoEta)

    if recoAbsEta > 2.5: continue

    recoIsTight = tree.Muon_tightId[iTag]
    recoIsLoose = tree.Muon_looseId[iTag]

    ## Require tag muon to pass Muon POG tight ID
    if not recoIsTight: continue

    ## Require prompt muons
    recoDxy = tree.Muon_dxy[iTag]
    recoDz  = tree.Muon_dz[iTag]
    if abs(recoDxy) > 0.2 or abs(recoDz) > 0.5: continue

    muon_passed_trigger = passedTrig(recoEta, recoPhi, tree.TrigObj_eta, tree.TrigObj_phi, tree.TrigObj_id, tree.TrigObj_filterBits)
    if not muon_passed_trigger: continue

    ## Require tag muon to pass pT and isolation cuts
    if tree.Muon_pt[iTag]  < TAG_PT: continue

    # find matching L1 muon
    for iL1 in range(tree.nL1Mu):
      if tree.L1Mu_hwQual[iL1] < L1_QUAL: continue
      if tree.L1Mu_pt[iL1]   < TAG_PT - 4.01: continue
      l1_eta = tree.L1Mu_etaAtVtx[iL1]
      l1_phi = tree.L1Mu_phiAtVtx[iL1]
      if CalcDR( l1_eta, l1_phi, recoEta, recoPhi ) > MAX_dR_L1_T: continue
      if not iTag in iTags:
        iTags.append(iTag)
        iL1Tags.append(iL1)

  ## End loop: for iTag in range(reco_tree.Muon.nMuons):

  ## Quit the event if there are no tag muons
  if len(iTags) == 0: continue

  ############################################################
  ###  Loop over RECO muons to find all valid probe muons  ###
  ############################################################

  xTags = []
  for iProbe in range(tree.nMuon):

    ## Compute tag muon coordinates at 2nd station, require to be valid
    recoEta = tree.Muon_eta[iProbe]
    recoPhi = tree.Muon_phi[iProbe]
    recoPt = tree.Muon_pt[iProbe]

    recoAbsEta = abs(recoEta)

    if recoAbsEta > 2.5: continue

    recoIsTight = tree.Muon_tightId[iProbe]
    recoIsLoose = tree.Muon_looseId[iProbe]

    ## Require probe muon to pass Muon POG tight ID
    if not recoIsTight: continue

    ## Require prompt muons
    recoDxy = tree.Muon_dxy[iProbe]
    recoDz  = tree.Muon_dz[iProbe]
    if abs(recoDxy) > 0.2 or abs(recoDz) > 0.5: continue


    ## Try to find at least on valid tag muon for this probe
    xTag = -1
    ## Loop over tag muon candidates
    for iTag in iTags:
        ## Make sure tag and probe are not the same muon
        if iTag == iProbe: continue
        ## Require tag and probe muons to be well separated
        tagEta = tree.Muon_eta[iTag]
        tagPhi = tree.Muon_phi[iTag]
        tagPt = tree.Muon_pt[iTag]

        if CalcDR( tagEta, tagPhi, recoEta, recoPhi ) < MIN_dR_TP: continue
        if REQ_ZMASS:
          lorentz_v_tag = ROOT.TLorentzVector()
          lorentz_v_probe = ROOT.TLorentzVector()

          lorentz_v_tag.SetPtEtaPhiM (tagPt, tagEta, tagPhi, tree.Muon_mass[iTag])
          lorentz_v_probe.SetPtEtaPhiM (recoPt, recoEta, recoPhi, tree.Muon_mass[iProbe])

          lorentz_tot = lorentz_v_tag + lorentz_v_probe
          mass = lorentz_tot.M()
          if mass < 85 or mass > 95: continue
          
        ## If tag passes requirements, store its index and quit the loop
        xTag = iTag
        break
    ## End loop: for iTag in iTags

    ## Valid probe only if there is a corresponding tag
    if xTag < 0: continue
    iProbes.append(iProbe)
    xTags.append(xTag)

  ## End loop: for iProbe in range(tree.muon_size):

  ## Quit the event if there are no probe muons
  if(len(iProbes) == 0): continue

  ## loop over probe muons to find matched L1 muon
  for iProbe in iProbes:

    matched_tag = xTags[iProbes.index(iProbe)]
    matched_tag_l1 = iL1Tags[iTags.index(matched_tag)]

    recoEta = tree.Muon_eta[iProbe]
    recoPhi = tree.Muon_phi[iProbe]
    recoAbsEta = abs(recoEta)
    recoPt = tree.Muon_pt[iProbe]

    for tf in trig_TF:
      ## Only consider reco muons in TF eta range
      if not (recoAbsEta > trig_TF[tf][0] and recoAbsEta < trig_TF[tf][1]): continue
      for WP in trig_WP.keys():
        for pt in trg_pt[WP]:

          wp_val = trig_WP[WP][0]
          # only allow WP > 12 for BMTF
          if wp_val > 12 and tf != "BMTF":
            continue

          key = tf + '_' + WP

          matched = False  
          # look for L1 muons to match
          for iL1 in range(tree.nL1Mu):
            if iL1 == matched_tag_l1: continue
            l1_eta = tree.L1Mu_etaAtVtx[iL1]
            l1_phi = tree.L1Mu_phiAtVtx[iL1]
            l1_pt = tree.L1Mu_pt[iL1]

            if matched: continue
            h_dr[key].Fill(CalcDR( l1_eta, l1_phi, recoEta, recoPhi ))
            if CalcDR( l1_eta, l1_phi, recoEta, recoPhi ) <= MAX_dR_L1_P and tree.L1Mu_bx[iL1] == 0 and tree.L1Mu_hwQual[iL1] >= trig_WP[WP][0] and tree.L1Mu_pt[iL1] >= pt:
              matched = True
          h_eff_pt[key].Fill(matched,recoPt)
          h_eff_pt_2[key].Fill(matched,recoPt)
          if recoPt > pt + 4:
            h_eff_eta[key].Fill(matched,recoEta)
            h_eff_phi[key].Fill(matched,recoPhi)
            # h_eff_nPV[key].Fill(matched,nPV)  
            h_eff_phi_eta[key].Fill(matched,recoEta,recoPhi)

            ## End loop: for iL1 in range(tree.l1mu_size):
          ## End loop: for pt in trg_pt[WP]:
        ## End loop: for WP in trig_WP.keys():
    ## End loop: for iProbe in iProbes:
  ## End loop: for iEvt in range(tree.GetEntries()):

## write output to root file
out_file = ROOT.TFile(output_dir + "/" + os.path.basename(input_file), "RECREATE")
out_file.cd()

for tf in trig_TF:
  for WP in trig_WP.keys():
    for pt in trg_pt[WP]:
      key = tf + '_' + WP

      wp_val = trig_WP[WP][0]
      # only allow WP > 12 for BMTF
      if wp_val > 12 and tf != "BMTF":
        continue

      # pt2
      h_eff_pt_2[key].Draw()
      h_passed_pt2 = h_eff_pt_2[key].GetPassedHistogram()
      h_passed_pt2.SetName(key + "_pt2_passed")
      h_passed_pt2.Write()
      h_total_pt2 = h_eff_pt_2[key].GetTotalHistogram()
      h_total_pt2.SetName(key + "_pt2_total")
      h_total_pt2.Write()

      # pt
      h_eff_pt[key].Draw()
      h_passed_pt = h_eff_pt[key].GetPassedHistogram()
      h_passed_pt.SetName(key + "_pt_passed")
      h_passed_pt.Write()
      h_total_pt = h_eff_pt[key].GetTotalHistogram()
      h_total_pt.SetName(key + "_pt_total")
      h_total_pt.Write()

      # eta
      h_eff_eta[key].Draw()
      h_passed_eta = h_eff_eta[key].GetPassedHistogram()
      h_passed_eta.SetName(key + "_eta_passed")
      h_passed_eta.Write()
      h_total_eta = h_eff_eta[key].GetTotalHistogram()
      h_total_eta.SetName(key + "_eta_total")
      h_total_eta.Write()

      # phi
      h_eff_phi[key].Draw()
      h_passed_phi = h_eff_phi[key].GetPassedHistogram()
      h_passed_phi.SetName(key + "_phi_passed")
      h_passed_phi.Write()
      h_total_phi = h_eff_phi[key].GetTotalHistogram()
      h_total_phi.SetName(key + "_phi_total")
      h_total_phi.Write()

      # # nPV
      # h_eff_nPV[key].Draw()
      # h_passed_nPV = h_eff_nPV[key].GetPassedHistogram()
      # h_passed_nPV.SetName(key + "_nPV_passed")
      # h_passed_nPV.Write()
      # h_total_nPV = h_eff_nPV[key].GetTotalHistogram()
      # h_total_nPV.SetName(key + "_nPV_total")
      # h_total_nPV.Write()

      # phi_eta
      h_eff_phi_eta[key].Draw()
      h_passed_phi_eta = h_eff_phi_eta[key].GetPassedHistogram()
      h_passed_phi_eta.SetName(key + "_phi_eta_passed")
      h_passed_phi_eta.Write()
      h_total_phi_eta = h_eff_phi_eta[key].GetTotalHistogram()
      h_total_phi_eta.SetName(key + "_phi_eta_total")
      h_total_phi_eta.Write()

      h_dr[key].Draw()
      h_dr[key].Write()
out_file.Write()