import argparse
import sys
import os
import subprocess

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exec', type=str, help='executable')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--submit', help='submit to condor',action='store_true')
parser.add_argument('--jobFlav', type=str, help='condor job flavour', default="espresso")
parser.add_argument('-o','--output', type=str, help='output dir')
parser.add_argument('--submitName', type=str, help='name of the condor submit file', default="submit_nano.sh")
parser.add_argument('--nFiles', type=str, help='number of files to run')
args = parser.parse_args()

## load golden json file
pwd = os.getcwd()
json_files = {
  "2023B": pwd + "/../JSON/Cert_Collisions2023_eraB_366403_367079_Golden.json",
  "2023C": pwd + "/../JSON/Cert_Collisions2023_eraC_367095_368823_Golden.json",
  "2023D": pwd + "/../JSON/Cert_Collisions2023_eraD_369803_370790_Golden.json",
  "2024B": pwd + "/../JSON/2024B_Golden.json",
  "2024C": pwd + "/../JSON/2024C_Golden.json",
  "2024D": pwd + "/../JSON/2024D_Golden.json",
  "2024E": pwd + "/../JSON/2024E_Golden.json",
  "2024F": pwd + "/../JSON/2024F_Golden.json",
  "2024G": pwd + "/../JSON/2024G_Golden.json",
  "2024H": pwd + "/../JSON/2024H_Golden.json",
  "2024I": pwd + "/../JSON/2024I_Golden.json",
  "2025B": pwd + "/../JSON/Cert_Collisions2025_391658_398860_Golden.json",  #391548 to 391950
  "2025C": pwd + "/../JSON/Cert_Collisions2025_391658_398860_Golden.json",  #392174 to 393087
  "2025D": pwd + "/../JSON/Cert_Collisions2025_391658_398860_Golden.json",  #394393 to 395948
  "2025E": pwd + "/../JSON/Cert_Collisions2025_391658_398860_Golden.json",  #395982 to 396422
  "2025F": pwd + "/../JSON/Cert_Collisions2025_391658_398860_Golden.json",  #396629 to 397853
  "2025G": pwd + "/../JSON/Cert_Collisions2025_391658_398860_Golden.json",  #397954 to 398903 (23/11)
#   "2025G": pwd + "/../JSON/2025_lowPU.json",  #397954 to 398903 (23/11)
}

if args.exec == None:
    print("you need to specify the executable")
    sys.exit(-1)

if args.dataset == None:
    print("you need to specify the dataset")
    sys.exit(-1)

if args.output == None:
    print("you need to specify the output dir")
    sys.exit(-1)

executable = args.exec
dataset = args.dataset

## find files using DAS
das_query = 'dasgoclient --query="file dataset=' + dataset + '"'
if args.nFiles:
    das_query += " -limit " + args.nFiles
query_out = os.popen(das_query)
files_found = ['root://xrootd-cms.infn.it/'+_file.strip() for _file in query_out]

print("Will run " + executable)
print("Dataset " + dataset)
print(str(len(files_found)) + " files found")
if not os.path.exists(args.output):
    os.makedirs(args.output)
print("Will write output to " + args.output)
era = dataset[dataset.find("Run")+3:+dataset.find("Run")+8] #find era from dataset name
print("Era",era,"found")
json_file_path = json_files[era]
print("json file", json_file_path)
json_file_str = json_file_path if json_file_path is not None else ""
exec_name=executable[:-3]
muon = dataset[dataset.find("Muon")+4]

## create the file list at the end of the submit file
def format_files_in_queue(files_found):
    file_string = ""
    for index, file_ in enumerate(files_found):
        if index != len(files_found) - 1: file_string += file_ + ","
        else: file_string += file_
    return file_string

## Create log directory
log_dir = f"log/{era}/Muon{muon}/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

## write condor submit file
condor_submit_file = open(args.submitName,"w")
condor_submit_file.write('''
executable = ''' + os.getcwd() + "/run.sh" '''
use_x509userproxy = true

arguments = ''' + executable + ''' $(Item) ''' + args.output + ''' ''' + pwd + ''' ''' + json_file_str +''' 

error   = ''' +log_dir+'''/''' + exec_name + '''/_$(Process).err
output  = ''' +log_dir+'''/''' + exec_name + '''/_$(Process).out
log     = ''' +log_dir+'''/''' + exec_name + '''/_$(Process).log

JobBatchName = muonDPG_''' + era + '''_''' +muon+'''_''' + exec_name + '''
+JobFlavour = "''' + args.jobFlav + '''"
    
queue 1 in (''' + format_files_in_queue(files_found) + ''')  
''')

condor_submit_file.close()

## submit jobs to condor
if args.submit:
    query_out = os.popen("condor_submit " + args.submitName)
    print("Jobs submitted to condor")
else: print(args.submitName + " created but jobs have not been submitted to condor")