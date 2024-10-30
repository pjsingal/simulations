import sys, os
sys.path.append(os.getcwd()+"cantera/build/python")
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy
import scipy.optimize
from scipy.optimize import curve_fit
import numpy as np
import argparse
import csv

# Function to save data to CSV
def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

name = 'MBR_BurkeSong'

# models = {
#           'Alzueta':"chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml",            
#           'Ar':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allAR.yaml",
#           r'H$_2$O':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allH2O.yaml",
#           'LMR-R':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR.yaml", 
#           }

models = {    
          # 'Alzueta':"chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml",  
          # 'Alzueta-300K':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_epsNH3_T=300K.yaml",  
          # 'Alzueta-2000K':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_epsNH3_T=2000K.yaml",            
          # 'Ar':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allAR.yaml",
          # r'H$_2$O':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allH2O.yaml",
          # 'LMR-R':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR.yaml", 
            'LMR-R-PLOG':"chemical_mechanisms/Alzueta-2023/LMRtests/LMRtest_PLOG_M.yaml",
            'LMR-R-Troe':"chemical_mechanisms/Alzueta-2023/LMRtests/LMRtest_Troe_M.yaml",
            'LMR-R-Cheb':"chemical_mechanisms/Alzueta-2023/LMRtests/LMRtest_cheb_M.yaml",   
          }

# models = {    
#           'LMR-R':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR.yaml", 
#           }

parser = argparse.ArgumentParser()
parser.add_argument('--gridsz', type=int, help="gridsz = ", default=10)
parser.add_argument('--date', type=str, help="sim date = ",default='May28')
parser.add_argument('--slopeVal', type=float, help="slope value = ")
parser.add_argument('--curveVal', type=float, help="curve value = ")
parser.add_argument('--transport', type=str, help="flame transport = ")
args = parser.parse_args()

# gridsz = 10
gridsz = args.gridsz
date=args.date
fratio=3
fslope=args.slopeVal
fcurve=args.curveVal
ftransport=args.transport


colors = ['xkcd:purple','r','b']

for i, m in enumerate(list(models.keys())):
    # p_list = np.linspace(8,12,gridsz)[1:]
    p_list = np.linspace(0.001,20,gridsz)[1:]
    T = 295  # unburned gas temperature [K]
    reactants = 'H2:0.1071, O2:0.1785, He:0.7144'  # premixed gas composition
    width = 0.03  # m
    loglevel = 1  # amount of diagnostic output (0 to 8)
    mbr = []
    for p in p_list:
        gas = ct.Solution(list(models.values())[i])
        gas.TPX = T, p*ct.one_atm, reactants
        f = ct.FreeFlame(gas, width=width)
        f.set_refine_criteria(ratio=fratio, slope=fslope, curve=fcurve)
        f.transport_model = ftransport
        f.soret_enabled = True
        f.solve(loglevel=loglevel, auto=True)
        mbr.append(f.velocity[0]*f.density[0] / 10) # g/cm2*s

    path=f'PCI-ESSCI/BurkeSongResults_'+date+f' (slope={fslope} curve={fcurve})'
    os.makedirs(path,exist_ok=True)
    csv_filename =path+f'/{m}_data.csv'
    mbr_data = zip(p_list, mbr)
    save_to_csv(csv_filename, mbr_data)