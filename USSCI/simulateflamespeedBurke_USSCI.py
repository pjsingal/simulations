import sys, os
sys.path.append(os.getcwd())
sys.path.append("simulations/cantera/build/python")
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

parser = argparse.ArgumentParser()
parser.add_argument('--gridsz', type=int, help="gridsz = ", default=10)
parser.add_argument('--date', type=str)
parser.add_argument('--slopeVal', type=float, help="slope value = ")
parser.add_argument('--curveVal', type=float, help="curve value = ")
parser.add_argument('--transport', type=str, help="flame transport = ")
args = parser.parse_args()
gridsz = args.gridsz
date=args.date
fratio=3
fslope=args.slopeVal
fcurve=args.curveVal
ftransport=args.transport

models = {
    'Alzueta-2023': {
        'base': r'chemical_mechanisms\\Alzueta-2023\\alzuetamechanism.yaml',
        'LMRR': r'factory_mechanisms\\alzuetamechanism_LMRR.yaml',
        'LMRR-allP': r'factory_mechanisms\\alzuetamechanism_LMRR_allP.yaml',
                },
    'Mei-2019': {
        'base': r'chemical_mechanisms\\Mei-2019\\mei-2019.yaml',
        'LMRR': r'factory_mechanisms\\mei-2019_LMRR.yaml',
        'LMRR-allP': r'factory_mechanisms\\mei-2019_LMRR_allP.yaml',
                },
    'Zhang-2017': {
        'base': r"chemical_mechanisms\\Zhang-2017\\zhang-2017.yaml",
        'LMRR': r"factory_mechanisms\\zhang-2017_LMRR.yaml",
        'LMRR-allP': r"factory_mechanisms\\zhang-2017_LMRR_allP.yaml",
                },
    'Otomo-2018': {
        'base': r"chemical_mechanisms\\Otomo-2018\\otomo-2018.yaml",
        'LMRR': r"factory_mechanisms\\otomo-2018_LMRR.yaml",
        'LMRR-allP': r"factory_mechanisms\\otomo-2018_LMRR_allP.yaml",
                },
    'Stagni-2020': {
        'base': r"chemical_mechanisms\\Stagni-2020\\stagni-2020.yaml",
        'LMRR': r"factory_mechanisms\\stagni-2020_LMRR.yaml",
        'LMRR-allP': r"factory_mechanisms\\stagni-2020_LMRR_allP.yaml",
                },
}

for z, n in enumerate(models):
    for k, m in enumerate(models[n]):
        p_list = np.linspace(0.001,20,gridsz)[1:]
        T = 295  # unburned gas temperature [K]
        reactants = 'H2:0.1071, O2:0.1785, He:0.7144'  # premixed gas composition
        width = 0.03  # m
        loglevel = 1  # amount of diagnostic output (0 to 8)
        mbr = []
        for p in p_list:
            gas = ct.Solution(list(models[n].values())[k])
            gas.TPX = T, p*ct.one_atm, reactants
            f = ct.FreeFlame(gas, width=width)
            f.set_refine_criteria(ratio=fratio, slope=fslope, curve=fcurve)
            f.transport_model = ftransport
            f.soret_enabled = True
            f.solve(loglevel=loglevel, auto=True)
            mbr.append(f.velocity[0]*f.density[0] / 10) # g/cm2*s

        path=f'USSCI\\data\\Burke\\'+args.date
        os.makedirs(path,exist_ok=True)
        csv_filename =path+f'\\{n}_{m}.csv'
        mbr_data = zip(p_list, mbr)
        save_to_csv(csv_filename, mbr_data)