from __future__ import division
from __future__ import print_function
import sys,os
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib as mpl
import numpy as np
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--figwidth', type=float, help="figwidth = ")
parser.add_argument('--figheight', type=float, help="figheight = ")
parser.add_argument('--fsz', type=float, help="mpl.rcParams['font.size'] = ", default=8)
parser.add_argument('--fszxtick', type=float, help="mpl.rcParams['xtick.labelsize'] = ", default=7)
parser.add_argument('--fszytick', type=float, help="mpl.rcParams['ytick.labelsize'] = ", default=7)
parser.add_argument('--fszaxlab', type=float, help="mpl.rcParams['axes.labelsize'] = ", default=8)
parser.add_argument('--lw', type=float, help="lw = ", default=0.7)
parser.add_argument('--mw', type=float, help="mw = ", default=0.5)
parser.add_argument('--msz', type=float, help="msz = ", default=2.5)
parser.add_argument('--lgdw', type=float, help="lgdw = ", default=0.6)
parser.add_argument('--lgdfsz', type=float, help="lgdw = ", default=5)
parser.add_argument('--dpi', type=int, help="dpi = ", default=1000)
parser.add_argument('--date', type=str, help="sim date = ")

import matplotlib.ticker as ticker

args = parser.parse_args()
lw=args.lw
mw=args.mw
msz=args.msz
dpi=args.dpi
lgdw=args.lgdw
lgdfsz=args.lgdfsz

mpl.rc('font',family='Times New Roman')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = args.fsz
mpl.rcParams['xtick.labelsize'] = args.fszxtick
mpl.rcParams['ytick.labelsize'] = args.fszytick
from matplotlib.legend_handler import HandlerTuple
plt.rcParams['axes.labelsize'] = args.fszaxlab
# mpl.rcParams['xtick.major.width'] = 0.5  # Width of major ticks on x-axis
# mpl.rcParams['ytick.major.width'] = 0.5  # Width of major ticks on y-axis
# mpl.rcParams['xtick.minor.width'] = 0.5  # Width of minor ticks on x-axis
# mpl.rcParams['ytick.minor.width'] = 0.5  # Width of minor ticks on y-axis
# mpl.rcParams['xtick.major.size'] = 2.5  # Length of major ticks on x-axis
# mpl.rcParams['ytick.major.size'] = 2.5  # Length of major ticks on y-axis
# mpl.rcParams['xtick.minor.size'] = 1.5  # Length of minor ticks on x-axis
# mpl.rcParams['ytick.minor.size'] = 1.5  # Length of minor ticks on y-axis


########################################################################################
title=r'ST Shao'
folder='Shao-2019'
name='Fig4c'
exp=False
dataLabel='Liu et al. (2019)'
data=['4c_2phi.csv']

T=1196
X_H2O2 = 1163e-6
X_H2O = 1330e-6
X_O2 = 665e-6
X_CO2= 0.2*(1-X_H2O2-X_H2O-X_O2)
X_Ar = 1-X_CO2
X = {'H2O2':X_H2O2, 'H2O':X_H2O, 'O2':X_O2, 'CO2':X_CO2, 'AR':X_Ar}
P=2.127
refSpecies='H2O'
Xlim=[0.0001,299.999]
Ylim=[0.120001,0.269999]

models = {
    'Klippenstein-CNF2018': {
        'submodels': {
            'base': r"chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
                    },
    },
    'Glarborg-2025': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allPLOG.yaml",
                    },
    },
}

lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3

########################################################################################
def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getTimeHistory(gas):
    gas.TPX = T, P*ct.one_atm, X
    r = ct.Reactor(contents=gas,energy="on")
    reactorNetwork = ct.ReactorNet([r]) # this will be the only reactor in the network
    estIgnitDelay = 1
    t = 0
    counter = 1
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    while t < estIgnitDelay:
        t = reactorNetwork.step()
        if not counter % 1:
            timeHistory.append(r.thermo.state, t=t)
        counter += 1
    return timeHistory.t, timeHistory(refSpecies).X.flatten()

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    timeHistory=getTimeHistory(gas)
    data = zip(timeHistory[0],timeHistory[1])
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/ST/{m}'
    os.makedirs(simOutPath,exist_ok=True)
    save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')

print(folder)
tic1=time.time()
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        sims=generateData(model,m) 
        simFile=f'USSCI/data/{args.date}/{folder}/{model}/ST/{m}/{name}.csv'
        # if not os.path.exists(simFile):
        #     sims=generateData(model,m)  
        sims=pd.read_csv(simFile)
        label = f'{model}' if k == 0 else None
        ax.semilogy(sims.iloc[:,0]*1e6,sims.iloc[:,1]*100, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
        if exp and j==len(models)-1 and k==2:
            dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[0]}',header=None)
            ax.semilogy(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylim)
        ax.tick_params(axis='both',direction='in')
        ax.set_xlabel(r'Time [$\mathdefault{\mu s}$]')
        ax.set_ylabel(r'Mole fraction [%]')
        print('  > Data added to plot')
ax.annotate(f'{title}', xy=(0.97, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+1)

ax.legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/ST'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
