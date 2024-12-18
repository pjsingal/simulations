from __future__ import division
from __future__ import print_function
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from joblib import Parallel, delayed
import matplotlib as mpl
import argparse
import csv
import warnings

warnings.filterwarnings("ignore", message="NasaPoly2::validate")
warnings.filterwarnings("ignore", message=".*discontinuity.*detected.*")
warnings.filterwarnings("ignore", message=".*return _ForkingPickler.loads.*")
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
parser.add_argument('--gridsz', type=int, help="gridsz = ", default=10)
parser.add_argument('--dpi', type=int, help="dpi = ", default=1000)
parser.add_argument('--date', type=str, help="sim date = ")
args = parser.parse_args()
lw=args.lw
mw=args.mw
msz=args.msz
dpi=args.dpi
lgdw=args.lgdw
lgdfsz=args.lgdfsz
gridsz=args.gridsz
from matplotlib.legend_handler import HandlerTuple
mpl.rc('font',family='Times New Roman')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = args.fsz
mpl.rcParams['xtick.labelsize'] = args.fszxtick
mpl.rcParams['ytick.labelsize'] = args.fszytick
plt.rcParams['axes.labelsize'] = args.fszaxlab
mpl.rcParams['xtick.major.width'] = 0.5  # Width of major ticks on x-axis
mpl.rcParams['ytick.major.width'] = 0.5  # Width of major ticks on y-axis
mpl.rcParams['xtick.minor.width'] = 0.5  # Width of minor ticks on x-axis
mpl.rcParams['ytick.minor.width'] = 0.5  # Width of minor ticks on y-axis
mpl.rcParams['xtick.major.size'] = 2.5  # Length of major ticks on x-axis
mpl.rcParams['ytick.major.size'] = 2.5  # Length of major ticks on y-axis
mpl.rcParams['xtick.minor.size'] = 1.5  # Length of minor ticks on x-axis
mpl.rcParams['ytick.minor.size'] = 1.5  # Length of minor ticks on y-axis



########################################################################################


folder='HO2'
P_list = np.linspace(1,100,gridsz)
T_list=[700,1500,2000]
reactionList=['H + O2 (+M) <=> HO2 (+M)','H + O2 <=> HO2','H + O2 (+M) <=> HO2 (+M)']
Xlim=[0,60]
Ylim=[1e0,1e3]
collider='AR'
X={collider:1}
title=f'H + O2 (+Ar) <=> HO2 (+Ar)'

models = {
    'Troe (ThInK 1.0)': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10/think.yaml",
                    },
    },
    r'PLOG (ThInK 1.0)': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10_HO2plog/think_ho2plog.yaml",
                    },
    },
    'Troe (Burke-2013)': {
        'submodels': {
            'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*3

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def getRateConstant(gas,T,P,reaction):
    gas.TPX = T, P*ct.one_atm,X
    k=gas.forward_rate_constants[gas.reaction_equations().index(reaction)]*1e-6
    Rjoule = 8.31446261815324 # [J/K/mol]
    conc = np.multiply(gas[collider].X, np.divide(P*ct.one_atm,np.multiply(Rjoule,T)))
    return k/conc

def generateData(model,m,reaction,T):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    # save_to_csv('data.csv', gas.reaction_equations())
    # print(gas.reaction_equations())
    k_list=[getRateConstant(gas,T,P,reaction) for P in P_list]
    data = zip(P_list,k_list)
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/RC_vs_P/{m}/{reaction}'
    os.makedirs(simOutPath,exist_ok=True)
    save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)

import matplotlib.ticker as ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
for j,model in enumerate(models):
    ax.plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for i, T in enumerate(T_list):
        name=f'K_vs_P_{collider}_{T}K'
        reaction = reactionList[j]
        print(f'Model: {model}')
        print(reaction)
        
        for k,m in enumerate(models[model]['submodels']):
            print(f' Submodel: {m}')
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/RC_vs_P/{m}/{reaction}/{name}.csv'
            # if not os.path.exists(simFile):
            sims=generateData(model,m,reaction,T)  
            sims=pd.read_csv(simFile)
            label = f'{T}K'
            ax.semilogy(sims.iloc[:,0],sims.iloc[:,1], color=colors[j], linestyle=lstyles[i], linewidth=lw, label=label)
            ax.set_xlim(Xlim)
            ax.set_ylim(Ylim)
            ax.tick_params(axis='both',direction='in')
            ax.set_xlabel('Pressure [atm]')
            print('  > Data added to plot')
ax.annotate(f'{title}', xy=(0.05, 0.95), xycoords='axes fraction',ha='left', va='top',fontsize=lgdfsz)
ax.legend(fontsize=lgdfsz,frameon=False,loc='lower left', handlelength=lgdw,ncol=1,bbox_to_anchor=(1,0))
ax.set_ylabel(f'k/[AR] [/s]')
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/RC'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')


