from __future__ import division
from __future__ import print_function
import os
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib as mpl
import numpy as np

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
models = {
    'Stagni-2023': {
        'submodels': {
            # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allP.yaml",
                    },
        'fuels': ['H2','NH3'],
        'oxidizer':'O2:1.0, N2:3.76',
        'phi_list':[0.5, 2],
        'P_list':[20,40],
        'refSpecies': 'H2O',
        'estIgnitDelay':12100e-6,
        'xlim': [[60,160], [11000,12000],[60,160], [5250,5550]]
    },
    'Alzueta-2023': {
        'submodels': {
            # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
            'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
            'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
                    },
        'fuels': ['H2','NH3'],
        'oxidizer':'O2:1.0, N2:3.76',
        'phi_list':[0.5, 2],
        'P_list':[20,40],
        'refSpecies': 'H2O',
        'estIgnitDelay':23200e-6,
        'xlim': [[60,160], [12500,23000],[60,160], [5000,9500]]
    },
    'Glarborg-2018': {
        'submodels': {
            # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
                    },
        'fuels': ['H2','NH3'],
        'oxidizer':'O2:1.0, N2:3.76',
        'phi_list':[0.5, 2],
        'P_list':[20,40],
        'refSpecies': 'H2O',
        'estIgnitDelay':9100e-6,
        'xlim': [[60,200], [6000,9000],[60,200], [3400,4800]]
    },
    # 'Aramco-3.0': {
    #     'submodels': {
    #         # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
    #                 },
    #     'fuels': ['C2H2','CH3OH','C4H10'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'refSpecies': 'H2O',
    #     'estIgnitDelay':410e-6,
    #     'xlim': [[-10,200], [-10,125],[100,380], [0,300], [0,75], [0,220]],
    # }
}

lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
T=1196

########################################################################################

def getTimeHistory(gas,oxidizer,est):
    gas.set_equivalence_ratio(phi,fuel,oxidizer,basis='mole')
    gas.TP = T, P*ct.one_atm
    r = ct.Reactor(contents=gas,energy="on")
    reactorNetwork = ct.ReactorNet([r]) # this will be the only reactor in the network
    estIgnitDelay = est
    t = 0
    counter = 1
    timeHistoryStates=[]
    while t < estIgnitDelay:
        t = reactorNetwork.step()
        if not counter % 1:
            timeHistoryStates.append((r.thermo.state, t))
        counter += 1
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    for state, time in timeHistoryStates:
        timeHistory.append(state, t=time)
    return timeHistory


for model in models:
    print(f'Model: {model}')
    f, ax = plt.subplots(len(models[model]['P_list']), len(models[model]['fuels']), figsize=(args.figwidth, args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'ST: {model}', fontsize=10)
    counter=0
    for z, P in enumerate(models[model]['P_list']):
        print(f'Pressure: {P}atm')
        for w, fuel in enumerate(models[model]['fuels']):
            print(f'Fuel: {fuel}')
            for k,m in enumerate(models[model]['submodels']):
                print(f'Submodel: {m}')
                gas = ct.Solution(list(models[model]['submodels'].values())[k])
                for i, phi in enumerate(models[model]['phi_list']):
                    print(r'$\phi$: '+f'{phi}')
                    timeHistory = getTimeHistory(gas,models[model]['oxidizer'],models[model]['estIgnitDelay'])
                    ax[z,w].plot(timeHistory.t*1e6, timeHistory(models[model]['refSpecies']).X*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')
            ax[z,w].set_title(f"{fuel}/air ({T}K, {P}atm)",fontsize=8)
            ax[z,w].tick_params(axis='both',direction='in')
            ax[z,w].set_xlim(models[model]['xlim'][z+w+counter])
            ax[len(models[model]['P_list'])-1,w].set_xlabel(r'Time [$\mathdefault{\mu s}$]')
        counter+=len(models[model]['fuels'])-1
        ax[z,0].set_ylabel(r'$\rm H_2O$ mole fraction [%]')
    ax[len(models[model]['P_list'])-1,len(models[model]['fuels'])-1].legend(fontsize=6,frameon=False,loc='best',handlelength=lgdw)  
    path=f'USSCI/figures/'+args.date+'/shock-tube'
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+f'/{model}.png', dpi=500, bbox_inches='tight')
    print(f'Simulation has been stored at {path}/{model}.png\n')