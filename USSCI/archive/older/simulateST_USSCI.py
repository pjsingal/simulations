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
parser.add_argument('--gridsz', type=int, help="gridsz = ", default=10)
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
gridsz=args.gridsz

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
    # 'Stagni-2023': {
    #     # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allP.yaml",
    #             },
    # 'Alzueta-2023': {
    #     # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
    #             },
    # 'Glarborg-2018': {
    #     # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
    #             },
    'Aramco-3.0': {
        # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
                },
}

# model = 'aramco30'
refSpecies='H2O'
X_H2O2 = 1163e-6
X_H2O = 1330e-6
X_O2 = 665e-6
dilution = 1-X_H2O2-X_H2O-X_O2
codiluentList = [r'CO$_2$']
X_codiluentList = [0,0.25,0.5]
Tin_list = [1001,1300, 1600, 1900] # Kelvin
P_list = [1,10,25,50]
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3

# gas.TPX = 1196, 2.127*101325, X
########################################################################################

def getTimeHistory(gas):
    r = ct.Reactor(contents=gas,energy="on")
    reactorNetwork = ct.ReactorNet([r]) # this will be the only reactor in the network
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    estIgnitDelay = 0.1
    t = 0
    counter = 1
    while t < estIgnitDelay:
        t = reactorNetwork.step()
        if counter % 10 == 0:
            timeHistory.append(r.thermo.state, t=t)
        counter += 1
    return timeHistory

def generatePlot(model,codiluent,T_list,P_list):
    newCodiluent = codiluent.replace('$','').replace('_','')
    print(f'Codiluent: {codiluent}')
    f, ax = plt.subplots(len(P_list), len(T_list), figsize=(args.figwidth, args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f'Shock Tube: {model}', fontsize=10, y=0.91)
    for z, P in enumerate(P_list):
        print(f'Pressure: {P}atm')
        for w, T in enumerate(T_list):
            print(f'Temperature: {T}K')
            for k,m in enumerate(models[model]):
                print(f'Submodel: {m}')
                for i, X_codiluent in enumerate(X_codiluentList):
                    print(f'{newCodiluent}: {X_codiluent}%')
                    
                    X = {'H2O2':X_H2O2, 'H2O':X_H2O, 'O2':X_O2, 'AR':dilution*(1-X_codiluent), newCodiluent:dilution*X_codiluent}
                    gas = ct.Solution(list(models[model].values())[k])
                    gas.TPX = T, P*ct.one_atm, X
                    timeHistory = getTimeHistory(gas)
                    
                    ax[z,w].plot(timeHistory.t*1e6, timeHistory(refSpecies).X*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {X_codiluent}% {codiluent}')
            ax[z,w].set_title(f"{P}atm, {T}K\n{int(X_H2O2*1e6)} H$_2$O$_2$/{int(X_H2O*1e6)} H$_2$O/{int(X_O2*1e6)} O$_2$\n{round(dilution*100,2)}% {codiluent}/Ar",fontsize=5.6,y=0.01,x=0.97,loc='right')
            ax[z,w].tick_params(axis='both',direction='in')
            
            ax[z,w].set_xlim([0.0001,299.999])
            ax[len(P_list)-1,w].set_xlabel(r'Time [$\mathdefault{\mu s}$]')
        ax[z,0].set_ylabel(r'$\rm H_2O$ mole fraction [%]')
    ax[len(P_list)-1,len(T_list)-1].legend(fontsize=lgdfsz,frameon=False,loc='center',bbox_to_anchor=(0.5, 0.45), handlelength=lgdw)  
    path=f'USSCI/figures/'+args.date+'/shock-tube'
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+f'/{model}_{newCodiluent}.png', dpi=500, bbox_inches='tight')
    print(f'Simulation has been stored at {path}/{model}_{newCodiluent}.png\n')

for model in models:
    print(f'Model: {model}')
    for codiluent in codiluentList:
        generatePlot(model,codiluent,Tin_list,P_list)