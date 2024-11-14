import os
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd 
import time
import numpy as np
import matplotlib as mpl
import argparse

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
parser.add_argument('--dpi', type=int, help="dpi = ", default=500)
parser.add_argument('--date', type=str)

args = parser.parse_args()
lw=args.lw
mw=args.mw
msz=args.msz
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
    # 'Stagni-2020': {
    #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
    #             },
    # 'Alzueta-2023': {
    #     # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
    #             },
    # 'Glarborg-2018': {
    #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
    #             },
    'Aramco-3.0': {
        # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
                },
}

# fuelList=['H2','C2H2','CH3OH','C4H10','NH3']
fuelList=['C2H2','CH3OH']
oxidizer={'O2':1, 'N2': 3.76}
T_list = np.linspace(830,1667,10)
phi_list = [1,4]
P_list = [1,10]
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3


def getIDT(gas,T_list,P):
    def ignitionDelay(states, species):
        # i_ign = np.gradient(states(species).Y.T[0]).argmax()
        # i_ign = np.gradient(states(species).Y.T[0]).argmax()
        i_ign = states(species).Y.T[0].argmax()
        # print(np.gradient(states(species).Y.T[0]).argmax())
        return states.t[i_ign]    
    estimatedIgnitionDelayTimes = np.ones(len(T_list))
    # estimatedIgnitionDelayTimes[:6] = 6 * [0.1]
    # estimatedIgnitionDelayTimes[-4:-2] = 10
    # estimatedIgnitionDelayTimes[-2:] = 100
    estimatedIgnitionDelayTimes[:]=0.1
    ignitionDelays_RG = np.zeros(len(T_list))
    for j, T in enumerate(T_list):
        gas.TP = T, P*ct.one_atm
        # r = ct.Reactor(contents=gas)
        r = ct.IdealGasReactor(contents=gas, name="Batch Reactor")
        # r = ct.Reactor(contents=gas, name="Batch Reactor")
        reactorNetwork = ct.ReactorNet([r])
        timeHistory = ct.SolutionArray(gas, extra=['t'])
        t0 = time.time()
        t = 0
        # counter = 1
        while t < estimatedIgnitionDelayTimes[j]:
            t = reactorNetwork.step()
            # if counter % 1 == 0:
            timeHistory.append(r.thermo.state, t=t)
            # counter += 1
        # print(timeHistory('o').Y)
        tau = ignitionDelay(timeHistory, 'oh')
        t1 = time.time()
        ignitionDelays_RG[j] = tau
    return ignitionDelays_RG

for model in models:
    print(f'Model: {model}')
    f, ax = plt.subplots(len(P_list), len(fuelList), figsize=(args.figwidth, args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'IDT: {model}', fontsize=10, y=0.91)
    for z, P in enumerate(P_list):
        print(f'Pressure: {P}atm')
        for w, fuel in enumerate(fuelList):
            print(f'Fuel: {fuel}')
            for k,m in enumerate(models[model]):
                print(f'Submodel: {m}')
                gas = ct.Solution(list(models[model].values())[k])
                for i, phi in enumerate(phi_list):
                    print(r'$\phi$: '+f'{phi}')
                    gas.set_equivalence_ratio(phi,fuel,oxidizer)
                    ignitionDelays = getIDT(gas,T_list,P)
                    ax[z,w].semilogy(T_list, ignitionDelays*1e6, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')
            ax[z,w].set_title(f"{fuel}/air ({P}atm)",fontsize=8)
            ax[z,w].tick_params(axis='both',direction='in')
            ax[len(P_list)-1,w].set_xlabel('Temperature [K]')
        ax[z,0].set_ylabel(r'Ignition delay [$\mathdefault{\mu s}$]')
    ax[len(P_list)-1,len(fuelList)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)  
    path=f'USSCI/figures/'+args.date+'/IDT'
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+f'/{model}.png', dpi=500, bbox_inches='tight')
    print(f'Simulation has been stored at {path}/{model}.png\n')