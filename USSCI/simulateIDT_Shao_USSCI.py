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
mpl.rcParams['xtick.major.width'] = 0.5  # Width of major ticks on x-axis
mpl.rcParams['ytick.major.width'] = 0.5  # Width of major ticks on y-axis
mpl.rcParams['xtick.minor.width'] = 0.5  # Width of minor ticks on x-axis
mpl.rcParams['ytick.minor.width'] = 0.5  # Width of minor ticks on y-axis
mpl.rcParams['xtick.major.size'] = 2.5  # Length of major ticks on x-axis
mpl.rcParams['ytick.major.size'] = 2.5  # Length of major ticks on y-axis
mpl.rcParams['xtick.minor.size'] = 1.5  # Length of minor ticks on x-axis
mpl.rcParams['ytick.minor.size'] = 1.5  # Length of minor ticks on y-axis

lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3




models = {
    # 'Alzueta-2023': {
    #     # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
    #             },
    # 'Mei-2019': {
    #     # 'base': r'chemical_mechanisms/Mei-2019/mei-2019.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR_allP.yaml',
    #             },
    'Zhang-2017': {
        'base': r"chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR_allP.yaml",
                },
    # 'Otomo-2018': {
    #     'base': r"chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
    #     # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR.yaml",
    #     # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR_allP.yaml",
    #             },
    # 'Stagni-2020': {
    #     'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
    #     # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
    #             },
    # 'Han-2021': {
    #     'base': r"chemical_mechanisms/Han-2021/han-2021.yaml",
    #     # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/han-2021_LMRR.yaml",
    #     # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/han-2021_LMRR_allP.yaml",
    #             },
}

conditions = {
    'Zhang-2017': {
        # 'X': {'N2O':0.02, 'AR': 0.98},
        'X': {'H2':0.0005, 'CO': 0.03, 'N2O': 0.01, 'AR': 0.9595},
        'T': np.linspace(1000/0.4,1000/0.75, 25),
        # 'P': [3,12]
        'P': [1.4,10]
                },
}

colours = ["xkcd:grey",'xkcd:purple','r']
lstyles = ["solid","dashed","dotted"]

def ignitionDelay(states, species):
    # i_ign = np.gradient(states(species).Y.T[0]).argmax()
    i_ign = np.gradient(states(species).Y.T[0]).argmax()
    # i_ign = states(species).Y.T[0].argmax()
    # print(np.gradient(states(species).Y.T[0]).argmax())
    return states.t[i_ign]

for z, n in enumerate(models):
    plt.figure()
    for q,P in enumerate(conditions[n]['P']):
        for k, m in enumerate(models[n]):
            estimatedIgnitionDelayTimes = np.ones(len(conditions[n]['T']))
            # estimatedIgnitionDelayTimes[:6] = 6 * [0.1]
            # estimatedIgnitionDelayTimes[-4:-2] = 10
            # estimatedIgnitionDelayTimes[-2:] = 100
            estimatedIgnitionDelayTimes[:]=0.1
            ignitionDelays_RG = np.zeros(len(conditions[n]['T']))
            for j, T in enumerate(conditions[n]['T']):
                gas = ct.Solution(list(models[n].values())[k])
                gas.TPX = T, P*ct.one_atm, conditions[n]['X']
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
            plt.semilogy(np.divide(1000,conditions[n]['T']), 1e6*ignitionDelays_RG, linestyle=lstyles[q], color=colours[k], label=f'{m} {P}atm')

    plt.ylabel(r'Ignition delay [$\mathdefault{\mu s}$]')
    plt.xlabel('Temperature [K]')
    X_str = "_".join(f"{key}{value}" for key, value in conditions[n]['X'].items())
    plt.title(f'IDT {n} (X={X_str})',fontsize=10)

    plt.legend(fontsize=10, frameon=False, loc='upper right')

    path=f'USSCI/figures/{args.date}/IDT'
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+f'/{n}_{X_str}.png', dpi=500, bbox_inches='tight')