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

lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
models = {
    'AramcoMech3.0': {
        # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
        'LMRR': f'USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml',
        'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml',
                },
    'Alzueta-2023': {
        # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
        'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
        'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
                },
    # 'Mei-2019': {
    #     # 'base': r'chemical_mechanisms/Mei-2019/mei-2019.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR_allP.yaml',
    #             },
    # 'Zhang-2017': {
    #     # 'base': r"chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR_allP.yaml",
    #             },
    # 'Otomo-2018': {
    #     'base': r"chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR_allP.yaml",
    #             },
    # 'Stagni-2020': {
    #     'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
    #             },
    # 'Han-2021': {
    #     'base': r"chemical_mechanisms/Han-2021/han-2021.yaml",
    #     # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/han-2021_LMRR.yaml",
    #     # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/han-2021_LMRR_allP.yaml",
    #             },
}

conditions = {
    'AramcoMech3.0': {
        # 'X': {'N2O':0.02, 'AR': 0.98},
        'X': {'C4H6':0.01, 'O2': 0.11, 'AR': 0.88},
        'T': [1232],
        # 'P': [3,12],
        'P': [1.2],
        'phi_list': [0.5],
                },
    # 'Zhang-2017': {
    #     # 'X': {'N2O':0.02, 'AR': 0.98},
    #     'X': {'H2':0.0005, 'CO': 0.03, 'N2O': 0.01, 'AR': 0.9595},
    #     'T': [11],
    #     # 'P': [3,12],
    #     'P': [1.4,10],
    #     'phi_list': [0.063],
    #             },
}


##Shao Mixture
# # X_H2O2 = 1163e-6
# # X_H2O = 1330e-6
# # X_O2 = 665e-6
# # X_CO2= 0.2*(1-X_H2O2-X_H2O-X_O2)
# # X_Ar = 1-X_CO2
# # X = {'H2O2':X_H2O2, 'H2O':X_H2O, 'O2':X_O2, 'CO2':X_CO2, 'AR':X_Ar}
# #Ronney Mixture
# x_fuel = (phi*(1/0.75)*0.21)/(1+phi*(1/0.75)*0.21)
# x_o2 = 0.21*(1-x_fuel)
# x_n2 = 0.79*(1-x_fuel)

def cp(T,P,X,model):
  gas_stream = ct.Solution(model)
  gas_stream.TPX = T, P*1e5, {X:1}
  return gas_stream.cp_mole # [J/kmol/K]

for z, n in enumerate(models):
    for phi in conditions[n]['phi_list']:
        for P in conditions[n]['P']:
            # fig, ax = plt.subplots(1, len(models.keys()),figsize=(args.figwidth, args.figheight))
            plt.figure()
            for k,m in enumerate(models[n]):
                for T in conditions[n]['T']:
                    gas = ct.Solution(list(models[n].values())[k])
                    gas.TPX = T, P*ct.one_atm, conditions[n]['X']
                    r = ct.Reactor(contents=gas,energy="on")
                    reactorNetwork = ct.ReactorNet([r])
                    timeHistory = ct.SolutionArray(gas, extra=['t'])
                    estIgnitDelay = 1
                    t = 0
                    counter = 1
                    while t < estIgnitDelay:
                        t = reactorNetwork.step()
                        if counter % 10 == 0:
                            timeHistory.append(r.thermo.state, t=t)
                        counter += 1
                    plt.plot(timeHistory.t*1e6, timeHistory('H2O').X*100, color=colors[k],linestyle=lstyles[k],linewidth=lw,label=f'{m} {P}atm')
                plt.legend(fontsize=lgdfsz,handlelength=lgdw, frameon=False, loc='lower right')
                plt.title(f"{n} ({T}K,{P}atm,{phi}phi)")
                plt.ylabel(r'$\rm H_2O$ mole fraction [%]')
                plt.xlabel(r'Time [$\mathdefault{\mu s}$]')
                X_str = "_".join(f"{key}{value}" for key, value in conditions[n]['X'].items())
                path=f'USSCI/figures/{args.date}/shock-tube'
                os.makedirs(path,exist_ok=True)
                plt.savefig(path+f'/{n}_{X_str}.png', dpi=500, bbox_inches='tight')