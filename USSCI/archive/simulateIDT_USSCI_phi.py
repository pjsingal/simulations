import os
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd 
import time
import numpy as np
from joblib import Parallel, delayed
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
    # 'Stagni-2023': {
    #     'submodels': {
    #         # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allPLOG.yaml",
    #         # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allP.yaml",
    #                 },
    #     'fuels': ['H2','NH3'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'T_range':[[1223,1490],
    #                [1188,1520]],
    #     'data': ['NH3']
    # },
    # 'Alzueta-2023': {
    #     'submodels': {
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
    #                 },
    #     'fuels': ['H2','NH3'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'T_range':[[1223,1490],
    #                [1188,1520]],
    #     'data': []
    # },
    # 'Glarborg-2018': {
    #     'submodels': {
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allPLOG.yaml",
    #                 },
    #     'fuels': ['H2','NH3'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'T_range':[[1223,1490],
    #                [1188,1520]],
    #     'data': []
    # },
    'Merchant-2015-a': {
        'submodels': {
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR_allPLOG.yaml",
                    },
        'fuels': ['CH4', 'C2H2','ethanol', 'C2H4', 'C2H6','H2'],
        # rm  
        # 'oxidizer':'O2:1.0, N2:3.76',
        'oxidizer':'O2:1.0, He:8, H2O:2',
        'phi_list':[4,2],
        'P_list':[20,100],
        'T_range':[[500,2000],[500,2000],[500,2000],[500,2000]],
        'data': []
    },
    # 'Merchant-2015-c': {
    #     'submodels': {
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR_allPLOG.yaml",
    #                 },
    #     'fuels': ['CH3OH','C3H8','C3H6','allyl-alcohol','CH3OCH3', 'CH2O'],
    #     # rm  
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5,1],
    #     'P_list':[20,100],
    #     'T_range':[[500,1600],[500,1600],[500,1600],[500,1600]],
    #     'data': []
    # },
    # 'Merchant-2015-b': {
    #     'submodels': {
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR_allPLOG.yaml",
    #                 },
    #     'fuels': ['C2H5', 'C3H8','CH3OH','CH3OCH3','H2','CH2O'],
    #     #other fuel options are 'CH4', 'C2H2','ethanol','allyl-alcohol', 'C2H6', and 'C2H4', but the present conditions make sims break
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5,1],
    #     'P_list':[10,30],
    #     'T_range':[[500,1000],[500,1000],
    #                [625,1000],[625,1000]],
    #     'data': []
    # },
    # 'Aramco-3.0-a': {
    #     'submodels': {
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allPLOG.yaml",
    #                 },
    #     'fuels': ['CH4', 'C2H2','C2H5OH', 'C2H4', 'C2H6','C3H6', 'C3H8','CH3OH','CH3OCH3', 'H2','CH2O'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5,1],
    #     'P_list':[1,20],
    #     'T_range':[[500,1000],[500,1000],
    #                [625,1000],[625,1000]],
    #     'data': []
    # },
    # 'Aramco-3.0-b': {
    #     'submodels': {
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allPLOG.yaml",
    #                 },
    #     'fuels': ['C2H5', 'C3H8','CH3OH','CH3OCH3','H2','CH2O'],
    #     #other fuel options are 'CH4', 'C2H2','ethanol','allyl-alcohol', 'C2H6', and 'C2H4', but the present conditions make sims break
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5,1],
    #     'P_list':[10,30],
    #     'T_range':[[500,1000],[500,1000],
    #                [625,1000],[625,1000]],
    #     'data': []
    # },
}


# fuelList=['H2','NH3']
# # fuelList=['C2H2','CH3OH','C4H10']
# oxidizer={'O2':1, 'N2': 3.76}
# T_list = np.linspace(2000,2500,gridsz)
# phi_list = [1,4]
# P_list = [1,10]
lstyles = ["solid","dashed","dotted"]
colors = ["xkcd:grey","xkcd:purple", "xkcd:teal", "orange", "r", "b", "xkcd:lime green", "xkcd:magenta", "xkcd:navy blue","xkcd:grey","cyan"]*2

def getTimeHistory(phi,fuel,oxidizer,T,P):
    def ignitionDelay(states, species):
        i_ign = np.gradient(states(species).Y.T[0]).argmax()
        # i_ign = states(species).Y.T[0].argmax()
        # print(np.gradient(states(species).Y.T[0]).argmax())
        return states.t[i_ign]   
    gas.set_equivalence_ratio(phi,fuel,oxidizer,basis='mole')
    gas.TP = T, P*ct.one_atm
    r = ct.Reactor(contents=gas)
    reactorNetwork = ct.ReactorNet([r])
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    t_end=1
    t=0
    while t < t_end:
        t=reactorNetwork.step()
        timeHistory.append(r.thermo.state, t=t)
    # # print(timeHistory('o').Y)
    # # oh, oh*, h, o, pressure
    # # max gradient of Xoh, max gradient of Yoh, max value of Xoh, max gradient of Yog
    return ignitionDelay(timeHistory, 'oh')


# def getIDT(gas,T_list,P):
def getTempHistory(phi,fuel,oxidizer,T_list,P):
    IDT_list = Parallel(n_jobs=-1)(  # Use all available cores; adjust n_jobs if needed
        delayed(getTimeHistory)(phi,fuel,oxidizer,T,P)
        for T in T_list
    )
    return np.array(IDT_list)


for model in models:
    tic = time.time()
    print(f'Model: {model}')
    f, ax = plt.subplots(len(models[model]['P_list']), len(models[model]['phi_list']), figsize=(args.figwidth, args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'IDT: {model}', fontsize=10)
    for z, P in enumerate(models[model]['P_list']):
        print(f'Pressure: {P}atm')
        T_list = np.linspace(models[model]['T_range'][z][0],models[model]['T_range'][z][1],gridsz)
        # for w, fuel in enumerate(models[model]['fuels']):
        for w, phi in enumerate(models[model]['phi_list']):
            # print(f'Fuel: {fuel}')
            print(r'$\phi$: '+f'{phi}')
            for k,m in enumerate(models[model]['submodels']):
                print(f'Submodel: {m}')
                gas = ct.Solution(list(models[model]['submodels'].values())[k])
                mkrs=['o','x']
                for i, fuel in enumerate(models[model]['fuels']):
                # for i, phi in enumerate(models[model]['phi_list']):
                    IDT = getTempHistory(phi, fuel, models[model]['oxidizer'], T_list, P)
                    print(f'Fuel: {fuel}')
                    ax[z,w].semilogy(T_list, IDT*1e3, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{fuel}')#label=f'{m} '+r'$\phi$='+f'{phi}')
                    # if fuel in models[model]['data']:
                    #     dat = pd.read_csv(f'USSCI/graph-reading/{model}/IDT/{P}bar_{phi}phi.csv',header=None)
                    #     ax[z,w].semilogy(dat.iloc[:,0],dat.iloc[:,1],mkrs[i],fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=r'$\phi$='+f'{phi}')
            ax[z,w].set_title(r'$\phi$='+f'{phi} ({P}atm)',fontsize=8)
            # ax[z,w].set_ylim([1e-3,1e5])
            ax[z,w].tick_params(axis='both',direction='in')
            ax[len(models[model]['P_list'])-1,w].set_xlabel('Temperature [K]')
        ax[z,0].set_ylabel(r'Ignition delay [ms]')
    ax[len(models[model]['P_list'])-1,len(models[model]['phi_list'])-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=2)  
    path=f'USSCI/figures/'+args.date+'/IDT'
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+f'/{model}.png', dpi=500, bbox_inches='tight')
    toc = time.time()
    print(f'Simulation completed in {toc-tic}s and stored at {path}/{model}.png\n')

#     /home/pjs/simulations/USSCI/graph-reading/Stagni-2023/20bar_0,5phi.csv
# 'USSCI/graph-reading/Stagni-2023/20bar_0.5phi.csv

