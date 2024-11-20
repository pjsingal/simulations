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
# models = {
#     'Stagni-2020': {
#         # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
#         'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
#         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
#                 },
#     # 'Alzueta-2023': {
#     #     # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
#     #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
#     #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
#     #             },
#     # 'Glarborg-2018': {
#     #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
#     #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
#     #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
#     #             },
#     # 'Aramco-3.0': {
#     #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
#     #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
#     #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
#     #             },
# }

models = {
    'Stagni-2020': {
        'submodels': {
            # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
                    },
        'fuels': ['H2','NH3'],
        'oxidizer':'O2:1.0, N2:3.76',
        'phi_list':[0.5, 2],
        'P_list':[20,40],
        'T_range':[[1223,1490],
                   [1188,1520]]
    }
}


# fuelList=['H2','NH3']
# # fuelList=['C2H2','CH3OH','C4H10']
# oxidizer={'O2':1, 'N2': 3.76}
# T_list = np.linspace(2000,2500,gridsz)
# phi_list = [1,4]
# P_list = [1,10]
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3


# def getIDT(gas,T_list,P):
def getIDT(phi,fuel,oxidizer,T_list,P):
    def ignitionDelay(states, species):
        i_ign = np.gradient(states(species).Y.T[0]).argmax()
        # i_ign = states(species).Y.T[0].argmax()
        # print(np.gradient(states(species).Y.T[0]).argmax())
        return states.t[i_ign]    
    
    # def ignitionDelay(timeHistory, species):
    #     time = timeHistory.t
    #     X_species = timeHistory(species).X.flatten()
    #     dOHdt = np.gradient(X_species, time)
    #     ignition_delay_index = np.argmax(dOHdt)
    #     ignition_delay_time = time[ignition_delay_index]
    #     return ignition_delay_time

    IDT_list = []
    for j, T in enumerate(T_list):
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
        IDT_list.append(ignitionDelay(timeHistory, 'oh'))
    return np.array(IDT_list)

for model in models:
    print(f'Model: {model}')
    f, ax = plt.subplots(len(models[model]['P_list']), len(models[model]['fuels']), figsize=(args.figwidth, args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'IDT: {model}', fontsize=10)
    for z, P in enumerate(models[model]['P_list']):
        print(f'Pressure: {P}atm')
        T_list = np.linspace(models[model]['T_range'][z][0],models[model]['T_range'][z][1],gridsz)
        for w, fuel in enumerate(models[model]['fuels']):
            print(f'Fuel: {fuel}')
            for k,m in enumerate(models[model]['submodels']):
                print(f'Submodel: {m}')
                gas = ct.Solution(list(models[model]['submodels'].values())[k])
                mkrs=['o','x']
                for i, phi in enumerate(models[model]['phi_list']):
                    print(r'$\phi$: '+f'{phi}')
                    # gas.set_equivalence_ratio(phi,fuel,models[model]['oxidizer'],basis='mole')
                    ignitionDelays = getIDT(phi,fuel,models[model]['oxidizer'],T_list,P)
                    ax[z,w].semilogy(T_list, ignitionDelays*1e3, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')
                    
                    if fuel=='NH3':
                        dat = pd.read_csv(f'USSCI/graph-reading/{model}/{P}bar_{phi}phi.csv',header=None)
                        ax[z,w].semilogy(dat.iloc[:,0],dat.iloc[:,1],mkrs[i],fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=r'$\phi$='+f'{phi}', zorder=120)
            
            ax[z,w].set_title(f"{fuel}/air ({P}atm)",fontsize=8)
            ax[z,w].tick_params(axis='both',direction='in')
            ax[len(models[model]['P_list'])-1,w].set_xlabel('Temperature [K]')
        ax[z,0].set_ylabel(r'Ignition delay [ms]')
    ax[len(models[model]['P_list'])-1,len(models[model]['fuels'])-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)  
    path=f'USSCI/figures/'+args.date+'/IDT'
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+f'/{model}.png', dpi=500, bbox_inches='tight')
    print(f'Simulation has been stored at {path}/{model}.png\n')

#     /home/pjs/simulations/USSCI/graph-reading/Stagni-2020/20bar_0,5phi.csv
# 'USSCI/graph-reading/Stagni-2020/20bar_0.5phi.csv