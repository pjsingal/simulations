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
X={'H2':0.04,'H2O':0.2,'AR':0.76}
P=40
T_list = np.linspace(1060,1280,gridsz)
data=['4H2_45CO2_Ar_2pt5bar.csv']
models = {
    # 'Stagni-2023': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Alzueta-2023': {
        'submodels': {
            'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
                    },
    },
    'Glarborg-2018': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allPLOG.yaml",
                    },
    },
    'Merchant-2015': {
        'submodels': {
            'base': r"chemical_mechanisms/Merchant-2015/merchant-2015.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR_allPLOG.yaml",
                    },
    },
    # 'Aramco-3.0': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/AramcoMech30/aramco30.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allPLOG.yaml",
    #                 },
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

def getTimeHistory(X,T,P):
    def ignitionDelay(states, species):
        i_ign = np.gradient(states(species).Y.T[0]).argmax()
        # i_ign = states(species).Y.T[0].argmax()
        # print(np.gradient(states(species).Y.T[0]).argmax())
        return states.t[i_ign]   
    gas.TPX = T, P*ct.one_atm, X
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
    return ignitionDelay(timeHistory, 'o')


# def getIDT(gas,T_list,P):
def getTempHistory(X,T_list,P):
    IDT_list = Parallel(n_jobs=-1)(  # Use all available cores; adjust n_jobs if needed
        delayed(getTimeHistory)(X,T,P)
        for T in T_list
    )
    return np.array(IDT_list)
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight))
tic = time.time()
for j,model in enumerate(models):
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f'Submodel: {m}')
        gas = ct.Solution(list(models[model]['submodels'].values())[k])
        IDT = getTempHistory(X, T_list, P)
        ax.semilogy(T_list, IDT*1e3, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=f'{model}-{m}')#label=f'{m} '+r'$\phi$='+f'{phi}')
        # if fuel in models[model]['data']:
        #     dat = pd.read_csv(f'USSCI/graph-reading/{model}/IDT/{P}bar_{phi}phi.csv',header=None)
        #     ax.semilogy(dat.iloc[:,0],dat.iloc[:,1],mkrs[i],fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=r'$\phi$='+f'{phi}')
ax.set_title(r'Beigzadeh et al.',fontsize=8)
# ax.set_ylim([60,6000])
ax.tick_params(axis='both',direction='in')
ax.set_xlabel('Temperature [K]')
ax.set_ylabel(r'Ignition delay [ms]')
ax.legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=2)  
path=f'USSCI/figures/'+args.date+'/Beigzadeh-2023'
os.makedirs(path,exist_ok=True)
plt.savefig(path+f'/4H2_45CO2_Ar_2pt5bar.png', dpi=500, bbox_inches='tight')
toc = time.time()
print(f'Simulation completed in {toc-tic}s and stored at {path}/4H2_45CO2_Ar_2pt5bar.png\n')

#     /home/pjs/simulations/USSCI/graph-reading/Stagni-2023/20bar_0,5phi.csv
# 'USSCI/graph-reading/Stagni-2023/20bar_0.5phi.csv
