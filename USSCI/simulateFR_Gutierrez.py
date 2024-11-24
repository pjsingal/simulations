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
models = {
    # 'Stagni-2023': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Alzueta-2023': {
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Gutierrez-2025': {
        'submodels': {
            'base': r"chemical_mechanisms/Gutierrez-2025/gutierrez-2025.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/gutierrez-2025_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/gutierrez-2025_LMRR_allPLOG.yaml",
                    },
    },
    # 'Arunthanayothin-2021': {
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Glarborg-2018': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Merchant-2015': {
        'submodels': {
            'base': r"chemical_mechanisms/Merchant-2015/merchant-2015.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR_allPLOG.yaml",
                    },
    },
    # 'Bugler-2016': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Bugler-2016/bugler-2016.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Song-2019': {
        'submodels': {
            'base': r"chemical_mechanisms/Song-2019/song-2019.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR_allPLOG.yaml",
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

P=1
X={'NH3':923e-6,'CH3OCH3':943e-6,'O2':3855e-6,'Ar':1-923e-6-943e-6-3855e-6}
# T_list = np.linspace(999,1001,gridsz)
T_list = np.linspace(900,1300,gridsz)
data=['XCH4_90CH4_10NH3.csv','XNO_90CH4_10NH3.csv']
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*3

Xspecies=['NH3','CH3OCH3']

########################################################################################

# length = 200e-3  # *approximate* PFR length [m]
length = 20e-2  # *approximate* PFR length [m]
diameter=0.0087 # PFR diameter [m]
n_steps = 2000
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
Q_tn = 1000 #nominal gas flow rate @ STP [mL/min]

def getXvsT(T):
    gas = ct.Solution(list(models[model]['submodels'].values())[k])
    gas.TPX = T, P*ct.one_atm, X
    area = np.pi*(diameter/2)**2
    u0 = Q_tn*1e-6/60/area
    n_steps=2000
    mass_flow_rate = u0 * gas.density * area
    flowReactor = ct.IdealGasConstPressureReactor(gas)
    reactorNetwork = ct.ReactorNet([flowReactor])
    # tau=192.097*P*ct.one_atm/1e5*1000/Q_tn/T #residence time formula from Gutierrez-2025
    tau=length/u0
    dt = tau/n_steps
    t1 = (np.arange(n_steps) + 1) * dt
    states = ct.SolutionArray(flowReactor.thermo)
    for n, t_i in enumerate(t1):
        reactorNetwork.advance(t_i)
    # print(f'{t1[-1]}={tau}')
    states.append(flowReactor.thermo.state)
    Xvec=[]
    for species in Xspecies:
        Xvec.append(states(species).X.flatten()[0])
    # print(f'{Xspecies[0]}:{Xvec[0]:.8e}, {Xspecies[1]}:{Xvec[1]:.8e}')
    return Xvec

f, ax = plt.subplots(1,len(Xspecies), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
tic = time.time()
for j,model in enumerate(models):
    print(f'\nModel: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f'Submodel: {m}')
        # gas = ct.Solution(list(models[model]['submodels'].values())[k])
        # X_history = Parallel(n_jobs=-1)(
        #     delayed(getXvsT)(gas,T)
        #     for T in T_list
        # )
        X_history=[]
        for T in T_list:
            X_history.append(getXvsT(T))
        for z, species in enumerate(Xspecies):
            Xi_history = [item[z]*1e6 for item in X_history]
            if k==0:
                label=f'{model}'
            else:
                label=None
            if species=='NO':
                ax[z].plot(T_list,Xi_history, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
                ax[z].set_ylabel(f'X-{species} [ppm]')
            else:
                ax[z].plot(T_list,Xi_history, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
                ax[z].set_ylabel(f'X-{species} [ppm]')
        # if j==len(list(models.keys()))-1:
        #     dat = pd.read_csv(f'USSCI/graph-reading/Manna-2024/{data[z]}',header=None)
        #     ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Lavadera et al.')
        # ax[z].set_xlim([700,1200])
        ax[z].tick_params(axis='both',direction='in')
        ax[z].set_xlabel('Temperature [K]')
plt.suptitle(r'Jet-stirred reactor: (90% CH4/10% NH3)/O2/N2, (1.16atm, phi=0.8)',fontsize=10)
ax[len(Xspecies)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1)  
path=f'USSCI/figures/'+args.date+'/Gutierrez-2025'
os.makedirs(path,exist_ok=True)
name=f'Fig10.png'
plt.savefig(f'{path}/{name}', dpi=500, bbox_inches='tight')
toc = time.time()
print(f'Simulation completed in {toc-tic}s and stored at {path}/{name}\n')