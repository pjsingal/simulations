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
title1='Jian-2024 using Model of Glarborg-2025'
title2='FR: 890ppm NH3/10% O2/N2'
folder='Jian-2024'
name='Fig6_12'
exp=True
override=False
dataLabel='Jian et al. (2024)'
data=['nh3.csv','no.csv']
observables=['NH3','NO']

X={'NH3':890e-6,'O2':0.1,'N2':1-890e-6-0.1}
P=1.18
T=1279
Xlim=[0,0.3]
Ylim=[0,15]
length = 20e-2  # [m]
diameter=0.0087 # [m]
n_steps = 2000
Q_tn = 1000 #nominal gas flow rate @ STP [mL/min]

models = {
    'Glarborg-2025': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allPLOG.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allP.yaml",
                    },
    },
    # 'Jian-2024': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Jian-2024/jian-2024.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/jian-2024_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/jian-2024_LMRR_allPLOG.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/jian-2024_LMRR_allP.yaml",
    #                 },
    # },
    # 'Klippenstein-CNF2018': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allP.yaml",
    #                 },
    # },
    # 'Stagni-2023': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allP.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allP.yaml",
    #                 },
    # },
    # 'Alzueta-2023': {
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml",
    #                 },
    # },
}
########################################################################################
lstyles = ["solid","dashed","dotted", "dashdot"]*6
colors = ["xkcd:purple","r","xkcd:teal",'orange']*3

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getTimeHistory(gas,T):
    gas.TPX = T, P*ct.one_atm, X
    area = np.pi*(diameter/2)**2
    u0 = Q_tn*1e-6/60/area
    mass_flow_rate = u0 * gas.density * area
    flowReactor = ct.IdealGasConstPressureReactor(gas)
    reactorNetwork = ct.ReactorNet([flowReactor])
    tau=length/u0
    dt = tau/n_steps
    t1 = (np.arange(n_steps) + 1) * dt
    Xlist=[]
    tList=[]
    for n, t_i in enumerate(t1):
        states = ct.SolutionArray(flowReactor.thermo)
        states.append(flowReactor.thermo.state)
        reactorNetwork.advance(t_i)
        Xvec=[states(species).X.flatten()[0] for species in observables]
        Xlist.append(Xvec)
        tList.append(t_i)
    return tList,Xlist

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    X_history=getTimeHistory(gas,T)
    for z, species in enumerate(observables):
        Xi_history = [item[z] for item in X_history[1]]
        data = zip(X_history[0],Xi_history)
        simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/FR/{m}/{species}'
        os.makedirs(simOutPath,exist_ok=True)
        save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
f, ax = plt.subplots(len(observables),1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'Model: {model}')
    ax[0].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        flag=False
        while not flag:
            for z, species in enumerate(observables):
                simFile=f'USSCI/data/{args.date}/{folder}/{model}/FR/{m}/{species}/{name}.csv'
                if not os.path.exists(simFile) or override:
                    sims=generateData(model,m) 
                    flag=True
            flag=True
        for z, species in enumerate(observables):  
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/FR/{m}/{species}/{name}.csv' 
            sims=pd.read_csv(simFile)
            label = f'{m}'
            ax[z].plot(sims.iloc[:,0],sims.iloc[:,1]*1e6, color=colors[k], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[z].set_ylabel(f'X-{species} [-]')
            if exp and j==len(models)-1 and k==len(models[model]['submodels'])-1:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
                ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[z].set_xlim(Xlim)
            # ax[z].set_ylim(Ylim)
            ax[z].tick_params(axis='both',direction='in')
        ax[1].set_xlabel('Time [s]')
        print('  > Data added to plot')
# ax[1].annotate(f'{title2}', xy=(0.97, 0.1), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz+2)
ax[0].legend(fontsize=lgdfsz-1,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/FR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
