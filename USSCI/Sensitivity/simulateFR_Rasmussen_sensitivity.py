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
title=f'FR:\n518ppm CO/\n453ppm H2/\n1.53% O2/\n36ppm NO2/\n113ppm NO/N2'
folder='Rasmussen-2008'
name='Fig8_sensitivity'
observables=['CO', 'NO', 'NO2']
cutoff=10
threshold=0.01
override=False

X1=518e-6
X2=453e-6
X3=0.0153
X4=36e-6
X5=113e-6
Xdil=1-X1-X2-X3-X4-X5
X={'CO':X1,'H2':X2,'O2':X3,'NO2':X4,'NO':X5,'N2':Xdil}
P=20/1.01325
T=800
Xlim=[0,0.3]
Ylim=[[-0.5,0.5],[-17,17],[-0.75,0.75]]
length = 50e-2  # [m]
diameter=0.008 # [m]
n_steps = 100
Q_std = 3000 #nominal gas flow rate @ STP [mL/min]

models = {
    # 'Glarborg-2025': {
    #     'submodels': {
    #         # 'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
    #         # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR.yaml",
    #         # 'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allPLOG.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allP.yaml",
    #                 },
    # },
    'Glarborg-2025-original': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2025-original/glarborg-2025-original.yaml",
            # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-original_LMRR.yaml",
            # 'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-original_LMRR_allPLOG.yaml",
            # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-original_LMRR_allP.yaml",
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted", "dashdot"]*6
# colors = ["xkcd:purple","r","xkcd:teal",'orange']*3
# colours=[(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(len(list(models.keys())))]

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getSensitivity(gas):
    Q = Q_std*(1e5/(P*ct.one_atm))*(273.15/T)
    gas.TPX = T, P*ct.one_atm, X
    area = np.pi*(diameter/2)**2
    u0 = Q*1e-6/60/area
    mass_flow_rate = u0 * gas.density * area
    flowReactor = ct.IdealGasConstPressureReactor(gas)
    reactorNetwork = ct.ReactorNet([flowReactor])
    for i in range(len(gas.reaction_equations())):
        flowReactor.add_sensitivity_reaction(i)
    # reactorNetwork.rtol = 1.0e-6
    # reactorNetwork.atol = 1.0e-15
    # reactorNetwork.rtol_sensitivity = 1.0e-6
    # reactorNetwork.atol_sensitivity = 1.0e-6
    # states = ct.SolutionArray(gas, extra=['t']+observables)
    tau=length/u0
    dt = tau/n_steps
    t1 = (np.arange(n_steps) + 1) * dt
    Xlist=[]
    tList=[]
    for n, t_i in enumerate(t1):
        reactorNetwork.advance(t_i)
        Xvec=[]
        for species in observables:
            sens_list = []
            for i in range(len(gas.reaction_equations())):
                sens_list.append(reactorNetwork.sensitivity(species, i))
            Xvec.append(sens_list)
        Xlist.append(Xvec)
        tList.append(t_i)
    Xlist_reduced=[]
    
    for j, species in enumerate(observables):
        dict1={}
        dict2={}
        for k, rxn in enumerate(gas.reaction_equations()):
            list1 = []
            for i, t in enumerate(tList):
                list1.append(Xlist[i][j][k])
            sumList=abs(sum(list1))
            if sumList>threshold:
                dict1[rxn]=list1
                dict2[rxn]=abs(sum(list1))
        # sorted_keys = list(sorted(dict2, key=dict2.get,reverse=True))
        dict2=dict(sorted(dict2.items(), key=lambda item: item[1],reverse=True))
        sorted_keys=list(dict2.keys())
        reduced_dict={}
        for i in range(cutoff):
            reduced_dict[sorted_keys[i]]=dict1[sorted_keys[i]]
        Xlist_reduced.append(reduced_dict)
    return tList,Xlist_reduced


def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    X_history=getSensitivity(gas)
    for z, species in enumerate(observables):
        for key in X_history[1][z].keys():
            # Xi_rxn_history = [item[i] for item in Xi_history]
            # print(Xi_rxn_history)
            data = zip(X_history[0],X_history[1][z][key])
            simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/FR-sensitivity/{m}/{species}/{key}'
            os.makedirs(simOutPath,exist_ok=True)
            save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,len(observables),figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=1)
for j,model in enumerate(models):
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        for z, species in enumerate(observables):
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/FR-sensitivity/{m}/{species}'
            if not os.path.exists(simFile) or override:
                sims=generateData(model,m)
        for z, species in enumerate(observables): 
            ax[z].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/FR-sensitivity/{m}/{species}'  
            for key in os.listdir(simFile):
                simFile2=simFile+f'/{key}/{name}.csv'
                sims=pd.read_csv(simFile2)
                label = f'{key}' if k==0 else None
                ax[z].plot(sims.iloc[:,0],sims.iloc[:,1], color=(np.random.rand(), np.random.rand(), np.random.rand()), linestyle=lstyles[k], linewidth=lw, label=label)
                ax[z].set_ylabel(f'{species} sensitivity')
                # ax[z].set_xlim(Xlim)
                ax[z].set_ylim(Ylim[z])
                ax[z].tick_params(axis='both',direction='in')
                ax[z].legend(fontsize=lgdfsz-1,frameon=False,loc='best', handlelength=lgdw,ncol=1,bbox_to_anchor=(1,1)) 
        ax[1].set_xlabel('Time [s]')
        print('  > Data added to plot')
# ax[1].annotate(f'{title2}', xy=(0.97, 0.1), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz+2)

toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/FR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
