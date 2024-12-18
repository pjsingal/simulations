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
folder='Zhang-2018'
name='Fig10_sensitivity'
observables=['CH4','C2H2','C2H4','C2H6']
cutoff=12
threshold=0.001
override=False
X={'CH3CH2OH':0.002,'O2':0.02,'N2':1-0.002-0.02} #ethanol mixed with o2 and n2 bath
P=10
T_list = np.linspace(775,1100,gridsz)
Xlim=[773,1100]
Ylim=[[-4,6],[-0.25,0.5],[-0.25,0.25],[-0.25,0.25]]
tau=0.07
diameter=0.04 #m
t_max=50
title=r'JSR: 0.2% ethanol/2% O$_2$/N$_2$'+f'\n{P} atm'
models = {
    'ThInK 1.0': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10/think.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/think_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/think_LMRR_allPLOG.yaml",
            # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/think_LMRR_allP.yaml",
                    },
    },
    r'ThInK 1.0 (HO2-PLOG)': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10_HO2plog/think_ho2plog.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR_allPLOG.yaml",
            # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR_allP.yaml",
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted", "dashdot"]*6
# colors = ["xkcd:purple","r","xkcd:teal",'orange']*3
# colours=[(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(len(list(models.keys())))]
V = 4/3*np.pi*(diameter/2)**2 #JSR volume

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getStirredReactor(gas):
    h = 79.5 # heat transfer coefficient W/m2/K
    K = 2e-5 # pressureValveCoefficient
    reactorRadius = (V*3/4/np.pi)**(1/3) # [m3]
    reactorSurfaceArea =4*np.pi*reactorRadius**2 # [m3]
    fuelAirMixtureTank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)
    env = ct.Reservoir(gas)
    reactor = ct.IdealGasReactor(gas, energy='on', volume=V)
    ct.MassFlowController(upstream=fuelAirMixtureTank,
                          downstream=reactor,
                          mdot=reactor.mass/tau)
    ct.Valve(upstream=reactor,
             downstream=exhaust,
             K=K)
    ct.Wall(reactor, env, A=reactorSurfaceArea, U=h)
    return reactor

def getSensitivity_T(gas,T):
    gas.TPX = T, P*ct.one_atm, X
    stirredReactor = getStirredReactor(gas)
    reactorNetwork = ct.ReactorNet([stirredReactor])
    for i in range(len(gas.reaction_equations())):
        stirredReactor.add_sensitivity_reaction(i)
    Xvec=[]
    t=0
    while t<t_max:
        t=reactorNetwork.step()
    for species in observables:
        sens_list = []
        for i in range(len(gas.reaction_equations())):
            sens_list.append(reactorNetwork.sensitivity(species, i))
        Xvec.append(sens_list)
    return Xvec

def getSensitivity(gas):
    Xlist = Parallel(n_jobs=len(T_list))(
        delayed(getSensitivity_T)(gas,T)
        for T in T_list
    )
    Xlist_reduced=[]
    for j, species in enumerate(observables):
        dict1={}
        dict2={}
        for k, rxn in enumerate(gas.reaction_equations()):
            list1 = []
            for i, T in enumerate(T_list):
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
    return T_list,Xlist_reduced


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
            simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/JSR-sensitivity/{m}/{species}/{key}'
            os.makedirs(simOutPath,exist_ok=True)
            save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,len(observables),figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'Model: {model}')
    ax[0].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    ax[1].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    ax[2].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        for z, species in enumerate(observables):
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/JSR-sensitivity/{m}/{species}'
            if not os.path.exists(simFile) or override:
                sims=generateData(model,m)
        for z, species in enumerate(observables): 
            # ax[z].plot(0, 0, '.', color='white',markersize=0.1,label=f'{m}') 
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/JSR-sensitivity/{m}/{species}'  
            for key in os.listdir(simFile):
                simFile2=simFile+f'/{key}/{name}.csv'
                sims=pd.read_csv(simFile2)
                label = f'{key}' if k==0 else None
                ax[z].plot(sims.iloc[:,0],sims.iloc[:,1], color=(np.random.rand(), np.random.rand(), np.random.rand()), linestyle=lstyles[k], linewidth=lw, label=label)
                ax[z].set_ylabel(f'{species} sensitivity')
                ax[z].set_xlim(Xlim)
                ax[z].set_ylim(Ylim[z])
                ax[z].tick_params(axis='both',direction='in')
            
        ax[1].set_xlabel('Temperature [K]')
        print('  > Data added to plot')
# ax[1].annotate(f'{title2}', xy=(0.97, 0.1), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz+2)
plt.suptitle(title,y=0.95)
ax[0].legend(fontsize=lgdfsz-2,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
ax[1].legend(fontsize=lgdfsz-2,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
ax[2].legend(fontsize=lgdfsz-2,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/JSR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
